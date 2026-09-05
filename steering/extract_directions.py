"""One-shot direction extractor (CPU) — SPRINT_oneshot_redesign.md §6.

Reads:   the substrate given by --substrate-root (released layout:
         $SCA_DATA_ROOT/substrate/{model}/{game}/{results.parquet, acts.npz})
         data/games/game_features.csv
Writes:  <SCA_DATA_ROOT>/steering/directions/{model}/directions.npz
         <SCA_DATA_ROOT>/steering/directions/{model}/manifest.json

Directions per capture layer L:
  REQUIRED-ESTIMABLE (abort at a steer layer if zero/non-finite/missing):
    d_inc_l{L}            univariate covariance E[X*delta1_c] (sign-aligned), P1-baseline
    d_trait_l{L}_{trait}  paired-within-(game,cb) mean(X[cue] - X[baseline])
  GATED (may be NOT_ESTIMABLE — skip, never abort):
    d_choice_l{L}         cross-game diff-of-means toward y=1[decoded==canonical_p1]
    d_choice_perp_l{L}    d_choice ⟂ d_inc (the steering vector; zero-norm gate)
    d_opp_l{L}            P1-baseline residual -> sign(delta2_c), nuisance-controlled
  ALWAYS:
    d_random_l{L}         deterministic random unit vector

d_gate (H0) is NOT written here — it is rebuilt per prompt at run time from the
unembedding matrix.

delta1_c / delta2_c use PLAYER-SPECIFIC empirical q-hat from the NEW one-shot
baselines (a player's q-hat is its opponent's act0 rate). No old behavioral data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from strategic_anatomy.config import (
    games_root,
    manifests_root,
    repo_root,
    results_root,
    steering_root,
    substrate_root,
)
from collection.oneshot_common import (  # noqa: E402
    canonical_sign,
    delta1,
    delta2,
    deterministic_random_unit_vector,
    load_game_vec,
)

# Verbatim from analysis/block_b/_common.py:61 (private repo, commit 1f47050).
# Inlined rather than imported: _common.py is excluded from the release (plan §5) and
# creates four output directories as an import-time side effect.
HIDDEN = {"qwen": 8192, "qwen_instruct": 8192, "llama31_instruct": 8192, "gptoss": 2880}

ROOT = repo_root()
GAME_FEATURES = games_root() / "game_features.csv"
CONFIG_PATH = manifests_root() / "oneshot_config.json"
DIR_ROOT = steering_root() / "directions"
SUMMARY_DIR = results_root() / "steering" / "summary"

MODELS = ("qwen", "qwen_instruct", "llama31_instruct", "gptoss")
TRAITS = ("risk_aversion", "loss_aversion", "inequity_aversion", "maximin", "selfish_maximizer")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False, cwd=str(ROOT)).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        raise ValueError("zero or non-finite vector")
    return (v / n).astype(np.float32)


def _vector_sha256(v: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(v, dtype=np.float32).tobytes())
    return h.hexdigest()


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for s in pos:
        wins += float((neg < s).sum()) + 0.5 * float((neg == s).sum())
    return wins / (len(pos) * len(neg))


def _groupkfold_by_game(games: np.ndarray, k: int = 5, seed: int = 0):
    uniq = np.unique(games)
    K = max(2, min(k, len(uniq)))
    perm = uniq.copy()
    np.random.default_rng(seed).shuffle(perm)
    for fold in np.array_split(perm, K):
        if len(fold):
            yield np.isin(games, fold)


# ---------------------------------------------------------------------------
# NOTE (phase-4): the ported default named the superseded A/B-matrix substrate
# (`output/oneshot_main/substrate`), which is excluded from the release and therefore
# resolved to a path that cannot exist in a clone — `_iter_games` silently yielded
# nothing. Repointed to the released Akata substrate (approved 2026-08-16, resolves
# phase-2 open question 10). The flag name and semantics are unchanged; only the value
# a bare invocation resolves to moved, from the private-repo layout to the deposit layout.
SUBSTRATE_ROOT = substrate_root()   # override via --substrate-root


def _iter_games(model: str):
    base = SUBSTRATE_ROOT / model
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name != "_tmp" and (d / "_DONE").exists():
            yield d


def load_behavior(model: str) -> pd.DataFrame:
    dfs = [pd.read_parquet(d / "results.parquet") for d in _iter_games(model)]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_gen_moves(moves_root: str, model: str) -> pd.DataFrame:
    """Concatenate finalized generation-move tables (corrected decoder) for `model`."""
    base = Path(moves_root) / model
    if not base.exists():
        raise SystemExit(f"--moves-root has no '{model}' dir: {base}")
    dfs = [pd.read_parquet(d / "moves.parquet") for d in sorted(base.iterdir())
           if d.is_dir() and d.name != "_tmp" and (d / "_DONE").exists()]
    if not dfs:
        raise SystemExit(f"--moves-root {base}: no finalized (_DONE) games")
    return pd.concat(dfs, ignore_index=True)


def gen_qhat_series(mv: pd.DataFrame, player: int) -> pd.Series:
    """Per-game action-0 rate from the committed generation move (parse_ok baseline only)."""
    sub = mv[(mv["player"] == player) & (mv["condition"] == "baseline") & (mv["parse_ok"])]
    return sub.groupby("game_code")["parsed_action"].apply(lambda s: float((s == 0).mean()))


def gen_action_map(mv: pd.DataFrame) -> dict:
    """(game_code, cb, player, condition) -> committed parsed_action, parse_ok rows only."""
    ok = mv[mv["parse_ok"]]
    return {(r.game_code, int(r.cb_id), int(r.player), r.condition): int(r.parsed_action)
            for r in ok.itertuples()}


def verify_prompt_alignment(model: str, mv: pd.DataFrame) -> dict:
    """HARD safety gate: the generation moves and the residual substrate MUST be on the same
    prompt. Compare prompt_hash per (game, cb, player, condition); abort on ANY mismatch
    (e.g. chat moves paired with a raw substrate -> residual/label pairing would be invalid)."""
    sub_hash = {}
    for d in _iter_games(model):
        df = pd.read_parquet(d / "results.parquet")
        if "prompt_hash" not in df.columns:
            return {"checked": 0, "mismatch": 0, "note": "substrate lacks prompt_hash; unverified"}
        for _, r in df.iterrows():
            sub_hash[(r["game_code"], int(r["counterbalance_id"]), int(r["player"]),
                      r["condition"])] = r["prompt_hash"]
    if "prompt_hash" not in mv.columns:
        return {"checked": 0, "mismatch": 0, "note": "moves lack prompt_hash; unverified"}
    checked = mismatch = 0
    examples = []
    for r in mv.itertuples():
        h = sub_hash.get((r.game_code, int(r.cb_id), int(r.player), r.condition))
        if h is None:
            continue
        checked += 1
        if h != r.prompt_hash:
            mismatch += 1
            if len(examples) < 5:
                examples.append((r.game_code, int(r.cb_id), int(r.player), r.condition))
    if mismatch:
        raise SystemExit(
            f"[{model}] PROMPT MISALIGNMENT {mismatch}/{checked}: moves.prompt_hash != "
            f"substrate.prompt_hash — the generation moves were produced on a DIFFERENT prompt "
            f"than the captured residuals; pairing them is invalid. Examples: {examples}. "
            f"(llama: recapture the residual substrate with --chat before extracting.)")
    return {"checked": int(checked), "mismatch": 0}


def build_delta_tables(model: str, canon1: dict, canon2: dict,
                       gen_qhat_p1=None, gen_qhat_p2=None) -> pd.DataFrame:
    """Per-game canonical-signed incentive gaps with player-specific hard q-hat (§6).

    With gen_qhat_* (from the corrected generation moves) q̂ is the committed move's act0
    rate; otherwise it falls back to the substrate slot-argmax `decoded_action` (legacy)."""
    if gen_qhat_p1 is not None:
        qhat_p1, qhat_p2 = gen_qhat_p1, gen_qhat_p2
    else:
        beh = load_behavior(model)
        if beh.empty:
            return pd.DataFrame()
        p1b = beh[(beh["player"] == 1) & (beh["condition"] == "baseline")]
        p2b = beh[(beh["player"] == 2) & (beh["condition"] == "baseline")]
        qhat_p1 = p1b.groupby("game_code")["decoded_action"].apply(lambda s: float((s == 0).mean()))
        qhat_p2 = p2b.groupby("game_code")["decoded_action"].apply(lambda s: float((s == 0).mean()))
    rows = []
    for g in sorted(set(qhat_p1.index) & set(qhat_p2.index)):
        vec = load_game_vec(g)
        c1 = canon1.get(g); c2 = canon2.get(g)
        d1 = delta1(vec, qhat_p2[g])              # P1's incentive; belief = P2 act0 rate
        d2 = delta2(vec, qhat_p1[g])              # P2's incentive; belief = P1 act0 rate
        d1c = canonical_sign(d1, c1) if c1 is not None else np.nan
        d2c = canonical_sign(d2, c2) if c2 is not None else np.nan
        rows.append({"game_code": g, "q_hat_p1": qhat_p1[g], "q_hat_p2": qhat_p2[g],
                     "delta1_emp": d1, "delta1_c": d1c, "delta2_emp": d2, "delta2_c": d2c,
                     "canonical_action_p1": -1 if c1 is None else int(c1),
                     "canonical_action_p2": -1 if c2 is None else int(c2)})
    return pd.DataFrame(rows)


def _load_layer(model: str, layer: int, want, canon1: dict, action_by_key=None):
    """want: iterable of (player, condition). Returns {(player,condition):(X, meta_df)}.

    With action_by_key (corrected generation moves), per-row `decoded_action` is the
    committed move (-1 when missing/unparsed -> excluded from d_choice only); otherwise it
    is the substrate slot-argmax (legacy). Residuals (X) are untouched in both paths."""
    buckets = {pc: ([], []) for pc in want}
    for d in _iter_games(model):
        df = pd.read_parquet(d / "results.parquet")
        z = np.load(d / "acts.npz")
        try:
            for (player, condition) in want:
                sub = df[(df["player"] == player) & (df["condition"] == condition)]
                for _, r in sub.iterrows():
                    cb = int(r["counterbalance_id"])
                    key = f"p{player}_{condition}_cb{cb}_l{layer}"
                    if key not in z.files:
                        continue
                    g = r["game_code"]
                    if action_by_key is not None:
                        da = int(action_by_key.get((g, cb, player, condition), -1))
                    else:
                        da = int(r["decoded_action"])
                    buckets[(player, condition)][0].append(np.asarray(z[key], dtype=np.float32))
                    buckets[(player, condition)][1].append({
                        "game_code": g, "counterbalance_id": cb,
                        "decoded_action": da,
                        "canonical_action_p1": int(canon1.get(g, -1)),
                    })
        finally:
            z.close()
    out = {}
    for pc, (Xs, ms) in buckets.items():
        X = np.vstack(Xs).astype(np.float32) if Xs else np.zeros((0, 0), np.float32)
        out[pc] = (X, pd.DataFrame(ms).reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
def fit_d_inc(X: np.ndarray, meta: pd.DataFrame, d1c_by_game: dict) -> tuple[np.ndarray, dict]:
    y = meta["game_code"].map(d1c_by_game).to_numpy(dtype=float)
    keep = np.isfinite(y)
    Xs, ys = X[keep], y[keep]
    if len(Xs) == 0:
        raise ValueError("d_inc: no rows with finite delta1_c")
    # CENTERED covariance Cov(X, delta1c): the incentive axis is the direction along which the
    # incentive signal VARIES. The old uncentered E[X*delta1c] = Cov(X,delta1c) + mean(X)*mean(delta1c);
    # since delta1c is canonical-signed (nonzero mean) that second term injects the mean-activation
    # direction and pushes the axis ~80-90deg off d_dec. Centering isolates the true incentive axis and
    # MATCHES the Layer B geometry d_inc, so geometry == steered axis (the causal-geometry bridge).
    Xc = Xs - Xs.mean(axis=0)
    yc = ys - ys.mean()
    direction = (Xc * yc[:, None]).mean(axis=0)
    corr = float(np.corrcoef(Xs @ direction, ys)[0, 1]) if len(Xs) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    if corr < 0:
        direction = -direction
    dn = float(np.linalg.norm(direction))
    if dn == 0.0 or not np.isfinite(dn):
        # degenerate layer (e.g. L0 embedding of the shared slot token -> zero variance): flag, don't crash.
        return np.zeros_like(direction, dtype=np.float32), {
            "definition": "centered_covariance_Cov(X,delta1c)_sign_aligned_p1_baseline",
            "n_rows": int(keep.sum()), "n_games": int(meta.loc[keep, "game_code"].nunique()),
            "status": "NOT_ESTIMABLE", "reason": "zero/non-finite incentive direction (degenerate layer)",
        }
    return _unit(direction), {
        "definition": "centered_covariance_Cov(X,delta1c)_sign_aligned_p1_baseline",
        "n_rows": int(keep.sum()), "n_games": int(meta.loc[keep, "game_code"].nunique()),
        "sign_aligned_correl": float(abs(corr)), "counterbalance_policy": "akata_4cell",
        "status": "OK",
    }


def fit_d_choice(X: np.ndarray, meta: pd.DataFrame, min_games_per_class: int,
                 min_auc: float, seed: int) -> tuple[np.ndarray, dict]:
    m = meta[(meta["canonical_action_p1"] >= 0) & (meta["decoded_action"] >= 0)].copy()
    Xs = X[m.index.to_numpy()]
    y = (m["decoded_action"].to_numpy() == m["canonical_action_p1"].to_numpy()).astype(int)
    diag = {"definition": "cross_game_diff_of_means_to_canonical_choice_H2a",
            "counterbalance_policy": "factorial_16", "n_rows": int(len(m))}
    g_pos = m.loc[y == 1, "game_code"].nunique()
    g_neg = m.loc[y == 0, "game_code"].nunique()
    diag.update({"n_games_class1": int(g_pos), "n_games_class0": int(g_neg)})
    if g_pos < min_games_per_class or g_neg < min_games_per_class:
        return np.zeros(X.shape[1], np.float32), {
            **diag, "status": "NOT_ESTIMABLE", "auc_groupkfold": float("nan"),
            "reason": f"games/class ({g_pos},{g_neg}) < {min_games_per_class}"}
    games = m["game_code"].to_numpy()
    # Leak-free GroupKFold: fit diff-of-means on the TRAIN games only, score held-out games.
    aucs = []
    for held in _groupkfold_by_game(games, seed=seed):
        tr = ~held
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[held])) < 2:
            continue
        d_tr = _unit(Xs[tr][y[tr] == 1].mean(0) - Xs[tr][y[tr] == 0].mean(0))
        a = _auc(Xs[held] @ d_tr, y[held])
        if np.isfinite(a):
            aucs.append(a)
    auc = float(np.mean(aucs)) if aucs else float("nan")
    diag["auc_groupkfold"] = auc
    if not np.isfinite(auc) or auc < min_auc:
        return np.zeros(X.shape[1], np.float32), {
            **diag, "status": "NOT_ESTIMABLE", "reason": f"AUC {auc:.3f} < {min_auc}"}
    # Gate passed -> refit the stored direction on ALL games.
    direction = _unit(Xs[y == 1].mean(0) - Xs[y == 0].mean(0))
    diag["status"] = "OK"
    return direction, diag


def make_d_choice_perp(d_choice: np.ndarray, d_inc: np.ndarray, min_perp_ratio: float):
    """Per-layer d_choice ⟂ d_inc with a zero-norm gate (§6)."""
    nc = float(np.linalg.norm(d_choice))
    if nc == 0.0 or not np.isfinite(nc):
        return np.zeros_like(d_choice), {"status": "NOT_ESTIMABLE", "reason": "d_choice unavailable"}
    proj = float(np.dot(d_choice, d_inc)) * d_inc
    perp = d_choice - proj
    ratio = float(np.linalg.norm(perp) / nc)
    if ratio < min_perp_ratio:
        return np.zeros_like(d_choice), {
            "status": "NOT_ESTIMABLE", "reason": f"perp_ratio {ratio:.3f} < {min_perp_ratio}",
            "perp_ratio": ratio, "cos_with_d_inc": float(np.dot(d_choice, d_inc))}
    return _unit(perp), {"status": "OK", "perp_ratio": ratio,
                         "cos_with_d_inc": float(np.dot(d_choice, d_inc)),
                         "definition": "unit(d_choice - proj_onto(d_inc))"}


def fit_d_trait(X_base, M_base, X_cue, M_cue, trait: str) -> tuple[np.ndarray, dict]:
    """Paired-within-(game, cb) mean(X[cue] - X[baseline])."""
    if len(M_cue) == 0 or len(M_base) == 0:
        return np.zeros(X_base.shape[1] if X_base.size else 1, np.float32), {
            "definition": f"paired_within_game_cb_cue_minus_baseline_{trait}",
            "status": "NOT_ESTIMABLE", "reason": "missing cue/baseline rows", "n_games": 0}
    base_idx = {(r.game_code, r.counterbalance_id): i for i, r in M_base.iterrows()}
    deltas, games = [], set()
    for i, r in M_cue.iterrows():
        j = base_idx.get((r.game_code, r.counterbalance_id))
        if j is None:
            continue
        deltas.append(X_cue[i] - X_base[j])
        games.add(r.game_code)
    if not deltas:
        return np.zeros(X_base.shape[1], np.float32), {
            "definition": f"paired_within_game_cb_cue_minus_baseline_{trait}",
            "status": "NOT_ESTIMABLE", "reason": "no matched (game,cb) pairs", "n_games": 0}
    return _unit(np.mean(np.stack(deltas), axis=0)), {
        "definition": f"paired_within_game_cb_cue_minus_baseline_{trait}",
        "status": "OK", "n_pairs": len(deltas), "n_games": len(games),
        "counterbalance_policy": "factorial_16", "trait": trait}


def _opp_covariates(meta: pd.DataFrame, dom1: dict) -> np.ndarray:
    """P1-side, features-only nuisance covariates (with intercept)."""
    g = meta["game_code"]
    d1c = meta["delta1_c"].to_numpy(dtype=float)
    ca1 = meta["canonical_action_p1"].to_numpy(dtype=float)
    dom = g.map(lambda x: dom1.get(x, -1)).to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(meta)), d1c, ca1,
        (dom == 0).astype(float),   # dominant action is act0 (reference = no dominant)
        (dom == 1).astype(float),   # dominant action is act1
    ])


def fit_d_opp(X, meta, dom1: dict, min_auc: float, seed: int) -> tuple[np.ndarray, dict]:
    """P1-baseline residual -> sign(delta2_c); train-fold features-only nuisance
    residualization; GroupKFold-by-game AUC gate; then refit on all games (§6)."""
    m = meta[np.isfinite(meta["delta2_c"].to_numpy(dtype=float))].copy().reset_index(drop=True)
    diag = {"definition": "p1_baseline_residual_to_sign_delta2c_nuisance_controlled",
            "nuisance_controls": ["delta1_c", "canonical_action_p1", "dominant_action_p1"],
            "counterbalance_policy": "factorial_16", "n_rows": int(len(m))}
    if len(m) == 0:
        return np.zeros(X.shape[1], np.float32), {**diag, "status": "NOT_ESTIMABLE",
                                                  "reason": "no rows with finite delta2_c",
                                                  "auc_groupkfold": float("nan")}
    Xm = X[m["_xrow"].to_numpy()]
    y = (m["delta2_c"].to_numpy(dtype=float) > 0).astype(int)
    games = m["game_code"].to_numpy()
    diag["n_games"] = int(pd.Series(games).nunique())
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], np.float32), {**diag, "status": "NOT_ESTIMABLE",
                                                  "reason": "single sign(delta2_c) class",
                                                  "auc_groupkfold": float("nan")}
    C = _opp_covariates(m, dom1)
    aucs = []
    for held in _groupkfold_by_game(games, seed=seed):
        tr = ~held
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[held])) < 2:
            continue
        beta, *_ = np.linalg.lstsq(C[tr], Xm[tr], rcond=None)  # nuisance fit on TRAIN only
        Xtr = Xm[tr] - C[tr] @ beta
        Xte = Xm[held] - C[held] @ beta
        dirn = _unit(Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0))
        a = _auc(Xte @ dirn, y[held])
        if np.isfinite(a):
            aucs.append(a)
    auc = float(np.mean(aucs)) if aucs else float("nan")
    diag["auc_groupkfold"] = auc
    if not np.isfinite(auc) or auc < min_auc:
        return np.zeros(X.shape[1], np.float32), {**diag, "status": "NOT_ESTIMABLE",
                                                  "reason": f"AUC {auc:.3f} < {min_auc}"}
    # Gate passed -> refit final steering direction on ALL games (nuisance fit on all).
    beta_all, *_ = np.linalg.lstsq(C, Xm, rcond=None)
    Xres = Xm - C @ beta_all
    direction = _unit(Xres[y == 1].mean(0) - Xres[y == 0].mean(0))  # +dose -> opp toward canonical
    diag["status"] = "OK"
    return direction, diag


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--out-root", default=str(DIR_ROOT))
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--moves-root", default="",
                   help="re-point q̂ + d_choice to the corrected generation moves under "
                        "{moves-root}/{model}/{game}/moves.parquet (SPEC_oneshot_FINAL §9). "
                        "Default empty = legacy substrate slot-argmax.")
    p.add_argument("--substrate-root", default="",
                   help="override the substrate base dir (default: $SCA_DATA_ROOT/substrate, the "
                        "released corrected substrate). Its decoded_action is ALREADY the "
                        "generate->parse J/P decision, so --moves-root is not needed.")
    args = p.parse_args()
    if args.substrate_root:
        global SUBSTRATE_ROOT
        SUBSTRATE_ROOT = Path(args.substrate_root)

    cfg = json.loads(Path(args.config).read_text())
    h2, h3 = cfg.get("h2", {}), cfg.get("h3", {})
    min_games_per_class = int(h2.get("min_games_per_class", 10))
    min_auc_h2 = float(h2.get("min_groupkfold_auc", 0.55))
    min_perp_ratio = float(h2.get("min_perp_ratio", 0.10))
    min_auc_h3 = float(h3.get("min_groupkfold_auc", 0.55))
    boot_seed = int(cfg.get("bootstrap", {}).get("seed", 0))

    feats = pd.read_csv(GAME_FEATURES)
    canon1 = {r.game_code: int(r.canonical_action_p1) for r in feats.itertuples()
              if pd.notna(r.canonical_action_p1)}
    canon2 = {r.game_code: int(r.canonical_action_p2) for r in feats.itertuples()
              if pd.notna(r.canonical_action_p2)}
    dom1 = {r.game_code: int(r.dominant_action_p1) for r in feats.itertuples()
            if pd.notna(r.dominant_action_p1)}

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    overall = {}

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        capture_layers = list(cfg["models"][model]["capture_layers"])
        steer_layers = list(cfg["models"][model]["steer_layers"])
        hidden = int(HIDDEN[model])
        mdir = out_root / model
        mdir.mkdir(parents=True, exist_ok=True)

        n_games_done = sum(1 for _ in _iter_games(model))
        if args.preflight_only:
            overall[model] = {"games_done": n_games_done, "capture_layers": capture_layers}
            continue
        if n_games_done == 0:
            overall[model] = {"status": "NO_SUBSTRATE", "games_done": 0}
            print(f"[{model}] no finalized substrate — skipping")
            continue

        gen_qhat_p1 = gen_qhat_p2 = action_by_key = None
        decoder_info = {"decoder_source": "substrate_slot_argmax"}
        if args.moves_root:
            mv = load_gen_moves(args.moves_root, model)
            decoder_info = {"decoder_source": "generation_moves",
                            "moves_root": str(Path(args.moves_root) / model)}
            decoder_info.update(verify_prompt_alignment(model, mv))
            gen_qhat_p1 = gen_qhat_series(mv, 1)
            gen_qhat_p2 = gen_qhat_series(mv, 2)
            action_by_key = gen_action_map(mv)
            if gen_qhat_p1.empty or gen_qhat_p2.empty:
                raise SystemExit(f"[{model}] --moves-root has no parsed baseline moves "
                                 f"(p1={gen_qhat_p1.size}, p2={gen_qhat_p2.size})")
            decoder_info.update({"gen_games_p1_baseline": int(gen_qhat_p1.size),
                                 "gen_games_p2_baseline": int(gen_qhat_p2.size)})
            print(f"[{model}] decoder=generation_moves; prompt_alignment "
                  f"{decoder_info.get('checked')} checked / {decoder_info.get('mismatch')} mismatch; "
                  f"q-hat games p1={gen_qhat_p1.size} p2={gen_qhat_p2.size}")
        deltas = build_delta_tables(model, canon1, canon2, gen_qhat_p1, gen_qhat_p2)
        deltas.to_csv(SUMMARY_DIR / f"delta_tables_{model}.csv", index=False)
        d1c_by_game = dict(zip(deltas["game_code"], deltas["delta1_c"]))
        d2c_by_game = dict(zip(deltas["game_code"], deltas["delta2_c"]))

        directions, diag = {}, {}
        for L in capture_layers:
            want = [(1, "baseline")] + [(1, f"cue_{t}") for t in TRAITS]
            loaded = _load_layer(model, L, want, canon1, action_by_key)
            Xb, Mb = loaded[(1, "baseline")]
            if len(Mb) == 0:
                raise RuntimeError(f"[{model}] layer {L}: empty P1 baseline residual matrix")

            # Degenerate-layer guard: L0 is the embedding of the shared slot token -> zero residual
            # variance -> no estimable axis. Flag NOT_ESTIMABLE (zeros) and skip; never a steer layer.
            if float(Xb.std(axis=0).max()) < 1e-8:
                z = np.zeros(Xb.shape[1], dtype=np.float32)
                ne = {"status": "NOT_ESTIMABLE",
                      "reason": "degenerate layer (zero residual variance, e.g. L0 slot embedding)"}
                for key in ([f"d_inc_l{L}", f"d_choice_l{L}", f"d_choice_perp_l{L}", f"d_opp_l{L}"]
                            + [f"d_trait_l{L}_{t}" for t in TRAITS]):
                    directions[key], diag[key] = z, dict(ne)
                continue

            # d_inc (required)
            v, d = fit_d_inc(Xb, Mb, d1c_by_game)
            directions[f"d_inc_l{L}"], diag[f"d_inc_l{L}"] = v, d

            # d_choice (gated) + d_choice_perp (steering vector)
            vc, dc = fit_d_choice(Xb, Mb, min_games_per_class, min_auc_h2, boot_seed)
            directions[f"d_choice_l{L}"], diag[f"d_choice_l{L}"] = vc, dc
            vperp, dperp = make_d_choice_perp(vc, v, min_perp_ratio)
            directions[f"d_choice_perp_l{L}"], diag[f"d_choice_perp_l{L}"] = vperp, dperp

            # d_trait_{trait} (required)
            for t in TRAITS:
                Xc, Mc = loaded[(1, f"cue_{t}")]
                vt, dt_ = fit_d_trait(Xb, Mb, Xc, Mc, t)
                directions[f"d_trait_l{L}_{t}"], diag[f"d_trait_l{L}_{t}"] = vt, dt_

            # d_opp (gated): attach per-row delta1_c/delta2_c + an X-row index, then fit
            Mb2 = Mb.copy()
            Mb2["_xrow"] = np.arange(len(Mb2))
            Mb2["delta1_c"] = Mb2["game_code"].map(d1c_by_game)
            Mb2["delta2_c"] = Mb2["game_code"].map(d2c_by_game)
            vo, do = fit_d_opp(Xb, Mb2, dom1, min_auc_h3, boot_seed)
            directions[f"d_opp_l{L}"], diag[f"d_opp_l{L}"] = vo, do

            # d_random
            directions[f"d_random_l{L}"] = deterministic_random_unit_vector(
                hidden, seed_keys=(model, L, "d_random"))
            diag[f"d_random_l{L}"] = {"definition": "deterministic_random_unit_vector", "status": "OK"}
            print(f"[{model}] L{L}: d_inc✓ d_choice={dc['status']} perp={dperp['status']} "
                  f"d_opp={do['status']}")

        # Abort gate: REQUIRED directions (d_inc, d_trait_*) must be OK at every steer layer.
        required_fail = []
        for L in steer_layers:
            for key in [f"d_inc_l{L}"] + [f"d_trait_l{L}_{t}" for t in TRAITS]:
                vnorm = float(np.linalg.norm(directions[key]))
                if diag[key].get("status") != "OK" or vnorm == 0 or not np.isfinite(vnorm):
                    required_fail.append(key)
        if required_fail:
            raise RuntimeError(f"[{model}] REQUIRED directions degenerate at steer layers: {required_fail}")

        np.savez_compressed(mdir / "directions.npz",
                            layers=np.array(capture_layers, dtype=np.int32),
                            steer_layers=np.array(steer_layers, dtype=np.int32),
                            **{k: v.astype(np.float32) for k, v in directions.items()})
        manifest = {
            "git_commit": _git_commit(), "created_at": _now(), "model": model,
            "capture_layers": capture_layers, "steer_layers": steer_layers, "hidden_dim": hidden,
            "n_games_substrate": n_games_done, "direction_keys": sorted(directions),
            "data_root": str(SUBSTRATE_ROOT / model),
            "decoder": decoder_info,
            "per_direction": diag,
            "vector_norms": {k: float(np.linalg.norm(v)) for k, v in directions.items()},
            "vector_sha256": {k: _vector_sha256(v) for k, v in directions.items()},
        }
        (mdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        overall[model] = {"status": "OK", "n_directions": len(directions),
                          "npz": str(mdir / "directions.npz")}
        print(f"[{model}] wrote {len(directions)} directions")

    (out_root / "extract_summary.json").write_text(json.dumps(overall, indent=2))
    print(f"wrote {out_root / 'extract_summary.json'}")


if __name__ == "__main__":
    main()
