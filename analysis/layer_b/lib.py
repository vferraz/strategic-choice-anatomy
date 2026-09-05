#!/usr/bin/env python3
"""Shared spine for Layer B Final (one-shot substrate, neural). Self-contained.

Reuses, BY IMPORT (read-only, no core-file edits):
  - analysis/oneshot/_oneshot_common: probe_auc (game-grouped CV OOF AUC + bootstrap-by-game
    CI), game_meta, delta1_per_game (uniform-q sign), trait_targets, pca_fit, lda_fit,
    parse_8vec, style, MODELS/SHORT/NICE/COL.
  - analysis/block_b/levelk_geometry_oneshot: attach_manifest, oof_proba, run_n5, run_n6
    (decision-slot residual machinery; already one-shot-native).
  - analysis/block_b/oneshot_common: delta1, delta2, canonical_sign, u_mats.
  - analysis/layer_a/src/shared_data: corrected integrated-root decisions. Dense models use
    parse_ok-gated decoded_action; GPT-OSS uses realized_action with mixed rows retained as
    resolved 0/1 behavior and none rows dropped.

RE-IMPLEMENTS (copied verbatim-in-math from analysis/block_b/c6_brain_core.py, cited inline):
  the pure decision/incentive geometry. c6's functions are entangled with the design_v2
  ``_common`` loader (decision_slot_matrix / load_unified), so we copy the short pure math and
  feed it the one-shot cached residuals instead.

Hard constraints (docs/METHODS.md): CANONICAL ACTION AXIS everywhere; decision/commit-slot residuals
only (never _seq); payoff_multiplier=1; corrected Akata 4-cell substrate;
bootstrap-by-game CIs; AUC magnitudes are confounded by the cb density -> report per-model
PATTERN + cross-model RANK, never absolute AUC as a finding.

Two incentive conventions, used deliberately (documented in README honesty invariants):
  - DECODABILITY (B1a): UNIFORM-belief (q=0.5) sign of Delta1c / Delta2c -- a pure function of
    the payoff matrix (the "stimulus" framing the organizing principle rests on; model-indep).
  - GEOMETRY / RECRUITMENT (B1c, B2): EMPIRICAL-belief Delta1c from corrected-root realized
    P1/P2 baseline actions, the SAME canonical-signed incentive the behavioural lambda recipe
    uses, so angle<->lambda is coherent.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---- reuse (read-only imports) ----------------------------------------------------------
from analysis.probe_common import (  # noqa: E402
    probe_auc, game_meta as _game_meta, delta1_per_game, trait_targets,
    pca_fit, lda_fit, parse_8vec, style, MODELS, SHORT, NICE, COL,
)
from collection.oneshot_common import delta1, delta2, canonical_sign, u_mats  # noqa: E402
from analysis.layer_a.src.shared_data import load_decisions  # noqa: E402
from strategic_anatomy.config import (game_features_csv, layer_b_cache_root, repo_root,
                                       results_root, substrate_root)

ROOT = repo_root()


# ---- paths ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SUBSTRATE = substrate_root()
# Path-indirection miss of the same class as rebuild_lib.py's FEATURES: this
# was the only site still resolving the residual cache under results_root(), while every other
# consumer -- build_fusion_figures.py, confirm_null_spread.py, check_complexity_rho.py,
# rebuild_missing_tables.py, rebuild_bridge_variants.py -- plus config.py, docs/DATA.md and
# REPRODUCING.md all use layer_b_cache_root(). The split meant build_residual_cache.py wrote a
# cache no fusion builder could read, so a fresh Tier-2 run silently fell back to whatever the
# data_heavy/layer_b_cache shim pointed at.
DATA_DIR = layer_b_cache_root()
BASE_CACHE = DATA_DIR / "baseline"
CUE_CACHE = DATA_DIR / "cues"
PANEL_DIR = HERE / "_paneldata"
TAB_DIR = results_root() / "layer_b"
FIG_DIR = HERE / "figures"
FEATURES = game_features_csv()
for _d in (BASE_CACHE, CUE_CACHE, PANEL_DIR, TAB_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- model presentation -----------------------------------------------------------------
DENSE = ["qwen_instruct", "qwen", "llama31_instruct"]  # 8192-dim; gptoss is 2880-dim MoE
DEEPEST = {"qwen_instruct": 80, "qwen": 80, "llama31_instruct": 80, "gptoss": 36}
TRAITS = ["risk_aversion", "loss_aversion", "inequity_aversion", "selfish_maximizer", "maximin"]
PLACEBO = "length_match_null"
ALL_CONDS = ["baseline"] + [f"cue_{t}" for t in TRAITS] + [f"cue_{PLACEBO}"]
DEFAULT_STEER_LAYERS = {
    "qwen_instruct": [30, 65, 75, 80],
    "qwen": [30, 65, 75, 80],
    "llama31_instruct": [30, 50, 65, 80],
    "gptoss": [9, 22, 24, 30, 36],
}

# seeds (match Layer A / c6 discipline)
BOOT_SEED = 20260520
CV_SEED = 0
N_BOOT = 2000
N_SPLITS = 5
PCA_K = 64
DATA_ROOT_REGISTRY_VERSION = "2026-06-07"


def game_dirs(model: str) -> list[Path]:
    base = SUBSTRATE / model
    if not base.exists():
        return []
    return [
        d for d in sorted(base.iterdir())
        if d.is_dir() and d.name != "_tmp" and (d / "results.parquet").exists()
        and (d / "acts.npz").exists()
    ]


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


_LAYER_CACHE: dict[str, list[int]] = {}


def available_layers(model: str) -> list[int]:
    """Derive captured residual layers from the integrated Akata acts.npz files/configs."""
    if model in _LAYER_CACHE:
        return _LAYER_CACHE[model]
    dirs = game_dirs(model)
    if not dirs:
        _LAYER_CACHE[model] = []
        return []
    pat = re.compile(r"^p1_baseline_cb\d+_l(\d+)$")
    with np.load(dirs[0] / "acts.npz") as z:
        layers = sorted({int(m.group(1)) for k in z.files for m in [pat.match(k)] if m})
    if not layers:
        cfg = _read_config(dirs[0] / "config.json")
        n = int(cfg.get("n_layers_incl_l0", 0))
        layers = list(range(n))
    _LAYER_CACHE[model] = layers
    return layers


def capture_layers(model: str) -> list[int]:
    return available_layers(model)


def steer_layers(model: str) -> list[int]:
    have = set(available_layers(model))
    return [L for L in DEFAULT_STEER_LAYERS.get(model, []) if L in have]


def deepest_layer(model: str) -> int:
    layers = available_layers(model)
    return max(layers) if layers else DEEPEST[model]


def router_layers() -> list[int]:
    dirs = game_dirs("gptoss")
    if not dirs:
        return []
    cfg = _read_config(dirs[0] / "config.json")
    return [int(x) for x in cfg.get("router_layers", [])]


_RAW_CACHE: dict[str, pd.DataFrame] = {}
_DEC_CACHE: dict[str, pd.DataFrame] = {}


def raw_results(model: str) -> pd.DataFrame:
    """Raw integrated results, lightly normalized for joins and emitted-label controls."""
    if model in _RAW_CACHE:
        return _RAW_CACHE[model].copy()
    frames = []
    for d in game_dirs(model):
        df = pd.read_parquet(d / "results.parquet").rename(columns={"counterbalance_id": "cb"})
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        _RAW_CACHE[model] = out
        return out.copy()
    if model == "gptoss":
        out["emitted_label"] = out.get("move_letter", "").fillna("").astype(str)
        out["raw_realized_action"] = pd.to_numeric(out["realized_action"], errors="coerce")
        out["raw_decoded_action"] = pd.to_numeric(out["decoded_action"], errors="coerce")
    else:
        out["emitted_label"] = out.get("decoded_label", "").fillna("").astype(str)
        out["raw_realized_action"] = np.nan
        out["raw_decoded_action"] = pd.to_numeric(out["decoded_action"], errors="coerce")
    _RAW_CACHE[model] = out
    return out.copy()


def decision_rows(model: str, *, ok_only: bool = False) -> pd.DataFrame:
    """Corrected behavior rows with downstream-compatible decoded_action semantics.

    ``decoded_action`` here is deliberately the realized behavioral action: dense
    parse_ok-gated generated decisions; GPT-OSS final-channel ``realized_action`` with mixed
    rows retained as resolved 0/1 behavior and ``none`` rows marked unusable.
    """
    if model not in _DEC_CACHE:
        dec = load_decisions(model).rename(columns={"source": "action_source"}).copy()
        raw = raw_results(model)
        join_cols = ["game_code", "player", "condition", "cb"]
        raw_keep = [c for c in [
            *join_cols, "emitted_label", "raw_decoded_action", "raw_realized_action",
            "parse_ok", "prompt_hash", "move_letter", "decoded_label",
        ] if c in raw.columns]
        dec = dec.merge(raw[raw_keep], on=join_cols, how="left")
        dec["decoded_action"] = pd.to_numeric(dec["action"], errors="coerce")
        dec["decoded_action"] = dec["decoded_action"].where(dec["decoded_action"].isin([0, 1]))
        dec["decoded_label"] = dec["emitted_label"].fillna("").astype(str)
        dec["counterbalance_id"] = dec["cb"].astype(int)
        dec["action_source"] = dec["action_source"].fillna(str(SUBSTRATE))
        for col in ("commit_type", "stated_p_act0", "prob_source"):
            if col not in dec:
                dec[col] = np.nan
        _DEC_CACHE[model] = dec
    out = _DEC_CACHE[model].copy()
    if ok_only:
        out = out[out["ok"].fillna(False)].copy()
    return out


# =========================================================================================
# Pure geometry (copied from analysis/block_b/c6_brain_core.py; math identical)
# =========================================================================================
def _unit(v: np.ndarray) -> np.ndarray:
    """c6_brain_core._unit (193-196)."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.zeros_like(v)


def decision_axis_diffmeans(X: np.ndarray, aligned: np.ndarray) -> np.ndarray:
    """d_dec: canonical-vs-non-canonical difference-of-means direction (c6 decision_axis, 202-229,
    the primary 'diffmeans' branch). Translation-invariant; lives in raw residual space."""
    y = pd.to_numeric(pd.Series(aligned), errors="coerce").to_numpy()
    keep = np.isfinite(y)
    if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
        return np.zeros(X.shape[1])
    Xk, yk = X[keep].astype(np.float64), y[keep].astype(int)
    return _unit(Xk[yk == 1].mean(0) - Xk[yk == 0].mean(0))


def incentive_axis_cov(X: np.ndarray, d1c: np.ndarray) -> np.ndarray:
    """d_inc: columnwise centered OLS slope of residual on canonical-signed Delta1c
    (c6 incentive_axis, 232-247). X centered per column, d1c centered; NO standardisation,
    so d_inc and d_dec share the same raw residual space and their angle is meaningful."""
    d1c = np.asarray(d1c, dtype=np.float64)
    keep = np.isfinite(d1c)
    if keep.sum() < 5 or np.nanstd(d1c[keep]) == 0:
        return np.zeros(X.shape[1])
    Xc = X[keep].astype(np.float64) - X[keep].astype(np.float64).mean(axis=0)
    xc = d1c[keep] - d1c[keep].mean()
    beta = (Xc.T @ xc) / float(xc @ xc)
    return _unit(beta)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """c6_brain_core._angle_deg (319-323)."""
    a, b = _unit(a), _unit(b)
    if not a.any() or not b.any():
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _slope_boot_by_game(x, y, games, *, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI on the OLS slope, resampling whole games (c6 _slope_boot_by_game, 284-299)."""
    x = np.asarray(x, float); y = np.asarray(y, float); games = np.asarray(games)
    gu = np.unique(games)
    rows = [np.where(games == g)[0] for g in gu]
    rng = np.random.default_rng(seed)
    out = []
    for pick in rng.integers(0, len(gu), size=(n_boot, len(gu))):
        idx = np.concatenate([rows[k] for k in pick])
        if np.std(x[idx]) == 0:
            continue
        out.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def neural_gain(proj, d1c, games, *, n_boot=N_BOOT):
    """OLS slope of the d_dec projection on canonical-signed Delta1c, game-clustered CI + a
    scale-free Pearson r companion (c6 neural_gain, 302-316)."""
    proj = np.asarray(proj, float); d1c = np.asarray(d1c, float); games = np.asarray(games)
    keep = np.isfinite(proj) & np.isfinite(d1c)
    out = dict(slope=np.nan, ci_lo=np.nan, ci_hi=np.nan, r=np.nan,
               n=int(keep.sum()), n_games=int(np.unique(games[keep]).size) if keep.any() else 0)
    if keep.sum() < 5 or np.std(d1c[keep]) == 0:
        return out
    out["slope"] = float(np.polyfit(d1c[keep], proj[keep], 1)[0])
    out["ci_lo"], out["ci_hi"] = _slope_boot_by_game(d1c[keep], proj[keep], games[keep], n_boot=n_boot)
    if np.std(proj[keep]) > 0:
        out["r"] = float(np.corrcoef(d1c[keep], proj[keep])[0, 1])
    return out


# =========================================================================================
# Game-grouped out-of-fold helpers (folds by GAME; cb cells are near-duplicates)
# =========================================================================================
def game_folds(games: np.ndarray, n_splits: int = N_SPLITS, seed: int = CV_SEED):
    """Mirror analysis/oneshot/_oneshot_common._game_folds: GroupKFold-by-game."""
    rng = np.random.default_rng(seed)
    gid = pd.factorize(games)[0]
    ug = rng.permutation(np.unique(gid))
    for ch in np.array_split(ug, n_splits):
        te = np.where(np.isin(gid, ch))[0]
        tr = np.setdiff1d(np.arange(len(games)), te)
        yield tr, te


def oof_axis_projection(X, y, games, *, kind="diffmeans", n_splits=N_SPLITS, seed=CV_SEED):
    """Leak-free OOF 1-D projection onto a per-fold-fit axis. ``kind`` in {diffmeans, cov}:
    diffmeans -> decision axis on aligned label; cov -> incentive axis on continuous Delta1c.
    Axes are fit on TRAIN games only and used to project TEST games (no in-sample circularity
    in the displayed map). Returns OOF score per row (NaN where unfittable)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    games = np.asarray(games)
    out = np.full(len(y), np.nan)
    for tr, te in game_folds(games, n_splits, seed):
        ytr = y[tr]
        keep = np.isfinite(ytr)
        if keep.sum() < 4:
            continue
        mu = X[tr][keep].mean(0)
        if kind == "diffmeans":
            yb = ytr[keep].astype(int)
            if len(np.unique(yb)) < 2:
                continue
            d = decision_axis_diffmeans(X[tr][keep], yb)
        else:
            if np.nanstd(ytr[keep]) == 0:
                continue
            d = incentive_axis_cov(X[tr][keep], ytr[keep])
        if not d.any():
            continue
        out[te] = (X[te].astype(np.float64) - mu) @ d
    return out


def probe_auc_multiclass(X, y_class, games, *, classes=None, pca_k=PCA_K,
                         n_splits=N_SPLITS, seed=CV_SEED, n_boot=1000) -> dict:
    """3-class macro one-vs-rest decodability: pooled OOF predict_proba over game-grouped folds,
    macro-OvR AUC + per-class OvR AUC + balanced accuracy, bootstrap-by-game CI on the macro AUC.
    Mirrors c6_brain_core._oof_multiclass (361-431) but folds by GAME via game_folds."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, float)
    y_class = np.asarray(y_class)
    games = np.asarray(games)
    keep = pd.notna(pd.Series(y_class)).to_numpy()
    X, y_class, games = X[keep], y_class[keep], games[keep]
    if classes is None:
        classes = sorted(pd.unique(y_class).tolist())
    col_of = {c: i for i, c in enumerate(classes)}
    out = dict(macro_auc=np.nan, macro_lo=np.nan, macro_hi=np.nan, bal_acc=np.nan,
               per_class={}, n=int(len(y_class)), n_games=int(pd.unique(games).size),
               classes=[str(c) for c in classes])
    if len(y_class) < 2 * n_splits or len(classes) < 2:
        return out
    n = len(y_class)
    proba = np.full((n, len(classes)), np.nan)
    pred = np.full(n, None, dtype=object)
    for tr, te in game_folds(games, n_splits, seed):
        if len(np.unique(y_class[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        k = min(pca_k, X[tr].shape[0] - 1, X[tr].shape[1])
        pca = PCA(n_components=k, random_state=seed).fit(sc.transform(X[tr]))
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
            pca.transform(sc.transform(X[tr])), y_class[tr])
        P = clf.predict_proba(pca.transform(sc.transform(X[te])))
        block = np.zeros((len(te), len(classes)))
        for j, cl in enumerate(clf.classes_):
            if cl in col_of:
                block[:, col_of[cl]] = P[:, j]
        proba[te] = block
        pred[te] = clf.predict(pca.transform(sc.transform(X[te])))
    used = np.array([p is not None for p in pred])
    if used.sum() < n_splits:
        return out
    yb = pd.get_dummies(pd.Categorical(y_class[used], categories=classes)).to_numpy()
    Pp = proba[used]
    present = yb.sum(0) > 0
    if present.sum() >= 2:
        try:
            out["macro_auc"] = float(roc_auc_score(yb[:, present], Pp[:, present], average="macro"))
        except Exception:
            pass
    for c in classes:
        j = col_of[c]
        if yb[:, j].sum() > 0 and yb[:, j].sum() < len(yb):
            try:
                out["per_class"][str(c)] = float(roc_auc_score(yb[:, j], Pp[:, j]))
            except Exception:
                out["per_class"][str(c)] = np.nan
    correct = (pred[used] == y_class[used]).astype(float)
    bal = [float(correct[y_class[used] == c].mean()) for c in classes if (y_class[used] == c).sum()]
    out["bal_acc"] = float(np.mean(bal)) if bal else np.nan
    # bootstrap-by-game CI on macro AUC
    gg = games[used]
    ug = np.unique(gg)
    if ug.size >= 2 and present.sum() >= 2:
        idx_by_g = {g: np.where(gg == g)[0] for g in ug}
        rng = np.random.default_rng(BOOT_SEED)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(ug, size=ug.size, replace=True)
            ii = np.concatenate([idx_by_g[g] for g in pick])
            ybi, ppi = yb[ii][:, present], Pp[ii][:, present]
            if (ybi.sum(0) > 0).all():
                try:
                    boots.append(roc_auc_score(ybi, ppi, average="macro"))
                except Exception:
                    pass
        if boots:
            out["macro_lo"] = float(np.percentile(boots, 2.5))
            out["macro_hi"] = float(np.percentile(boots, 97.5))
    return out


# =========================================================================================
# Cache I/O  (built by build_residual_cache.py; never re-reads the 71 GB substrate)
# =========================================================================================
def baseline_done(model: str) -> bool:
    return (BASE_CACHE / f"_DONE_{model}").exists()


def cues_done(model: str) -> bool:
    return (CUE_CACHE / f"_DONE_{model}").exists()


def load_baseline(model: str, layers=None):
    """Return (meta, {L: X}) for the P1-baseline decision slot. meta is reset-indexed and X rows
    align 1:1 by position (the load_p1_baseline contract -> directly usable as run_n5/n6 ``pre``)."""
    meta = pd.read_parquet(BASE_CACHE / f"meta_{model}.parquet")
    avail = sorted(int(p.stem.split("_l")[-1]) for p in BASE_CACHE.glob(f"X_{model}_l*.npy"))
    layers = avail if layers is None else [int(L) for L in layers]
    X = {L: np.load(BASE_CACHE / f"X_{model}_l{L}.npy", mmap_mode="r") for L in layers}
    return meta, X


def attach_baseline_commit_type(meta: pd.DataFrame, model: str) -> pd.DataFrame:
    """Attach GPT-OSS pure/mixed capture type to cached P1-baseline rows.

    Choice-dependent GPT-OSS analyses must distinguish literal pure commitments from
    seeded resolutions of mixed answers.  Dense rows have a single answer-slot site and
    are labelled ``parsed_generated_choice`` for a uniform downstream scope column.
    """
    out = meta.reset_index(drop=True).copy()
    if model != "gptoss":
        out["commit_type"] = "parsed_generated_choice"
        return out

    # A cache written by the current build_residual_cache.py always carries a commit_type
    # column (unconditional meta key); the 2026-07-10 cache predates it. Drop the
    # cache-borne copy so the authoritative merge from results.parquet below can never
    # collide into commit_type_x/_y.
    out = out.drop(columns=["commit_type"], errors="ignore")

    raw = raw_results(model)
    commit = raw[(raw["player"] == 1) & (raw["condition"] == "baseline")][
        ["game_code", "cb", "commit_type"]
    ].copy()
    if commit.duplicated(["game_code", "cb"]).any():
        raise AssertionError("duplicate GPT-OSS P1-baseline commit rows")
    out = out.merge(
        commit,
        left_on=["game_code", "counterbalance_id"],
        right_on=["game_code", "cb"],
        how="left",
        validate="one_to_one",
    ).drop(columns="cb")
    if out["commit_type"].isna().any():
        raise AssertionError("GPT-OSS cache rows are missing commit_type provenance")
    return out


def load_cues(model: str):
    """Return (meta, X) at the deepest layer for conditions baseline + 5 cues + placebo."""
    meta = pd.read_parquet(CUE_CACHE / f"meta_{model}.parquet")
    L = deepest_layer(model)
    X = np.load(CUE_CACHE / f"X_{model}_l{L}.npy", mmap_mode="r")
    return meta, X


# =========================================================================================
# Per-game targets  (the single source for all probe labels)
# =========================================================================================
def game_meta() -> pd.DataFrame:
    """144-row metadata (reuse _oneshot_common.game_meta: canonical_action_p1/p2, num_pure_ne,
    dominance_profile, complexity_score, nagel_lk_type, canonical_8vec, delta1_canonical,
    sign_delta1c at q=0.5)."""
    return _game_meta()


def target_table() -> pd.DataFrame:
    """Per-game probe targets (game-level; broadcast to rows by game_code on join).

    UNIFORM-belief (stimulus) incentive signs for decodability; structural targets for B1d.
    Columns: game_code, canonical_action_p1, sign_delta1c (q=0.5), sign_delta2c (q=0.5),
    dominant_action_p1 (NaN where none, n=72), num_pure_ne (3-class eq type), dominance_profile,
    nagel_lk_type, stim_cell00 (3-class P1 payoff in canonical cell(0,0)),
    stim_cell01_ismax (binary, P1 cell(0,1)==4)."""
    gm = game_meta().copy()
    rows = []
    for _, r in gm.drop_duplicates("game_code").iterrows():
        v = parse_8vec(r["canonical_8vec"])
        u1, u2 = u_mats(v)                       # u1,u2 indexed [p1_action, p2_action]
        c1 = int(r["canonical_action_p1"]) if pd.notna(r["canonical_action_p1"]) else np.nan
        c2 = int(r["canonical_action_p2"]) if pd.notna(r["canonical_action_p2"]) else np.nan
        # uniform-belief opponent incentive sign (pure stimulus): EU2(act0)-EU2(act1) @ q=0.5
        d2u = delta2(v, 0.5)
        s2 = canonical_sign(d2u, c2) if pd.notna(c2) else np.nan
        rows.append(dict(
            game_code=r["game_code"],
            sign_delta2c=int(s2 > 0) if pd.notna(s2) else np.nan,
            stim_cell00=int(round(u1[0, 0])),            # canonical (0,0) payoff rank, 3-class
            stim_cell01_ismax=int(round(u1[0, 1]) == 4), # binary is-max floor (72/72)
        ))
    extra = pd.DataFrame(rows)
    keep = ["game_code", "canonical_action_p1", "sign_delta1c", "dominant_action_p1",
            "num_pure_ne", "dominance_profile", "nagel_lk_type"]
    out = gm[keep].drop_duplicates("game_code").merge(extra, on="game_code", how="left")
    return out


def empirical_delta1c(model: str) -> pd.DataFrame:
    """Per-game EMPIRICAL-belief canonical-signed Delta1c / Delta2c for ``model`` (q-hat from the
    model's own corrected baseline opponent rate) -- the incentive the behavioural lambda
    recipe uses. This intentionally avoids the legacy slot-argmax fallback in
    ``extract_causal_directions_oneshot``."""
    gm = game_meta().drop_duplicates("game_code").set_index("game_code")
    dec = decision_rows(model, ok_only=True)
    p1b = dec[(dec["player"] == 1) & (dec["condition"] == "baseline")]
    p2b = dec[(dec["player"] == 2) & (dec["condition"] == "baseline")]
    qhat_p1 = p1b.groupby("game_code")["decoded_action"].apply(lambda s: float((s == 0).mean()))
    qhat_p2 = p2b.groupby("game_code")["decoded_action"].apply(lambda s: float((s == 0).mean()))

    rows = []
    for g in sorted(set(qhat_p1.index) & set(qhat_p2.index) & set(gm.index)):
        r = gm.loc[g]
        c1 = int(r["canonical_action_p1"]) if pd.notna(r["canonical_action_p1"]) else None
        c2 = int(r["canonical_action_p2"]) if pd.notna(r["canonical_action_p2"]) else None
        vec = parse_8vec(r["canonical_8vec"])
        d1 = delta1(vec, float(qhat_p2[g]))
        d2 = delta2(vec, float(qhat_p1[g]))
        rows.append({
            "game_code": g,
            "q_hat_p1": float(qhat_p1[g]),
            "q_hat_p2": float(qhat_p2[g]),
            "delta1_emp": float(d1),
            "delta1_c": float(canonical_sign(d1, c1)) if c1 is not None else np.nan,
            "delta2_emp": float(d2),
            "delta2_c": float(canonical_sign(d2, c2)) if c2 is not None else np.nan,
            "canonical_action_p1": -1 if c1 is None else int(c1),
            "canonical_action_p2": -1 if c2 is None else int(c2),
        })
    return pd.DataFrame(rows)
