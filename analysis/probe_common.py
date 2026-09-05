"""Shared spine for the one-shot Layer A + B reproduction.

Single source of truth for: game metadata (canonical 8-vec, lk class, human refs,
canonical action), the canonical incentive Delta1^c, Nash coords, trait targets,
a leak-free game-grouped probe, plotting style, and the CI-based verdict rule.

All cross-game aggregation is on the CANONICAL ACTION AXIS (docs/METHODS.md): never pool
raw ``decoded_action == 0`` across games. Delta1^c = EU1(canonical) - EU1(non-canonical)
under uniform belief q=0.5 (positive => incentive favours the canonical action).

This module is additive and read-only w.r.t. the core tree: it *imports* the locked
estimators (``delta1``/``fit_lambda``/``boot_cluster_lambda``), the supervised-axis
helpers (``oof_dir``/``auc``/``_ellipse``), and the equilibrium engine; it copies the
two tiny LDA helpers from ``fig_lda_dispositions`` because that module runs code on
import.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from strategic_anatomy.config import game_features_csv, repo_root, taxonomy_dir

ROOT = repo_root()



FEATURES = str(game_features_csv())
EQUIV = str(taxonomy_dir() / "equivalence_per_canonical.csv")

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
PANEL_DIR = os.path.join(HERE, "paneldata")
VERDICT_DIR = os.path.join(PANEL_DIR, "verdicts")
for _d in (FIG_DIR, PANEL_DIR, VERDICT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- reuse locked estimators from the active probe program -------------------
from analysis.layer_a.layer1_lambda_delta1 import delta1, fit_lambda, boot_cluster_lambda  # noqa: E402
from analysis.layer_a.decision_state import oof_dir, auc, _ellipse  # noqa: E402
from strategic_anatomy import eq_engine as EQ  # noqa: E402

# ---- model presentation ------------------------------------------------------
MODELS = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss"]
SHORT = {"qwen_instruct": "Qwen-I", "qwen": "Qwen-B",
         "llama31_instruct": "Llama", "gptoss": "GPT-OSS"}
NICE = {"qwen_instruct": "Qwen2.5-72B-Instruct", "qwen": "Qwen2.5-72B (base)",
        "llama31_instruct": "Llama-3.1-70B-Instruct", "gptoss": "GPT-OSS-120B (MoE)"}
COL = {"qwen_instruct": "#1f77b4", "qwen": "#2ca02c",
       "llama31_instruct": "#d62728", "gptoss": "#9467bd"}
DENSE_MODELS = ["qwen_instruct", "qwen", "llama31_instruct"]  # exclude gptoss MoE

# trait cue dispositions (oneshot raw condition names without the cue_ prefix)
TRAITS = ["risk_aversion", "loss_aversion", "inequity_aversion",
          "selfish_maximizer", "maximin"]
PLACEBO = "length_match_null"
LK_ORDER = ["DD", "OD1", "OD2", "CO1", "CO2", "MP"]

# ---- design_v2 reference numbers (locked storyline; for verdicts) ------------
DV2 = {
    "lambda": {  # hard, headline (empirical-q)
        "qwen_instruct": (0.91, 0.50, 1.53),
        "qwen": (0.45, -0.04, 0.93),
        "llama31_instruct": (-0.01, -0.51, 0.47),
        "gptoss": (-0.04, -0.36, 0.32),
    },
    "B3_angle": {"qwen_instruct": 44.0, "llama31_instruct": 81.0, "gptoss": 72.0},
    "B4_rho_angle": -0.80,
    "B4_rho_gain": 1.0,
    "B6_acc": {"qwen_instruct": (0.93, 1.00), "qwen": (0.93, 1.00),
               "llama31_instruct": (0.93, 1.00), "gptoss": (0.55, 0.55)},
}


# ---- 8-vector parsing --------------------------------------------------------
def parse_8vec(s) -> list[float]:
    """canonical_8vec is a comma-separated string ('1,4,3,2,2,4,3,1') -- NOT JSON."""
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    return [float(x) for x in str(s).split(",")]


# ---- game metadata: the single join -----------------------------------------
def game_meta() -> pd.DataFrame:
    """144 Bruns games: canonical_8vec, nagel_lk_type, human refs, canonical action,
    structural predictors, and the canonical incentive Delta1^c (q=0.5)."""
    equiv = pd.read_csv(EQUIV)
    feat = pd.read_csv(FEATURES)
    equiv = equiv.rename(columns={"bruns_name": "game_code"})
    m = equiv.merge(feat, on="game_code", how="left", suffixes=("", "_feat"))

    # hard constraint: canonical action axis must be defined everywhere
    bad = m["canonical_action_p1"].isna() | ~m["canonical_action_p1"].isin([0, 1])
    if bad.any():
        raise AssertionError(
            f"canonical_action_p1 not in {{0,1}} for {int(bad.sum())} games: "
            f"{sorted(m.loc[bad, 'game_code'].tolist())[:10]}")

    d1 = delta1_per_game(m)
    m = m.merge(d1, on="game_code", how="left")
    return m


def delta1_per_game(meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per game: delta1_uniform = EU1(act0)-EU1(act1) @q=0.5 (positional), and the
    canonical-signed Delta1^c = where(canon==0, +gap, -gap). Mirrors the exact flip
    in layer1_lambda_delta1.compute_layer2_robustness."""
    if meta is None:
        equiv = pd.read_csv(EQUIV).rename(columns={"bruns_name": "game_code"})
        feat = pd.read_csv(FEATURES)[["game_code", "canonical_action_p1"]]
        meta = equiv.merge(feat, on="game_code", how="left")
    rows = []
    for _, r in meta.drop_duplicates("game_code").iterrows():
        vec = parse_8vec(r["canonical_8vec"])
        gap = delta1(vec, 0.5)            # EU1(act0) - EU1(act1) at q=0.5
        canon = int(r["canonical_action_p1"])
        d_can = gap if canon == 0 else -gap
        rows.append({"game_code": r["game_code"],
                     "delta1_uniform": float(gap),
                     "delta1_canonical": float(d_can),
                     "abs_delta1": float(abs(gap)),
                     "sign_delta1c": int(d_can > 0)})
    return pd.DataFrame(rows)


def ne_pairs_per_game() -> dict[str, tuple[list[tuple[float, float]], str]]:
    """game_code -> ([(P1 act0-prob, P2 act0-prob), ...], kind) from the 8-vec."""
    meta = pd.read_csv(EQUIV).rename(columns={"bruns_name": "game_code"})
    out = {}
    for _, r in meta.iterrows():
        p1, p2 = EQ.matrices_from_8vec(parse_8vec(r["canonical_8vec"]))
        pairs, kind = EQ.enumerate_nash(p1, p2)
        out[r["game_code"]] = (pairs, kind)
    return out


def _br(vals, *, high: bool = True) -> float:
    """Best-response action index (argmax/argmin); NaN on a tie."""
    if abs(float(vals[0]) - float(vals[1])) < 1e-12:
        return np.nan
    return float(np.argmax(vals) if high else np.argmin(vals))


def trait_targets() -> pd.DataFrame:
    """Per game preferred action a_<trait> and signed dose d_<trait> (d>0 => act0),
    reimplementing analysis/layer_a/trait_steering_proper.trait_targets over the
    oneshot canonical_8vec (the original reads the design_v2 UNIFIED parquet)."""
    meta = pd.read_csv(EQUIV).rename(columns={"bruns_name": "game_code"})
    out = []
    for _, r in meta.iterrows():
        v = np.array(parse_8vec(r["canonical_8vec"]), float)
        p1, p2 = v[:4].reshape(2, 2), v[4:].reshape(2, 2)
        worst = [float(p1[i, :].min()) for i in range(2)]                       # risk/loss/maximin
        expo = [float(p1[i, :].mean()) for i in range(2)]                       # selfish (BR uniform)
        gap = [float(np.mean(np.abs(p1[i, :] - p2[i, :]))) for i in range(2)]   # inequity
        a_worst, a_expo, a_gap = _br(worst), _br(expo), _br(gap, high=False)
        out.append({
            "game_code": r["game_code"],
            "a_risk_aversion": a_worst, "d_risk_aversion": worst[0] - worst[1],
            "a_loss_aversion": a_worst, "d_loss_aversion": worst[0] - worst[1],
            "a_maximin": a_worst, "d_maximin": worst[0] - worst[1],
            "a_selfish_maximizer": a_expo, "d_selfish_maximizer": expo[0] - expo[1],
            "a_inequity_aversion": a_gap, "d_inequity_aversion": gap[1] - gap[0],
        })
    return pd.DataFrame(out)


# ---- copied LDA helpers (fig_lda_dispositions runs code on import) -----------
def pca_fit(Xtr: np.ndarray, k: int):
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    return mu, Vt[:k]


def lda_fit(Z: np.ndarray, y: np.ndarray, reg: float = 1e-2) -> np.ndarray:
    cls = np.unique(y)
    d = Z.shape[1]
    mu_all = Z.mean(0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in cls:
        Zc = Z[y == c]
        mc_ = Zc.mean(0)
        D = Zc - mc_
        Sw += D.T @ D
        dd = (mc_ - mu_all)[:, None]
        Sb += len(Zc) * (dd @ dd.T)
    Sw += reg * np.trace(Sw) / d * np.eye(d)
    ev, evec = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(ev.real)[::-1]
    W = evec[:, order[:2]].real
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)


# ---- leak-free game-grouped probe (per-fold PCA -> logistic, OOF AUC) --------
def _game_folds(games: np.ndarray, n_splits: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    gid = pd.factorize(games)[0]
    ug = rng.permutation(np.unique(gid))
    chunks = np.array_split(ug, n_splits)
    folds = []
    for ch in chunks:
        te = np.where(np.isin(gid, ch))[0]
        tr = np.setdiff1d(np.arange(len(games)), te)
        folds.append((tr, te))
    return folds


def probe_auc(X: np.ndarray, y: np.ndarray, games: np.ndarray, *,
              pca_k: int = 64, n_splits: int = 5, seed: int = 0,
              n_boot: int = 1000) -> dict:
    """Out-of-fold AUC of a linear probe with PCA fit INSIDE each train fold
    (no leakage) and folds GROUPED BY GAME (cb are near-duplicate renderings).
    Bootstrap-by-game CI on the pooled OOF AUC. Returns NaNs on degenerate input."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    games = np.asarray(games)
    out = {"auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
           "n": int(len(y)), "n_games": int(pd.unique(games).size),
           "base_rate": float(np.mean(y)) if len(y) else np.nan}
    if len(y) < 2 * n_splits or len(set(y.tolist())) < 2:
        return out

    score = np.full(len(y), np.nan)
    for tr, te in _game_folds(games, n_splits, seed):
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        k = min(pca_k, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=k, random_state=seed).fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000,
                                 random_state=seed).fit(pca.transform(Xtr), y[tr])
        col = list(clf.classes_).index(1)
        score[te] = clf.predict_proba(pca.transform(Xte))[:, col]

    used = np.isfinite(score)
    if used.sum() < 2 or len(set(y[used].tolist())) < 2:
        return out
    s, yy, gg = score[used], y[used], games[used]
    out["auc"] = float(roc_auc_score(yy, s))

    # bootstrap by game
    ug = np.unique(gg)
    idx_by_g = {g: np.where(gg == g)[0] for g in ug}
    rng = np.random.default_rng(seed + 1)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        if len(set(yy[ii].tolist())) < 2:
            continue
        boots.append(roc_auc_score(yy[ii], s[ii]))
    if boots:
        out["auc_lo"] = float(np.percentile(boots, 2.5))
        out["auc_hi"] = float(np.percentile(boots, 97.5))
    return out


# ---- CI-based verdict rule ---------------------------------------------------
def _ci_class(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "NA"
    if lo > 0:
        return "POS"
    if hi < 0:
        return "NEG"
    return "NULL"


def classify_cell(new, dv2, *, expected_sign: int | None = None) -> str:
    """One cell verdict. new/dv2 are (val, lo, hi). expected_sign in {+1,-1,None}.

    HOLD: same sign-class AND CIs overlap.
    STRENGTHENED: same sign but more decisive (NULL->sig in expected dir, or same
        sig class with larger |effect| AND tighter-or-equal CI, or disjoint same-sign
        with larger |effect|).
    CHANGED: sign flip with a CI excluding 0, sig->NULL collapse, or disjoint CIs
        that are not a clean strengthening.
    """
    vn, lon, hin = new
    vd, lod, hid = dv2
    cn, cd = _ci_class(lon, hin), _ci_class(lod, hid)
    if "NA" in (cn, cd):
        return "NA"
    sn = {"POS": 1, "NEG": -1, "NULL": 0}[cn]
    sd = {"POS": 1, "NEG": -1, "NULL": 0}[cd]
    overlap = (lon <= hid) and (lod <= hin)

    # sign flip, both significant
    if sn != 0 and sd != 0 and sn != sd:
        return "CHANGED"
    # significant -> null collapse
    if sd != 0 and sn == 0:
        return "CHANGED"
    # emergence: null -> significant
    if sd == 0 and sn != 0:
        if expected_sign is None or sn == expected_sign:
            return "STRENGTHENED"
        return "CHANGED"
    # both null
    if sn == 0 and sd == 0:
        return "HOLD"
    # same significant sign
    more_decisive = (abs(vn) >= abs(vd)) and ((hin - lon) <= (hid - lod) + 1e-9)
    if overlap:
        return "STRENGTHENED" if (more_decisive and abs(vn) > abs(vd)) else "HOLD"
    # disjoint, same sign
    return "STRENGTHENED" if abs(vn) > abs(vd) else "CHANGED"


def aggregate_verdict(cells: list[str]) -> str:
    cells = [c for c in cells if c in ("HOLD", "STRENGTHENED", "CHANGED")]
    if not cells:
        return "NA"
    counts = {v: cells.count(v) for v in ("HOLD", "STRENGTHENED", "CHANGED")}
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    if len(winners) == 1:
        return winners[0]
    return "CHANGED"  # ties -> conservative


def write_verdict(vid: str, title: str, dv2_finding: str, oneshot_finding: str,
                  metric_table: list[dict], verdict: str) -> None:
    rec = {"id": vid, "title": title, "dv2_finding": dv2_finding,
           "oneshot_finding": oneshot_finding, "metric_table": metric_table,
           "verdict": verdict}
    with open(os.path.join(VERDICT_DIR, f"{vid}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[verdict] {vid}: {verdict}")


# ---- style -------------------------------------------------------------------
def style():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    from strategic_anatomy import paper_style as PS
    PS.apply()
    return PS
