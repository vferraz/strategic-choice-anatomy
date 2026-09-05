#!/usr/bin/env python3
"""Level-k geometry: the neural identification test (SPEC §2, N1-N6).

Behavioral identification of reasoning depth tau vs response precision lambda is
weak in 2x2 choices. This module tests whether the belief/level latent is
linearly decodable from the one-shot P1-baseline decision-slot residuals,
separately from the incentive-precision axis -- the activations break the
degeneracy the choice data cannot.

Reads decision-slot residuals from
  $SCA_DATA_ROOT/substrate/{model}/{game}/acts.npz  (keys p1_baseline_cb{cb}_l{L})
and the per-game revealed-level manifest from
  analysis/block_a/tables/levelk_revealed_level.csv      (built by §1, Module A).

P1 BASELINE ONLY. CANONICAL AXIS throughout (docs/METHODS.md HC-2). Game-grouped CV and
bootstrap-by-game CIs (reuse analysis/oneshot/_oneshot_common.probe_auc). Decision-
slot residuals only -- one-shot acts.npz has no _seq keys, so the constraint is
satisfied by construction; loaders only ever request ``..._l{L}`` keys.

Tests
  N1  level is a readable internal state  (decode revealed_level: L1 vs L2+)
  N2  the belief axis econ can't identify (sign(gap_L1) decodable; L1>EQ contrast)
  N3  reasoning depth maps onto network depth (crystallization layer vs complexity)
  N4  two axes, not one  (cos(d_belief_L1, d_inc) per model/layer)
  N5  distributed vs localized code (participation ratio + greedy ablation)
  N6  two-choice dynamics across depth (proxy; faithful logit-lens optional)

Outputs -> analysis/block_b/tables/levelk/*.csv, analysis/block_b/figures/n{1..6}_*.{png,pdf}

CPU only. The existing crystallization machinery (c1_crystallization.py /
fig_main_layer_b.py) is bound to the design_v2 substrate (keys r{round}_p{p}_l{L},
output/design_v2_main) and cannot be called on the one-shot substrate; we reuse the
*concept* (per-layer AUC -> onset layer, ONSET_AUC=0.65 from c1_crystallization.py:64)
but compute it via the one-shot loader + probe_auc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.probe_common import probe_auc, MODELS, SHORT, COL, style  # noqa: E402
from steering.extract_directions import (  # noqa: E402
    _iter_games, build_delta_tables,
)
from strategic_anatomy.config import (
    game_features_csv, manifests_root, repo_root, results_root, substrate_root,
)

ROOT = repo_root()


GAME_FEATURES = game_features_csv()
MANIFEST = results_root() / "layer_a" / "levelk_revealed_level.csv"
CONFIG = manifests_root() / "oneshot_config.json"
TAB_DIR = results_root() / "layer_b" / "levelk"
FIG_DIR = ROOT / "analysis" / "layer_b" / "figures"

ONSET_AUC = 0.65  # crystallization threshold; mirrors c1_crystallization.py:64
PCA_K = 64
N_SPLITS = 5
SEED = 0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _capture_layers(model: str) -> list[int]:
    import json
    cfg = json.loads(CONFIG.read_text())
    return list(cfg["models"][model]["capture_layers"])


def _steer_layers(model: str) -> list[int]:
    import json
    cfg = json.loads(CONFIG.read_text())
    return list(cfg["models"][model]["steer_layers"])


def load_p1_baseline(model: str, layers: list[int]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Open each game's acts.npz ONCE and pull all requested layers for the P1
    baseline decision slot. Returns (meta, {L: X [n x d]}) with meta rows aligned
    to every X. Only (game, cb) rows present at ALL requested layers are kept."""
    feats = pd.read_csv(GAME_FEATURES)
    canon1 = {r.game_code: int(r.canonical_action_p1) for r in feats.itertuples()
              if pd.notna(r.canonical_action_p1)}
    metas: list[dict] = []
    Xacc: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    for d in _iter_games(model):
        df = pd.read_parquet(d / "results.parquet")
        sub = df[(df["player"] == 1) & (df["condition"] == "baseline")]
        sub = sub[sub["decoded_action"].isin([0, 1])]
        if sub.empty:
            continue
        z = np.load(d / "acts.npz")
        try:
            files = set(z.files)
            for _, r in sub.iterrows():
                cb = int(r["counterbalance_id"])
                keys = {L: f"p1_baseline_cb{cb}_l{L}" for L in layers}
                if not all(k in files for k in keys.values()):
                    continue
                for L in layers:
                    Xacc[L].append(np.asarray(z[keys[L]], dtype=np.float32))
                g = r["game_code"]
                metas.append({"game_code": g, "counterbalance_id": cb,
                              "decoded_action": int(r["decoded_action"]),
                              "canonical_action_p1": int(canon1.get(g, -1))})
        finally:
            z.close()
    meta = pd.DataFrame(metas).reset_index(drop=True)
    X = {L: (np.vstack(Xacc[L]).astype(np.float32) if Xacc[L] else np.zeros((0, 0), np.float32))
         for L in layers}
    return meta, X


def attach_manifest(meta: pd.DataFrame, model: str) -> pd.DataFrame:
    """Join the §1 revealed-level manifest and derive the canonical-alignment label."""
    man = pd.read_csv(MANIFEST)
    man = man[man["model"] == model].set_index("game_code")
    meta = meta.copy()
    for col in ["revealed_level", "gap_L1", "gap_EQ", "complexity_score",
                "iesds_depth", "diagnostic", "nagel_lk_type"]:
        meta[col] = meta["game_code"].map(man[col])
    meta = meta[meta["canonical_action_p1"] >= 0].copy()
    meta["aligned"] = (meta["decoded_action"] == meta["canonical_action_p1"]).astype(int)
    meta["level_bin"] = (meta["revealed_level"] != "L1").astype(int)  # 0=L1, 1=L2+
    return meta


def _prep(model: str, layers: list[int]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Load once + attach manifest. Pass the result as ``pre`` to every run_nX to
    avoid re-opening all 144 acts.npz per test (6x redundant I/O otherwise)."""
    meta, X = load_p1_baseline(model, layers)
    return attach_manifest(meta, model), X


# ---------------------------------------------------------------------------
# Shared cross-validated probe internals (mirror probe_auc; return OOF probs)
# ---------------------------------------------------------------------------
def _game_folds(games: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    gid = pd.factorize(games)[0]
    ug = rng.permutation(np.unique(gid))
    for ch in np.array_split(ug, n_splits):
        te = np.where(np.isin(gid, ch))[0]
        tr = np.setdiff1d(np.arange(len(games)), te)
        yield tr, te


def oof_proba(X: np.ndarray, y: np.ndarray, games: np.ndarray, *,
              pca_k: int = PCA_K, n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold P(y==1) with per-fold StandardScaler+PCA+logistic, folds grouped
    by game (no leakage). NaN where a sample's fold had a single train class."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    out = np.full(len(y), np.nan)
    if len(y) < 2 * n_splits or len(set(y.tolist())) < 2:
        return out
    for tr, te in _game_folds(games, n_splits, seed):
        if len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        k = min(pca_k, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=k, random_state=seed).fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(pca.transform(Xtr), y[tr])
        col = list(clf.classes_).index(1)
        out[te] = clf.predict_proba(pca.transform(Xte))[:, col]
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if (n > 0 and np.isfinite(n)) else v


def _standardize(X: np.ndarray) -> np.ndarray:
    """Per-dim standardize (z-score). Removes the residual-stream mean / rogue-dim
    domination that makes uncentered E[X*y] directions collinear across targets."""
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _dir_cov(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale-fair belief/incentive axis on standardized Z: dir ∝ E[(Z)(y-ȳ)],
    unit-normalized, sign-aligned so corr(Z@dir, y) > 0. Returns (unit_dir, corr).

    NB: the repo's ``fit_d_inc`` computes the *uncentered* E[X*y] -- correct as a
    steering vector but dominated by the residual-stream mean, so it is unfit for
    axis-angle geometry (all targets collapse onto one direction). We center here."""
    y = np.asarray(y, float)
    keep = np.isfinite(y)
    Zk, yk = Z[keep], y[keep]
    yc = yk - yk.mean()
    d = (Zk * yc[:, None]).mean(0)
    n = float(np.linalg.norm(d))
    if n == 0 or not np.isfinite(n):
        return np.zeros(Z.shape[1]), 0.0
    d = d / n
    corr = float(np.corrcoef(Zk @ d, yk)[0, 1]) if len(yk) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    if corr < 0:
        d, corr = -d, -corr
    return d, corr


def _dir_diff(Z: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Unit diff-of-means choice axis on standardized Z (translation-invariant)."""
    if pos.sum() == 0 or neg.sum() == 0:
        return np.zeros(Z.shape[1])
    return _unit(Z[pos].mean(0) - Z[neg].mean(0))


# ---------------------------------------------------------------------------
# N1 -- level is a readable internal state
# ---------------------------------------------------------------------------
def run_n1(model: str, layers: list[int], pre=None) -> pd.DataFrame:
    meta, X = pre if pre is not None else _prep(model, layers)
    idx = meta.index.to_numpy()
    games = meta["game_code"].to_numpy()
    y = meta["level_bin"].to_numpy()
    # shuffle control: permute level labels within game-class (nagel_lk_type)
    rng = np.random.default_rng(SEED + 7)
    y_shuf = y.copy()
    for _, gi in meta.groupby("nagel_lk_type").groups.items():
        pos = meta.index.get_indexer(gi)
        y_shuf[pos] = rng.permutation(y_shuf[pos])
    rows = []
    for L in layers:
        Xl = X[L][idx]
        r = probe_auc(Xl, y, games, pca_k=PCA_K, n_splits=N_SPLITS, seed=SEED)
        rs = probe_auc(Xl, y_shuf, games, pca_k=PCA_K, n_splits=N_SPLITS, seed=SEED)
        rows.append({"model": model, "layer": L, "depth_frac": L / max(layers),
                     "auc": r["auc"], "auc_lo": r["auc_lo"], "auc_hi": r["auc_hi"],
                     "auc_shuffled": rs["auc"], "base_rate_L2plus": r["base_rate"],
                     "n": r["n"], "n_games": r["n_games"]})
    df = pd.DataFrame(rows)
    print(f"[{SHORT.get(model, model)}] N1 level(L1 vs L2+): "
          f"max AUC={np.nanmax(df['auc']):.3f} (shuffled max={np.nanmax(df['auc_shuffled']):.3f}), "
          f"base_rate(L2+)={df['base_rate_L2plus'].iloc[0]:.2f}")
    return df


# ---------------------------------------------------------------------------
# N2 -- the belief axis (L1 vs EQ)
# ---------------------------------------------------------------------------
def run_n2(model: str, layers: list[int], pre=None) -> pd.DataFrame:
    meta, X = pre if pre is not None else _prep(model, layers)
    idx = meta.index.to_numpy()
    games = meta["game_code"].to_numpy()
    gapL1 = meta["gap_L1"].to_numpy(float)
    gapEQ = meta["gap_EQ"].to_numpy(float)
    al = meta["aligned"].to_numpy()
    # disagreement set (rows): sign(gap_L1) != sign(gap_EQ), both nonzero
    disagree = (np.sign(gapL1) != np.sign(gapEQ)) & (gapL1 != 0) & (gapEQ != 0)
    n_dis_games = int(meta.loc[disagree, "game_code"].nunique())
    rows = []
    for L in layers:
        Z = _standardize(X[L][idx])
        # belief axes (scale-fair centered covariance) + their decodability
        dL1, cL1 = _dir_cov(Z, gapL1)
        dEQ, cEQ = _dir_cov(Z, gapEQ)
        dchoice = _dir_diff(Z, al == 1, al == 0)  # choice axis (canonical vs not)
        # decodability of belief sign via the leak-free game-grouped probe (headline)
        aL1 = probe_auc(X[L][idx], (gapL1 > 0).astype(int), games, pca_k=PCA_K, seed=SEED)
        aEQ = probe_auc(X[L][idx], (gapEQ > 0).astype(int), games, pca_k=PCA_K, seed=SEED)
        # sharp test on disagreement games: decode sign(gap_L1) there (AUC>0.5 =>
        # the residual belief sign sides with L1 over EQ -> identification by activation).
        # Need enough distinct GAMES for the grouped CV (folds are by game), not just rows.
        aDis = np.nan
        if (n_dis_games >= N_SPLITS and disagree.sum() >= 2 * N_SPLITS
                and len(set((gapL1[disagree] > 0).astype(int))) == 2):
            try:
                aDis = probe_auc(X[L][idx][disagree], (gapL1[disagree] > 0).astype(int),
                                 games[disagree], pca_k=min(PCA_K, 32), seed=SEED)["auc"]
            except Exception:
                aDis = np.nan
        rows.append({
            "model": model, "layer": L, "depth_frac": L / max(layers),
            "dir_fit_corr_L1_insample": cL1, "dir_fit_corr_EQ_insample": cEQ,
            "auc_sign_gapL1": aL1["auc"], "auc_sign_gapL1_lo": aL1["auc_lo"], "auc_sign_gapL1_hi": aL1["auc_hi"],
            "auc_sign_gapEQ": aEQ["auc"],
            "cos_dL1_dEQ": float(np.dot(dL1, dEQ)),
            "cos_dL1_choice": float(np.dot(dL1, dchoice)),
            "cos_dEQ_choice": float(np.dot(dEQ, dchoice)),
            "auc_disagree_sideL1": aDis, "n_disagree_games": n_dis_games,
        })
    df = pd.DataFrame(rows)
    best = df.loc[df["auc_sign_gapL1"].idxmax()]
    print(f"[{SHORT.get(model, model)}] N2 belief axis: max AUC sign(gap_L1)={best['auc_sign_gapL1']:.3f} "
          f"(EQ={df['auc_sign_gapEQ'].max():.3f}) | "
          f"cos(d_L1,choice)={best['cos_dL1_choice']:+.3f} vs cos(d_EQ,choice)={best['cos_dEQ_choice']:+.3f} | "
          f"disagree games(L1!=EQ)={n_dis_games}")
    return df


# ---------------------------------------------------------------------------
# N3 -- reasoning depth maps onto network depth (crystallization)
# ---------------------------------------------------------------------------
def run_n3(model: str, layers: list[int], pre=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr
    layers = sorted(layers)
    meta, X = pre if pre is not None else _prep(model, layers)
    idx = meta.index.to_numpy()
    games = meta["game_code"].to_numpy()
    y = meta["canonical_action_p1"].to_numpy().astype(int)  # normative target
    Lmax = max(layers)

    # per-layer correctness score s = p if canon==1 else 1-p, then per-game mean
    per_layer_curve = []
    s_by_layer = {}
    for L in layers:
        p = oof_proba(X[L][idx], y, games, pca_k=PCA_K, seed=SEED)
        s = np.where(y == 1, p, 1.0 - p)
        s_by_layer[L] = s
        # global per-layer AUC curve (for the figure / cross-check)
        a = probe_auc(X[L][idx], y, games, pca_k=PCA_K, seed=SEED)
        per_layer_curve.append({"model": model, "layer": L, "depth_frac": L / Lmax, "auc": a["auc"]})
    curve = pd.DataFrame(per_layer_curve)

    # per-game crystallization layer = shallowest L where mean correctness >= 0.65
    rows = []
    gmeta = meta.drop_duplicates("game_code").set_index("game_code")
    for g in pd.unique(games):
        gm = games == g
        onset = np.nan
        for L in layers:
            sl = s_by_layer[L][gm]
            if np.isfinite(sl).any() and np.nanmean(sl) >= ONSET_AUC:
                onset = L / Lmax
                break
        rows.append({"model": model, "game_code": g, "onset_frac": onset,
                     "complexity_score": float(gmeta.loc[g, "complexity_score"]),
                     "iesds_depth": int(gmeta.loc[g, "iesds_depth"])})
    per_game = pd.DataFrame(rows)

    sub = per_game.dropna(subset=["onset_frac"])
    rho, pval = (spearmanr(sub["onset_frac"], sub["complexity_score"]) if len(sub) > 3 else (np.nan, np.nan))
    rho_ie, _ = (spearmanr(sub["onset_frac"], sub["iesds_depth"]) if len(sub) > 3 else (np.nan, np.nan))
    # bootstrap-by-game slope of onset ~ complexity
    slopes = []
    if len(sub) > 5:
        rng = np.random.default_rng(SEED)
        gv = sub["game_code"].to_numpy()
        for _ in range(2000):
            pick = rng.choice(gv, size=len(gv), replace=True)
            d = sub.set_index("game_code").loc[pick]
            if d["complexity_score"].std() > 0:
                slopes.append(np.polyfit(d["complexity_score"], d["onset_frac"], 1)[0])
    slope = float(np.mean(slopes)) if slopes else np.nan
    slo, shi = (float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))) if slopes else (np.nan, np.nan)
    summary = pd.DataFrame([{
        "model": model, "n_crystallized": int(len(sub)), "n_games": int(len(per_game)),
        "rho_onset_complexity": float(rho), "rho_pval": float(pval),
        "rho_onset_iesds": float(rho_ie),
        "slope_onset_complexity": slope, "slope_lo": slo, "slope_hi": shi,
        "median_onset_frac": float(sub["onset_frac"].median()) if len(sub) else np.nan,
    }])
    print(f"[{SHORT.get(model, model)}] N3 crystallization: {len(sub)}/{len(per_game)} games crystallized | "
          f"rho(onset, complexity)={rho:.3f} (p={pval:.3g}) | slope={slope:.3f} [{slo:.3f},{shi:.3f}]")
    per_game["_curve_tag"] = "per_game"
    return summary, pd.concat([per_game, curve.assign(onset_frac=np.nan, game_code=None, _curve_tag="curve")],
                              ignore_index=True)


# ---------------------------------------------------------------------------
# N4 -- two axes, not one
# ---------------------------------------------------------------------------
def run_n4(model: str, layers: list[int], d1c_by_game: dict, pre=None) -> pd.DataFrame:
    meta, X = pre if pre is not None else _prep(model, layers)
    idx = meta.index.to_numpy()
    games = meta["game_code"].to_numpy()
    gapL1 = meta["gap_L1"].to_numpy(float)                       # L1 belief (q=0.5)
    d1c = meta["game_code"].map(d1c_by_game).to_numpy(float)     # incentive (empirical q-hat)
    rows = []
    for L in layers:
        Z = _standardize(X[L][idx])
        dbelief, _ = _dir_cov(Z, gapL1)
        dinc, _ = _dir_cov(Z, d1c)
        # decodability = leak-free game-grouped CV AUC (NOT the in-sample direction
        # fit, which overfits at d=8192 and would falsely credit the null model)
        aB = probe_auc(X[L][idx], (gapL1 > 0).astype(int), games, pca_k=PCA_K, seed=SEED)["auc"]
        aI = probe_auc(X[L][idx], (d1c > 0).astype(int), games, pca_k=PCA_K, seed=SEED)["auc"]
        rows.append({"model": model, "layer": L, "depth_frac": L / max(layers),
                     "cos_dbelief_dinc": float(np.dot(dbelief, dinc)),
                     "decodability_dbelief_auc": aB, "decodability_dinc_auc": aI})
    df = pd.DataFrame(rows)
    print(f"[{SHORT.get(model, model)}] N4 two-axes: mean cos(d_belief,d_inc)={df['cos_dbelief_dinc'].mean():+.3f} | "
          f"decode-AUC d_belief={df['decodability_dbelief_auc'].max():.3f} d_inc={df['decodability_dinc_auc'].max():.3f}")
    return df


# ---------------------------------------------------------------------------
# N5 -- distributed vs localized code
# ---------------------------------------------------------------------------
def _participation_ratio(X: np.ndarray, y: np.ndarray) -> dict:
    """Scale-fair PR on a_i=(w_i*sigma_i)^2. Fitting on standardized features makes
    the logistic weight already = w_raw_i*sigma_i, so a_i = coef_i^2."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    if len(set(y.tolist())) < 2:
        return {"pr_over_d": np.nan, "frac90": np.nan}
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xs, y)
    a = clf.coef_.ravel() ** 2
    sa = a.sum()
    if sa <= 0:
        return {"pr_over_d": np.nan, "frac90": np.nan}
    pr = (sa ** 2) / float((a ** 2).sum())
    asort = np.sort(a)[::-1]
    n90 = int(np.searchsorted(np.cumsum(asort), 0.9 * sa) + 1)
    return {"pr_over_d": pr / len(a), "frac90": n90 / len(a)}


def _greedy_ablation(X: np.ndarray, y: np.ndarray, games: np.ndarray, full_auc: float) -> dict:
    """Rank dims by univariate |corr|, add greedily, GroupKFold AUC at log-spaced k;
    report fraction of dims to reach 90% of full AUC."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if not np.isfinite(full_auc) or len(set(y.tolist())) < 2:
        return {"dims_to_90pct_full_auc": np.nan}
    yc = y - y.mean()
    corr = np.abs((X - X.mean(0)).T @ yc) / (X.std(0) * np.sqrt(len(y)) + 1e-12)
    order = np.argsort(corr)[::-1]
    d = X.shape[1]
    ks = sorted({k for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, d] if k <= d})
    target = 0.9 * full_auc
    hit = np.nan
    for k in ks:
        cols = order[:k]
        oof = np.full(len(y), np.nan)
        for tr, te in _game_folds(games):
            if len(set(y[tr].tolist())) < 2:
                continue
            sc = StandardScaler().fit(X[tr][:, cols])
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(sc.transform(X[tr][:, cols]), y[tr])
            ci = list(clf.classes_).index(1)
            oof[te] = clf.predict_proba(sc.transform(X[te][:, cols]))[:, ci]
        u = np.isfinite(oof)
        if u.sum() > 2 and len(set(y[u].tolist())) == 2:
            auc = roc_auc_score(y[u], oof[u])
            if auc >= target:
                hit = k / d
                break
    return {"dims_to_90pct_full_auc": hit}


def run_n5(model: str, layers: list[int], pre=None) -> pd.DataFrame:
    meta, X = pre if pre is not None else _prep(model, layers)
    idx = meta.index.to_numpy()
    games = meta["game_code"].to_numpy()
    axes = {
        "decision": meta["canonical_action_p1"].to_numpy().astype(int),
        "incentive": (meta["gap_L1"].to_numpy(float) > 0).astype(int),
    }
    rows = []
    for L in layers:
        Xl = X[L][idx]
        for name, y in axes.items():
            full = probe_auc(Xl, y, games, pca_k=PCA_K, seed=SEED)["auc"]
            pr = _participation_ratio(Xl, y)
            abl = _greedy_ablation(Xl, y, games, full)
            rows.append({"model": model, "layer": L, "depth_frac": L / max(layers),
                         "axis": name, "full_auc": full, **pr, **abl})
    df = pd.DataFrame(rows)
    for name in axes:
        sub = df[df["axis"] == name]
        print(f"[{SHORT.get(model, model)}] N5 {name}: mean PR/d={sub['pr_over_d'].mean():.3f}, "
              f"mean dims->90%AUC={sub['dims_to_90pct_full_auc'].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# N6 -- two-choice dynamics across depth (proxy)
# ---------------------------------------------------------------------------
def run_n6(model: str, layers: list[int], faithful: bool = False, pre=None) -> pd.DataFrame:
    layers = sorted(layers)
    meta, X = pre if pre is not None else _prep(model, layers)
    idx = meta.index.to_numpy()
    games = meta["game_code"].to_numpy()
    aligned = meta["aligned"].to_numpy()       # 1 if model chose the canonical option
    y = aligned                                 # proxy target: did the model commit to canonical?
    Lmax = max(layers)
    rows = []
    for L in layers:
        p = oof_proba(X[L][idx], y, games, pca_k=PCA_K, seed=SEED)  # P(model chose canonical)
        for choice, mask in [("canonical", aligned == 1), ("non_canonical", aligned == 0)]:
            pm = p[mask]
            pm = pm[np.isfinite(pm)]
            if len(pm):
                rows.append({"model": model, "layer": L, "depth_frac": L / Lmax,
                             "actual_choice": choice,
                             "p_canonical_option": float(pm.mean()),
                             "margin": float(2 * pm.mean() - 1), "n": int(len(pm))})
    df = pd.DataFrame(rows)
    # commitment = shallowest depth where the two choice subsets' readouts SEPARATE,
    # i.e. P(canonical|chose canonical) - P(canonical|chose non-canonical) >= 0.5.
    # (A bare margin>0 fires at layer 0 because the prompt embedding already carries
    # some predictive signal; the separation is the honest "decision is decodable" mark.)
    piv = (df.pivot_table(index=["layer", "depth_frac"], columns="actual_choice",
                          values="p_canonical_option").reset_index().sort_values("layer"))
    piv["separation"] = piv.get("canonical", np.nan) - piv.get("non_canonical", np.nan)
    crossed = piv.loc[piv["separation"] >= 0.5, "depth_frac"]
    commit_frac = float(crossed.iloc[0]) if len(crossed) else np.nan
    late_sep = float(piv["separation"].iloc[-1]) if len(piv) else np.nan
    df = df.merge(piv[["layer", "separation"]], on="layer", how="left")
    print(f"[{SHORT.get(model, model)}] N6 proxy: commitment(depth_frac@sep>=0.5)={commit_frac:.3f} | "
          f"late separation={late_sep:+.3f}")
    if faithful:
        print(f"[{SHORT.get(model, model)}] N6 faithful logit-lens requested but not wired: "
              "requires HF weights (final norm + option-letter unembedding rows) present in the env; "
              "see module docstring. Shipping proxy only.")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, stem: str) -> None:
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {FIG_DIR / (stem + '.png')}")


def fig_n1(n1: pd.DataFrame, models: list[str]) -> None:
    style(); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        s = n1[n1["model"] == m].sort_values("layer")
        ax.plot(s["depth_frac"], s["auc"], label=SHORT.get(m, m), color=COL.get(m))
    ax.axhline(0.5, color="grey", lw=0.7, ls="--")
    ax.set_xlabel("network depth (fraction)"); ax.set_ylabel("level-decode AUC (L1 vs L2+)")
    ax.set_title("N1: reasoning level is a readable internal state")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); _save(fig, "n1_level_decode")


def fig_n3(curve: pd.DataFrame, models: list[str]) -> None:
    style(); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in models:
        c = curve[(curve["model"] == m) & (curve["_curve_tag"] == "curve")].sort_values("layer")
        axes[0].plot(c["depth_frac"], c["auc"], label=SHORT.get(m, m), color=COL.get(m))
        pg = curve[(curve["model"] == m) & (curve["_curve_tag"] == "per_game")].dropna(subset=["onset_frac"])
        axes[1].scatter(pg["complexity_score"], pg["onset_frac"], s=10, alpha=0.5,
                        color=COL.get(m), label=SHORT.get(m, m))
    axes[0].axhline(ONSET_AUC, color="grey", lw=0.7, ls="--")
    axes[0].set_xlabel("network depth (fraction)"); axes[0].set_ylabel("canonical-action AUC")
    axes[0].set_title("N3: crystallization curve"); axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_xlabel("complexity_score"); axes[1].set_ylabel("crystallization depth (fraction)")
    axes[1].set_title("N3: deeper games crystallize later"); axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); _save(fig, "n3_depth_vs_complexity")


def fig_n6(n6: pd.DataFrame, models: list[str]) -> None:
    style(); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        for choice, ls in [("canonical", "-"), ("non_canonical", "--")]:
            s = n6[(n6["model"] == m) & (n6["actual_choice"] == choice)].sort_values("layer")
            ax.plot(s["depth_frac"], s["p_canonical_option"], ls, color=COL.get(m),
                    label=f"chose {choice}")
        ax.axhline(0.5, color="grey", lw=0.7, ls=":")
        ax.set_title(SHORT.get(m, m)); ax.set_xlabel("depth (frac)")
        ax.legend(fontsize=6, frameon=False)
    axes[0].set_ylabel("P(canonical option)")
    fig.suptitle("N6: option competition across depth, split by the model's actual choice")
    fig.tight_layout(); _save(fig, "n6_option_dynamics_proxy")


# ---------------------------------------------------------------------------
def _verdict(n1: pd.DataFrame, n2: pd.DataFrame, models: list[str]) -> None:
    print("\n=== SPEC §3 verdict ===")
    for m in models:
        a1 = np.nanmax(n1[n1["model"] == m]["auc"]) if len(n1) else np.nan
        n2m = n2[n2["model"] == m]
        if len(n2m):
            best = n2m.loc[n2m["auc_sign_gapL1"].idxmax()]  # most belief-informative layer
            aL1, aEQ = best["auc_sign_gapL1"], best["auc_sign_gapEQ"]
            cL1, cEQ = best["cos_dL1_choice"], best["cos_dEQ_choice"]
        else:
            aL1 = aEQ = cL1 = cEQ = np.nan
        n1_pass = np.isfinite(a1) and a1 > 0.6
        # identification-by-activation: the L1 belief sign is decodable AND more so
        # than the EQ belief sign (the network represents L1 better than equilibrium).
        n2_pass = bool(np.isfinite(aL1) and aL1 > 0.6 and aL1 > aEQ)
        tag = ("LEVEL DECODED + belief axis represents L1 > EQ" if (n1_pass and n2_pass)
               else "behavioral L1 stands; activation-ID claim weak")
        print(f"  {SHORT.get(m, m):9s} N1 maxAUC={a1:.3f} {'PASS' if n1_pass else 'weak'} | "
              f"N2 AUC L1={aL1:.3f}/EQ={aEQ:.3f} (cos(L1,ch)={cL1:+.3f}/cos(EQ,ch)={cEQ:+.3f}) "
              f"{'PASS' if n2_pass else 'weak'} -> {tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Level-k geometry neural tests (one-shot, SPEC N1-N6).")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--tests", nargs="*", default=["n1", "n2", "n3", "n4", "n5", "n6"])
    ap.add_argument("--quick", action="store_true", help="use only steer layers (fast smoke test)")
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="explicit layer list override")
    ap.add_argument("--faithful", action="store_true", help="attempt the faithful N6 logit-lens (needs HF weights)")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(GAME_FEATURES)
    canon1 = {r.game_code: int(r.canonical_action_p1) for r in feats.itertuples() if pd.notna(r.canonical_action_p1)}
    canon2 = {r.game_code: int(r.canonical_action_p2) for r in feats.itertuples() if pd.notna(r.canonical_action_p2)}

    # phase-4: guard repointed from the superseded A/B root (output/oneshot_main/substrate,
    # which cannot exist in a clone, so `models` was always empty) to the released root that
    # the imported _iter_games() actually reads.
    models = [m for m in args.models if (substrate_root() / m).exists()]
    n1_all, n2_all, n3_all, n3_curve, n4_all, n5_all, n6_all = ([] for _ in range(7))

    for m in models:
        layers = (args.layers if args.layers is not None
                  else (_steer_layers(m) if args.quick else _capture_layers(m)))
        # N5 is the expensive test -> always restrict to steer layers
        n5_layers = [L for L in (args.layers if args.layers is not None else _steer_layers(m))
                     if L in layers]
        # load every game's acts.npz ONCE for all requested layers, share across tests
        load_layers = sorted(set(layers) | set(n5_layers))
        print(f"\n[{SHORT.get(m, m)}] loading P1 baseline residuals: {len(load_layers)} layers ...")
        pre = _prep(m, load_layers)
        print(f"[{SHORT.get(m, m)}] loaded {pre[0].shape[0]} rows x {pre[1][load_layers[0]].shape[1]} dims")
        # each test is isolated: a failure on one model/test must not discard the rest
        def _try(name, fn):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[{SHORT.get(m, m)}] {name} FAILED: {type(exc).__name__}: {exc}")
                return None
        if "n1" in args.tests:
            r = _try("N1", lambda: run_n1(m, layers, pre=pre)); n1_all.append(r) if r is not None else None
        if "n2" in args.tests:
            r = _try("N2", lambda: run_n2(m, layers, pre=pre)); n2_all.append(r) if r is not None else None
        if "n3" in args.tests:
            r = _try("N3", lambda: run_n3(m, layers, pre=pre))
            if r is not None:
                n3_all.append(r[0]); n3_curve.append(r[1])
        if "n4" in args.tests:
            dt = build_delta_tables(m, canon1, canon2)
            d1c = dict(zip(dt["game_code"], dt["delta1_c"]))
            r = _try("N4", lambda: run_n4(m, layers, d1c, pre=pre)); n4_all.append(r) if r is not None else None
        if "n5" in args.tests:
            r = _try("N5", lambda: run_n5(m, n5_layers, pre=pre)); n5_all.append(r) if r is not None else None
        if "n6" in args.tests:
            r = _try("N6", lambda: run_n6(m, layers, faithful=args.faithful, pre=pre))
            n6_all.append(r) if r is not None else None
        del pre

    def _dump(frames, name):
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(TAB_DIR / name, index=False)
            print(f"wrote {TAB_DIR / name}")
            return df
        return pd.DataFrame()

    n1 = _dump(n1_all, "n1_level_decode.csv")
    n2 = _dump(n2_all, "n2_belief_axis.csv")
    _dump(n3_all, "n3_crystallization_summary.csv")
    n3c = _dump(n3_curve, "n3_crystallization_detail.csv")
    _dump(n4_all, "n4_axes.csv")
    _dump(n5_all, "n5_distributedness.csv")
    n6 = _dump(n6_all, "n6_option_dynamics.csv")

    if not args.no_fig:
        if len(n1):
            fig_n1(n1, models)
        if len(n3c):
            fig_n3(n3c, models)
        if len(n6):
            fig_n6(n6, models)
    if len(n1) and len(n2):
        _verdict(n1, n2, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
