#!/usr/bin/env python3
"""Layer A final — attribution & variance decomposition (Notebook 2 / Supplementary).

Rebuild of ``analysis/block_a/layer1_conformity_attribution.ipynb`` on the corrected
integrated-root REALIZED DECISIONS (dense=parse-ok decoded action, GPT-OSS=resolved
realized action with mixed rows included and no-commit rows dropped). Produces the headline
partition the main text cites and a Supplementary figure:

  * conformity dataset: per usable P1 cell across conditions (7 conditions x 4 cb x
    144 games per model, minus no-commit rows), y = played-canonical on the realized
    decision, with a STRUCTURE block (|Delta1^c|, num_pure_ne, complexity, iesds depth,
    level-k class), a TRAIT block (cue family), and MODEL identity.
  * variance_partition(): gradient-boosted classifier, GroupKFold-by-game OOF, grouped
    permutation importance -> structure vs trait vs model %, trait-block dAUC + boot CI.
  * shap_attribution(): TreeSHAP of |Delta1^c| (the quantal curve as attribution).
  * glmm(): aligned ~ |Delta1^c| + trait family + C(lk) + (1|game).

Tables: s_variance_partition.csv, s_attribution_shap.csv, s_glmm.csv.
Figure:  figS_attribution.{png,pdf}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from collection.oneshot_common import delta1, canonical_sign  # noqa: E402
from steering.extract_directions import load_behavior  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


DATA = SD.DATA
TABLES = results_root() / "layer_a"
FIGURES = ROOT / "analysis" / "layer_a" / "figures"

CONDITIONS = ["baseline", "cue_risk_aversion", "cue_loss_aversion",
              "cue_inequity_aversion", "cue_selfish_maximizer", "cue_maximin",
              "cue_length_match_null"]
LK = SD.LK_ORDER
RNG = np.random.default_rng(SD.RNG_SEED)


# ---- dataset ----------------------------------------------------------------
def conformity_dataset() -> pd.DataFrame:
    """Per P1 cell across all conditions: y_canon + structure + trait + model."""
    m = SD.master()
    meta = m[["game_code", "canonical_action_p1", "canonical_8vec", "nagel_lk_type",
              "num_pure_ne", "complexity_score", "iesds_depth"]].copy()
    # per-game |Delta1^c| at q=0.5 (structure incentive clarity)
    gap = []
    for _, r in meta.iterrows():
        vec = SD.parse_8vec(r["canonical_8vec"])
        gap.append(abs(canonical_sign(delta1(vec, 0.5), int(r["canonical_action_p1"]))))
    meta["abs_delta1c"] = gap
    canon = dict(zip(meta["game_code"], meta["canonical_action_p1"]))

    # y = played-canonical on the corrected integrated-root REALIZED DECISION.
    # Dense rows are parse_ok-gated; GPT-OSS no-commit rows are dropped and mixed rows
    # are retained after their resolved 0/1 realized_action.
    frames = []
    for mod in SD.MODELS:
        dec = SD.load_decisions(mod)
        p1 = dec[(dec["player"] == 1) & (dec["condition"].isin(CONDITIONS)) & (dec["ok"])].copy()
        p1["model"] = mod
        p1["y"] = (p1["action"].astype(int) == p1["game_code"].map(canon).astype(int)).astype(int)
        frames.append(p1[["model", "game_code", "condition", "y"]])
    df = pd.concat(frames, ignore_index=True).merge(
        meta[["game_code", "nagel_lk_type", "num_pure_ne", "complexity_score",
              "iesds_depth", "abs_delta1c"]], on="game_code", how="left")
    # trait family from condition
    df["trait"] = df["condition"].str.replace("cue_", "", regex=False)
    return df


def _design(df: pd.DataFrame):
    """Return X (DataFrame), y, groups, and the column->group map."""
    struct = ["abs_delta1c", "num_pure_ne", "complexity_score", "iesds_depth"]
    lk_oh = pd.get_dummies(df["nagel_lk_type"], prefix="lk")
    trait_oh = pd.get_dummies(df["trait"], prefix="trait")
    model_oh = pd.get_dummies(df["model"], prefix="model")
    X = pd.concat([df[struct], lk_oh, trait_oh, model_oh], axis=1).astype(float)
    groups_map = {}
    for c in struct: groups_map[c] = "structure"
    for c in lk_oh.columns: groups_map[c] = "structure"
    for c in trait_oh.columns: groups_map[c] = "trait"
    for c in model_oh.columns: groups_map[c] = "model"
    return X, df["y"].to_numpy(int), df["game_code"].to_numpy(), groups_map


# ---- variance partition (grouped permutation importance, GroupKFold-by-game) -
def variance_partition(df: pd.DataFrame | None = None, n_splits: int = 5):
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier

    if df is None:
        df = conformity_dataset()
    X, y, games, gmap = _design(df)
    cols = list(X.columns)
    Xv = X.to_numpy(float)
    blocks = ["structure", "trait", "model"]
    block_cols = {b: [i for i, c in enumerate(cols) if gmap[c] == b] for b in blocks}

    pred_base = np.full(len(y), np.nan)
    pred_perm = {b: np.full(len(y), np.nan) for b in blocks}
    rng = np.random.default_rng(SD.RNG_SEED)
    for tr, te in GroupKFold(n_splits).split(Xv, y, games):
        clf = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(Xv[tr], y[tr])
        pred_base[te] = clf.predict_proba(Xv[te])[:, 1]
        for b in blocks:
            Xp = Xv[te].copy()
            perm = rng.permutation(len(te))
            for j in block_cols[b]:
                Xp[:, j] = Xp[perm, j]
            pred_perm[b][te] = clf.predict_proba(Xp)[:, 1]

    auc_base = roc_auc_score(y, pred_base)
    dauc = {b: auc_base - roc_auc_score(y, pred_perm[b]) for b in blocks}
    tot = sum(max(0.0, v) for v in dauc.values()) or 1.0
    pct = {b: 100.0 * max(0.0, dauc[b]) / tot for b in blocks}

    # bootstrap-by-game CI on each block's dAUC
    ug = np.unique(games)
    idx_by_g = {g: np.where(games == g)[0] for g in ug}
    boots = {b: [] for b in blocks}
    for _ in range(SD.BOOT_N // 2):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        ab = roc_auc_score(y[ii], pred_base[ii])
        for b in blocks:
            boots[b].append(ab - roc_auc_score(y[ii], pred_perm[b][ii]))

    rows = []
    for b in blocks:
        lo, hi = (np.percentile(boots[b], [2.5, 97.5]) if boots[b] else (np.nan, np.nan))
        rows.append({"block": b, "pct": pct[b], "dAUC": dauc[b],
                     "dAUC_lo": float(lo), "dAUC_hi": float(hi)})
    out = pd.DataFrame(rows)
    out.attrs["auc_base"] = auc_base
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "s_variance_partition.csv", index=False)
    return out


def shap_attribution(df: pd.DataFrame | None = None, n_sample: int = 12000):
    """TreeSHAP attribution: mean |SHAP| per feature (the quantal curve re-expressed as
    attribution -> |Delta1^c| should dominate the structural drivers)."""
    import shap
    from lightgbm import LGBMClassifier
    if df is None:
        df = conformity_dataset()
    X, y, games, gmap = _design(df)
    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
    clf.fit(X, y)
    rng = np.random.default_rng(SD.RNG_SEED)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    Xs = X.iloc[idx]
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv          # class-1 contributions
    msh = np.abs(sv).mean(0)
    rows = [{"feature": c, "block": gmap[c], "mean_abs_shap": float(msh[i])}
            for i, c in enumerate(X.columns)]
    out = (pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "s_attribution_shap.csv", index=False)
    return out


def glmm(df: pd.DataFrame | None = None):
    """Mixed logistic: aligned ~ |Delta1^c| + trait family + C(lk) + (1|game).
    Tries BinomialBayesMixedGLM (vb); falls back to game-clustered Logit."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    if df is None:
        df = conformity_dataset()
    d = df.copy()
    d["trait"] = pd.Categorical(d["trait"], categories=["baseline", "risk_aversion", "loss_aversion",
                                "inequity_aversion", "selfish_maximizer", "maximin", "length_match_null"])
    formula = "y ~ abs_delta1c + C(trait) + C(nagel_lk_type)"
    method = "binomial_bayes_mixed_glm_vb"
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        md = BinomialBayesMixedGLM.from_formula(formula, {"game": "0 + C(game_code)"}, data=d)
        res = md.fit_vb()
        params = pd.Series(res.fe_mean, index=res.model.exog_names[:len(res.fe_mean)])
        sds = pd.Series(res.fe_sd, index=res.model.exog_names[:len(res.fe_sd)])
        tab = pd.DataFrame({"coef": params, "sd": sds})
        tab["lo"] = tab["coef"] - 1.96 * tab["sd"]
        tab["hi"] = tab["coef"] + 1.96 * tab["sd"]
        tab = tab.reset_index().rename(columns={"index": "term"})
    except Exception as e:
        method = f"game_clustered_logit (GLMM fallback: {type(e).__name__})"
        res = smf.logit(formula, data=d).fit(disp=0, cov_type="cluster",
                                             cov_kwds={"groups": d["game_code"]})
        ci = res.conf_int()
        tab = pd.DataFrame({"term": res.params.index, "coef": res.params.values,
                            "lo": ci[0].values, "hi": ci[1].values})
    tab["method"] = method
    TABLES.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TABLES / "s_glmm.csv", index=False)
    return tab, method


def render_figure(vp, shap_tab, glmm_tab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from strategic_anatomy import paper_style as S
    S.apply()
    BLOCKCOL = {"structure": S.GOOD_GREEN, "trait": S.SOFT_BLUE, "model": S.GREY}

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 0.8, 1.1], wspace=0.55,
                           left=0.13, right=0.985, top=0.84, bottom=0.30)

    # (a) SHAP bar — top features
    axa = fig.add_subplot(gs[0])
    top = shap_tab.head(9)[::-1]
    axa.barh(range(len(top)), top["mean_abs_shap"], color=[BLOCKCOL[b] for b in top["block"]])
    axa.set_yticks(range(len(top))); axa.set_yticklabels(top["feature"], fontsize=S.FS_FOOT)
    axa.set_xlabel("mean |SHAP|", fontsize=S.FS_AXIS); axa.tick_params(labelsize=S.FS_TICK)
    S.panel_title(axa, "a", "Attribution (TreeSHAP)", fontsize=S.FS_PANEL)

    # (b) variance partition
    axb = fig.add_subplot(gs[1])
    blocks = ["structure", "trait", "model"]
    vv = vp.set_index("block").reindex(blocks)
    x0 = 0
    for b in blocks:
        axb.bar(0, vv.loc[b, "pct"], bottom=x0, color=BLOCKCOL[b], width=0.6,
                label=f"{b} {vv.loc[b,'pct']:.0f}%")
        x0 += vv.loc[b, "pct"]
    axb.set_ylim(0, 100); axb.set_xlim(-0.6, 0.6); axb.set_xticks([])
    axb.set_ylabel("variance share (%)", fontsize=S.FS_AXIS); axb.tick_params(labelsize=S.FS_TICK)
    axb.legend(fontsize=S.FS_FOOT, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    S.panel_title(axb, "b", "Variance partition", fontsize=S.FS_PANEL)

    # (c) GLMM coefficients
    axc = fig.add_subplot(gs[2])
    keep = glmm_tab[~glmm_tab["term"].str.contains("Intercept|game", case=False, regex=True)].copy()
    keep["short"] = (keep["term"].str.replace("C(trait)[T.", "", regex=False)
                     .str.replace("C(nagel_lk_type)[T.", "lk:", regex=False)
                     .str.replace("]", "", regex=False).str.replace("_aversion", "", regex=False))
    keep = keep.iloc[::-1]
    yy = range(len(keep))
    axc.errorbar(keep["coef"], yy, xerr=[keep["coef"] - keep["lo"], keep["hi"] - keep["coef"]],
                 fmt="o", ms=3, color=S.INK, ecolor=S.SUBTLE, elinewidth=0.8, capsize=1.5)
    axc.axvline(0, color=S.REF_RED, lw=0.7, ls="--")
    axc.set_yticks(list(yy)); axc.set_yticklabels(keep["short"], fontsize=5.6)
    axc.set_xlabel("logit coefficient", fontsize=S.FS_AXIS); axc.tick_params(labelsize=S.FS_TICK)
    S.panel_title(axc, "c", "GLMM (conformity)", fontsize=S.FS_PANEL)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"figS_attribution.{ext}", dpi=600 if ext == "png" else 300,
                    bbox_inches="tight")
    plt.close(fig)
    print("[attribution] wrote figS_attribution.{png,pdf}")


def main(force: bool = False) -> int:
    df = conformity_dataset()
    print(f"[attribution] conformity dataset: {len(df)} P1 cells "
          f"({df['model'].nunique()} models x {df['game_code'].nunique()} games x "
          f"{df['condition'].nunique()} conditions), base rate y={df['y'].mean():.3f}")

    vp_cache = TABLES / "s_variance_partition.csv"
    if vp_cache.exists() and not force:
        vp = pd.read_csv(vp_cache)
        print("[attribution] variance partition loaded from cache")
    else:
        vp = variance_partition(df)
    print("[attribution] variance partition (grouped permutation importance, GroupKFold-by-game):")
    for _, r in vp.iterrows():
        print(f"   {r['block']:10s} {r['pct']:5.1f}%   dAUC={r['dAUC']:+.3f} "
              f"[{r['dAUC_lo']:+.3f}, {r['dAUC_hi']:+.3f}]")

    shap_tab = shap_attribution(df)
    print("\n[attribution] top SHAP features:")
    print(shap_tab.head(6).to_string(index=False))

    glmm_tab, method = glmm(df)
    print(f"\n[attribution] GLMM ({method}): key terms")
    show = glmm_tab[glmm_tab["term"].str.contains("delta1c|trait|maximin", case=False, regex=True)]
    print(show[["term", "coef", "lo", "hi"]].round(3).to_string(index=False))

    render_figure(vp, shap_tab, glmm_tab)
    print(f"\nwrote {TABLES/'s_variance_partition.csv'} ; {TABLES/'s_attribution_shap.csv'} ; {TABLES/'s_glmm.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
