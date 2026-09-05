#!/usr/bin/env python3
"""Layer A final attribution analysis for Figure 2 and supplement.

All models use corrected integrated-root decisions from
``$SCA_DATA_ROOT/substrate``. Dense/chat models use parse-ok decoded actions;
GPT-OSS uses resolved final-channel actions, including mixed rows and dropping only
no-commit rows.
The outcome is canonical conformity: generated/commit action equals the
canonical action. The primary predictor set is deliberately lean: one
canonical-signed incentive gradient, one complexity composite, maximin geometry,
game regime, Pareto-rankability, randomized cue, and model identity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.probe_common import game_meta  # noqa: E402
from analysis.layer_a.build_trait_steering_oneshot import load_p1  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


LAYERA = ROOT / "analysis" / "layer_a"
TAB_DIR = results_root() / "layer_a"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
PANEL_DIR = FIG_DIR / "_attrib_panel_data"

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
DENSE_MODELS = ["qwen", "qwen_instruct", "llama31_instruct"]

FAMILY = {
    "baseline": "baseline",
    "cue_risk_aversion": "preference",
    "cue_loss_aversion": "preference",
    "cue_inequity_aversion": "preference",
    "cue_selfish_maximizer": "preference",
    "cue_maximin": "rule",
    "cue_length_match_null": "placebo",
}
TRAIT_NAME = {
    "cue_risk_aversion": "risk_aversion",
    "cue_loss_aversion": "loss_aversion",
    "cue_inequity_aversion": "inequity_aversion",
    "cue_selfish_maximizer": "selfish_maximizer",
    "cue_maximin": "maximin",
    "cue_length_match_null": "length_match_null",
}
LK_ORDER = ["CO1", "CO2", "DD", "MP", "OD1", "OD2"]
TRAIT_ORDER = [
    "length_match_null",
    "maximin",
    "selfish_maximizer",
    "risk_aversion",
    "inequity_aversion",
    "loss_aversion",
]

PRIMARY_NUMERIC_STRUCTURE = [
    "dInc_signed_unif",          # Δ_c = EU(canonical) - EU(non-canonical), q=0.5
    "complexity_score",          # composite complexity index; do not also include components
    "maximin_to_ne_action_distance",
    "maximin_to_ne_payoff_distance",
    "maximin_is_ne",
]
PRIMARY_CAT_STRUCTURE = [
    "nagel_lk_type",
    "ne_pareto_rankable",
]
STEP_LABELS = [
    ("M0", "model identity"),
    ("M1", "+ structure"),
    ("M2", "+ cue"),
    ("M3", "+ structure x cue"),
]
SEED = 20260520
N_BOOT = 2000


def _bool_cat(x) -> str:
    if pd.isna(x):
        return "na"
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in {"true", "1", "yes"}:
            return "true"
        if xs in {"false", "0", "no"}:
            return "false"
        if xs in {"", "nan", "none", "na"}:
            return "na"
    return "true" if bool(x) else "false"


def _bool_float(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in {"true", "1", "yes"}:
            return 1.0
        if xs in {"false", "0", "no"}:
            return 0.0
        return np.nan
    return float(bool(x))


def build_long() -> pd.DataFrame:
    """Build the corrected one-shot canonical-conformity table."""
    meta = game_meta().copy()
    meta["nagel_lk_type"] = meta["nagel_lk_type"].fillna(meta.get("lk_type", "unknown")).astype(str)
    meta["ne_pareto_rankable"] = meta["ne_pareto_rankable"].map(_bool_cat)
    meta["maximin_is_ne"] = meta["maximin_is_ne"].map(_bool_float)

    meta_idx = meta.set_index("game_code")
    canon = meta_idx["canonical_action_p1"].astype(int).to_dict()

    frames: list[pd.DataFrame] = []
    for model in MODELS:
        p1 = load_p1(model)
        if p1.empty:
            continue
        d = p1[p1["decoded_action"].isin([0, 1])].copy()
        d["model"] = model
        d["canon"] = d["game_code"].map(canon).astype(int)
        d["y"] = (d["decoded_action"].astype(int) == d["canon"]).astype(int)
        d["analysis_set"] = "gptoss_full" if model == "gptoss" else "dense_primary"
        d["family"] = d["condition"].map(FAMILY).fillna("other")
        d["trait_name"] = d["condition"].map(TRAIT_NAME)

        d["dInc_signed_unif"] = d["game_code"].map(meta_idx["delta1_canonical"]).astype(float)

        keep = [
            "nagel_lk_type",
            "complexity_score",
            "maximin_to_ne_action_distance",
            "maximin_to_ne_payoff_distance",
            "maximin_is_ne",
            "ne_pareto_rankable",
        ]
        d = d.merge(meta[["game_code"] + keep], on="game_code", how="left")
        frames.append(d)

    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    for c in PRIMARY_NUMERIC_STRUCTURE:
        long[c] = pd.to_numeric(long[c], errors="coerce")
    long = long[long["family"].isin(["baseline", "placebo", "preference", "rule"])].copy()
    long = long.dropna(subset=["y", "game_code", "model"]).reset_index(drop=True)
    return long


def _dummy(series: pd.Series, prefix: str, order: list[str] | None = None) -> pd.DataFrame:
    vals = series.astype("string").fillna("na")
    d = pd.get_dummies(vals, dtype=float)
    if order is not None:
        for c in order:
            if c not in d:
                d[c] = 0.0
        extra = [c for c in d.columns if c not in order]
        d = d[order + extra]
    d = d.rename(columns={c: f"{prefix}={c}" for c in d.columns})
    return d.reset_index(drop=True)


def _model_dummies(long: pd.DataFrame) -> pd.DataFrame:
    return _dummy(long["model"], "model", MODELS)


def _trait_dummies(long: pd.DataFrame) -> pd.DataFrame:
    vals = long["trait_name"].astype("string")
    d = pd.get_dummies(vals.fillna("__baseline__"), dtype=float)
    d = d.drop(columns=["__baseline__"], errors="ignore")
    for c in TRAIT_ORDER:
        if c not in d:
            d[c] = 0.0
    d = d[[c for c in TRAIT_ORDER if c in d] + [c for c in d.columns if c not in TRAIT_ORDER]]
    return d.rename(columns={c: f"trait={c}" for c in d.columns}).reset_index(drop=True)


def _structure_matrix(long: pd.DataFrame) -> pd.DataFrame:
    numeric = long[PRIMARY_NUMERIC_STRUCTURE].astype(float).reset_index(drop=True)
    cats: list[pd.DataFrame] = []
    if "nagel_lk_type" in PRIMARY_CAT_STRUCTURE:
        cats.append(_dummy(long["nagel_lk_type"], "nagel_lk_type", LK_ORDER))
    if "ne_pareto_rankable" in PRIMARY_CAT_STRUCTURE:
        cats.append(_dummy(long["ne_pareto_rankable"], "ne_pareto_rankable", ["false", "true", "na"]))
    return pd.concat([numeric, *cats], axis=1)


def full_design(long: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    struct = _structure_matrix(long)
    trait = _trait_dummies(long)
    model = _model_dummies(long)
    X = pd.concat([struct, trait, model], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    blocks = {
        "structure": list(struct.columns),
        "trait": list(trait.columns),
        "model": list(model.columns),
    }
    return X, blocks


def design_matrix(long: pd.DataFrame, step: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    model = _model_dummies(long)
    struct = _structure_matrix(long)
    trait = _trait_dummies(long)
    if step in {"M0", "M1", "M2", "M3"}:
        parts.append(model)
    if step in {"M1", "M2", "M3"}:
        parts.append(struct)
    if step in {"M2", "M3"}:
        parts.append(trait)
    if step == "M3":
        inter = {}
        for s in struct.columns:
            sv = struct[s].to_numpy(float)
            for t in trait.columns:
                inter[f"{s} x {t}"] = sv * trait[t].to_numpy(float)
        parts.append(pd.DataFrame(inter))
    if not parts:
        return pd.DataFrame(index=long.index)
    X = pd.concat(parts, axis=1)
    return X.replace([np.inf, -np.inf], np.nan)


def _constant_oof(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan, float)
    uniq = np.unique(groups)
    k = max(2, min(5, len(uniq)))
    for tr, te in GroupKFold(n_splits=k).split(np.zeros((len(y), 1)), y, groups):
        out[te] = float(np.mean(y[tr]))
    return out


def _hgbm_classifier(*, interactions: bool) -> HistGradientBoostingClassifier:
    kw = dict(
        max_iter=300,
        learning_rate=0.05,
        random_state=SEED,
        early_stopping=False,
        l2_regularization=1.0,
    )
    if interactions:
        kw.update(max_depth=3, max_leaf_nodes=31)
    else:
        kw.update(max_depth=None, max_leaf_nodes=2, interaction_cst="no_interactions")
    return HistGradientBoostingClassifier(**kw)


def oof_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0 or all(X[c].nunique(dropna=False) <= 1 for c in X.columns):
        return _constant_oof(y, groups)
    uniq = np.unique(groups)
    k = max(2, min(5, len(uniq)))
    out = np.full(len(y), np.nan, float)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            out[te] = float(np.mean(y[tr]))
            continue
        clf = _hgbm_classifier(interactions=False)
        clf.fit(X.iloc[tr], y[tr])
        out[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    keep = np.isfinite(p)
    if keep.sum() < 2 or len(np.unique(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], p[keep]))


def _auc_ci_by_game(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    games = np.unique(groups)
    by_game = {g: np.flatnonzero(groups == g) for g in games}
    vals = []
    for _ in range(N_BOOT):
        draw = rng.choice(games, size=len(games), replace=True)
        idx = np.concatenate([by_game[g] for g in draw])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def nested_auc(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = [
        ("dense_primary", DENSE_MODELS, "dense primary"),
        ("gptoss_full", ["gptoss"], "GPT-OSS full corrected-root sample"),
    ]
    for set_id, models, label in sets:
        d = long[long["model"].isin(models)].copy().reset_index(drop=True)
        if d.empty:
            continue
        y = d["y"].to_numpy(int)
        groups = d["game_code"].to_numpy()
        prev_auc = np.nan
        for step, step_label in STEP_LABELS:
            X = design_matrix(d, step)
            p = oof_predict(X, y, groups) if X.shape[1] else _constant_oof(y, groups)
            auc = _auc(y, p)
            lo, hi = _auc_ci_by_game(y, p, groups)
            rows.append({
                "analysis_set": set_id,
                "analysis_label": label,
                "step": step,
                "step_label": step_label,
                "auc": auc,
                "ci_lo": lo,
                "ci_hi": hi,
                "delta_prev": auc - prev_auc if np.isfinite(prev_auc) else np.nan,
                "n_games": int(d["game_code"].nunique()),
                "n_rows": int(len(d)),
                "models": ",".join(models),
            })
            prev_auc = auc
    out = pd.DataFrame(rows)
    out.to_csv(TAB_DIR / "f2_nested_auc.csv", index=False)

    dense = out[out["analysis_set"].eq("dense_primary")].copy()
    panel = pd.DataFrame({
        "step": ["M0  model only", "M1  + structure", "M2  + cue", "M3  + interactions"],
        "auc": dense["auc"].to_numpy(float),
        "ci_lo": dense["ci_lo"].to_numpy(float),
        "ci_hi": dense["ci_hi"].to_numpy(float),
    })
    panel.to_csv(PANEL_DIR / "panelB_nested_auc.csv", index=False)
    return out


def variance_partition(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X, blocks = full_design(long)
    y = long["y"].to_numpy(int)
    groups = long["game_code"].to_numpy()
    rng = np.random.default_rng(SEED)
    k = max(2, min(5, len(np.unique(groups))))
    oof = np.full(len(y), np.nan)
    drops = {b: [] for b in blocks}
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        clf = _hgbm_classifier(interactions=True)
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        a0 = roc_auc_score(y[te], p) if len(np.unique(y[te])) > 1 else np.nan
        for b, cols in blocks.items():
            Xp = X.iloc[te].copy()
            for c in cols:
                Xp[c] = rng.permutation(Xp[c].to_numpy())
            ap = roc_auc_score(y[te], clf.predict_proba(Xp)[:, 1]) if len(np.unique(y[te])) > 1 else np.nan
            drops[b].append(max(0.0, a0 - ap) if np.isfinite(a0) and np.isfinite(ap) else np.nan)

    mean_drop = {b: float(np.nanmean(drops[b])) for b in blocks}
    tot = sum(v for v in mean_drop.values() if np.isfinite(v)) or 1.0
    out = pd.DataFrame({
        "component": ["structure", "trait", "model"],
        "auc_drop": [mean_drop["structure"], mean_drop["trait"], mean_drop["model"]],
        "pct": [
            100.0 * mean_drop["structure"] / tot,
            100.0 * mean_drop["trait"] / tot,
            100.0 * mean_drop["model"] / tot,
        ],
    })
    out.attrs["base_auc"] = _auc(y, oof)
    return out, X, y, groups


def block_shares(shap_tab: pd.DataFrame) -> pd.DataFrame:
    out = (
        shap_tab.groupby("block", as_index=False)["mean_abs_shap"]
        .sum()
        .rename(columns={"mean_abs_shap": "sum_mean_abs_shap"})
    )
    total = float(out["sum_mean_abs_shap"].sum())
    out["pct"] = np.where(total > 0, 100.0 * out["sum_mean_abs_shap"] / total, np.nan)
    order = {"structure": 0, "trait": 1, "model": 2}
    out["order"] = out["block"].map(order).fillna(99)
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def _blockof(feature: str) -> str:
    if feature.startswith("model="):
        return "model"
    if feature.startswith("trait="):
        return "trait"
    return "structure"


def _as_binary_shap(values) -> np.ndarray:
    sv = values[1] if isinstance(values, list) else values
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv


def shap_attribution(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    try:
        import shap
    except Exception as e:  # noqa: BLE001
        print(f"  [shap] unavailable ({e}); skipping SHAP tables")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), float("nan")

    k = max(2, min(5, len(np.unique(groups))))
    sv = np.full(X.shape, np.nan, float)
    try:
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            clf = _hgbm_classifier(interactions=True)
            clf.fit(X.iloc[tr], y[tr])
            expl = shap.TreeExplainer(clf)
            sv[te] = _as_binary_shap(expl.shap_values(X.iloc[te]))
    except Exception as e:  # noqa: BLE001
        print(f"  [shap] TreeExplainer failed ({e}); skipping SHAP tables")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), float("nan")

    mean_abs = np.abs(sv).mean(axis=0)
    shap_tab = (
        pd.DataFrame({
            "feature": X.columns,
            "block": [_blockof(c) for c in X.columns],
            "mean_abs_shap": mean_abs,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    x_s = X.reset_index(drop=True)
    sv_df = pd.DataFrame(sv, columns=X.columns)
    dep = pd.DataFrame({
        "delta_c": x_s["dInc_signed_unif"].astype(float),
        "shap_delta_c": sv_df["dInc_signed_unif"].astype(float),
        "model": long["model"].to_numpy(object),
    })
    rho = float(dep[["delta_c", "shap_delta_c"]].corr(method="spearman").iloc[0, 1])
    dep.to_csv(PANEL_DIR / "panelA_shap_delta1.csv", index=False)
    with open(PANEL_DIR / "panelA_shap_delta1_meta.json", "w", encoding="utf-8") as f:
        json.dump({"rho": rho}, f)

    top = shap_tab.head(12)["feature"].tolist()
    rows = []
    for feat in top:
        vals = x_s[feat].to_numpy(float)
        lo, hi = np.nanpercentile(vals, [1, 99])
        denom = hi - lo if hi > lo else 1.0
        norm = np.clip((vals - lo) / denom, 0, 1)
        for sh, fv in zip(sv_df[feat].to_numpy(float), norm):
            rows.append({
                "feature": feat,
                "block": _blockof(feat),
                "shap": float(sh),
                "fval_norm": float(fv),
            })
    bees = pd.DataFrame(rows)
    shap_tab.to_csv(PANEL_DIR / "panelD_shap_meanabs.csv", index=False)
    bees.to_csv(PANEL_DIR / "panelD_beeswarm.csv", index=False)
    return shap_tab, dep, bees, rho


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty_like(adj)
    tmp[order] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def _short_glmm_term(term: str) -> str | None:
    if term == "Intercept":
        return None
    if "trait_family" in term and ":dInc_signed_unif" not in term:
        if "[T.placebo]" in term:
            return "trait:placebo"
        if "[T.preference]" in term:
            return "trait:preference"
        if "[T.rule]" in term:
            return "trait:rule"
    if "nagel_lk_type" in term:
        for lk in ["CO2", "DD", "MP", "OD1", "OD2"]:
            if f"[T.{lk}]" in term:
                return f"lk:{lk}"
    if term == "dInc_signed_unif":
        return "dInc_signed_unif"
    if "trait_family" in term and ":dInc_signed_unif" in term:
        if "[T.placebo]" in term:
            return "trait:placebo x Δc"
        if "[T.preference]" in term:
            return "trait:preference x Δc"
        if "[T.rule]" in term:
            return "trait:rule x Δc"
    return term


def glmm(long: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    d = long.copy()
    d["trait_family"] = pd.Categorical(d["family"], categories=["baseline", "placebo", "preference", "rule"])
    d["nagel_lk_type"] = pd.Categorical(d["nagel_lk_type"], categories=LK_ORDER)
    formula = (
        "y ~ C(trait_family, Treatment(reference='baseline'))"
        " + C(nagel_lk_type, Treatment(reference='CO1'))"
        " + dInc_signed_unif"
        " + C(trait_family, Treatment(reference='baseline')):dInc_signed_unif"
    )
    try:
        res = smf.logit(formula, data=d).fit(
            disp=0,
            maxiter=250,
            cov_type="cluster",
            cov_kwds={"groups": d["game_code"]},
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [glmm] fit failed ({e})")
        return pd.DataFrame()

    ci = res.conf_int()
    raw = pd.DataFrame({
        "raw_term": res.params.index,
        "coef": res.params.values,
        "ci_lo": ci[0].values,
        "ci_hi": ci[1].values,
        "p": res.pvalues.values,
    })
    raw["term"] = raw["raw_term"].map(_short_glmm_term)
    raw = raw.dropna(subset=["term"])
    raw["p_bh"] = _bh(raw["p"].to_numpy(float))
    order = [
        "trait:placebo",
        "trait:preference",
        "trait:rule",
        "lk:CO2",
        "lk:DD",
        "lk:MP",
        "lk:OD1",
        "lk:OD2",
        "dInc_signed_unif",
        "trait:placebo x Δc",
        "trait:preference x Δc",
        "trait:rule x Δc",
    ]
    raw["order"] = raw["term"].map({t: i for i, t in enumerate(order)}).fillna(999)
    out = raw.sort_values("order")[["term", "coef", "ci_lo", "ci_hi", "p_bh"]].reset_index(drop=True)
    out.to_csv(PANEL_DIR / "panelC_glmm.csv", index=False)
    return out


def _pretty_feature(f: str) -> str:
    d1 = "Δ₁"
    names = {
        "model=qwen": "Model: Qwen (base)",
        "model=qwen_instruct": "Model: Qwen-Instruct",
        "model=llama31_instruct": "Model: Llama-3.1",
        "model=gptoss": "Model: GPT-OSS",
        "dInc_signed_unif": f"Incentive {d1} → canonical (uniform)",
        "complexity_score": "Game complexity",
        "maximin_to_ne_payoff_distance": "Maximin-NE payoff gap",
        "maximin_to_ne_action_distance": "Maximin-NE action gap",
        "maximin_is_ne": "Maximin is NE",
        "trait=maximin": "Cue: maximin rule",
        "trait=inequity_aversion": "Cue: inequity aversion",
        "trait=selfish_maximizer": "Cue: selfish",
        "trait=loss_aversion": "Cue: loss aversion",
        "trait=risk_aversion": "Cue: risk aversion",
        "trait=length_match_null": "Cue: procedural control",
    }
    if f in names:
        return names[f]
    if f.startswith("nagel_lk_type="):
        return "Game type: " + f.split("=", 1)[1]
    if f.startswith("ne_pareto_rankable="):
        return "NE Pareto-rankable = " + f.split("=", 1)[1]
    return f


def _pretty_glmm(t: str) -> str:
    d1 = "Δ₁"
    names = {
        "trait:placebo": "Cue: procedural control",
        "trait:preference": "Cue: disposition wording",
        "trait:rule": "Cue: maximin rule",
        "dInc_signed_unif": f"Incentive {d1} → canonical",
    }
    if t in names:
        return names[t]
    if t.startswith("lk:"):
        return "Game type " + t[3:]
    return t.replace("trait:", "Cue: ").replace(" x Δc", f" × {d1}")


def figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from strategic_anatomy import paper_style as S

    S.apply()
    block_c = {"structure": S.GOOD_GREEN, "trait": S.SOFT_BLUE, "model": S.GREY}

    A = pd.read_csv(PANEL_DIR / "panelA_shap_delta1.csv")
    B = pd.read_csv(PANEL_DIR / "panelB_nested_auc.csv")
    C = pd.read_csv(PANEL_DIR / "panelC_glmm.csv")
    rho = json.loads((PANEL_DIR / "panelA_shap_delta1_meta.json").read_text())["rho"]

    fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, 4.65))
    gs = GridSpec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.08],
        wspace=0.36,
        hspace=0.55,
        left=0.185,
        right=0.975,
        top=0.90,
        bottom=0.115,
    )
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    axA.scatter(A["delta_c"], A["shap_delta_c"], s=7, color=S.GREY, alpha=0.28, edgecolor="none", zorder=2)
    edges = np.linspace(A["delta_c"].min(), A["delta_c"].max(), 9)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    gi = np.clip(np.digitize(A["delta_c"], edges) - 1, 0, len(ctr) - 1)
    binm = [A["shap_delta_c"][gi == k].mean() if (gi == k).any() else np.nan for k in range(len(ctr))]
    axA.plot(ctr, binm, color=block_c["structure"], lw=1.8, zorder=4, solid_capstyle="round")
    axA.axhline(0, color=S.GREY, lw=0.75, ls=(0, (4, 3)), zorder=1)
    axA.set_xlabel(r"$\Delta_c$  (canonical incentive)", fontsize=S.NHB_FS_AXIS)
    axA.set_ylabel(r"SHAP for $\Delta_c$", fontsize=S.NHB_FS_AXIS)
    axA.text(0.97, 0.06, rf"$\rho = {rho:+.2f}$", transform=axA.transAxes, ha="right",
             va="bottom", fontsize=S.NHB_FS_LEGEND, color=S.INK)
    S.nhb_panel_title(axA, "a", "Payoff-gradient reliance", y=1.07)

    xb = np.arange(len(B))
    yb = B["auc"].to_numpy(float)
    axB.errorbar(
        xb,
        yb,
        yerr=[yb - B["ci_lo"].to_numpy(float), B["ci_hi"].to_numpy(float) - yb],
        fmt="o-",
        color=block_c["structure"],
        ecolor=S.GREY,
        elinewidth=1.0,
        capsize=2.5,
        ms=4.8,
        lw=1.6,
        zorder=3,
    )
    axB.axhline(0.5, color=S.GREY, lw=0.7, ls=(0, (2, 2)), zorder=1)
    axB.set_xticks(xb)
    axB.set_xticklabels(["M0\nmodel", "M1\nstructure", "M2\ncue", "M3\ninteract."],
                        fontsize=S.NHB_FS_TICK)
    axB.set_ylabel("held-out AUC", fontsize=S.NHB_FS_AXIS, labelpad=5)
    axB.yaxis.set_label_position("right")
    axB.yaxis.set_label_coords(1.08, 0.5)
    axB.tick_params(axis="x", length=0)
    S.nhb_panel_title(axB, "b", "Nested variance partition", y=1.07)

    Cf = C.copy()
    Cf["lab"] = Cf["term"].map(_pretty_glmm)

    def _tcol(t: str) -> str:
        tl = t.lower()
        if tl.startswith("trait"):
            return block_c["trait"]
        if tl.startswith("lk:") or "dinc" in tl:
            return block_c["structure"]
        return S.GREY

    yc = np.arange(len(Cf))[::-1]
    axC.axvline(0, color=S.GREY, lw=0.75, ls=(0, (4, 3)), zorder=1)
    for yv, (_, r) in zip(yc, Cf.iterrows()):
        col = _tcol(str(r["term"]))
        axC.plot([r["ci_lo"], r["ci_hi"]], [yv, yv], color=col, lw=1.55, zorder=3, solid_capstyle="round")
        axC.scatter([r["coef"]], [yv], s=18, color=col, zorder=4, edgecolor="white", linewidth=0.5)
    axC.set_yticks(yc)
    axC.set_yticklabels(Cf["lab"], fontsize=S.NHB_FS_FOOT)
    axC.set_xlabel("log-odds (95% CI)", fontsize=S.NHB_FS_AXIS)
    S.nhb_panel_title(axC, "c", "GLMM cross-check", y=1.07)

    for ax in (axA, axB, axC):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.55)
        ax.tick_params(labelsize=S.NHB_FS_TICK, width=0.5, length=2.2)
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.4, alpha=0.75)

    with matplotlib.rc_context({"savefig.bbox": None}):
        for stem in ("figS_attribution", "fig_appendix_attribution"):
            for ext in ("png", "pdf"):
                fig.savefig(
                    FIG_DIR / f"{stem}.{ext}",
                    dpi=600 if ext == "png" else 300,
                    bbox_inches=None,
                )
    plt.close(fig)


def main() -> int:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    long = build_long()
    print(f"corrected attribution table: {len(long)} P1 decisions, "
          f"{long['model'].nunique()} models, {long['game_code'].nunique()} games")

    coverage = (
        long.groupby(["model", "analysis_set"], as_index=False)
        .agg(n_rows=("y", "size"), n_games=("game_code", "nunique"), p_canonical=("y", "mean"))
    )
    print("\n=== coverage and canonical conformity ===")
    print(coverage.round({"p_canonical": 3}).to_string(index=False))

    X0, blocks0 = full_design(long)
    print("\n=== primary feature set ===")
    for block in ["structure", "trait", "model"]:
        cols = blocks0[block]
        print(f"{block:9s} n={len(cols):2d}: " + ", ".join(cols))

    auc_tab = nested_auc(long)
    print("\n=== nested held-out AUC (GroupKFold by game) ===")
    print(auc_tab[["analysis_set", "step", "auc", "ci_lo", "ci_hi", "delta_prev",
                   "n_games", "n_rows"]].round(3).to_string(index=False))

    part, X, y, groups = variance_partition(long)
    part.to_csv(TAB_DIR / "s_variance_partition.csv", index=False)
    print(f"\n=== variance partition (corrected; base OOF AUC={part.attrs['base_auc']:.3f}) ===")
    print(part.round(3).to_string(index=False))

    shap_tab, dep, bees, rho = shap_attribution(X, y, groups, long)
    if not shap_tab.empty:
        shap_tab.to_csv(TAB_DIR / "s_attribution_shap.csv", index=False)
        dep.to_csv(TAB_DIR / "s_shap_delta1_dependence.csv", index=False)
        bees.to_csv(TAB_DIR / "s_shap_beeswarm.csv", index=False)
        shares = block_shares(shap_tab)
        print(f"\n=== OOF TreeSHAP: canonical incentive dependence ===")
        print(f"Spearman rho(delta_c, SHAP_delta_c) = {rho:+.3f}")
        print("\n=== OOF TreeSHAP block shares ===")
        print(shares.round(3).to_string(index=False))
        print("\n=== top SHAP features ===")
        print(shap_tab.head(12).round(4).to_string(index=False))

    g = glmm(long)
    if not g.empty:
        g.to_csv(TAB_DIR / "s_glmm.csv", index=False)
        print("\n=== GLMM fixed effects ===")
        print(g.round(3).to_string(index=False))

    if not shap_tab.empty and not g.empty:
        figure()
        print(f"\nwrote {FIG_DIR/'figS_attribution.png'} (+pdf)")
        print(f"wrote {FIG_DIR/'fig_appendix_attribution.png'} (+pdf)")
        print(f"wrote {PANEL_DIR}/panel[A-D]_*.csv")
    print(f"wrote {TAB_DIR/'f2_nested_auc.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
