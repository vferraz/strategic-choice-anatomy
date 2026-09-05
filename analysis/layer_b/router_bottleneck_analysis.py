#!/usr/bin/env python3
"""GPT-OSS router bottleneck analysis.

This is a descriptive MoE-native analysis on the corrected integrated one-shot
root. It asks a sharper question than raw router decodability:

1. Does the full router gate vector carry a strategic incentive signal?
2. How much of that signal is visible in the actual top-k selected experts?
3. Does gate-level strategic evidence track the final-channel decision process?

No causal or router-edit claim is made. GPT-OSS MXFP4 routing is treated as
read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from strategic_anatomy.config import results_root

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

from strategic_anatomy import paper_style as S  # noqa: E402
from analysis.layer_b.router_oneshot_audit import (  # noqa: E402
    _analysis_rows,
    _layer_feature_bundle,
    all_results,
    cv_binary_scores,
)

S.apply()

TABLES = results_root() / "layer_b"
FIGS = HERE / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

LAYER = 18
SEED = 20260629
INK, GREY, FAINT = S.INK, S.GREY, S.FAINT
# This is a single-model (GPT-OSS) diagnostic.  Router feature sets therefore
# use distinguishable GPT-OSS-purple tones, rather than colors assigned to
# other models elsewhere in the paper.
COL_GATE = "#9467bd"
COL_TOPKW = "#c6b0dc"
COL_TOPK = "#6b4a85"
COL_RESID = "#777777"
COL_GAP = "#4e315f"


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if keep.sum() <= X.shape[1] + 2:
        return out
    Xs = StandardScaler().fit_transform(X[keep])
    fit = LinearRegression().fit(Xs, y[keep])
    out[keep] = y[keep] - fit.predict(Xs)
    return out


def partial_pearson(x: np.ndarray, y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    rx = residualize(np.asarray(x, dtype=float), X)
    ry = residualize(np.asarray(y, dtype=float), X)
    keep = np.isfinite(rx) & np.isfinite(ry)
    if keep.sum() < 5:
        return np.nan, np.nan
    return tuple(float(v) for v in pearsonr(rx[keep], ry[keep]))


def model_r2(y: np.ndarray, X: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xs = StandardScaler().fit_transform(X[keep])
    fit = LinearRegression().fit(Xs, y[keep])
    return float(r2_score(y[keep], fit.predict(Xs)))


def signed_margin(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Positive if the OOF score points toward the true incentive sign."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    return (2 * y - 1) * (score - 0.5)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dec = pd.read_csv(TABLES / "router_oneshot_decodability.csv")
    depth = dec[dec.target.eq("incentive_sign_q05")].pivot(
        index="layer", columns="feature_set", values="auc"
    )
    depth = depth.reset_index()
    depth["gate_minus_topk_binary_auc"] = (
        depth["router_gate"] - depth["router_topk_binary"]
    )
    depth["gate_minus_topk_weight_auc"] = (
        depth["router_gate"] - depth["router_topk_weight"]
    )
    depth.to_csv(TABLES / "router_bottleneck_depth.csv", index=False)

    df = all_results()
    base = _analysis_rows(df)
    bundle = _layer_feature_bundle(LAYER, base)
    y = base["sign_delta1c_q05"].to_numpy(dtype=int)
    games = base["game_code"].to_numpy()
    scores = {
        "gate": cv_binary_scores(
            bundle["router_gate"], y, games, pca_dim=None, seed=SEED + 1
        ),
        "topk_binary": cv_binary_scores(
            bundle["router_topk_binary"], y, games, pca_dim=None, seed=SEED + 2
        ),
        "topk_weight": cv_binary_scores(
            bundle["router_topk_weight"], y, games, pca_dim=None, seed=SEED + 3
        ),
        "residual": cv_binary_scores(
            bundle["residual"], y, games, pca_dim=64, seed=SEED + 4
        ),
    }
    rows = base.copy()
    for key, score in scores.items():
        rows[f"{key}_score"] = score
        rows[f"{key}_margin"] = signed_margin(y, score)
    rows["gate_minus_topk_margin"] = (
        rows["gate_margin"] - rows["topk_binary_margin"]
    )

    per_game = rows.groupby("game_code").agg(
        gate_margin=("gate_margin", "mean"),
        topk_binary_margin=("topk_binary_margin", "mean"),
        topk_weight_margin=("topk_weight_margin", "mean"),
        residual_margin=("residual_margin", "mean"),
        gate_minus_topk_margin=("gate_minus_topk_margin", "mean"),
        median_n_new_tokens=("n_new_tokens", "median"),
        mixed_rate=("mixed_commit", "mean"),
        pure_canonical_rate=("aligned_pure", "mean"),
        abs_delta1c_q05=("abs_delta1c_q05", "first"),
        complexity_score=("complexity_score", "first"),
        dominance_profile=("dominance_profile", "first"),
        num_pure_ne=("num_pure_ne", "first"),
    ).reset_index()
    per_game["log_median_n_new_tokens"] = np.log1p(per_game["median_n_new_tokens"])
    per_game.to_csv(TABLES / "router_bottleneck_game_level.csv", index=False)

    controls = per_game[
        ["abs_delta1c_q05", "complexity_score", "dominance_profile", "num_pure_ne"]
    ].to_numpy(dtype=float)
    summary_rows = []
    base_r2 = model_r2(per_game["log_median_n_new_tokens"].to_numpy(), controls)
    for metric in [
        "gate_margin",
        "topk_binary_margin",
        "topk_weight_margin",
        "residual_margin",
        "gate_minus_topk_margin",
    ]:
        for outcome in [
            "log_median_n_new_tokens",
            "mixed_rate",
            "pure_canonical_rate",
        ]:
            keep = (
                np.isfinite(per_game[metric].to_numpy())
                & np.isfinite(per_game[outcome].to_numpy())
            )
            rho, sp = spearmanr(per_game.loc[keep, metric], per_game.loc[keep, outcome])
            pr, pp = partial_pearson(
                per_game[metric].to_numpy(), per_game[outcome].to_numpy(), controls
            )
            summary_rows.append(
                {
                    "layer": LAYER,
                    "metric": metric,
                    "outcome": outcome,
                    "spearman_rho": float(rho),
                    "spearman_p": float(sp),
                    "partial_pearson_r_control_objective": pr,
                    "partial_pearson_p_control_objective": pp,
                    "n_games": int(keep.sum()),
                }
            )

    for metric in [
        "gate_margin",
        "topk_binary_margin",
        "topk_weight_margin",
        "residual_margin",
        "gate_minus_topk_margin",
    ]:
        X_aug = np.column_stack([controls, per_game[metric].to_numpy(dtype=float)])
        r2 = model_r2(per_game["log_median_n_new_tokens"].to_numpy(), X_aug)
        summary_rows.append(
            {
                "layer": LAYER,
                "metric": metric,
                "outcome": "log_median_n_new_tokens",
                "objective_control_r2": base_r2,
                "augmented_r2": r2,
                "delta_r2_over_objective": r2 - base_r2,
                "n_games": int(len(per_game)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "router_bottleneck_summary.csv", index=False)
    return depth, per_game, summary


def clean(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=S.NHB_FS_TICK, colors=INK, length=2.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999")


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    S.nhb_panel_title(ax, letter, title, title_x=0.12, fontsize=S.NHB_FS_PANEL)


def make_figure(depth: pd.DataFrame, per_game: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, 3.05), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 3, left=0.080, right=0.985, top=0.80, bottom=0.22,
        wspace=0.52, width_ratios=[1.04, 1.08, 0.92]
    )
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in (axa, axb, axc):
        clean(ax)

    # A: AUC depth curves
    axa.axhline(0.5, color=GREY, ls=(0, (4, 3)), lw=0.8)
    axa.plot(depth["layer"], depth["router_gate"], color=COL_GATE, lw=1.8, marker="o", ms=3, label="gate logits")
    axa.plot(depth["layer"], depth["router_topk_weight"], color=COL_TOPKW, lw=1.6, ls="--", marker="s", ms=2.7, label="top-k weights")
    axa.plot(depth["layer"], depth["router_topk_binary"], color=COL_TOPK, lw=1.6, ls=":", marker="^", ms=2.9, label="top-k set")
    axa.plot(depth["layer"], depth["residual"], color=COL_RESID, lw=1.2, ls="-.", alpha=0.75, label="residual")
    axa.axvline(LAYER, color="#bbbbbb", lw=0.8)
    axa.set_ylim(0.45, 0.96)
    axa.set_xlabel("GPT-OSS router layer", fontsize=S.NHB_FS_AXIS)
    axa.set_ylabel("incentive-sign AUC", fontsize=S.NHB_FS_AXIS)
    axa.text(35.0, depth["residual"].iloc[-1] + 0.008, "residual",
             color=COL_RESID, fontsize=S.NHB_FS_FOOT, ha="right", va="bottom")
    axa.text(35.0, depth["router_gate"].iloc[-1] - 0.002, "gate",
             color=COL_GATE, fontsize=S.NHB_FS_FOOT, ha="right", va="top")
    axa.text(35.0, depth["router_topk_weight"].iloc[-1] - 0.006, "top-k wt.",
             color=COL_TOPKW, fontsize=S.NHB_FS_FOOT, ha="right", va="top")
    axa.text(35.0, depth["router_topk_binary"].iloc[-1] - 0.008, "top-k set",
             color=COL_TOPK, fontsize=S.NHB_FS_FOOT, ha="right", va="top")
    panel_title(axa, "a", "Router gate-route gap")

    # B: gate evidence vs process length
    sc = axb.scatter(
        per_game["gate_margin"],
        per_game["median_n_new_tokens"],
        c=per_game["mixed_rate"],
        cmap="viridis",
        s=26,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
    )
    x = per_game["gate_margin"].to_numpy()
    y = np.log1p(per_game["median_n_new_tokens"].to_numpy())
    keep = np.isfinite(x) & np.isfinite(y)
    b1, b0 = np.polyfit(x[keep], y[keep], 1)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    axb.plot(xs, np.expm1(b0 + b1 * xs), color=INK, lw=1.1)
    axb.set_yscale("log")
    axb.set_xlabel("gate strategic evidence\n(held-out signed margin)", fontsize=S.NHB_FS_AXIS)
    axb.set_ylabel("median new tokens to commit", fontsize=S.NHB_FS_AXIS)
    cax = fig.add_axes([0.450, 0.930, 0.135, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0.0, 0.75])
    cb.ax.tick_params(labelsize=S.NHB_FS_INSET_STAT, length=1.2, pad=1)
    cax.set_title("mixed rate", fontsize=S.NHB_FS_INSET_STAT, color=S.SUBTLE, pad=1.0)
    row = summary[
        summary["metric"].eq("gate_margin")
        & summary["outcome"].eq("log_median_n_new_tokens")
        & summary["spearman_rho"].notna()
    ].iloc[0]
    axb.text(
        0.03,
        0.05,
        f"Spearman ρ={row['spearman_rho']:.2f}\npartial r={row['partial_pearson_r_control_objective']:.2f}",
        transform=axb.transAxes,
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="left",
        va="bottom",
    )
    panel_title(axb, "b", "Evidence and commitment")

    # C: objective-controlled process gain
    r2 = summary.dropna(subset=["delta_r2_over_objective"]).copy()
    order = ["gate_margin", "topk_binary_margin", "gate_minus_topk_margin", "residual_margin"]
    labels = ["gate", "topk", "gap", "resid."]
    colors = [COL_GATE, COL_TOPK, COL_GAP, COL_RESID]
    vals = [float(r2[r2.metric.eq(m)]["delta_r2_over_objective"].iloc[0]) for m in order]
    xx = np.arange(len(order))
    axc.bar(xx, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.4)
    axc.axhline(0, color="#999", lw=0.8)
    axc.set_xticks(xx)
    axc.set_xticklabels(labels, fontsize=S.NHB_FS_TICK_SMALL)
    axc.set_ylabel("ΔR² for log(tokens)", fontsize=S.NHB_FS_AXIS)
    axc.set_ylim(0, max(vals) * 1.35)
    for i, v in enumerate(vals):
        axc.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=S.NHB_FS_FOOT, color=INK)
    axc.text(
        0.02,
        0.94,
        "controls: |Δ₁ᶜ|, complexity,\ndominance, # pure NE",
        transform=axc.transAxes,
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        va="top",
    )
    panel_title(axc, "c", "Process gain")

    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(FIGS / "fig_router_bottleneck.pdf", dpi=300, bbox_inches=None)
        fig.savefig(FIGS / "fig_router_bottleneck.png", dpi=600, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    depth, per_game, summary = build_tables()
    make_figure(depth, per_game, summary)
    print(f"wrote {FIGS / 'fig_router_bottleneck.png'}")


if __name__ == "__main__":
    main()
