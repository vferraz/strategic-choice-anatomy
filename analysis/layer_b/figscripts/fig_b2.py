#!/usr/bin/env python3
"""Figure B2 — recruitment is the gate (the load-bearing, competitor-differentiating figure).

(a) WITHIN-MODEL recruitment bridge (n=144 games, CONFIRMATORY): the slope of per-game
    P(canonical) on the per-game strength of the REPRESENTED incentive (z-scored OOF projection
    onto d_inc). Confound-free: it links representation strength to behavioural outcome, not
    decodability of a stimulus. Steep positive in Qwen-I/Qwen-B; ~0 in Llama (represented but not
    recruited); GPT-OSS partial in the dense residual (router-mediated).
(b) cross-model geometry <-> λ summary (DESCRIPTIVE, n<=4): angle<->λ and gain<->λ.
(c) option dynamics across depth (the neural quantal parameter): P(canonical option) by depth,
    SPLIT BY the model's actual choice -> the readout reverses (drops below 0.5) when the model
    actually played non-canonical, proving it tracks the decision, not the normative answer.

  .venv/bin/python analysis/layer_b/figscripts/fig_b2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

PS = lib.style()
import matplotlib.pyplot as plt  # noqa: E402

ORDER = list(lib.MODELS); COL = lib.COL; SHORT = lib.SHORT
TAB = lib.TAB_DIR; FIG = lib.FIG_DIR


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}")


def main():
    bridge = pd.read_csv(TAB / "b2_within_model_bridge.csv").set_index("model")
    geo = pd.read_csv(TAB / "b2_geometry_lambda.csv").set_index("model")
    od = pd.read_csv(TAB / "b2_option_dynamics.csv")

    fig = plt.figure(figsize=(11.0, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.7,
                          top=0.86, bottom=0.10, left=0.08, right=0.97)

    # ── (a) within-model recruitment bridge (n=144) ───────────────────────
    axa = fig.add_subplot(gs[0, 0:2])
    y = np.arange(len(ORDER))[::-1]
    for yi, m in zip(y, ORDER):
        s = bridge.loc[m, "slope_per_sd"]; lo = bridge.loc[m, "slope_lo"]; hi = bridge.loc[m, "slope_hi"]
        axa.barh(yi, s, height=0.55, color=COL[m], edgecolor="white", lw=0.5, zorder=3)
        axa.errorbar(s, yi, xerr=[[s - lo], [hi - s]], fmt="none", ecolor=PS.INK,
                     elinewidth=0.7, capsize=3, zorder=5)
        axa.text(0.005, yi + 0.34, f"r={bridge.loc[m,'pearson_r']:+.2f}", fontsize=PS.FS_FOOT, color=PS.FAINT)
    axa.axvline(0, color=PS.GREY, lw=0.7, ls=":")
    axa.set_yticks(y); axa.set_yticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_AXIS)
    axa.set_xlabel("Δ P(canonical) per SD of represented incentive", fontsize=PS.FS_AXIS)
    axa.tick_params(labelsize=PS.FS_TICK)
    PS.panel_title(axa, "a", "Within-model recruitment bridge (n=144 games, confirmatory)")

    # ── (b) geometry ↔ λ (descriptive, n≤4) ───────────────────────────────
    axb = fig.add_subplot(gs[0, 2])
    for m in ORDER:
        axb.scatter(geo.loc[m, "lambda_L1"], geo.loc[m, "angle_deg"], s=70, color=COL[m],
                    edgecolors=PS.INK, lw=0.5, zorder=5)
        axb.annotate(SHORT[m], (geo.loc[m, "lambda_L1"], geo.loc[m, "angle_deg"]),
                     xytext=(3, 3), textcoords="offset points", fontsize=PS.FS_FOOT, color=PS.SUBTLE)
    from scipy.stats import spearmanr
    rho = spearmanr(geo["lambda_L1"], geo["angle_deg"]).correlation
    axb.set_xlabel("behavioural λ", fontsize=PS.FS_AXIS); axb.set_ylabel("angle(d_inc,d_dec) [°]", fontsize=PS.FS_AXIS)
    axb.tick_params(labelsize=PS.FS_TICK)
    PS.panel_title(axb, "b", f"Angle↔λ  (ρ={rho:+.2f})")

    axc = fig.add_subplot(gs[0, 3])
    dense = geo[geo.index.isin(lib.DENSE)]
    for m in dense.index:
        axc.errorbar(geo.loc[m, "lambda_L1"], geo.loc[m, "gain_slope"],
                     yerr=[[geo.loc[m, "gain_slope"] - geo.loc[m, "gain_lo"]],
                           [geo.loc[m, "gain_hi"] - geo.loc[m, "gain_slope"]]],
                     fmt="o", color=COL[m], markersize=8, markeredgecolor=PS.INK, capsize=3, zorder=5)
        axc.annotate(SHORT[m], (geo.loc[m, "lambda_L1"], geo.loc[m, "gain_slope"]),
                     xytext=(3, 3), textcoords="offset points", fontsize=PS.FS_FOOT, color=PS.SUBTLE)
    axc.axhline(0, color=PS.GREY, lw=0.5, ls=":")
    axc.set_xlabel("behavioural λ", fontsize=PS.FS_AXIS); axc.set_ylabel("neural gain", fontsize=PS.FS_AXIS)
    axc.tick_params(labelsize=PS.FS_TICK)
    PS.panel_title(axc, "c", "Gain↔λ (dense, n=3)")

    # ── (d) option dynamics across depth, split by actual choice ──────────
    sub = gs[1, 0:4].subgridspec(1, 4, wspace=0.28)
    for k, m in enumerate(ORDER):
        ax = fig.add_subplot(sub[0, k])
        for choice, ls, lab in [("canonical", "-", "chose canonical"),
                                ("non_canonical", "--", "chose non-canonical")]:
            s = od[(od.model == m) & (od.actual_choice == choice)].sort_values("depth_frac")
            ax.plot(s.depth_frac, s.p_canonical_option, ls, color=COL[m], lw=1.4,
                    label=lab if k == 0 else None)
        ax.axhline(0.5, color=PS.GREY, lw=0.7, ls=":")
        ax.set_ylim(0.0, 1.0); ax.set_title(SHORT[m], fontsize=PS.FS_TICK, color=PS.INK)
        ax.set_xlabel("depth (frac)", fontsize=PS.FS_TICK); ax.tick_params(labelsize=PS.FS_TICK)
        if k == 0:
            ax.set_ylabel("P(canonical option)", fontsize=PS.FS_TICK)
            ax.legend(fontsize=PS.FS_FOOT, frameon=False, loc="center right")
    fig.text(0.08, 0.46, "(d)  Option dynamics across depth — readout reverses when the model plays non-canonical",
             fontsize=PS.FS_PANEL, fontweight="bold", color=PS.INK)

    fig.text(0.5, 0.02, "Causal steering (incentive injection) is the reserved capstone — NOT yet run; "
             "B2 is correlational-but-confound-free. No do()-claim is made.",
             ha="center", fontsize=PS.FS_FOOT, color=PS.FAINT)
    PS.figure_titles(fig, "Recruitment is the gate",
                     f"within-model strength→behaviour bridge (confirmatory){PS.SEP}geometry↔λ "
                     f"(descriptive){PS.SEP}the neural quantal parameter across depth")
    _save(fig, "fig_b2_recruitment")


if __name__ == "__main__":
    main()
