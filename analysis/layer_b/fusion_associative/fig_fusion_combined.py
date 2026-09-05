#!/usr/bin/env python3
"""Single combined figure: the associative fusion transition across architectures.

Row 1 (a-d): decision-incentive angle vs depth for all four models (residual stream,
objective incentive). Row 2 (e-g): GPT-OSS MoE deep-dive comparing residual and router-gate
geometry under objective and model-specific opponent beliefs. The spare cell holds the
shared legend.

Reads precomputed tables (no recompute):
  tables/fusion_depth_table.csv, tables/fusion_depth_empirical.csv, tables/oss_router_fusion.csv
  python analysis/fusion_associative/fig_fusion_combined.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib
from strategic_anatomy.config import results_root
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
# phase-4: git does not track empty directories, so figures/ and tables/ do not exist in a
# fresh clone and savefig() would raise FileNotFoundError. Create them up front.
(HERE / "figures").mkdir(parents=True, exist_ok=True)
(HERE / "tables").mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[2]
from strategic_anatomy import paper_style as PS  # noqa: E402
PS.apply()

COL = {"qwen_instruct": "#1f77b4", "qwen": "#2ca02c",
       "llama31_instruct": "#d62728", "gptoss": "#9467bd"}
PURPLE = "#9467bd"

U = pd.read_csv(results_root() / "layer_b" / "fusion" / "fusion_depth_table.csv")
E = pd.read_csv(results_root() / "layer_b" / "fusion" / "fusion_depth_empirical.csv")
R = pd.read_csv(results_root() / "layer_b" / "fusion" / "oss_router_fusion.csv")

# (dataframe, colour, letter, title) for each of the 7 plotted panels
panels = [
    (U[U.model == "qwen_instruct"], COL["qwen_instruct"], "a", "Qwen2.5-Instruct"),
    (U[U.model == "qwen"], COL["qwen"], "b", "Qwen2.5 base"),
    (U[U.model == "llama31_instruct"], COL["llama31_instruct"], "c", "Llama-3.1-Instruct"),
    (U[U.model == "gptoss"], COL["gptoss"], "d", "GPT-OSS · uniform site"),
    (E[E.model == "gptoss"], PURPLE, "e", "residual · own belief"),
    (R[(R.belief == "uniform")], PURPLE, "f", "router · objective"),
    (R[(R.belief == "empirical")], PURPLE, "g", "router · own belief"),
]

fig, axes = plt.subplots(2, 4, figsize=(PS.NHB_TEXTWIDTH_IN, 4.5), sharey=True, sharex=True)
flat = axes.flatten()
for ax, (df, col, letter, title) in zip(flat[:7], panels):
    s = df[df.layer > 0].sort_values("depth_frac")
    x = s.depth_frac.to_numpy()
    ax.fill_between(x, s.null_lo, s.null_hi, color="0.86", lw=0)
    ax.plot(x, s.null_med, color=PS.GREY, lw=0.8, ls=(0, (4, 2)))
    ax.plot(x, s.angle_deg, color=col, lw=1.6)
    sg = s[s.perm_p < 0.05]
    ax.scatter(sg.depth_frac, sg.angle_deg, color=col, s=10, zorder=5,
               edgecolor="white", linewidths=0.3)
    ax.axhline(90, color=PS.INK, lw=0.6, ls=":", alpha=0.55)
    ax.set_xlim(0, 1); ax.set_ylim(0, 135); ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=PS.NHB_FS_TICK)
    if letter in "abcd":
        PS.nhb_panel_title(ax, letter, title, title_x=0.085, fontsize=PS.NHB_FS_MINI_TITLE)
    else:  # row 2: descriptor on line 1, model tag on line 2 (avoids overrun)
        ax.text(PS.NHB_PANEL_LETTER_X, 1.16, f"({letter})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=PS.NHB_FS_MINI_TITLE, fontweight="bold",
                color=PS.INK, clip_on=False)
        ax.text(0.075, 1.16, title, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=PS.NHB_FS_MINI_TITLE, fontweight="bold", color=PS.INK, clip_on=False)
        ax.text(0.075, 1.02, "(GPT-OSS)", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=PS.NHB_FS_FOOT, color=PS.SUBTLE, clip_on=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# y-axis labels on the two left panels; x-axis labels on the bottom-adjacent panels
axes[0, 0].set_ylabel("decision–incentive\nangle (°)", fontsize=PS.NHB_FS_AXIS)
axes[1, 0].set_ylabel("decision–incentive\nangle (°)", fontsize=PS.NHB_FS_AXIS)
for ax in (flat[4], flat[5], flat[6], flat[3]):
    ax.set_xlabel("relative depth", fontsize=PS.NHB_FS_AXIS)
axes[0, 0].text(0.03, 92, "orthogonal", fontsize=PS.NHB_FS_FOOT, color=PS.FAINT, va="top")
axes[0, 0].text(0.03, 4, "fused", fontsize=PS.NHB_FS_FOOT, color=PS.FAINT, va="bottom")

# spare cell (row 2, col 4): legend + one-line reading aids
lg = flat[7]
lg.axis("off")
handles = [
    Line2D([0], [0], color=PS.INK, lw=1.6, label="observed angle"),
    Patch(facecolor="0.86", label="permutation null (95%)"),
    Line2D([0], [0], color=PS.GREY, lw=0.8, ls=(0, (4, 2)), label="null median"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=PS.INK,
           markeredgecolor="white", markersize=4, label="angle < null (p < 0.05)"),
    Line2D([0], [0], color=PS.INK, lw=0.6, ls=":", alpha=0.55, label="90° (independent-vector)"),
]
lg.legend(handles=handles, fontsize=PS.NHB_FS_LEGEND, loc="upper center",
          frameon=False, handlelength=1.7, labelspacing=0.5, borderpad=0.2)
lg.text(0.5, 0.30, "Row 1: residual stream,\nobjective incentive.\nGPT-OSS: 541 stated-policy rows,\nuniform pre-answer site.",
        transform=lg.transAxes, ha="center", va="top",
        fontsize=PS.NHB_FS_FOOT, color=PS.SUBTLE)

fig.subplots_adjust(left=0.095, right=0.99, top=0.945, bottom=0.095, wspace=0.13, hspace=0.42)
fig.savefig(
    HERE / "figures" / "fig_fusion_combined.pdf",
    dpi=300,
    facecolor="white",
    transparent=False,
)
fig.savefig(
    HERE / "figures" / "fig_fusion_combined.png",
    dpi=600,
    facecolor="white",
    transparent=False,
)
print("wrote figures/fig_fusion_combined.{pdf,png}")
