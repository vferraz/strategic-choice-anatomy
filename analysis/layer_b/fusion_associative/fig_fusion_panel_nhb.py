#!/usr/bin/env python3
"""NHB-formatted main-text panel: decision-incentive angle across depth (uniform belief).

Caption-led house style (analysis/_shared/_paper_style.py): no in-figure master title
or footnote, only panel letters; locked NHB text width + type scale; canonical model
colours. Reads the precomputed table from build_fusion_figures.py.

  python analysis/fusion_associative/fig_fusion_panel_nhb.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
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
ROOT = HERE.parents[1]
from strategic_anatomy import paper_style as PS  # noqa: E402
PS.apply()

# canonical palette / names (mirror analysis/oneshot/_oneshot_common.py)
MODELS = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss"]
COL = {"qwen_instruct": "#1f77b4", "qwen": "#2ca02c",
       "llama31_instruct": "#d62728", "gptoss": "#9467bd"}
TITLE = {"qwen_instruct": "Qwen2.5-Instruct", "qwen": "Qwen2.5 base",
         "llama31_instruct": "Llama-3.1-Instruct", "gptoss": "GPT-OSS-120B"}
LETTERS = ["a", "b", "c", "d"]

df = pd.read_csv(results_root() / "layer_b" / "fusion" / "fusion_depth_table.csv")
df = df[df.angle_deg.notna()]

fig, axes = plt.subplots(1, 4, figsize=(PS.NHB_TEXTWIDTH_IN, 2.45), sharey=True)
for ax, m, L in zip(axes, MODELS, LETTERS):
    s = df[(df.model == m) & (df.layer > 0)].sort_values("depth_frac")
    x = s.depth_frac.to_numpy()
    ax.fill_between(x, s.null_lo, s.null_hi, color="0.86", lw=0)
    ax.plot(x, s.null_med, color=PS.GREY, lw=0.8, ls=(0, (4, 2)))
    ax.plot(x, s.angle_deg, color=COL[m], lw=1.6)
    sig = s[s.perm_p < 0.05]
    ax.scatter(sig.depth_frac, sig.angle_deg, color=COL[m], s=9, zorder=5,
               edgecolor="white", linewidths=0.3)
    ax.axhline(90, color=PS.INK, lw=0.6, ls=":", alpha=0.55)
    ax.set_xlim(0, 1); ax.set_ylim(10, 96)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=PS.NHB_FS_TICK)
    ax.set_xlabel("relative depth", fontsize=PS.NHB_FS_AXIS)
    PS.nhb_panel_title(ax, L, TITLE[m], title_x=0.10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].set_ylabel("decision–incentive\nangle (°)", fontsize=PS.NHB_FS_AXIS)
# minimal in-panel anchors (kept tiny; full description lives in the caption)
axes[0].text(0.03, 92.5, "orthogonal", fontsize=PS.NHB_FS_FOOT, color=PS.FAINT, va="top")
axes[0].text(0.03, 14.5, "fused", fontsize=PS.NHB_FS_FOOT, color=PS.FAINT, va="bottom")

handles = [
    Line2D([0], [0], color=PS.INK, lw=1.6, label="observed angle"),
    Patch(facecolor="0.86", label="permutation null (95%)"),
    Line2D([0], [0], color=PS.GREY, lw=0.8, ls=(0, (4, 2)), label="null median"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=PS.INK,
           markeredgecolor="white", markersize=4, label="angle < null (p < 0.05)"),
]
fig.legend(handles=handles, fontsize=PS.NHB_FS_LEGEND, loc="lower center", ncol=4,
           frameon=False, handlelength=1.7, columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))

fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.30, wspace=0.12)
fig.savefig(HERE / "figures" / "fig_fusion_panel_nhb.pdf", dpi=300)
fig.savefig(HERE / "figures" / "fig_fusion_panel_nhb.png", dpi=600)
print("wrote figures/fig_fusion_panel_nhb.{pdf,png}")
