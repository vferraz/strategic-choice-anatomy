#!/usr/bin/env python3
"""House-style supplementary figure for the frozen null-spread geometry table.

The source table is derived by ``confirm_null_spread.py`` from the joint
game-level permutation table (``fusion_depth_table.csv``) and the frozen
baseline activation cache.  The renderer only reads that table; an explicit
``--input-table`` may be used when the frozen table is stored outside this
active analysis bundle.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
from strategic_anatomy.config import results_root
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
# phase-4: git does not track empty directories, so figures/ and tables/ do not exist in a
# fresh clone and savefig() would raise FileNotFoundError. Create them up front.
(HERE / "figures").mkdir(parents=True, exist_ok=True)
(HERE / "tables").mkdir(parents=True, exist_ok=True)
ROOT = HERE.parents[2]
from strategic_anatomy import paper_style as PS  # noqa: E402
PS.apply()

COL = {"qwen": "#1f77b4", "qwen_instruct": "#17becf",
       "llama31_instruct": "#2ca02c", "gptoss": "#9467bd"}
NICE = {"qwen": "Qwen2.5", "qwen_instruct": "Qwen2.5-I",
        "llama31_instruct": "Llama", "gptoss": "GPT-OSS"}
MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-table",
    type=Path,
    default=results_root() / "layer_b" / "fusion" / "null_spread_vs_geometry.csv",
    help="Frozen null_spread_vs_geometry.csv to render (read only).",
)
args = parser.parse_args()
df = pd.read_csv(args.input_table)

fig, ax = plt.subplots(1, 2, figsize=(PS.NHB_TEXTWIDTH_IN, 2.7))
for m in MODELS:
    s = df[df.model == m].sort_values("depth_frac")
    ax[0].plot(s.depth_frac, s.null_spread, color=COL[m], lw=1.6, marker="o", ms=2.5)
    ax[1].scatter(s.participation_ratio, s.null_spread, color=COL[m], s=12)
ax[0].set_xlabel("relative depth", fontsize=PS.NHB_FS_AXIS)
ax[0].set_ylabel("null spread (95%, °)", fontsize=PS.NHB_FS_AXIS)
ax[1].set_xscale("log")
ax[1].set_xlabel("participation ratio (effective dim.)", fontsize=PS.NHB_FS_AXIS)
for a in ax:
    a.tick_params(labelsize=PS.NHB_FS_TICK)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
PS.nhb_panel_title(ax[0], "a", "Null width varies with depth", title_x=0.10)
PS.nhb_panel_title(ax[1], "b", "Wider null, lower effective dimension", title_x=0.10)

handles = [Line2D([0], [0], color=COL[m], lw=1.6, marker="o", ms=3, label=NICE[m]) for m in MODELS]
ax[1].legend(handles=handles, fontsize=PS.NHB_FS_LEGEND, frameon=False, loc="upper right",
             handlelength=1.4, labelspacing=0.3)
fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.17, wspace=0.24)
fig.savefig(HERE / "figures" / "fig_nullspread_geometry_nhb.pdf", dpi=300, bbox_inches="tight")
fig.savefig(HERE / "figures" / "fig_nullspread_geometry_nhb.png", dpi=600, bbox_inches="tight")
print("wrote figures/fig_nullspread_geometry_nhb.{pdf,png}")
