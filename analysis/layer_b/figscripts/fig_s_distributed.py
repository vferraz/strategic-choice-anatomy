#!/usr/bin/env python3
"""Fig S-distributed (N5) — distributed vs localized code.

Pre-empts "is it a few memorized neurons?". (a) participation ratio PR/d on scale-fair logistic
weights; (b) greedy-ablation fraction of dims to reach 90% of full AUC. The point is the cross-
model CONTRAST: the decision/incentive code is similarly DISTRIBUTED (PR/d ~ 0.3) in every model
including Llama — so Llama's failure is not a missing code but a missing recruitment (B2).

Caveat (annotated): the ablation fraction is only interpretable where full AUC is meaningfully
above chance; near-chance axes (e.g. Llama incentive) make it degenerate.

  .venv/bin/python analysis/layer_b/figscripts/fig_s_distributed.py
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


def main():
    d = pd.read_csv(TAB / "s_distributed.csv")
    agg = d.groupby(["model", "axis"]).agg(pr_over_d=("pr_over_d", "mean"),
                                           frac90=("dims_to_90pct_full_auc", "mean"),
                                           full_auc=("full_auc", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.32, left=0.08, right=0.97)
    x = np.arange(len(ORDER)); w = 0.36
    for j, axis in enumerate(["decision", "incentive"]):
        sub = agg[agg.axis == axis].set_index("model")
        axes[0].bar(x + j * w, [sub.loc[m, "pr_over_d"] for m in ORDER], w,
                    label=axis, edgecolor="white", lw=0.4,
                    color=[COL[m] for m in ORDER], alpha=0.65 if j else 1.0, zorder=3)
        axes[1].bar(x + j * w, [sub.loc[m, "frac90"] * 100 for m in ORDER], w,
                    label=axis, edgecolor="white", lw=0.4,
                    color=[COL[m] for m in ORDER], alpha=0.65 if j else 1.0, zorder=3)
    axes[0].set_xticks(x + w / 2); axes[0].set_xticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_TICK)
    axes[0].set_ylabel("participation ratio  PR / d", fontsize=PS.FS_AXIS)
    axes[0].tick_params(labelsize=PS.FS_TICK)
    PS.panel_title(axes[0], "a", "Distributed code (higher = more distributed)")
    axes[1].set_xticks(x + w / 2); axes[1].set_xticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_TICK)
    axes[1].set_ylabel("% of dims to reach 90% of full AUC", fontsize=PS.FS_AXIS)
    axes[1].tick_params(labelsize=PS.FS_TICK)
    axes[1].legend(fontsize=PS.FS_LEGEND, frameon=False, title="axis", title_fontsize=PS.FS_FOOT)
    PS.panel_title(axes[1], "b", "Greedy ablation (degenerate where AUC≈chance)")
    fig.text(0.5, 0.02, "decision & incentive codes are similarly distributed across models — Llama's "
             "gap is recruitment (Fig B2), not a missing or localized code.",
             ha="center", fontsize=PS.FS_FOOT, color=PS.FAINT)
    PS.figure_titles(fig, "Supplementary — distributed vs localized strategic code",
                     f"scale-fair participation ratio + greedy ablation{PS.SEP}steer layers, "
                     f"decision-slot residuals")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_s_distributed.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_s_distributed.{pdf,png}")


if __name__ == "__main__":
    main()
