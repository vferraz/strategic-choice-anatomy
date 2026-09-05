#!/usr/bin/env python3
"""Supplementary — emitted-label control (the B1a inset, broken out).

Is a model's canonical-action decodability really a disguised J/P-label detector? We decode the
literal emitted label (decoded_label == 'J') and compare to the canonical-action AUC. GPT-OSS mixed
rows retain their realized action but are excluded from this label-control target because they did
not emit a pure J/P letter.

  .venv/bin/python analysis/layer_b/figscripts/fig_llama_label_control.py
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
    d = pd.read_csv(TAB / "b1_decodability.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.10, right=0.97)
    x = np.arange(len(ORDER)); w = 0.36
    can = [d[(d.model == m) & (d.probe == "canonical_action")]["auc"].values[0] for m in ORDER]
    lab = [d[(d.model == m) & (d.probe == "label_axis_J")]["auc"].values[0] for m in ORDER]
    jrate = [d[(d.model == m) & (d.probe == "label_axis_J")]["base_rate"].values[0] for m in ORDER]
    ax.bar(x, can, w, color=[COL[m] for m in ORDER], edgecolor="white", lw=0.4, label="canonical action", zorder=3)
    ax.bar(x + w, lab, w, color=[COL[m] for m in ORDER], alpha=0.55, edgecolor="white", lw=0.4,
           label="emitted label = J", zorder=3)
    ax.axhline(0.5, color=PS.GREY, lw=0.7, ls=":")
    for xi, jr in zip(x, jrate):
        ax.text(xi + w, 0.52, f"P(J)={jr:.2f}", fontsize=PS.FS_FOOT, color=PS.FAINT, ha="center", rotation=90)
    ax.set_xticks(x + w / 2); ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_TICK)
    ax.set_ylim(0.4, 1.03); ax.set_ylabel("AUC (out-of-fold)", fontsize=PS.FS_AXIS)
    ax.tick_params(labelsize=PS.FS_TICK); ax.legend(fontsize=PS.FS_LEGEND, frameon=False, loc="lower right")
    PS.figure_titles(fig, "Supplementary — emitted-label control",
                     f"is canonical-action decodability a disguised J/P detector?"
                     f"{PS.SEP}deepest layer, P1 baseline")
    fig.text(0.5, 0.015, "The J/P label axis is reported only as a control. Strategic interpretation remains in "
             "canonical action space; GPT-OSS mixed rows are behaviorally included but excluded from this label target.",
             ha="center", fontsize=PS.FS_FOOT, color=PS.FAINT)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_s_llama_label_control.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_s_llama_label_control.{pdf,png}")


if __name__ == "__main__":
    main()
