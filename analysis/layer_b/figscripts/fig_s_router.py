#!/usr/bin/env python3
"""Fig S-router — GPT-OSS router state on the corrected Akata substrate.

DESCRIPTIVE only. Reads ``analysis/layer_b/tables/s_router_decodability.csv``, which is
built from ``$SCA_DATA_ROOT/substrate/gptoss/*/router.npz`` gate logits with game-grouped CV.
The router is read, not edited; no router causal claim is made.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


from analysis.layer_b import lib  # noqa: E402

PS = lib.style()
import matplotlib.pyplot as plt  # noqa: E402


def main():
    src = lib.TAB_DIR / "s_router_decodability.csv"
    if not src.exists():
        raise FileNotFoundError(f"missing {src}; run analysis/layer_b/build_router.py first")
    d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.12, right=0.97)
    colors = {
        "canonical_action": lib.COL["gptoss"],
        "sign_delta1c": PS.SOFT_BLUE,
        "sign_delta2c": PS.GOOD_GREEN,
    }
    for target, sub in d.groupby("target"):
        sub = sub.sort_values("layer")
        ax.plot(sub["layer"], sub["auc"], marker="o", lw=1.6, ms=4.0,
                color=colors.get(target, PS.INK), label=sub["label"].iloc[0])
        ax.fill_between(sub["layer"], sub["lo"], sub["hi"],
                        color=colors.get(target, PS.INK), alpha=0.12, lw=0)
    if "shuffle_auc" in d:
        sh = d.groupby("layer", as_index=False)["shuffle_auc"].mean().sort_values("layer")
        ax.plot(sh["layer"], sh["shuffle_auc"], color=PS.FAINT, lw=1.0, ls=":",
                label="shuffled-label control")
    ax.axhline(0.5, color=PS.GREY, lw=0.7, ls="--")
    ax.set_xlabel("GPT-OSS router layer", fontsize=PS.FS_AXIS)
    ax.set_ylabel("AUC from router gate logits", fontsize=PS.FS_AXIS)
    ax.set_ylim(0.35, 1.02)
    ax.tick_params(labelsize=PS.FS_TICK)
    ax.legend(fontsize=PS.FS_LEGEND, frameon=False, loc="lower right")
    ax.grid(axis="y", color="#dddddd", lw=0.5)

    PS.figure_titles(
        fig,
        "Supplementary — GPT-OSS router state carries strategic structure",
        f"corrected Akata one-shot root{PS.SEP}read-only gate-logit probes, game-grouped CV",
    )
    fig.text(
        0.5, 0.035,
        "Router evidence is descriptive. Residual-only GPT-OSS nulls are not interpreted as "
        "absence of strategic computation; no router do()-claim is made.",
        ha="center", fontsize=PS.FS_FOOT, color=PS.FAINT,
    )
    for ext in ("pdf", "png"):
        fig.savefig(lib.FIG_DIR / f"fig_s_router.{ext}", dpi=300 if ext == "pdf" else 600,
                    bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_s_router.{pdf,png}")


if __name__ == "__main__":
    main()
