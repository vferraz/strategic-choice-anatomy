#!/usr/bin/env python3
"""Layer B main figure, manuscript build.

The figure is caption-led to match Layer A: no in-figure master title, subtitle,
row banner, or provenance footer. Scientific content is unchanged from the
Layer B redesign: broad representation, selective recruitment, a GPT-OSS
MoE-native router audit, and cue-gated use. Causal steering is not plotted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from strategic_anatomy.config import results_root

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
from strategic_anatomy import paper_style as S  # noqa: E402

S.apply()

LBF = str(results_root() / "layer_b")
LRF = str(results_root() / "layer_b" / "recruitment")
RBT = str(results_root() / "layer_b")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Model identity follows the Layer A paper figures.
MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
NAME = {
    "qwen": "Qwen2.5",
    "qwen_instruct": "Qwen2.5-I",
    "llama31_instruct": "Llama",
    "gptoss": "GPT-OSS",
}
COL = {
    "qwen": "#1f77b4",
    "qwen_instruct": "#17becf",
    "llama31_instruct": "#2ca02c",
    "gptoss": "#9467bd",
}
MAXL = {"qwen_instruct": 80, "qwen": 80, "llama31_instruct": 80, "gptoss": 36}
INK, FAINT, GREY = S.INK, S.FAINT, S.GREY

COL_GATE = "#9467bd"
COL_TOPKW = "#6aa6d8"
COL_TOPK = "#2f6db3"
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.15}


def style(ax: plt.Axes, *, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    ax.tick_params(labelsize=S.NHB_FS_TICK, colors=INK, length=2.0, width=0.5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color="#e1e1e1", linewidth=0.42, alpha=0.78)


def ptitle(ax: plt.Axes, letter: str, text: str, *, y: float = S.NHB_PANEL_TITLE_Y) -> None:
    S.nhb_panel_title(ax, letter, text, fontsize=S.NHB_FS_PANEL, y=y)


def add_shared_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=COL[m], lw=1.8, marker="o", ms=3.8, label=NAME[m])
        for m in MODELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, S.NHB_LEGEND_Y),
        ncol=4,
        frameon=False,
        fontsize=S.NHB_FS_LEGEND,
        handlelength=1.05,
        handletextpad=0.28,
        columnspacing=0.70,
        borderaxespad=0.0,
    )


def main() -> int:
    fig = plt.figure(
        figsize=(S.NHB_TEXTWIDTH_IN, S.NHB_TWO_ROW_HEIGHT_IN),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.62,
        wspace=0.43,
        left=0.085,
        right=0.985,
        top=0.875,
        bottom=0.125,
    )
    add_shared_legend(fig)

    # A. Representation inventory.
    axa = fig.add_subplot(gs[0, 0])
    style(axa)
    dec = pd.read_csv(os.path.join(LBF, "b1_decodability.csv"))
    targets = ["sign_delta1c", "canonical_action", "sign_delta2c", "stim_control_cell00"]
    tlab = ["own\nΔ₁ᶜ", "canonical\nchoice", "opp.\nΔ₂ᶜ", "stimulus\ncontrol"]
    x = np.arange(len(targets))
    width = 0.16
    for i, m in enumerate(MODELS):
        sub = dec[dec.model == m].set_index("probe")
        vals = [sub.loc[t, "auc"] for t in targets]
        lo = [sub.loc[t, "auc"] - sub.loc[t, "lo"] for t in targets]
        hi = [sub.loc[t, "hi"] - sub.loc[t, "auc"] for t in targets]
        axa.bar(
            x + (i - 1.5) * width,
            vals,
            width,
            yerr=[lo, hi],
            color=COL[m],
            capsize=1.2,
            error_kw={"lw": 0.55, "ecolor": "#555555"},
            edgecolor="white",
            linewidth=0.25,
        )
    axa.axhline(0.5, ls=(0, (4, 3)), c=GREY, lw=0.75)
    axa.text(
        3.43,
        0.512,
        "chance",
        fontsize=S.NHB_FS_FOOT,
        color=GREY,
        ha="right",
        bbox=LABEL_BOX,
    )
    axa.text(
        3.45,
        0.975,
        "payoff-content\ncontrol",
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="right",
        va="top",
        bbox=LABEL_BOX,
    )
    axa.set_xticks(x)
    axa.set_xticklabels(tlab, fontsize=S.NHB_FS_TICK_SMALL)
    axa.set_ylim(0.40, 1.0)
    axa.set_ylabel("decodability (AUC)", fontsize=S.NHB_FS_AXIS)
    axa.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ptitle(axa, "a", "Strategic decodability")

    # B. Depth crystallisation.
    axb = fig.add_subplot(gs[0, 1])
    style(axb)
    cry = pd.read_csv(os.path.join(LBF, "b1_crystallization.csv"))
    axb.axhspan(0.45, 0.55, color="#eeeeee", alpha=0.7, zorder=0)
    for m in MODELS:
        s = cry[cry.model == m].sort_values("depth_frac")
        axb.plot(s.depth_frac, s.auc, color=COL[m], lw=1.35, solid_capstyle="round")
        peak = s.loc[s.auc.idxmax()]
        axb.scatter(
            [peak.depth_frac],
            [peak.auc],
            color=COL[m],
            s=11,
            zorder=6,
            edgecolor="white",
            linewidth=0.5,
        )
    axb.axhline(0.5, ls=(0, (4, 3)), c=GREY, lw=0.75)
    axb.set_xlim(0, 1)
    axb.set_ylim(0.30, 0.97)
    axb.set_xlabel("neural-network depth\n(0=embedding, 1=final layer)", fontsize=S.NHB_FS_AXIS)
    axb.set_ylabel("canonical-action AUC", fontsize=S.NHB_FS_AXIS)
    axb.text(
        0.98,
        0.415,
        "embedding floor → late crystallisation",
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="right",
        va="bottom",
        style="italic",
        bbox=LABEL_BOX,
    )
    ptitle(axb, "b", "Choice readout across depth")

    # C. Decision-incentive geometry.
    axc = fig.add_subplot(gs[0, 2])
    style(axc)
    gd = pd.read_csv(os.path.join(LRF, "recruitment_geometry_depth.csv"))
    axc.axhline(90, ls=(0, (1, 2)), c="#8a7da6", lw=0.8, zorder=0)
    axc.text(
        0.985,
        91.5,
        "90° (independent vectors)",
        fontsize=S.NHB_FS_FOOT,
        color="#8a7da6",
        ha="right",
        va="center",
    )
    for m in MODELS:
        s = gd[gd.model == m].dropna(subset=["angle_decision_incentive_deg"]).copy()
        s["depth"] = s.layer / MAXL[m]
        s = s.sort_values("depth")
        axc.plot(s.depth, s.angle_decision_incentive_deg, color=COL[m], lw=1.35)
        axc.scatter(
            [s.depth.values[-1]],
            [s.angle_decision_incentive_deg.values[-1]],
            color=COL[m],
            s=15,
            zorder=6,
            edgecolor="white",
            linewidth=0.5,
        )
    axc.set_xlim(0, 1)
    axc.set_ylim(20, 97)
    axc.set_xlabel("neural-network depth\n(0=embedding, 1=final layer)", fontsize=S.NHB_FS_AXIS)
    axc.set_ylabel("decision-incentive angle (°)", fontsize=S.NHB_FS_AXIS)
    ptitle(axc, "c", "Decision-incentive axes")

    # D. GPT-OSS MoE gate-versus-route readout contrast.
    axd = fig.add_subplot(gs[1, 0])
    style(axd)
    bd = pd.read_csv(os.path.join(RBT, "router_bottleneck_depth.csv"))
    axd.plot(bd.layer, bd.router_gate, color=COL_GATE, lw=1.7, marker="o", ms=2.3)
    axd.plot(
        bd.layer,
        bd.router_topk_weight,
        color=COL_TOPKW,
        lw=1.35,
        marker="o",
        ms=2.1,
    )
    axd.plot(
        bd.layer,
        bd.router_topk_binary,
        color=COL_TOPK,
        lw=1.35,
        marker="o",
        ms=2.1,
    )
    axd.axhline(0.5, ls=(0, (4, 3)), c=GREY, lw=0.75)
    axd.axvline(18, color="#bbbbbb", lw=0.75)
    axd.text(34.6, 0.512, "chance", fontsize=S.NHB_FS_FOOT, color=GREY,
             ha="right", va="bottom")
    axd.set_xlim(0, 36)
    axd.set_ylim(0.45, 0.96)
    axd.set_xlabel("GPT-OSS router layer", fontsize=S.NHB_FS_AXIS)
    axd.set_ylabel("incentive-sign AUC", fontsize=S.NHB_FS_AXIS)
    axd.text(35.1, bd["router_gate"].iloc[-1] + 0.012, "gate", color=COL_GATE,
             fontsize=S.NHB_FS_FOOT, ha="right", va="bottom")
    axd.text(35.1, bd["router_topk_weight"].iloc[-1] - 0.006, "top-k wt.",
             color=COL_TOPKW, fontsize=S.NHB_FS_FOOT, ha="right", va="top")
    axd.text(35.1, bd["router_topk_binary"].iloc[-1] - 0.010, "top-k set",
             color=COL_TOPK, fontsize=S.NHB_FS_FOOT, ha="right", va="top")
    gap18 = bd[bd.layer == 18].iloc[0]
    axd.text(
        0.03,
        0.94,
        f"L18: gate {gap18.router_gate:.2f}\nvs top-k set {gap18.router_topk_binary:.2f}",
        transform=axd.transAxes,
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="left",
        va="top",
    )
    ptitle(axd, "d", "GPT-OSS gate vs route")

    # E. Recruitment bridge.
    axe = fig.add_subplot(gs[1, 1])
    style(axe, grid="x")
    bridge = pd.read_csv(os.path.join(LRF, "within_model_bridge.csv")).set_index("model")
    order = ["qwen_instruct", "gptoss", "llama31_instruct", "qwen"]
    ypos = np.arange(len(order))[::-1]
    for y, m in zip(ypos, order):
        row = bridge.loc[m]
        raw = row.slope_pcanonical_per_sd_neural_inc
        partial = row.partial_slope_control_delta1c
        sig = (row.partial_ci_lo > 0) or (row.partial_ci_hi < 0)
        axe.barh(
            y + 0.18,
            raw,
            height=0.32,
            color=COL[m],
            alpha=0.30,
            xerr=[[raw - row.ci_lo], [row.ci_hi - raw]],
            capsize=1.8,
            error_kw={"lw": 0.6, "ecolor": "#888888"},
        )
        axe.barh(
            y - 0.18,
            partial,
            height=0.32,
            color=COL[m],
            edgecolor="white",
            linewidth=0.25,
            xerr=[[partial - row.partial_ci_lo], [row.partial_ci_hi - partial]],
            capsize=1.8,
            error_kw={"lw": 0.6, "ecolor": "#555555"},
        )
        if sig:
            label_x = row.partial_ci_hi + 0.006
            ha = "left"
            label_color = INK
            label_weight = "bold"
            label = f"{partial:+.2f}"
        elif partial >= 0:
            label_x = row.partial_ci_hi + 0.006
            ha = "left"
            label_color = GREY
            label_weight = "normal"
            label = f"{partial:+.2f}"
        else:
            label_x = 0.135 if m == "qwen" else row.partial_ci_lo - 0.006
            ha = "left" if m == "qwen" else "right"
            label_color = GREY
            label_weight = "normal"
            label = f"{partial:+.2f}"
        axe.text(
            label_x,
            y - 0.18,
            label,
            va="center",
            ha=ha,
            fontsize=S.NHB_FS_FOOT,
            color=label_color,
            fontweight=label_weight,
            bbox=LABEL_BOX,
        )
    axe.axvline(0, color="#999999", lw=0.8)
    axe.set_yticks(ypos)
    axe.set_yticklabels([NAME[m] for m in order], fontsize=S.NHB_FS_TICK)
    axe.set_xlim(-0.05, 0.235)
    axe.set_ylim(-0.6, 4.05)
    axe.set_xlabel("change in P(canonical)\nper SD neural incentive", fontsize=S.NHB_FS_AXIS)
    axe.text(
        0.98,
        0.98,
        "pale: raw\nsolid: beyond Δ₁ᶜ",
        transform=axe.transAxes,
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="right",
        va="top",
        bbox=LABEL_BOX,
    )
    axe.text(
        0.98,
        0.03,
        "GPT: pure only",
        transform=axe.transAxes,
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="right",
        va="bottom",
        bbox=LABEL_BOX,
    )
    ptitle(axe, "e", "Incentive recruitment")

    # F. Cue axes and behavioural use.
    axf = fig.add_subplot(gs[1, 2])
    style(axf, grid="x")
    ds = pd.read_csv(os.path.join(LRF, "disposition_dissociation.csv"))
    trait_order = ["inequity", "risk", "loss", "maximin", "selfish"]
    trait_y = {t: len(trait_order) - 1 - i for i, t in enumerate(trait_order)}
    offset = {
        "qwen": 0.21,
        "qwen_instruct": 0.07,
        "llama31_instruct": -0.07,
        "gptoss": -0.21,
    }
    axf.axvspan(0, 0.14, color="#eeeeee", alpha=0.8, zorder=0)
    axf.axhspan(trait_y["inequity"] - 0.38, trait_y["inequity"] + 0.38,
                color="#f4edf5", alpha=0.95, zorder=0)
    for trait in trait_order:
        axf.axhline(trait_y[trait], color="#e5e5e5", lw=0.55, zorder=0)
    for m in MODELS:
        sub = ds[ds.model == m].set_index("trait")
        for trait in trait_order:
            row = sub.loc[trait]
            y = trait_y[trait] + offset[m]
            val = row.behavior_magnitude
            xerr = [[val - row.behavior_magnitude_lo], [row.behavior_magnitude_hi - val]]
            axf.errorbar(
                val,
                y,
                xerr=xerr,
                fmt="o",
                ms=3.2,
                color=COL[m],
                ecolor="#8b8b8b",
                elinewidth=0.58,
                capsize=1.2,
                mec="white",
                mew=0.28,
                alpha=0.96,
                zorder=4,
            )
    axf.axvline(0, color="#999999", lw=0.75)
    axf.set_xlim(0, 1.0)
    axf.set_ylim(-0.65, len(trait_order) - 0.08)
    axf.set_yticks([trait_y[t] for t in trait_order])
    axf.set_yticklabels(trait_order, fontsize=S.NHB_FS_TICK_SMALL)
    axf.set_xlabel("absolute cue-baseline shift", fontsize=S.NHB_FS_AXIS)
    axf.set_ylabel("decision cue", fontsize=S.NHB_FS_AXIS)
    lda_min = ds["lda_trait_accuracy"].min()
    lda_max = ds["lda_trait_accuracy"].max()
    axf.text(
        0.02,
        len(trait_order) - 0.24,
        f"LDA {lda_min:.2f}-{lda_max:.2f}",
        fontsize=S.NHB_FS_FOOT,
        color=FAINT,
        ha="left",
        va="top",
        bbox=LABEL_BOX,
    )
    axf.text(
        0.98,
        len(trait_order) - 0.24,
        "largest shift",
        fontsize=S.NHB_FS_FOOT,
        color=INK,
        ha="right",
        va="top",
        bbox=LABEL_BOX,
    )
    ptitle(axf, "f", "Cue encoding and use")

    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(os.path.join(OUT, "fig_layerB_main_v2.pdf"), dpi=300,
                    bbox_inches=None, facecolor="white", transparent=False)
        fig.savefig(os.path.join(OUT, "fig_layerB_main_v2.png"), dpi=600,
                    bbox_inches=None, facecolor="white", transparent=False)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig_layerB_main_v2.png"), "(+pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
