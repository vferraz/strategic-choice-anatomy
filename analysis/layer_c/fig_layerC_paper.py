"""Layer C paper figure (house style, matches Figs 1-5 / _paper_style).

Dense-model logit-lens output projections on the one-shot substrate.
(a) graded neural-lambda (final-token recruitment)  (b) own- vs opponent-payoff recruitment
(c) cue-target projection by position                (d) represent->project bridge (probe vs lens)

Reads analysis/layer_c/tables/*.csv. GPT-OSS excluded (dense only). Run from anywhere.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from strategic_anatomy.config import results_root

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
from strategic_anatomy import paper_style as S  # noqa: E402
S.apply()

T = str(results_root() / "layer_c")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
INK, FAINT, GREY = S.INK, S.FAINT, S.GREY

MODELS = ["qwen", "qwen_instruct", "llama31_instruct"]
NAME = {"qwen": "Qwen2.5", "qwen_instruct": "Qwen2.5-I", "llama31_instruct": "Llama"}
COL = {"qwen": "#1f77b4", "qwen_instruct": "#17becf", "llama31_instruct": "#2ca02c"}
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.15}


def style(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    ax.tick_params(labelsize=S.NHB_FS_TICK, colors=INK, length=2.0, width=0.5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color="#e1e1e1", linewidth=0.42, alpha=0.78)


def ptitle(ax, letter, text, *, y=S.NHB_PANEL_TITLE_Y):
    S.nhb_panel_title(ax, letter, text, fontsize=S.NHB_FS_PANEL, y=y)


def add_shared_legend(fig):
    handles = [Line2D([0], [0], color=COL[m], lw=1.8, marker="o", ms=3.8, label=NAME[m]) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, S.NHB_LEGEND_Y),
               ncol=3, frameon=False, fontsize=S.NHB_FS_LEGEND, handlelength=1.05,
               handletextpad=0.28, columnspacing=0.70, borderaxespad=0.0)


def main():
    nl = pd.read_csv(os.path.join(T, "stat_neural_lambda.csv"))
    rc = pd.read_csv(os.path.join(T, "stat_region_contrib.csv"))
    ho = pd.read_csv(os.path.join(T, "stat_heard_obeyed.csv"))
    br = pd.read_csv(os.path.join(T, "bridge_probe_vs_lens.csv"))

    fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, S.NHB_TWO_ROW_HEIGHT_IN), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.37, left=0.105, right=0.985, top=0.880, bottom=0.130)
    add_shared_legend(fig)

    # (a) graded neural-lambda (final-token), ordered by strength
    axa = fig.add_subplot(gs[0, 0]); style(axa, grid="x")
    order = ["qwen_instruct", "llama31_instruct", "qwen"]
    ys = np.arange(len(order))[::-1]
    for m, y in zip(order, ys):
        r = nl[(nl.model == m) & (nl.layer == 79)].iloc[0]
        axa.errorbar(r.lambda_lens, y, xerr=[[r.lambda_lens - r.lambda_lo], [r.lambda_hi - r.lambda_lens]],
                     fmt="o", ms=4.0, color=COL[m], ecolor="#8b8b8b", elinewidth=0.6, capsize=1.6,
                     mec="white", mew=0.3, zorder=3)
        axa.text(r.lambda_hi + 0.06, y, f"p_can={r.p_sign:.2f}", va="center", ha="left",
                 fontsize=S.NHB_FS_FOOT, color=FAINT)
    axa.axvline(0, color=GREY, lw=0.5, ls=(0, (3, 2)))
    axa.set_yticks(ys); axa.set_yticklabels([NAME[m] for m in order], fontsize=S.NHB_FS_TICK)
    axa.set_xlim(-0.15, 2.05); axa.set_ylim(-0.6, len(order) - 0.4)
    axa.set_xlabel(r"lens $\lambda$ at final token  (slope on $\Delta_1^{\,c}$)", fontsize=S.NHB_FS_AXIS)
    axa.text(0.98, 0.04, "behavioural $\\lambda$ ≈ 1.9–2.0 (all three)", transform=axa.transAxes,
             ha="right", va="bottom", fontsize=S.NHB_FS_FOOT, color=FAINT, bbox=LABEL_BOX)
    ptitle(axa, "a", "Incentive projection at commitment")

    # (b) own- vs opponent-payoff recruitment
    axb = fig.add_subplot(gs[0, 1]); style(axb, grid="x")
    regs = ["own_payoff", "opponent_payoff", "answer_prefix"]
    rlab = {"own_payoff": "own payoff", "opponent_payoff": "opp. payoff", "answer_prefix": "answer"}
    yreg = np.arange(len(regs))[::-1]; h = 0.24
    axb.axhspan(yreg[1] - 0.42, yreg[1] + 0.42, color="#f2f2f2", zorder=0)
    for i, m in enumerate(MODELS):
        sub = rc[rc.model == m].set_index("region").reindex(regs)
        off = (i - 1) * h
        axb.barh(yreg + off, sub["mean"], height=h, color=COL[m], edgecolor="white", linewidth=0.25,
                 xerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
                 error_kw={"lw": 0.55, "ecolor": "#888888"}, zorder=3)
    axb.axvline(0, color="#999999", lw=0.7)
    axb.set_yticks(yreg); axb.set_yticklabels([rlab[r] for r in regs], fontsize=S.NHB_FS_TICK)
    axb.set_xlabel("signed output-projection increment", fontsize=S.NHB_FS_AXIS)
    axb.text(0.02, 0.30, "non-zero opponent-payoff\nmean only in Qwen2.5-I", transform=axb.transAxes,
             fontsize=S.NHB_FS_FOOT, color=FAINT)
    ptitle(axb, "b", "Payoff-region projection increments")

    # (c) absolute cue-target projections at cue and final positions
    axc = fig.add_subplot(gs[1, 0]); style(axc)
    traits = ["risk_aversion", "loss_aversion", "maximin", "selfish_maximizer", "inequity_aversion"]
    tlab = ["risk", "loss", "maximin", "selfish", "inequity"]
    tx = np.arange(len(traits)); w = 0.26
    for i, m in enumerate(MODELS):
        sub = ho[ho.model == m].set_index("trait").reindex(traits)
        off = (i - 1) * w
        axc.bar(tx + off, sub["final_target_projection"], width=w, color=COL[m], edgecolor="white", linewidth=0.25,
                yerr=[sub["final_target_projection"] - sub["final_target_projection_lo"],
                      sub["final_target_projection_hi"] - sub["final_target_projection"]],
                error_kw={"lw": 0.55, "ecolor": "#888888"}, zorder=3)
        axc.scatter(tx + off, sub["cue_token_projection"], s=5, color=INK, zorder=5)
    axc.axhline(0, color="#999999", lw=0.7)
    axc.set_xticks(tx); axc.set_xticklabels(tlab, fontsize=S.NHB_FS_TICK_SMALL, rotation=0)
    axc.set_ylabel("absolute target-letter projection", fontsize=S.NHB_FS_AXIS)
    axc.text(0.015, 0.97, "• cue-token projection ≈ 0", transform=axc.transAxes, va="top",
             fontsize=S.NHB_FS_FOOT, color=FAINT)
    lr = ho[(ho.model == "llama31_instruct") & (ho.trait == "risk_aversion")].iloc[0]
    axc.annotate("Llama risk-wording\nfinal projection ≈ 0",
                 xy=(0 + w, lr.final_target_projection), xytext=(0.42, 0.62),
                 textcoords="axes fraction", fontsize=S.NHB_FS_FOOT, color=COL["llama31_instruct"],
                 arrowprops=dict(arrowstyle="->", color=COL["llama31_instruct"], lw=0.6))
    ptitle(axc, "c", "Cue-target projection by position")

    # (d) represent -> project bridge (probe vs lens), the punchline
    axd = fig.add_subplot(gs[1, 1]); style(axd)
    b = br[br.layer == 79]
    pos = {"own_payoff": 0.0, "final_answer": 1.0}
    dod = {"qwen": -0.16, "qwen_instruct": 0.0, "llama31_instruct": 0.16}
    for m in MODELS:
        for rg, x0 in pos.items():
            row = b[(b.model == m) & (b.region == rg)]
            if row.empty:
                continue
            r = row.iloc[0]
            pa, la = float(r.probe_auc), float(r.lens_auc)
            x = x0 + dod[m]
            axd.plot([x, x], [pa, la], color=COL[m], lw=0.7, zorder=2)
            axd.errorbar(x, pa, yerr=[[pa - r.probe_lo], [r.probe_hi - pa]], fmt="o", ms=4.2,
                         color=COL[m], ecolor=COL[m], elinewidth=0.6, capsize=1.3, zorder=3)   # probe (filled)
            axd.errorbar(x, la, yerr=[[la - r.lens_lo], [r.lens_hi - la]], fmt="o", ms=4.2,
                         mfc="white", mec=COL[m], mew=0.9, ecolor=COL[m], elinewidth=0.6,
                         capsize=1.3, zorder=3)                                                # lens (open)
    axd.axhline(0.5, color=GREY, lw=0.75, ls=(0, (4, 3)))
    axd.set_xticks([0, 1]); axd.set_xticklabels(["own-payoff\ntoken", "final\ntoken"], fontsize=S.NHB_FS_TICK)
    axd.set_xlim(-0.5, 1.5); axd.set_ylim(0.40, 1.02)
    axd.set_ylabel("decode incentive (AUC)", fontsize=S.NHB_FS_AXIS)
    hleg = [Line2D([0], [0], marker="o", ls="", mfc=INK, mec=INK, ms=4, label="probe (any dir.)"),
            Line2D([0], [0], marker="o", ls="", mfc="white", mec=INK, mew=0.9, ms=4, label="lens (choice axis)")]
    axd.legend(handles=hleg, loc="upper left", frameon=False, fontsize=S.NHB_FS_FOOT,
               handletextpad=0.3, borderaxespad=0.1)
    axd.text(0.0, 0.415, "represented\noff-axis", ha="center", va="bottom", fontsize=S.NHB_FS_FOOT, color=FAINT)
    axd.text(1.0, 0.415, "projected\non-axis", ha="center", va="bottom", fontsize=S.NHB_FS_FOOT, color=FAINT)
    ptitle(axd, "d", "Representation vs projection")

    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(os.path.join(OUT, "fig_layerC_paper.pdf"), dpi=300, bbox_inches=None)
        fig.savefig(os.path.join(OUT, "fig_layerC_paper.png"), dpi=600, bbox_inches=None)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "fig_layerC_paper.png"), "(+pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
