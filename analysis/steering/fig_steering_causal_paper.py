"""Causal-steering main figure (house style, matches Figs 1-5 / _paper_style).

Small-dose (linear-regime) steering result on the corrected Akata substrate.
(a) family-resolved canonical dose-slope gradient (d_inc, letter-orthogonalized,
    layer 65): the incentive push shifts choice toward the canonical action only
    in dominance-solvable games; random directions are flat.
(b) two-channel dissociation: the letter-axis slope collapses under
    orthogonalization at L79 (pure readout-axis overlap) but survives at L65
    (content-mediated).

Reads data/results/steering/smalldose_summary_q05/{family_canonical,summary}.csv
(the corrected q=0.5 incentive target). Override with --tables to render the
pre-correction empirical arm; --out-pdf redirects the output.
GPT-OSS excluded (MoE hold; dense models only). Run from anywhere.
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

# Default = the CORRECTED q=0.5 tables, i.e. the figure the paper ships.
# The pre-correction empirical arm is still readable via
#   --tables data/results/steering/smalldose_summary
T = str(results_root() / "steering" / "smalldose_summary_q05")
OUT = os.path.join(ROOT, 'analysis', 'steering', 'figures')
OUT_PDF = os.path.join(OUT, "fig_steering_causal.pdf")
INK, FAINT, GREY = S.INK, S.FAINT, S.GREY

MODELS = ["qwen", "qwen_instruct", "llama31_instruct"]
NAME = {"qwen": "Qwen2.5", "qwen_instruct": "Qwen2.5-I", "llama31_instruct": "Llama"}
COL = {"qwen": "#1f77b4", "qwen_instruct": "#17becf", "llama31_instruct": "#2ca02c"}

# family order: dominance-solvable (incentive determines choice) -> coordination -> matching pennies
FAM_ORDER = ["DD", "OD2", "OD1", "CO1", "CO2", "MP"]
DOM_FAMS = {"DD", "OD2", "OD1"}


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


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="causal-steering main figure (additive path overrides)")
    p.add_argument("--tables", default=T, help="directory holding family_canonical.csv + summary.csv")
    p.add_argument("--out-pdf", default=OUT_PDF, help="output PDF path (.png written alongside)")
    return p.parse_args(argv)


def _one(df, where, what):
    """Single-row lookup with a legible error instead of a bare IndexError."""
    if len(df) != 1:
        raise SystemExit(f"[fig_steering] expected exactly 1 row for {what} ({where}); got {len(df)}")
    return df.iloc[0]


def main(argv=None):
    global T, OUT_PDF
    a = _parse_args(argv)
    T, OUT_PDF = a.tables, a.out_pdf
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PDF)), exist_ok=True)
    print(f"[fig_steering] tables={T}\n[fig_steering] out={OUT_PDF}")
    fam = pd.read_csv(os.path.join(T, "family_canonical.csv"))
    summ = pd.read_csv(os.path.join(T, "summary.csv"))

    fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, 3.0), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.085, right=0.985,
                          top=0.800, bottom=0.175)

    # shared legend
    handles = [Line2D([0], [0], color=COL[m], lw=1.8, marker="o", ms=3.8, label=NAME[m])
               for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, S.NHB_LEGEND_Y),
               ncol=3, frameon=False, fontsize=S.NHB_FS_LEGEND, handlelength=1.05,
               handletextpad=0.28, columnspacing=0.90, borderaxespad=0.0)

    # ---- (a) family gradient: h1 (d_inc), letter-orthogonalized, L65 ----------
    axa = fig.add_subplot(gs[0, 0]); style(axa, grid="y")
    h1 = fam[fam["mode"] == "h1_dinc"].copy()
    xbase = np.arange(len(FAM_ORDER))
    dodge = {"qwen": -0.22, "qwen_instruct": 0.0, "llama31_instruct": 0.22}
    # shade the dominance-solvable region
    dom_idx = [i for i, f in enumerate(FAM_ORDER) if f in DOM_FAMS]
    axa.axvspan(min(dom_idx) - 0.5, max(dom_idx) + 0.5, color=S.BAND_DOMINANCE,
                zorder=0, alpha=0.9)
    axa.axhline(0, color="#bbbbbb", lw=0.6, zorder=1)
    for m in MODELS:
        xs, ys, lo, hi = [], [], [], []
        for i, f in enumerate(FAM_ORDER):
            r = h1[(h1["model"] == m) & (h1["family"] == f)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            xs.append(i + dodge[m]); ys.append(r["mean_canon_slope"])
            lo.append(r["mean_canon_slope"] - r["ci_lo"]); hi.append(r["ci_hi"] - r["mean_canon_slope"])
        axa.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3.6, color=COL[m],
                     ecolor=COL[m], elinewidth=0.9, capsize=1.6, capthick=0.9,
                     mec="white", mew=0.4, zorder=3)
    # random pooled reference band (canon, h1, random, L65) across models
    rnd = summ[(summ["mode"] == "h1_dinc") & (summ["component"] == "canon")
               & (summ["layer"] == 65) & (summ["variant"] == "random")]
    r_lo, r_hi = rnd["mean_slope"].min(), rnd["mean_slope"].max()
    axa.axhspan(r_lo, r_hi, color=GREY, alpha=0.16, zorder=1)
    axa.text(len(FAM_ORDER) - 1 + 0.30, (r_lo + r_hi) / 2, "random", color=FAINT,
             fontsize=S.NHB_FS_FOOT, va="center", ha="left", rotation=0)
    axa.set_xticks(xbase); axa.set_xticklabels(FAM_ORDER, fontsize=S.NHB_FS_TICK)
    axa.set_xlim(-0.6, len(FAM_ORDER) - 0.15)
    axa.set_ylabel("canonical dose-slope", fontsize=S.NHB_FS_AXIS)
    axa.set_xlabel("game family  (dominance-solvable  →  coordination  →  MP)",
                   fontsize=S.NHB_FS_TICK_SMALL)
    ptitle(axa, "a", "Family-resolved susceptibility")

    # ---- (b) two-channel dissociation: letter-axis slope main vs main_perp -----
    axb = fig.add_subplot(gs[0, 1]); style(axb, grid="y")
    lett = summ[(summ["mode"] == "h1_dinc") & (summ["component"] == "letter")].copy()
    layers = [79, 65]
    xL = {79: 0.0, 65: 1.4}
    dodge2 = {"qwen": -0.28, "qwen_instruct": 0.0, "llama31_instruct": 0.28}
    axb.axhline(0, color="#bbbbbb", lw=0.6, zorder=1)
    for L in layers:
        for m in MODELS:
            rm = _one(lett[(lett["model"] == m) & (lett["layer"] == L) & (lett["variant"] == "main")],
                      f"model={m} layer={L}", "variant=main letter slope")
            rp = _one(lett[(lett["model"] == m) & (lett["layer"] == L) & (lett["variant"] == "main_perp")],
                      f"model={m} layer={L}", "variant=main_perp letter slope")
            x = xL[L] + dodge2[m]
            v_main, v_perp = rm["mean_slope"], rp["mean_slope"]
            axb.plot([x, x], [v_perp, v_main], color=COL[m], lw=1.4, zorder=2,
                     solid_capstyle="round")
            axb.plot(x, v_main, "o", ms=4.2, color=COL[m], mec="white", mew=0.5, zorder=3)
            axb.plot(x, v_perp, "o", ms=4.2, mfc="white", mec=COL[m], mew=1.1, zorder=3)
    axb.set_xticks([xL[79], xL[65]])
    axb.set_xticklabels(["Layer 79\n(later site)", "Layer 65\n(earlier site)"], fontsize=S.NHB_FS_TICK)
    axb.set_xlim(-0.7, 2.1)
    axb.set_ylabel("letter-axis dose-slope", fontsize=S.NHB_FS_AXIS)
    ptitle(axb, "b", "Letter-component ablation by site")
    # legend for filled vs open
    lh = [Line2D([0], [0], marker="o", color=INK, lw=0, ms=4.2, mfc=INK, label="raw (d$_{inc}$)"),
          Line2D([0], [0], marker="o", color=INK, lw=0, ms=4.2, mfc="white", mec=INK,
                 mew=1.1, label="⊥ℓ (letter removed)")]
    axb.legend(handles=lh, loc="upper right", frameon=False, fontsize=S.NHB_FS_FOOT,
               handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2)

    out_pdf = OUT_PDF
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=200)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
