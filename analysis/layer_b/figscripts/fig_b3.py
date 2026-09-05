#!/usr/bin/env python3
"""Figure B3 — dispositions: clearly represented, weakly recruited ("heard, not obeyed").

(a) held-out LDA disposition separation: train on a game split, project on held-out games. 70B+
    models (incl. Llama) separate the 5 dispositions near-perfectly; the placebo (length-matched
    null) lands central (low centrality ratio) -> the discriminant is real, not prompt-perturbation.
(b) disposition axis ⊥ decision axis: the common cue-shift direction is ~orthogonal to the decision
    axis in every model -> "what to do" and "what kind of player" on independent tracks.
(c) representation-vs-recruitment dissociation: per (model, trait), held-out LDA accuracy
    (representation, x) vs signed behavioural aim (recruitment, y). Llama sits at the top edge
    (perfect representation) but aim≈0 (not recruited) — the one-panel statement of the thesis.

Reserved capstone slot (NOT built): the causal double dissociation (steering the disposition axis
moves the disposition readout but not the choice). No causal claim is made.

  .venv/bin/python analysis/layer_b/figscripts/fig_b3.py
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
TRAIT_MARK = {"risk_aversion": "o", "loss_aversion": "s", "inequity_aversion": "^",
              "selfish_maximizer": "D", "maximin": "v"}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}")


def main():
    lda = pd.read_csv(TAB / "b3_lda.csv")
    ortho = pd.read_csv(TAB / "b3_orthogonality.csv").set_index("model")
    diss = pd.read_csv(TAB / "b3_dissociation.csv")

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42, top=0.80, bottom=0.16, left=0.07, right=0.97)

    # ── (a) held-out LDA accuracy + placebo centrality ────────────────────
    axa = fig.add_subplot(gs[0])
    x = np.arange(len(ORDER))
    overall = [lda[(lda.model == m) & (lda.disposition == "OVERALL")]["held_out_acc"].values[0] for m in ORDER]
    axa.bar(x, overall, 0.6, color=[COL[m] for m in ORDER], edgecolor="white", lw=0.4, zorder=3)
    # per-trait points
    for xi, m in zip(x, ORDER):
        sub = lda[(lda.model == m) & (lda.disposition.isin(lib.TRAITS))]
        axa.scatter(np.full(len(sub), xi), sub["held_out_acc"], s=14, color=PS.INK, alpha=0.6, zorder=5)
        pc = ortho.loc[m, "placebo_centrality"]
        axa.text(xi, 0.46, f"placebo\n{pc:.2f}", fontsize=PS.FS_FOOT, color=PS.FAINT, ha="center")
    axa.axhline(0.2, color=PS.GREY, lw=0.7, ls=":")  # 5-class chance
    axa.set_xticks(x); axa.set_xticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_TICK)
    axa.set_ylim(0.4, 1.03); axa.set_ylabel("held-out LDA accuracy (5 dispositions)", fontsize=PS.FS_AXIS)
    axa.tick_params(labelsize=PS.FS_TICK)
    PS.panel_title(axa, "a", "Dispositions are represented (held-out)")

    # ── (b) disposition axis ⊥ decision axis ──────────────────────────────
    axb = fig.add_subplot(gs[1])
    ang = [ortho.loc[m, "angle_disposition_decision_deg"] for m in ORDER]
    axb.bar(x, ang, 0.6, color=[COL[m] for m in ORDER], edgecolor="white", lw=0.4, zorder=3)
    axb.axhline(90, color=PS.GREY, lw=0.9, ls="--")
    axb.text(len(ORDER) - 0.5, 91, "orthogonal", fontsize=PS.FS_FOOT, color=PS.FAINT, ha="right")
    axb.set_xticks(x); axb.set_xticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_TICK)
    axb.set_ylim(0, 110); axb.set_ylabel("angle(disposition, decision) [°]", fontsize=PS.FS_AXIS)
    axb.tick_params(labelsize=PS.FS_TICK)
    PS.panel_title(axb, "b", "Disposition axis ⊥ decision axis")

    # ── (c) representation vs recruitment dissociation ────────────────────
    axc = fig.add_subplot(gs[2])
    for m in ORDER:
        sub = diss[diss.model == m]
        for _, r in sub.iterrows():
            axc.scatter(r["lda_acc"], r["behavioral_aim"], s=42, color=COL[m],
                        marker=TRAIT_MARK.get(r["trait"], "o"), edgecolors=PS.INK, lw=0.4, zorder=5)
    axc.axhline(0, color=PS.GREY, lw=0.7, ls=":")
    axc.set_xlabel("disposition decodability (held-out LDA acc)", fontsize=PS.FS_AXIS)
    axc.set_ylabel("behavioural aim (signed cue→choice shift)", fontsize=PS.FS_AXIS)
    axc.tick_params(labelsize=PS.FS_TICK)
    # model legend (color) + trait legend (marker)
    from matplotlib.lines import Line2D
    mh = [Line2D([0], [0], marker="o", ls="", mfc=COL[m], mec="none", ms=6, label=SHORT[m]) for m in ORDER]
    axc.legend(handles=mh, fontsize=PS.FS_FOOT, frameon=False, loc="upper left", ncol=2)
    axc.text(0.98, 0.03, "Llama: high representation, ~zero aim", transform=axc.transAxes,
             ha="right", fontsize=PS.FS_FOOT, color=PS.FAINT)
    PS.panel_title(axc, "c", "Represented ≠ recruited")

    fig.text(0.5, 0.015, "Reserved capstone (NOT built): causal double dissociation — steering the "
             "disposition axis moves the readout but not the choice. No do()-claim yet.",
             ha="center", fontsize=PS.FS_FOOT, color=PS.FAINT)
    PS.figure_titles(fig, "Dispositions: clearly represented, weakly recruited",
                     f"deepest-layer cue-shift vectors{PS.SEP}5 dispositions + length-matched placebo"
                     f"{PS.SEP}held-out games")
    _save(fig, "fig_b3_dispositions")


if __name__ == "__main__":
    main()
