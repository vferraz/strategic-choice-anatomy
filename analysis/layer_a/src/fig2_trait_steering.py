#!/usr/bin/env python3
"""Figure 2 — personality cues: magnitude vs direction (one-shot, paper-final).

FROZEN DESIGN. Reuses the committed ``analysis/block_a/fig_main_trait_steering_v2.py``
panel/layout/style verbatim (``panel_A``, ``panel_B``, ``clarity_pooled``, ``_ci``,
colours, gridspec, legend) and re-points ONLY the data sources to the one-shot
trait-effect tables (built by ``trait_effects.py`` over ``decoded_action``) and the
output directory. The single content change is panel (c): its variance partition is
read from the ONE-SHOT ``s_variance_partition.csv`` (built by ``attribution.py``)
instead of the hard-coded design_v2 literals — those numbers are data, not style.

Panel (a) magnitude vs aim per trait per model (inequity aims; risk/loss are
anti-aimed -> they shift toward the equality action; placebo anchors x=0). Panel (b)
incentive-gating: magnitude and net steering collapse as the payoff gap sharpens.

Tables: f2_trait_aim.csv, f2_trait_gating.csv.
Citations: trait cues are model-internal manipulations; human refs not used here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from analysis.layer_a.src import trait_effects as TE  # noqa: E402
from analysis.layer_a.src import attribution as ATTR  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


DATA = SD.DATA
OUTDIR = ROOT / "analysis" / "layer_a" / "figures"
TABLES = results_root() / "layer_a"

# import the FROZEN figure module and re-point its data globals (the only changes)
import analysis.layer_a.frozen.fig_main_trait_steering_v2 as V2T  # noqa: E402
V2T.SUMMARY = DATA / "trait_oneshot_summary.csv"
V2T.PER_GAME = DATA / "trait_oneshot_per_game.csv"
V2T.UNIFIED = DATA / "trait_unified_8vec.parquet"
V2T.FIG_DIR = OUTDIR


def _partition_numbers():
    p = TABLES / "s_variance_partition.csv"
    if not p.exists():
        ATTR.variance_partition()
    vp = pd.read_csv(p).set_index("block")
    return vp


def write_tables(s1, cs):
    rows = []
    for model in V2T.MODELS:
        for t in V2T.TRAITS + [V2T.CONTROL]:
            r = s1[(s1.model == model) & (s1.trait == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            aim = r.dir_net / r.magnitude if r.magnitude > 1e-9 else 0.0
            rows.append(dict(model=model, trait=t, magnitude=r.magnitude,
                             magnitude_lo=r.get("magnitude_lo", np.nan),
                             magnitude_hi=r.get("magnitude_hi", np.nan),
                             net=r.dir_net, net_lo=r.dir_net_lo, net_hi=r.dir_net_hi,
                             aim=aim, base_pref=r.base_pref_mean,
                             placebo_magnitude=r.magnitude_null, n_games=r.get("n_games", np.nan)))
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "f2_trait_aim.csv", index=False)
    cs.to_csv(TABLES / "f2_trait_gating.csv", index=False)
    print(f"[fig2] wrote {TABLES/'f2_trait_aim.csv'} ; {TABLES/'f2_trait_gating.csv'}")


def main() -> int:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    S = V2T.S

    summ = pd.read_csv(V2T.SUMMARY)
    s1 = summ[summ["round"] == 1]
    pg = pd.read_csv(V2T.PER_GAME)
    cs, labels = V2T.clarity_pooled(pg)
    vp = _partition_numbers()

    # ===== figure body: VERBATIM from fig_main_trait_steering_v2.main(), only the
    #       panel-(c) numbers come from the one-shot partition =====
    fig = plt.figure(figsize=(7.2, 7.05))
    gs = gridspec.GridSpec(3, len(V2T.MODELS), height_ratios=[1.0, 1.0, 0.34], hspace=0.88, wspace=0.14,
                           left=0.10, right=0.985, top=0.855, bottom=0.155)
    axA0 = axB0 = None
    for i, model in enumerate(V2T.MODELS):
        srow = {t: s1[(s1.model == model) & (s1.trait == t)].iloc[0].to_dict()
                for t in V2T.TRAITS + [V2T.CONTROL] if not s1[(s1.model == model) & (s1.trait == t)].empty}
        plac_mag = float(s1[(s1.model == model) & (s1.trait == "risk")].iloc[0]["magnitude_null"])
        axA = fig.add_subplot(gs[0, i]); V2T.panel_A(axA, srow, plac_mag, model, show_y=(i == 0))
        axB = fig.add_subplot(gs[1, i]); V2T.panel_B(axB, cs, labels, model, show_y=(i == 0))
        if i == 0:
            axA0, axB0 = axA, axB

    # ---- panel (c): ONE-SHOT held-out variance partition of canonical conformity ----
    struct = float(vp.loc["structure", "pct"]); trait = float(vp.loc["trait", "pct"])
    model_pct = float(vp.loc["model", "pct"])
    tr_dauc = float(vp.loc["trait", "dAUC"]); tr_lo = float(vp.loc["trait", "dAUC_lo"]); tr_hi = float(vp.loc["trait", "dAUC_hi"])
    axC = fig.add_subplot(gs[2, :])
    _seg = [("Game structure", struct, S.GOOD_GREEN, "white"),
            ("Trait prompt",   trait,  S.SOFT_BLUE,  "white"),
            ("Model identity", model_pct, S.GREY,    "white")]
    _x = 0.0
    for _lab, _w, _col, _tc in _seg:
        axC.barh(0, _w, left=_x, height=1.05, color=_col, edgecolor="white", lw=1.8, zorder=3)
        if _w >= 12:
            axC.text(_x + _w / 2, 0, f"{_lab}   {_w:.0f}%", ha="center", va="center",
                     fontsize=S.FS_FOOT, color=_tc, fontweight="bold")
        else:
            axC.text(_x + _w / 2, 0, f"{_w:.0f}%", ha="center", va="center",
                     fontsize=S.FS_FOOT, color=_tc, fontweight="bold")
            axC.text(_x + _w / 2, 0.78, _lab, ha="center", va="bottom",
                     fontsize=S.FS_FOOT, color="#262626", fontweight="bold")
            axC.plot([_x + _w / 2, _x + _w / 2], [0.54, 0.75], color="#9a9a9a", lw=0.7, zorder=2)
        _x += _w
    axC.set_xlim(0, 100); axC.set_ylim(-1.05, 1.15); axC.axis("off")
    axC.text(50, -0.85,
             r"held-out variance partition (GroupKFold-by-game)   $\cdot$   "
             rf"trait block $\Delta$AUC ${tr_dauc:+.3f}$ [{tr_lo:.3f}, {tr_hi:.3f}]",
             ha="center", va="center", fontsize=S.FS_FOOT, color=S.FAINT)

    pa, pb = axA0.get_position(), axB0.get_position()
    pc = axC.get_position()
    fig.text(pa.x0, pa.y1 + 0.028, "(a)  Do prompts steer toward the trait's payoffs?",
             fontsize=S.FS_PANEL, fontweight="bold", color=S.INK, ha="left", va="bottom")
    fig.text(pb.x0, pb.y1 + 0.016, "(b)  Does the steering survive a clear incentive?",
             fontsize=S.FS_PANEL, fontweight="bold", color=S.INK, ha="left", va="bottom")
    fig.text(pc.x0, pc.y1 + 0.012, "(c)  What explains canonical conformity?",
             fontsize=S.FS_PANEL, fontweight="bold", color=S.INK, ha="left", va="bottom")

    handles_A = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markeredgecolor="#555", ms=7,
               label="trait (filled: aim sig.)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#555", ms=7,
               label="trait (aim n.s.)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=S.GOOD_GREEN, markeredgecolor=S.GOOD_GREEN, ms=7,
               label="maximin (control)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=V2T.GREY, markeredgecolor=V2T.GREY, ms=8,
               label="placebo"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#c2c2c2",
               markeredgewidth=2.0, ms=11, label="ceiling (≥0.8)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=S.SOFT_BLUE, markeredgecolor=S.SOFT_BLUE, ms=6,
               label="(b) net steering"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=V2T.GREY, markeredgecolor=V2T.GREY, ms=6,
               label="(b) magnitude"),
    ]
    leg = fig.legend(handles=handles_A, loc="lower center", ncol=4, frameon=False, fontsize=S.FS_LEGEND,
                     bbox_to_anchor=(0.5, 0.045), columnspacing=1.3, handletextpad=0.4,
                     title="(a) points = model × trait" + S.SEP + "(b) lines pooled over traits")
    leg.get_title().set_fontsize(S.FS_FOOT); leg.get_title().set_color(S.SUBTLE)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(OUTDIR / f"fig_main_trait_steering_v2clean.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[fig2] wrote fig_main_trait_steering_v2clean.{png,pdf}")
    print(f"[fig2 c] one-shot partition: structure {struct:.1f}% / trait {trait:.1f}% / "
          f"model {model_pct:.1f}%  (trait dAUC {tr_dauc:+.3f} [{tr_lo:.3f},{tr_hi:.3f}])")

    write_tables(s1, cs)
    # printed panel-A coordinates (findings)
    print("\n[fig2 a] panel-A coordinates (one-shot, round 1):")
    for model in V2T.MODELS:
        for t in V2T.TRAITS + [V2T.CONTROL]:
            r = s1[(s1.model == model) & (s1.trait == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            aim = r.dir_net / r.magnitude if r.magnitude > 1e-9 else 0
            sig = (r.dir_net_lo > 0) or (r.dir_net_hi < 0)
            print(f"  {SD.SHORT[model]:8s} {t:9s} mag={r.magnitude:.2f}  aim={aim:+.2f}  sig={int(sig)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
