"""Merged R1 behaviour figure — six panels (a-f) on ONE 2x3 canvas.

WP1 / decision D1 of the paper: the two R1
behaviour display items become one figure.  Stacking the two shipped PDFs does
not work -- the page is 466pt wide, so a stacked merge scales both graphics to
~0.60x and shrinks every label with them.  Redrawing the six panels on one
canvas keeps the type at its real size (8.7pt panel titles, 7pt ticks) and only
narrows each panel's data area.

ADDITIVE, by design:
  * `fig1_rationality.py` and `fig3_qre_to_levelk.py` are IMPORTED, never edited.
  * Panels a-d are drawn by their own functions from `fig1_rationality`, so they
    are byte-for-byte the approved panels, only in a narrower column.
  * Panels e-f are the `fig3_qre_to_levelk.make_figure` panel bodies, copied here
    because that code builds its own figure and is not factored per-axes.  The
    ONLY changes are layout furniture (annotation offsets, tick density, label
    wrapping) needed to survive a 1/3-width column.  No series, number, colour
    or model was changed.  A proper refactor of fig3 should replace this copy.

Data: cached tables + the cheap builders (no fits, no bootstraps re-run).
  f1_*.csv          <- fig1's build_all()   (13s, verified byte-identical)
  f3_lambda.csv     <- committed
  f3_fingerprint.csv<- committed
  agent_games       <- fig3's build_agent_games()  (7s, no fitting)

Writes: analysis/layer_a/figures/fig_behaviour_merged_2x3.{pdf,png}

Run:  .venv/bin/python -m analysis.layer_a.figscripts.fig_behaviour_merged_2x3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.special import expit  # noqa: E402

from strategic_anatomy import paper_style as S  # noqa: E402
from analysis.layer_a.figscripts import fig1_rationality as F1  # noqa: E402
from analysis.layer_a.figscripts import fig3_qre_to_levelk as F3  # noqa: E402
from analysis.layer_a.build_regime_rationality import build_all  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


TAB_DIR = results_root() / "layer_a"
OUT_DIR = ROOT / "analysis" / "layer_a" / "figures"
STEM = "fig_behaviour_merged_2x3"

# ── canvas ────────────────────────────────────────────────────────────────────
# 7.1in is the frozen NHB canvas width (S.NHB_TEXTWIDTH_IN); LaTeX renders it at
# \textwidth = 466.34pt, i.e. 0.912x.  Height 5.55in -> 364.6pt in the document,
# which is what is left of \textheight (647.0pt) once the merged caption is set
# \footnotesize (268pt) plus float skips.
FIG_W = S.NHB_TEXTWIDTH_IN
FIG_H = 5.55

# Per-row column widths.  The rows do NOT share column boundaries: panels a and c
# carry more furniture (grouped bars over three families; two stacked sub-charts
# with agent names) than b and d, and e needs right margin for its lambda labels.
ROW1_RATIOS = [1.05, 0.90, 1.25]     # a, b, c  -- c carries 6 agent labels,
ROW2_RATIOS = [0.90, 1.12, 1.10]     # d, e, f     two sub-charts and a legend
WSPACE = 0.46                        # gap must clear c's left-hand agent labels


def _panel_letters_ef(ax, letter: str, text: str) -> None:
    S.nhb_panel_title(ax, letter, text, fontsize=S.NHB_FS_PANEL)


# ═════════════════════════════════════════════════════════════════════════════
#  Panel (e) — incentive response.  Body copied from fig3_qre_to_levelk.make_figure
#  (panel "a" there).  Furniture-only deltas are marked  # [2x3].
# ═════════════════════════════════════════════════════════════════════════════
def plot_incentive_response(ax, agent_games, lambda_tbl) -> None:
    fs_axis = S.NHB_FS_AXIS_LARGE
    all_x = np.concatenate(
        [agent_games[ag]["delta1c_unif"].to_numpy(float) for ag in F3.AGENTS])
    xx = np.linspace(float(np.nanmin(all_x)) - 0.05, float(np.nanmax(all_x)) + 0.05, 240)
    label_x = xx[-1] + 0.07                      # [2x3] 0.09 -> 0.07, tighter margin
    lambda_label_y = {
        "gptoss": 0.985,
        "qwen_instruct": 0.945,
        "qwen": 0.875,
        "nagel": 0.815,
        "llama31_instruct": 0.650,
    }
    for ag in F3.AGENTS:
        d = agent_games[ag]
        row = lambda_tbl[lambda_tbl["agent"] == ag].iloc[0]
        col = F3.COLOR[ag]
        is_h = ag == "nagel"
        y_action = d["s"].to_numpy(float) / d["n"].to_numpy(float)
        ax.scatter(d["delta1c_unif"], y_action, s=6,          # [2x3] s=9 -> 6
                   alpha=0.30 if not is_h else 0.45,
                   color=col, edgecolors="none", zorder=2,
                   marker="o" if not is_h else "D")
        lam_soft, alpha_soft = row["lam_soft_calibration"], row["alpha_soft_calibration"]
        if not is_h and np.isfinite(lam_soft) and np.isfinite(alpha_soft):
            ax.plot(xx, expit(alpha_soft + lam_soft * xx), color=col, lw=1.25,
                    ls=(0, (3, 2)), alpha=0.28, zorder=3)
        lam, alpha = row["lam_action"], row["alpha_action"]
        if np.isfinite(lam) and np.isfinite(alpha):
            ls = (0, (3, 2)) if is_h else "-"
            ax.plot(xx, expit(alpha + lam * xx), color=col, lw=1.4, ls=ls, zorder=4)
            y_end = float(expit(alpha + lam * xx[-1]))
            y_label = lambda_label_y.get(ag, y_end)
            if abs(y_label - y_end) > 0.012:
                ax.plot([xx[-1], label_x - 0.02], [y_end, y_label],
                        color=col, lw=0.45, alpha=0.65, zorder=4, clip_on=False)
            ax.text(label_x, y_label, f"{F3._compact_decimal(lam)}", color=col,
                    fontsize=5.6, ha="left", va="center", clip_on=False,  # [2x3] 6.0 -> 5.6
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.12))
    ax.axhline(0.5, color="#999", lw=0.7, ls=":")
    ax.axvline(0.0, color="#999", lw=0.7, ls=":")
    ax.set_xlim(xx[0], xx[-1] + 0.34)            # [2x3] 0.42 -> 0.34
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-1.0, 0.0, 1.0, 2.0])         # [2x3] thin the x ticks
    ax.set_xlabel(r"canonical incentive gap  $\Delta_1^{c}$", fontsize=fs_axis)
    ax.set_ylabel(r"$P(\mathrm{canonical\ action})$", fontsize=fs_axis)
    _panel_letters_ef(ax, "e", "Incentive response (λ)")   # [2x3] shortened title


# ═════════════════════════════════════════════════════════════════════════════
#  Panel (f) — bounded-rationality map.  Body copied from make_figure panel "b".
# ═════════════════════════════════════════════════════════════════════════════
def plot_fingerprint(ax, agent_games, fingerprint_tbl) -> None:
    fs_axis = S.NHB_FS_AXIS_LARGE
    R = fingerprint_tbl.set_index("agent")
    lm = F3.qlk_landmarks(agent_games["qwen"])
    ax.axhline(0.5, color="#bbb", ls=":", lw=0.75, zorder=1)
    ax.text(3.12, 0.505, "chance", fontsize=5.4, color="#888", ha="right", va="bottom")

    ax.scatter([0, 1, 2, 3], [lm["L0"], lm["L1"], lm["L2"], lm["L3"]],
               marker="D", s=18, color="#8a8a8a", zorder=2)   # [2x3] s=22 -> 18
    # [2x3] the four landmark captions are collapsed to bare level names; the
    # "deterministic" qualifier and the idealised-limit gloss move to the caption
    # text that already carries them ("Grey diamonds show pure level-k reference
    # policies").  No landmark was removed.
    for x, y, lab, off, ha in [
        (0, lm["L0"], "L0", (7, 6), "left"),
        (1, lm["L1"], "L1", (-4, 6), "right"),
        (2, lm["L2"], "L2", (0, 6), "center"),
        (3, lm["L3"], "L3", (9, 6), "right"),
    ]:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                    ha=ha, va="bottom", fontsize=5.4, color="#777")
    ax.annotate("pure level-k refs", (3.12, 0.468), fontsize=4.9,
                color="#9a9a9a", ha="right")

    cluster = ["qwen", "qwen_instruct", "llama31_instruct", "nagel"]
    dlo = R.loc[cluster, "depth_pi_ci_lo"].min()
    dhi = R.loc[cluster, "depth_pi_ci_hi"].max()
    slo = R.loc[cluster, "sharpness_ci_lo"].min()
    shi = R.loc[cluster, "sharpness_ci_hi"].max()
    if np.all(np.isfinite([dlo, dhi, slo, shi])):
        ax.add_patch(Ellipse(((dlo + dhi) / 2, (slo + shi) / 2),
                             max(dhi - dlo, 0.01), max(shi - slo, 0.01),
                             facecolor="#888", alpha=0.09, edgecolor="#aaa",
                             ls="--", lw=0.65, zorder=1))
        ax.annotate("dense LLMs + human\n95% CIs overlap",
                    ((dlo + dhi) / 2, slo - 0.028), ha="center", va="top",
                    fontsize=5.0, color="#777")

    lab_off = {                                   # [2x3] retuned for the narrow axes
        "qwen": (6, 9),
        "qwen_instruct": (7, -12),
        "llama31_instruct": (-9, 12),
        "nagel": (-14, -1),
        "gptoss": (-7, 12),
    }
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.12)
    for ag, r in R.iterrows():
        x = float(r["depth_pi"])
        y = float(r["sharpness"])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        col = F3.COLOR[ag]
        xerr = np.array([[x - float(r["depth_pi_ci_lo"])],
                         [float(r["depth_pi_ci_hi"]) - x]])
        yerr = np.array([[y - float(r["sharpness_ci_lo"])],
                         [float(r["sharpness_ci_hi"]) - y]])
        xerr = np.where(np.isfinite(xerr) & (xerr >= 0), xerr, 0.0)
        yerr = np.where(np.isfinite(yerr) & (yerr >= 0), yerr, 0.0)
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=col,
                    elinewidth=0.75, alpha=0.75, capsize=1.6, zorder=4)
        ax.scatter(x, y, s=30 if ag == "gptoss" else 26, color=col,
                   marker="D" if ag == "nagel" else "o",
                   edgecolor="white", linewidth=0.5, zorder=5)
        dx, dy = lab_off.get(ag, (5, 4))
        ax.annotate(F3.DISPLAY[ag], (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha="left" if dx >= 0 else "right",
                    va="center", fontsize=5.5,
                    fontweight="bold" if ag == "gptoss" else "normal",
                    color=col, bbox=label_box,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.36,
                                    alpha=0.55, shrinkA=2, shrinkB=3)
                    if ag != "gptoss" else None)
    if "gptoss" in R.index:
        g = R.loc["gptoss"]
        ax.annotate(f"L2/L3 mix\nL0 ≈ {g['pi_L0'] * 100:.0f}%",
                    (float(g["depth_pi"]), float(g["sharpness"])),
                    textcoords="offset points", xytext=(7, -14),
                    ha="left", va="top", fontsize=5.0, color="#6b4fa0",
                    bbox=label_box,
                    arrowprops=dict(arrowstyle="-", color="#6b4fa0", lw=0.38,
                                    alpha=0.55, shrinkA=2, shrinkB=3))

    ax.set_xlim(-0.10, 3.18)
    ax.set_ylim(0.45, 0.995)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("reasoning depth\n(quantal level-k mixture)", fontsize=fs_axis)
    ax.set_ylabel("decision sharpness\nmean modal-action prob.", fontsize=fs_axis)
    _panel_letters_ef(ax, "f", "Bounded-rationality map")


def _finish_ef(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.tick_params(labelsize=S.NHB_FS_TICK_SMALL, width=0.55, length=2)
    ax.grid(axis="y", color="#dedede", linewidth=0.45, alpha=0.8)


def _oneline_ylabel(ax) -> None:
    """[2x3] Collapse a two-line y-label onto one line.

    A rotated label is bounded by the panel HEIGHT (~1.68in here), so this is only
    applied where the single line demonstrably fits; panels b and f keep their two
    lines because they do not.  Text is unchanged apart from the newline.
    """
    lab = ax.get_ylabel()
    if "\n" in lab:
        ax.set_ylabel(lab.replace("\n", " "), fontsize=ax.yaxis.label.get_fontsize())


def _align_top_row_titles(axes, y: float = 1.18) -> None:
    """[2x3] Put every top-row panel title on one line.

    Panel c needs two header rows (its panel title, then the Pareto-rankable /
    Distributional-conflict sub-titles), so its title sits at y=1.18 while a and b
    sit at 1.07.  Raising a and b to c's height aligns all three at the top rather
    than dropping c into its own sub-titles.
    """
    for ax, keys in axes:
        for t in ax.texts:
            if t.get_text() in keys:
                t.set_y(y)


def _retitle(ax, old: str, new: str) -> None:
    """[2x3] Shorten a panel title that no longer fits a third-width column.

    The title is a Text artist placed by the source script's panel_title(); only
    the string changes, position/size/weight/colour are untouched.
    """
    for t in ax.texts:
        if t.get_text() == old:
            t.set_text(new)
            return
    raise AssertionError(f"panel title {old!r} not found — did the source script change?")


def _compact_coordination_ticks(fig, before) -> None:
    """[2x3] Panel c: "50 (random)" x-labels are too wide for a narrow column.

    The labels become "0 / 50 / 100" and the random reference is named once, next
    to the dashed line it annotates -- the same idiom panel d already uses for its
    own random reference.  Same line, same value, same information.
    """
    new_axes = [ax for ax in fig.axes if ax not in before]
    assert len(new_axes) == 3, f"plot_coordination made {len(new_axes)} axes, expected 3"
    _outer, ax_rank, ax_conf = new_axes          # order fixed by the source script
    for sub in (ax_rank, ax_conf):
        sub.set_xticklabels(["0", "50", "100"])
    ax_rank.text(0.5, -0.145, "random", transform=ax_rank.get_xaxis_transform(),
                 ha="center", va="top", fontsize=5.6, color=S.GREY, clip_on=False)


def _rewrap_coordination_legend(fig, before) -> None:
    """[2x3] Re-flow panel c's 5-item segment legend from one row to two.

    `plot_coordination` places it as ncol=5 at 5.8pt, which needs ~2.6in -- more
    than a third-width column has.  Same handles, same labels, same font size;
    only the wrap and the vertical offset change.  Nothing is dropped.
    """
    new_axes = [ax for ax in fig.axes if ax not in before]
    if not new_axes:
        return
    outer = new_axes[0]
    leg = outer.get_legend()
    if leg is None:
        return
    handles = list(leg.legend_handles)
    labels = [t.get_text() for t in leg.get_texts()]
    leg.remove()
    outer.legend(handles=handles, labels=labels, loc="lower center",
                 bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False,
                 fontsize=5.8, handlelength=0.72, handletextpad=0.25,
                 columnspacing=0.48, labelspacing=0.22, borderaxespad=0.0)


def make_merged() -> None:
    S.apply()
    tables = build_all()
    agent_games = F3.build_agent_games()
    lambda_tbl = pd.read_csv(TAB_DIR / "f3_lambda.csv")
    fingerprint_tbl = pd.read_csv(TAB_DIR / "f3_fingerprint.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55,
                             left=0.072, right=0.985, top=0.855, bottom=0.085)
    top = outer[0, 0].subgridspec(1, 3, width_ratios=ROW1_RATIOS, wspace=WSPACE)
    bot = outer[1, 0].subgridspec(1, 3, width_ratios=ROW2_RATIOS, wspace=WSPACE)

    # ── row 1: a, b, c ───────────────────────────────────────────────────────
    ax_a = fig.add_subplot(top[0, 0])
    F1.plot_dominance(ax_a, tables["dominance"])
    _retitle(ax_a, "Unique-equilibrium conformity", "Equilibrium conformity")
    _oneline_ylabel(ax_a)
    # NB: a's "DD\nn=36" x-ticks stay on two lines -- "DD (n=36)" needs 0.53in per
    # tick against 0.52in of pitch, so the one-line form collides.
    ax_b = fig.add_subplot(top[0, 1])
    F1.plot_payoff(ax_b, tables["payoff"])
    ax_b.tick_params(axis="x", labelsize=S.NHB_FS_TICK_SMALL)
    # [2x3] One line instead of two.  The full "Payoff efficiency (0 worst, 1
    # best)" measures 1.87in against 1.68in of panel height; dropping "Payoff" --
    # already the panel title immediately above -- gives 1.49in.  The scale
    # definition, which is the informative part, is kept.  Mirrors panel a, whose
    # label is likewise "Conformity (...)" under the title "Equilibrium conformity".
    ax_b.set_ylabel("Efficiency (0 worst, 1 best)", fontsize=S.NHB_FS_AXIS)
    # [2x3] the equilibrium-reference caption is left-anchored at x=0.58 in the
    # source, which overhangs a third-width column; right-anchor it instead.
    for t in ax_b.texts:
        if t.get_text() == "rank-scale equilibrium":
            t.set_position((1.0, 0.90))
            t.set_ha("right")
        elif t.get_text() == "MP descriptive":
            # [2x3] left-anchored at the MP tick in the source; it now runs under
            # panel c's agent labels, so right-anchor it inside b's own axes.
            t.set_position((5.45, 0.05))
            t.set_ha("right")
    before = list(fig.axes)
    F1.plot_coordination(fig, top[0, 2], tables["coordination"])
    _compact_coordination_ticks(fig, before)
    _rewrap_coordination_legend(fig, before)
    _align_top_row_titles([(ax_a, {"(a)", "Equilibrium conformity"}),
                           (ax_b, {"(b)", "Payoff efficiency"})])

    # ── row 2: d, e, f ───────────────────────────────────────────────────────
    ax_d = fig.add_subplot(bot[0, 0])
    F1.plot_complexity(ax_d, tables["panel"])
    ax_e = fig.add_subplot(bot[0, 1])
    plot_incentive_response(ax_e, agent_games, lambda_tbl)
    _finish_ef(ax_e)
    ax_f = fig.add_subplot(bot[0, 2])
    plot_fingerprint(ax_f, agent_games, fingerprint_tbl)
    _finish_ef(ax_f)

    F1.add_shared_legend(fig)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"{STEM}.{ext}",
                        dpi=600 if ext == "png" else 300, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {OUT_DIR / (STEM + '.pdf')}  ({FIG_W}in x {FIG_H}in)")
    print(f"wrote {OUT_DIR / (STEM + '.png')}")


def main() -> int:
    make_merged()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
