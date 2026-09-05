#!/usr/bin/env python3
"""Main paper figure for prompt-trait steering (standalone; overwrites nothing).

Two panels carrying the whole logic:

  A) "Moves a lot, aimed little."  Per model, each dispositional trait is a point at
        x = placebo-corrected fraction of movement aimed at the trait's target
            (= net steering / magnitude, in [-1, 1]); filled if the net-steering CI
            excludes 0,
        y = magnitude of movement (mean |Δ freq of act0|).
     The maximin rule is shown as a control (same target as risk/loss, different
     framing); the content-free length-null prompt anchors x=0 (no target -> no aim).

  B) "...and only where incentives are weak."  Per model, magnitude and net steering
     (pooled over the four dispositional cues) vs how clearly the payoffs favor one
     action (Flat / Moderate / Clear).  Both collapse as the incentive sharpens - the
     quantal-response signature.

Reads the audited outputs of trait_steering_proper.py.  Round 1.
Outputs: analysis/layer_a/figures/fig_main_trait_steering_v2clean.{png,pdf}
"""
# Ported from analysis/block_a/ (private repo, commit 1f47050). Release plan §5 excludes
# "analysis/block_a figure scripts (superseded by layer_a figscripts)", but these three are
# NOT superseded: analysis/layer_a/src/fig1_rationality.py and src/fig2_trait_steering.py
# import them and re-point only the data source and output dir, reusing the frozen plotting
# body unchanged. Flagged in PROGRESS.md as a plan gap.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from strategic_anatomy.config import human_refs_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
adjust_text = lambda *a, **k: None
from strategic_anatomy import paper_style as S
S.apply()

ROOT = Path(".")
PER_GAME = human_refs_root() / "trait_steering_proper_per_game.csv"
SUMMARY = human_refs_root() / "trait_steering_proper_summary.csv"
UNIFIED = human_refs_root() / "unified_pairs.parquet"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"

RNG = np.random.default_rng(20260521)
N_BOOT = 4000

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
MODEL_LABEL = {"qwen": "Qwen2.5-72B", "qwen_instruct": "Qwen2.5-72B-Instruct",
               "llama31_instruct": "Llama-3.1-70B-Instruct", "gptoss": "GPT-OSS-120B"}
TRAITS = ["risk", "loss", "inequity", "selfish"]      # dispositional cues
CONTROL = "maximin"                                   # explicit-rule control
TRAIT_LABEL = {"risk": "Risk", "loss": "Loss", "inequity": "Inequity",
               "selfish": "Selfish", "maximin": "Maximin"}
TRAIT_COLOR = {"risk": S.REF_RED, "loss": "#e08a00", "inequity": "#1f6f8b",
               "selfish": "#7a4fa3", "maximin": S.GOOD_GREEN}
GREY = S.GREY


def _ci(vals):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    draws = vals[RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def incentive_gap():
    u = pd.read_parquet(UNIFIED, columns=["game_code", "matrix_8vec", "agent_kind", "condition"])
    g = u[(u.agent_kind == "llm") & (u.condition == "baseline")].drop_duplicates("game_code")
    rows = [{"game": r.game_code,
             "gap_EV": abs(np.array(json.loads(r.matrix_8vec), float)[:4].reshape(2, 2)[0].mean()
                           - np.array(json.loads(r.matrix_8vec), float)[:4].reshape(2, 2)[1].mean())}
            for _, r in g.iterrows()]
    return pd.DataFrame(rows)


def clarity_pooled(pg):
    """Pooled over the four dispositional cues: magnitude & net steering by clarity bin."""
    gap = incentive_gap()
    s = pg[(pg["round"] == 1) & pg.trait.isin(TRAITS)].merge(gap, on="game", how="left")
    labels = ["Flat\n(gap≈0)", "Moderate\n(gap≈1)", "Clear\n(gap≥2)"]
    s["clarity"] = pd.cut(s.gap_EV, [-0.1, 0.5, 1.5, 10], labels=labels)
    rows = []
    for m in MODELS:
        for k, lab in enumerate(labels):
            sub = s[(s.model == m) & (s.clarity == lab)]
            if sub.empty:
                continue
            mag, mlo, mhi = _ci(sub.abs_shift.to_numpy())
            net, nlo, nhi = _ci(sub.net_pref.to_numpy())
            rows.append({"model": m, "x": k, "clarity": lab, "n": len(sub),
                         "mag": mag, "mag_lo": mlo, "mag_hi": mhi,
                         "net": net, "net_lo": nlo, "net_hi": nhi})
    return pd.DataFrame(rows), labels


# --------------------------------------------------------------------------- #
def panel_A(ax, srow, plac_mag, model, show_y):
    ax.axvspan(0, 2, color="#edf6ed", zorder=0)
    ax.axvspan(-2, 0, color="#faf0ee", zorder=0)
    ax.axvline(0, color="#222", lw=0.9, zorder=2)
    texts, xs, ys = [], [], []
    # placebo anchor: a content-free prompt -> no target -> zero aim
    ax.scatter(0, plac_mag, s=46, marker="X", color=GREY, edgecolor="white",
               linewidth=0.6, alpha=0.85, zorder=4)
    lbl_bbox = dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.62)
    xs.append(0.0); ys.append(plac_mag)
    texts.append(ax.text(0, plac_mag, "placebo", fontsize=6.0, color=GREY, ha="center",
                         zorder=8, bbox=lbl_bbox))
    for t in TRAITS + [CONTROL]:
        r = srow.get(t)
        if r is None:
            continue
        mag = r["magnitude"]
        aim = np.clip(r["dir_net"] / mag, -1.1, 1.1) if mag > 1e-9 else 0.0
        sig = (r["dir_net_lo"] > 0) or (r["dir_net_hi"] < 0)
        col = TRAIT_COLOR[t]
        mk = "D" if t == CONTROL else "o"
        if r.get("base_pref_mean", 0) >= 0.8:   # baseline at ceiling -> aim unreliable
            ax.scatter(aim, mag, s=140, marker=mk, facecolor="none",
                       edgecolor="#c2c2c2", linewidth=1.8, zorder=5)
        ax.scatter(aim, mag, s=58 if t == CONTROL else 46, marker=mk,
                   facecolor=(col if sig else "white"), edgecolor=col,
                   linewidth=1.3 if t == CONTROL else 1.1, alpha=0.85, zorder=6)
        xs.append(aim); ys.append(mag)
        texts.append(ax.text(aim, mag, TRAIT_LABEL[t], fontsize=6.0, color=col,
                             fontweight="medium", zorder=8, bbox=lbl_bbox))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.02, 0.56)
    ax.set_xticks([-1, 0, 1])
    ax.tick_params(labelsize=S.FS_TICK)
    ax.set_title(MODEL_LABEL[model], fontsize=7.2, fontweight="normal", color="#333", pad=3)
    ax.set_xlabel("aimed fraction\n(away ←  0  → toward)", fontsize=6.8)
    if show_y:
        ax.set_ylabel("magnitude\nmean |Δ freq of act0|", fontsize=S.FS_AXIS)
    else:
        ax.set_yticklabels([])
    for sp in ax.spines.values():            # closed box, matching the rationality figure
        sp.set_visible(True); sp.set_linewidth(0.8)
    # dodge labels so they don't overlap each other or the markers
    adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.45, 1.9),
                force_text=(0.6, 0.9), force_static=(0.4, 0.7), force_pull=(0.005, 0.005),
                min_arrow_len=4,
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.5, shrinkA=2, shrinkB=5))


def panel_B(ax, cs, labels, model, show_y):
    d = cs[cs.model == model].sort_values("x")
    if d.empty:
        return
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.fill_between(d.x, d.mag_lo, d.mag_hi, color=GREY, alpha=0.18)
    ax.plot(d.x, d.mag, "-o", color=GREY, lw=1.8, ms=4.5, alpha=0.9, label="magnitude |Δ freq|")
    ax.fill_between(d.x, d.net_lo, d.net_hi, color=S.SOFT_BLUE, alpha=0.18)
    ax.plot(d.x, d.net, "-s", color=S.SOFT_BLUE, lw=1.8, ms=4.5, alpha=0.9, label="net steering toward trait")
    for _, r in d.iterrows():
        ax.annotate(f"n={r.n}", (r.x, r.mag), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=5.6, color="#666")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_xlim(-0.3, len(labels) - 0.7)
    ax.set_ylim(-0.36, 0.78)
    ax.tick_params(axis="y", labelsize=S.FS_TICK)
    ax.set_xlabel("payoff incentive clarity →", fontsize=6.8)
    if show_y:
        ax.set_ylabel("Δ choice frequency", fontsize=S.FS_AXIS)
    else:
        ax.set_yticklabels([])
    for sp in ax.spines.values():            # closed box, matching the rationality figure
        sp.set_visible(True); sp.set_linewidth(0.8)


def main():
    summ = pd.read_csv(SUMMARY)
    s1 = summ[summ["round"] == 1]
    pg = pd.read_csv(PER_GAME)
    cs, labels = clarity_pooled(pg)

    fig = plt.figure(figsize=(7.2, 7.05))
    gs = gridspec.GridSpec(3, len(MODELS), height_ratios=[1.0, 1.0, 0.34], hspace=0.88, wspace=0.14,
                           left=0.10, right=0.985, top=0.855, bottom=0.155)
    axA0 = axB0 = None
    for i, model in enumerate(MODELS):
        srow = {t: s1[(s1.model == model) & (s1.trait == t)].iloc[0].to_dict()
                for t in TRAITS + [CONTROL] if not s1[(s1.model == model) & (s1.trait == t)].empty}
        # length-null magnitude (trait-independent): take from risk row over all 76 games
        plac_mag = float(s1[(s1.model == model) & (s1.trait == "risk")].iloc[0]["magnitude_null"])
        axA = fig.add_subplot(gs[0, i]); panel_A(axA, srow, plac_mag, model, show_y=(i == 0))
        axB = fig.add_subplot(gs[1, i]); panel_B(axB, cs, labels, model, show_y=(i == 0))
        if i == 0:
            axA0, axB0 = axA, axB

    # ---- panel (c): held-out variance partition of canonical conformity (Layer-1 attribution) ----
    # Source: analysis/block_a/layer1_conformity_attribution.ipynb §3.2 — grouped permutation importance,
    # GroupKFold-by-game, out-of-fold, level-k scoped out. structure 36.8% / trait 7.6% / model 55.6%.
    axC = fig.add_subplot(gs[2, :])
    _seg = [("Game structure", 36.8, S.GOOD_GREEN, "white"),
            ("Trait prompt",   7.6,  S.SOFT_BLUE,  "white"),
            ("Model identity", 55.6, S.GREY,        "white")]
    _x = 0.0
    for _lab, _w, _col, _tc in _seg:
        axC.barh(0, _w, left=_x, height=1.05, color=_col, edgecolor="white", lw=1.8, zorder=3)
        if _w >= 12:                                  # label inside the wide segments
            axC.text(_x + _w / 2, 0, f"{_lab}   {_w:.0f}%", ha="center", va="center",
                     fontsize=S.FS_FOOT, color=_tc, fontweight="bold")
        else:                                         # narrow trait segment -> name above, % inside
            axC.text(_x + _w / 2, 0, f"{_w:.0f}%", ha="center", va="center",
                     fontsize=S.FS_FOOT, color=_tc, fontweight="bold")
            axC.text(_x + _w / 2, 0.78, _lab, ha="center", va="bottom",
                     fontsize=S.FS_FOOT, color="#262626", fontweight="bold")
            axC.plot([_x + _w / 2, _x + _w / 2], [0.54, 0.75], color="#9a9a9a", lw=0.7, zorder=2)
        _x += _w
    axC.set_xlim(0, 100); axC.set_ylim(-1.05, 1.15); axC.axis("off")
    axC.text(50, -0.85,
             r"held-out variance partition (GroupKFold-by-game)   $\cdot$   trait block $\Delta$AUC $+0.011$ [0.007, 0.015]",
             ha="center", va="center", fontsize=S.FS_FOOT, color=S.FAINT)

    # (a)/(b)/(c): bold left-aligned headers sitting tight on top of each row
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
        Line2D([0], [0], marker="X", color="w", markerfacecolor=GREY, markeredgecolor=GREY, ms=8,
               label="placebo"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="#c2c2c2",
               markeredgewidth=2.0, ms=11, label="ceiling (≥0.8)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=S.SOFT_BLUE, markeredgecolor=S.SOFT_BLUE, ms=6,
               label="(b) net steering"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markeredgecolor=GREY, ms=6,
               label="(b) magnitude"),
    ]
    leg = fig.legend(handles=handles_A, loc="lower center", ncol=4, frameon=False, fontsize=S.FS_LEGEND,
                     bbox_to_anchor=(0.5, 0.045), columnspacing=1.3, handletextpad=0.4,
                     title="(a) points = model × trait" + S.SEP + "(b) lines pooled over traits")
    leg.get_title().set_fontsize(S.FS_FOOT); leg.get_title().set_color(S.SUBTLE)
    for ext in ("png", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"fig_main_trait_steering_v2clean.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote analysis/layer_a/figures/fig_main_trait_steering_v2clean.{png,pdf}")
    print("\nPanel A coordinates (round 1):")
    for model in MODELS:
        for t in TRAITS + [CONTROL]:
            r = s1[(s1.model == model) & (s1.trait == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            aim = r.dir_net / r.magnitude if r.magnitude > 1e-9 else 0
            sig = (r.dir_net_lo > 0) or (r.dir_net_hi < 0)
            print(f"  {model:16s} {t:9s} mag={r.magnitude:.2f}  aim={aim:+.2f}  sig={sig}")


if __name__ == "__main__":
    main()
