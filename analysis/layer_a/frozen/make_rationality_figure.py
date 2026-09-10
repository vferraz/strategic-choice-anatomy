#!/usr/bin/env python3
"""Strategic-rationality-by-game-type figure.

R = 1 - d/d_chance, d = Euclidean distance from a play pair (P(P1 act0),P(P2 act0))
to the class equilibrium (point-to-set for CO), d_chance = distance of random play
(0.5,0.5) to it.  R: 1 = equilibrium, 0 = random, <0 = plays the equilibrium action
less than a coin (systematic bias).

Each bar carries three things:
  * solid bar   : R from the REALIZED choice frequencies (the analysis object)
  * faded bar   : R from the SOFTMAX preference pair (pref0; LLM only)
  * in-bar label: expected ordinal payoff pair (E[pi1]/E[pi2], ranks 1-4) under
                  independent play, on the shared canonical ordinal matrix

Benchmarks: DD/OD1/OD2 -> unique pure NE; CO1/CO2 -> the two pure NE (set);
MP -> mixed NE.  MP games whose mixed NE == (0.5,0.5) are excluded.
Unit of analysis = game; CIs = bootstrap over games (Griffiths variants collapsed
to per-canonical means first).
"""
# Ported from analysis/block_a/ (private repo, commit 8d8370e). Release plan §5 excludes
# "analysis/block_a figure scripts (superseded by layer_a figscripts)", but these three are
# NOT superseded: analysis/layer_a/src/fig1_rationality.py and src/fig2_trait_steering.py
# import them and re-point only the data source and output dir, reusing the frozen plotting
# body unchanged. Flagged in PROGRESS.md as a plan gap.
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from strategic_anatomy.config import human_refs_root, repo_root
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

ROOT = repo_root()
DATA = human_refs_root() / "unified_pairs.parquet"
OUT = ROOT / "analysis" / "layer_a" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260520)

# canonical ordinal matrices (shared rank scale for the payoff annotation)
from strategic_anatomy import games as _g
ORD = {k: [int(x) for x in v[0]] for k, v in _g.bruns_games.items()}

CLASSES = ["DD", "OD1", "OD2", "CO1", "CO2", "MP"]
CO = {"CO1", "CO2"}
AGENTS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss", "nagel", "griffiths"]
DISPLAY = {"qwen": "Qwen", "qwen_instruct": "Qwen-Instruct",
           "llama31_instruct": "Llama-3.1-Instruct", "gptoss": "GPT-OSS",
           "nagel": "Nagel (human)", "griffiths": "Griffiths (human)"}
COLOR = {"qwen": "#1f77b4", "qwen_instruct": "#17becf",
         "llama31_instruct": "#2ca02c", "gptoss": "#9467bd",
         "nagel": "#d62728", "griffiths": "#ff7f0e"}
IS_HUMAN = {"nagel": True, "griffiths": True}


def dist(ax, ay, bx, by):
    return float(np.hypot(ax - bx, ay - by))


def Rval(ox, oy, bench):
    if pd.isna(ox) or pd.isna(oy):
        return np.nan
    d_obs = min(dist(ox, oy, a, b) for a, b in bench)
    d_ch = min(dist(0.5, 0.5, a, b) for a, b in bench)
    return np.nan if d_ch < 1e-9 else 1 - d_obs / d_ch


def exp_payoff(p1, p2, vec):
    P = {(0, 0): p1 * p2, (0, 1): p1 * (1 - p2),
         (1, 0): (1 - p1) * p2, (1, 1): (1 - p1) * (1 - p2)}
    e1 = sum(P[(i, j)] * vec[i * 2 + j] for i in (0, 1) for j in (0, 1))
    e2 = sum(P[(i, j)] * vec[4 + i * 2 + j] for i in (0, 1) for j in (0, 1))
    return e1, e2


def load_rows() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df = df.loc[:, ~df.columns.duplicated()]
    llm = df[(df.agent_kind == "llm") & (df.condition == "baseline") & (df.in_model_set)]
    hum = df[(df.agent_kind == "human") & (df.in_model_set)]
    d = pd.concat([llm, hum], ignore_index=True)
    recs = []
    for r in d.itertuples():
        lk = r.nagel_lk_type
        bench = ([(r.ne1_p1, r.ne1_p2), (r.ne2_p1, r.ne2_p2)] if lk in CO
                 else [(r.ne1_p1, r.ne1_p2)])
        bench = [(a, b) for a, b in bench if pd.notna(a) and pd.notna(b)]
        if not bench:
            continue
        if min(dist(0.5, 0.5, a, b) for a, b in bench) < 1e-9:   # eq == random: undegradable
            continue
        vec = ORD.get(r.bruns_name)
        e1, e2 = exp_payoff(r.realized_p1, r.realized_p2, vec) if vec else (np.nan, np.nan)
        # equilibrium and anti-equilibrium (centre-reflected) payoffs — game-level reference
        if vec:
            eqpay = [exp_payoff(a, b, vec) for a, b in bench]            # CO: two pure NE
            eq1, eq2 = np.mean([p[0] for p in eqpay]), np.mean([p[1] for p in eqpay])
            # worst = least-rational pure profile = corner with minimum R
            # (opposite corner for DD/OD; off-diagonal miscoordination for CO)
            corners = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
            wc = min(corners, key=lambda c: Rval(c[0], c[1], bench))
            an1, an2 = exp_payoff(wc[0], wc[1], vec)
            rd1, rd2 = exp_payoff(0.5, 0.5, vec)                      # random-play payoff
            csum = [vec[i * 2 + j] + vec[4 + i * 2 + j] for i in (0, 1) for j in (0, 1)]
            lo_sum, hi_sum = float(min(csum)), float(max(csum))       # achievable total range
        else:
            eq1 = eq2 = an1 = an2 = rd1 = rd2 = lo_sum = hi_sum = np.nan
        rec = dict(sample=r.sample, cid=r.canonical_id, lk=lk,
                   R=Rval(r.realized_p1, r.realized_p2, bench),
                   R_soft=Rval(r.pref0_p1, r.pref0_p2, bench), e1=e1, e2=e2,
                   eq1=eq1, eq2=eq2, an1=an1, an2=an2,
                   rd1=rd1, rd2=rd2, lo_sum=lo_sum, hi_sum=hi_sum)
        if lk in CO:
            for k, (a, b) in enumerate(bench[:2], 1):
                rec[f"R{k}"] = Rval(r.realized_p1, r.realized_p2, [(a, b)])
        recs.append(rec)
    return pd.DataFrame(recs)


def boot_ci(vals, n=2000):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    means = vals[RNG.integers(0, len(vals), size=(n, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def agg_R(R, col):
    perg = R.dropna(subset=[col]).groupby(["sample", "lk", "cid"])[col].mean().reset_index()
    return {(s, lk): (*boot_ci(perg[(perg["sample"] == s) & (perg.lk == lk)][col].values), 0)
            for s in AGENTS for lk in CLASSES}


def agg_pay(R):
    perg = R.dropna(subset=["R"]).groupby(["sample", "lk", "cid"])[["e1", "e2"]].mean().reset_index()
    out = {}
    for s in AGENTS:
        for lk in CLASSES:
            sub = perg[(perg["sample"] == s) & (perg.lk == lk)]
            out[(s, lk)] = (sub.e1.mean(), sub.e2.mean()) if len(sub) else (np.nan, np.nan)
    return out


def agg_ref(R):
    # equilibrium and anti-equilibrium payoffs are game-level: collapse to unique games, mean per class
    g = R.dropna(subset=["R", "eq1"]).groupby(["lk", "cid"])[["eq1", "eq2", "an1", "an2"]].mean().reset_index()
    return {lk: tuple(g[g.lk == lk][["eq1", "eq2", "an1", "an2"]].mean()) for lk in CLASSES}


def main():
    R = load_rows()
    realA, softA = agg_R(R, "R"), agg_R(R, "R_soft")
    payA = agg_pay(R)
    refA = agg_ref(R)
    r1A, r2A = agg_R(R, "R1"), agg_R(R, "R2")

    # table out
    rows = []
    for s in AGENTS:
        for lk in CLASSES:
            m, lo, hi, _ = realA[(s, lk)]
            sm, slo, shi, _ = softA[(s, lk)]
            e1, e2 = payA[(s, lk)]
            rows.append(dict(agent=DISPLAY[s], cls=lk, R_realized=m, R_lo=lo, R_hi=hi,
                             R_softmax=sm, Epi1=e1, Epi2=e2))
    pd.DataFrame(rows).to_csv(OUT / "rationality_by_class.csv", index=False)

    # ---- figure ----
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.05], hspace=0.6,
                          top=0.79, bottom=0.085, left=0.07, right=0.985)
    ax = fig.add_subplot(gs[0])

    xpos = {"DD": 0, "OD1": 1, "OD2": 2, "CO1": 3.6, "CO2": 4.6, "MP": 6.0}
    bw = 0.13
    offs = np.array([-2.6, -1.6, -0.6, 0.4, 1.7, 2.7]) * bw
    YTOP = 1.62

    for x0, x1, c, lab in [(-0.5, 2.5, "#e8eef6", "DOMINANCE-SOLVABLE  (unique pure NE)"),
                           (3.1, 5.1, "#e6f4e6", "COORDINATION  (two pure NE)"),
                           (5.5, 6.5, "#fceadb", "NO PURE NE  (mixed NE)")]:
        ax.axvspan(x0, x1, color=c, zorder=0)
        ax.text((x0 + x1) / 2, 1.46, lab, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#3a3a3a")

    ax.axhline(1, color="#222", lw=1.5, zorder=1)
    ax.axhline(0, color="#888", lw=1.2, ls="--", zorder=1)
    wstroke = [pe.withStroke(linewidth=2, foreground="white")]
    ax.text(-0.62, 1.0, "on a Nash eq.", ha="left", va="center", fontsize=7.3, color="#222", path_effects=wstroke)
    ax.text(-0.62, 0.0, "random", ha="left", va="center", fontsize=7.5, color="#666", path_effects=wstroke)

    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    for j, s in enumerate(AGENTS):
        for lk in CLASSES:
            m, lo, hi, _ = realA[(s, lk)]
            if np.isnan(m):
                continue
            x = xpos[lk] + offs[j]
            ax.bar(x, m, width=bw, color=COLOR[s], alpha=0.9,
                   hatch="///" if IS_HUMAN.get(s) else None,
                   edgecolor="#333" if IS_HUMAN.get(s) else "white", lw=0.6, zorder=3,
                   label=DISPLAY[s] if lk == "DD" else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none",
                        ecolor="#222", elinewidth=0.8, capsize=2, zorder=5)
            # softmax overlay (LLM only)
            sm = softA[(s, lk)][0]
            if not np.isnan(sm):
                ax.bar(x, sm, width=bw, color=COLOR[s], alpha=0.28,
                       edgecolor=COLOR[s], ls=":", lw=1.0, zorder=4)
            # this agent's realized expected payoff pair, inside the bar
            e1, e2 = payA[(s, lk)]
            if not np.isnan(e1):
                yl = m * 0.5 if abs(m) > 0.34 else (m + 0.09 * np.sign(m or 1))
                ax.text(x, yl, f"{e1:.1f}/{e2:.1f}", rotation=90, ha="center",
                        va="center", fontsize=5.4, color="#111", path_effects=halo, zorder=6)

    # per-class equilibrium payoff, on the equilibrium (R=1) line
    for lk in CLASSES:
        eq1, eq2, _, _ = refA[lk]
        ax.text(xpos[lk], 1.06, f"{eq1:.1f}/{eq2:.1f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color="#0a5f3a", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="#eafaf0", ec="#0a5f3a", lw=0.6))

    ax.set_xticks([xpos[c] for c in CLASSES])
    depth = {"DD": "DD\ndominant", "OD1": "OD1\nL1", "OD2": "OD2\nL2",
             "CO1": "CO1\nselects", "CO2": "CO2\ncycles", "MP": "MP\nmix"}
    ax.set_xticklabels([depth[c] for c in CLASSES], fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Proximity to Nash equilibrium   (1 = on an equilibrium,  0 = random)", fontsize=12)
    ax.set_ylim(min(-1.15, R.R.min() - 0.05), YTOP)
    ax.set_xlim(-0.65, 6.7)

    fig.suptitle("Nash-equilibrium conformity of LLMs vs. humans, by game type",
                 fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.5, 0.918,
             "Distance of each agent's one-shot play to the NEAREST Nash equilibrium, normalized: 1 = on an equilibrium, 0 = no closer than random play, below 0 = farther from every equilibrium than random.\n"
             "For coordination games the nearer of the two equilibria is used — being far from one can mean being on the other.   "
             "Solid bar = realized choices · faded bar = softmax preference (LLMs) · number inside bar = that agent's expected payoff E[π1]/E[π2] · green box on the equilibrium line = payoff of equilibrium play (ranks 1–4) · whiskers = 95% CI.",
             ha="center", va="top", fontsize=9.6, color="#555")

    h, lab = ax.get_legend_handles_labels()
    h.append(Patch(facecolor="#777", alpha=0.28, edgecolor="#777", ls=":", label="softmax preference (overlay)"))
    lab.append("softmax preference (overlay)")
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.882), ncol=7,
               fontsize=9, frameon=False, handlelength=1.3, columnspacing=1.3,
               title="Models (solid)         Human reference (hatched)", title_fontsize=9.5)

    # ---- companion: CO report-both ----
    ax2 = fig.add_subplot(gs[1])
    co_x = {"CO1": 0.0, "CO2": 1.6}
    ax2.axhline(0, color="#888", lw=1.0, ls="--")
    ax2.axhline(1, color="#222", lw=1.0)
    for j, s in enumerate(AGENTS):
        for lk in CO:
            base = co_x[lk] + offs[j] * 1.6
            for marker, agg in (("o", r1A), ("^", r2A)):
                m = agg[(s, lk)][0]
                if not np.isnan(m):
                    ax2.plot(base, m, marker=marker, color=COLOR[s], ms=7,
                             markeredgecolor="#333", mew=0.5, zorder=3)
    ax2.set_xticks(list(co_x.values())); ax2.set_xticklabels(["CO1", "CO2"], fontsize=10.5)
    ax2.tick_params(axis="x", length=0)
    ax2.set_ylabel("proximity to each\npure equilibrium", fontsize=9.5)
    ax2.set_xlim(-0.7, 2.3); ax2.set_ylim(-1.0, 1.25)
    ax2.set_title("Coordination detail — proximity to each pure equilibrium separately "
                  "(● equilibrium 1, ▲ equilibrium 2):  near one ⇒ on that equilibrium,  near neither ⇒ off both",
                  fontsize=9.5, loc="left", color="#333", pad=6)

    fig.text(0.07, 0.016,
             "Data: one-shot baseline play, 76 shared games (LLM = round 1 over 3 seeds; Nagel n≈450/game; Griffiths on its cardinal payoffs).   "
             "CO = consistency with either pure NE (mixed excluded).   MP excludes 4 games whose mixed NE is uniform play.   Griffiths has no MP games.   "
             "Expected payoffs use independent play on the shared canonical ordinal matrix; for CO the equilibrium box averages the two pure NE.   n games/class: DD 20, OD1 18, OD2 18, CO1 6, CO2 5, MP 5.",
             fontsize=7.6, color="#666")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_rationality_by_class.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", OUT / "fig_rationality_by_class.png")
    print(pd.DataFrame(rows).pivot(index="agent", columns="cls", values="R_realized").round(2).to_string())


if __name__ == "__main__":
    main()
