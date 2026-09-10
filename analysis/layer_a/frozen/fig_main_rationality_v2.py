#!/usr/bin/env python3
"""FINAL rationality-by-class figure (harmonized visual standard).

Final generator for the two main rationality figures. Visually-harmonized copy of
make_rationality_figure_v3.py: identical analysis and layout, but it imports the
shared paper style (analysis/_shared/_paper_style.py) so it reads as one source with
fig_main_trait_steering.py — same fonts (editable Type-42 PDF text), shared
green/red/grey semantic colours, and the same type scale. Layout and data unchanged.

Variant 'raw'  : y = 1 − distance-to-nearest-NE (no chance normalization).
Variant 'max'  : y = 1 − distance/max-distance-to-NE.

Outputs fig_rationality_by_class_v3_raw_harmonized.{png,pdf} and _max_harmonized.{png,pdf}.
Does NOT touch make_rationality_figure_v3.py or v1/v2.
"""
# Ported from analysis/block_a/ (private repo, commit 8d8370e). Release plan §5 excludes
# "analysis/block_a figure scripts (superseded by layer_a figscripts)", but these three are
# NOT superseded: analysis/layer_a/src/fig1_rationality.py and src/fig2_trait_steering.py
# import them and re-point only the data source and output dir, reusing the frozen plotting
# body unchanged. Flagged in PROGRESS.md as a plan gap.
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import analysis.layer_a.frozen.make_rationality_figure as B
from strategic_anatomy import paper_style as S
S.apply()

CLASSES, CO, AGENTS = B.CLASSES, B.CO, B.AGENTS
DISPLAY, COLOR, IS_HUMAN, OUT, ORD = B.DISPLAY, B.COLOR, B.IS_HUMAN, B.OUT, B.ORD
DISPLAY = {**DISPLAY, "qwen": "Qwen2.5-72B", "qwen_instruct": "Qwen2.5-72B-Instruct",
           "llama31_instruct": "Llama-3.1-70B-Instruct", "gptoss": "GPT-OSS-120B"}  # correct model names
exp_payoff = B.exp_payoff
RNG = np.random.default_rng(20260520)
CORNERS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
SQRT2 = np.hypot(1, 1)


def d(ax, ay, bx, by):
    return float(np.hypot(ax - bx, ay - by))


def load_all() -> pd.DataFrame:
    df = pd.read_parquet(B.DATA)
    df = df.loc[:, ~df.columns.duplicated()]
    llm = df[(df.agent_kind == "llm") & (df.condition == "baseline") & (df.in_model_set)]
    hum = df[(df.agent_kind == "human") & (df.in_model_set)]
    data = pd.concat([llm, hum], ignore_index=True)
    rows = []
    for r in data.itertuples():
        lk = r.nagel_lk_type
        bench = ([(r.ne1_p1, r.ne1_p2), (r.ne2_p1, r.ne2_p2)] if lk in CO
                 else [(r.ne1_p1, r.ne1_p2)])
        bench = [(a, b) for a, b in bench if pd.notna(a) and pd.notna(b)]
        if not bench:
            continue
        d_obs = min(d(r.realized_p1, r.realized_p2, a, b) for a, b in bench)
        d_soft = (min(d(r.pref0_p1, r.pref0_p2, a, b) for a, b in bench)
                  if pd.notna(r.pref0_p1) else np.nan)
        d_ch = min(d(0.5, 0.5, a, b) for a, b in bench)
        d_max = max(min(d(cx, cy, a, b) for a, b in bench) for cx, cy in CORNERS)
        vec = ORD.get(r.bruns_name)
        e1, e2 = exp_payoff(r.realized_p1, r.realized_p2, vec) if vec else (np.nan, np.nan)
        if vec:
            eqp = [exp_payoff(a, b, vec) for a, b in bench]
            eq1, eq2 = np.mean([p[0] for p in eqp]), np.mean([p[1] for p in eqp])
            rd1, rd2 = exp_payoff(0.5, 0.5, vec)
            cs = [vec[i * 2 + j] + vec[4 + i * 2 + j] for i in (0, 1) for j in (0, 1)]
            lo_sum, hi_sum = float(min(cs)), float(max(cs))
        else:
            eq1 = eq2 = rd1 = rd2 = lo_sum = hi_sum = np.nan
        rec = dict(sample=r.sample, cid=r.canonical_id, lk=lk,
                   d_obs=d_obs, d_soft=d_soft, d_ch=d_ch, d_max=d_max,
                   e1=e1, e2=e2, eq1=eq1, eq2=eq2, rd1=rd1, rd2=rd2,
                   lo_sum=lo_sum, hi_sum=hi_sum)
        if lk in CO:
            rec["d_ne1"] = d(r.realized_p1, r.realized_p2, *bench[0])
            rec["d_ne2"] = d(r.realized_p1, r.realized_p2, *bench[1])
        rows.append(rec)
    return pd.DataFrame(rows)


def proximity(dist_col, df, norm):
    if norm == "raw":
        return 1 - df[dist_col]
    return 1 - df[dist_col] / df["d_max"]


def boot(vals, n=2000):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    m = vals[RNG.integers(0, len(vals), size=(n, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agg(df, col):
    g = df.dropna(subset=[col]).groupby(["sample", "lk", "cid"])[col].mean().reset_index()
    return {(s, lk): boot(g[(g["sample"] == s) & (g.lk == lk)][col].values)
            for s in AGENTS for lk in CLASSES}


def build(norm):
    df = load_all()
    df["R"] = proximity("d_obs", df, norm)
    df["Rs"] = proximity("d_soft", df, norm)
    df["Rrand"] = 1 - df["d_ch"] if norm == "raw" else 1 - df["d_ch"] / df["d_max"]
    realA, softA = agg(df, "R"), agg(df, "Rs")
    randref = df.dropna(subset=["R"]).groupby(["lk", "cid"])["Rrand"].mean().groupby("lk").mean().to_dict()
    ngames = df.dropna(subset=["R"]).groupby("lk")["cid"].nunique().to_dict()

    # CO per-equilibrium proximity (single corner NE -> d_max = sqrt2)
    if "d_ne1" in df.columns:
        df["Rne1"] = 1 - df["d_ne1"] if norm == "raw" else 1 - df["d_ne1"] / SQRT2
        df["Rne2"] = 1 - df["d_ne2"] if norm == "raw" else 1 - df["d_ne2"] / SQRT2
        r1A, r2A = agg(df, "Rne1"), agg(df, "Rne2")
    # welfare (norm-independent)
    w = df.dropna(subset=["R", "lo_sum"]).copy()
    w["eq_sum"], w["rd_sum"], w["re_sum"] = w.eq1 + w.eq2, w.rd1 + w.rd2, w.e1 + w.e2
    gl = w.groupby(["lk", "cid"])[["lo_sum", "hi_sum", "eq_sum", "rd_sum"]].mean().reset_index()
    wrefs = {lk: tuple(gl[gl.lk == lk][["lo_sum", "hi_sum", "eq_sum", "rd_sum"]].mean()) for lk in CLASSES}
    pa = w.groupby(["sample", "lk", "cid"])["re_sum"].mean().reset_index()
    wreal = {(s, lk): pa[(pa["sample"] == s) & (pa.lk == lk)].re_sum.mean() for s in AGENTS for lk in CLASSES}

    xpos = {"DD": 0, "OD1": 1, "OD2": 2, "CO1": 3.6, "CO2": 4.6, "MP": 6.0}
    bw = 0.13
    offs = np.array([-2.6, -1.6, -0.6, 0.4, 1.7, 2.7]) * bw
    # recenter each class's bars on its tick when an agent is absent (Griffiths has no MP games)
    coff = {}
    for lk in CLASSES:
        present = np.array([not np.isnan(realA[(s, lk)][0]) for s in AGENTS])
        coff[lk] = offs - (offs[present].mean() if present.any() else 0.0)
    bands = [(-0.5, 2.5, "#e8eef6", "DOMINANCE-SOLVABLE\n(unique pure NE)"),
             (3.1, 5.1, "#e6f4e6", "COORDINATION\n(two pure NE)"),
             (5.5, 6.5, "#fceadb", "NO PURE NE\n(mixed NE)")]
    depthword = {"DD": "dominant", "OD1": "L1", "OD2": "L2", "CO1": "selects", "CO2": "cycles", "MP": "mix"}
    star = {"CO1": "*", "CO2": "*"}   # nearest-NE caveat (see footnote)
    xlabels = [f"{c}{star.get(c, '')}\n{depthword[c]}\nN={ngames.get(c, 0)}" for c in CLASSES]
    ylab = ("Proximity to Nash equilibrium\n(1 = on equilibrium; raw distance)" if norm == "raw"
            else "Proximity to Nash equilibrium\n(1 = on equilibrium, 0 = farthest)")

    # ---------- figure: paper-width, two panels (fits a single text-column width) ----------
    fig = plt.figure(figsize=(7.2, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.5], hspace=0.42,
                          top=0.87, bottom=0.215, left=0.118, right=0.985)

    # ---- panel (a): proximity to the equilibrium ----
    ax = fig.add_subplot(gs[0])
    lo_vals = [realA[(s, lk)][1] for s in AGENTS for lk in CLASSES if not np.isnan(realA[(s, lk)][1])]
    lo_vals += [v for v in randref.values() if v is not None and not np.isnan(v)]
    ymin = min(-0.03, (min(lo_vals) - 0.05) if lo_vals else -0.03)   # no dead space below the data
    ytop = 1.26
    for x0, x1, c, lab in bands:
        ax.axvspan(x0, x1, color=c, zorder=0)
        ax.text((x0 + x1) / 2, 1.16, lab, ha="center", va="center", fontsize=6.6, fontweight="bold",
                color="#3a3a3a", linespacing=1.0)
    ax.axhline(1, color="#222", lw=1.3, zorder=1)
    ws = [pe.withStroke(linewidth=2, foreground="white")]
    ax.text(-0.60, 1.0, "on a Nash eq.", ha="left", va="center", fontsize=6.0, color="#222", path_effects=ws)
    # per-class RANDOM reference line
    for lk in CLASSES:
        rr = randref.get(lk, np.nan)
        if not np.isnan(rr):
            xc = xpos[lk]
            ax.plot([xc - 0.42, xc + 0.42], [rr, rr], color=S.REF_RED, lw=1.2, ls="--", zorder=2)
    for j, s in enumerate(AGENTS):
        for lk in CLASSES:
            m, lo, hi = realA[(s, lk)]
            if np.isnan(m):
                continue
            x = xpos[lk] + coff[lk][j]
            ax.bar(x, m, width=bw, color=COLOR[s], alpha=0.9, hatch="///" if IS_HUMAN.get(s) else None,
                   edgecolor="#333" if IS_HUMAN.get(s) else "white", lw=0.5, zorder=3,
                   label=DISPLAY[s] if lk == "DD" else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#222", elinewidth=0.7, capsize=1.5, zorder=5)
            sm = softA[(s, lk)][0]
            if not np.isnan(sm):
                ax.bar(x, sm, width=bw, color=COLOR[s], alpha=0.28, edgecolor=COLOR[s], ls=":", lw=0.9, zorder=4)
    ax.set_xticks([xpos[c] for c in CLASSES]); ax.tick_params(axis="x", length=0)
    ax.set_xticklabels(xlabels, fontsize=7.0); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel(ylab, fontsize=8.0); ax.set_ylim(ymin, ytop); ax.set_xlim(-0.65, 6.7)
    ax.set_title("(a)  Do they play the equilibrium?", fontsize=9.5, loc="left", fontweight="bold", color="#222", pad=6)

    # ---- panel (b): payoff ----
    ax2 = fig.add_subplot(gs[1]); ax2.set_xlim(-0.65, 6.7)  # noqa
    for x0, x1, c, _ in bands:
        ax2.axvspan(x0, x1, color=c, zorder=0)
    for lk in CLASSES:
        lo, hi, eqs, rds = wrefs[lk]; xc = xpos[lk]
        ax2.add_patch(plt.Rectangle((xc - 0.34, lo), 0.68, hi - lo, color="#cfd6dd", alpha=0.6, zorder=1))
        ax2.plot([xc - 0.34, xc + 0.34], [eqs, eqs], color=S.GOOD_GREEN, lw=1.8, zorder=2)
        ax2.plot([xc - 0.34, xc + 0.34], [rds, rds], color=S.GREY, lw=1.0, ls="--", zorder=2)
        ax2.text(xc + 0.37, hi, "best", fontsize=5.5, va="center", color="#555")
        ax2.text(xc + 0.37, lo, "worst", fontsize=5.5, va="center", color="#555")
        ax2.text(xc - 0.37, eqs, "eq", fontsize=5.5, va="center", ha="right", color=S.GOOD_GREEN, fontweight="bold")
        for j, s in enumerate(AGENTS):
            v = wreal[(s, lk)]
            if not np.isnan(v):
                ax2.plot(xc + coff[lk][j], v, "o", ms=4.0, color=COLOR[s], markeredgecolor="#333" if IS_HUMAN.get(s) else "white", mew=0.6, zorder=4)
    ax2.set_xticks([xpos[c] for c in CLASSES]); ax2.tick_params(axis="x", length=0)
    ax2.set_xticklabels(xlabels, fontsize=7.0); ax2.tick_params(axis="y", labelsize=7.5)
    ax2.set_ylabel("Total payoff  P1+P2\n(ranks summed, 2–8)", fontsize=8.0); ax2.set_ylim(1.6, 8.4)
    ax2.set_title("(b)  What does it pay?", fontsize=9.5, loc="left", fontweight="bold", color="#222", pad=3)

    # ---------- titles & legend ----------

    # legend — agents + softmax overlay + panel-a/b reference encodings
    h, lab = ax.get_legend_handles_labels()
    h += [Patch(facecolor="#777", alpha=0.28, edgecolor="#777", ls=":"),
          Line2D([0], [0], color=S.REF_RED, lw=1.2, ls="--"),
          Patch(facecolor="#cfd6dd", alpha=0.6),
          Line2D([0], [0], color=S.GOOD_GREEN, lw=1.8),
          Line2D([0], [0], color=S.GREY, lw=1.0, ls="--")]
    lab += ["softmax preference (overlay)",
            "random play  (panel a, per class)",
            "achievable payoff range  (panel b)",
            "equilibrium payoff  (panel b)",
            "random-play payoff  (panel b)"]
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, fontsize=6.8,
               frameon=False, handlelength=1.5, columnspacing=1.3, handletextpad=0.5,
               title="models = solid bars    ·    human reference = hatched bars",
               title_fontsize=7.6)
    for ext in ("png", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(OUT / f"fig_rationality_by_class_v2clean_{norm}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return df, realA, randref, ngames


def main():
    print("N games per class (ALL games, none dropped):")
    df, realA, randref, ng = build("raw")
    print(" ", {k: ng[k] for k in CLASSES}, " total =", sum(ng.values()))
    build("max")
    print("wrote v3_raw and v3_max")
    # interpretation check: per class, mean R across LLMs vs humans, both norms
    for norm in ("raw", "max"):
        d2 = load_all(); d2["R"] = proximity("d_obs", d2, norm)
        g = d2.dropna(subset=["R"]).groupby(["sample", "lk", "cid"]).R.mean().reset_index()
        g["kind"] = np.where(g["sample"].isin(["nagel", "griffiths"]), "human", "llm")
        piv = g.groupby(["kind", "lk"]).R.mean().unstack()[CLASSES].round(2)
        print(f"\n[{norm}] mean proximity, humans vs LLMs:")
        print(piv.to_string())


if __name__ == "__main__":
    main()
