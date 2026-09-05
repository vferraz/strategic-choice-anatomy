#!/usr/bin/env python3
"""Figure 1 — strategic behaviour & rationality (one-shot, paper-final).

FROZEN DESIGN. Panels (a) proximity-to-Nash by level-k class and (b) realised vs
equilibrium vs achievable payoff are produced by the committed
``analysis/block_a/fig_main_rationality_v2.py`` body with **only two globals
re-pointed**: the data source (-> the one-shot Fig-1 panel built by shared_data) and
the output directory. No layout/style/colour line is touched, so (a)/(b) render
pixel-identical to the committed ``fig_rationality_by_class_v2clean_*`` except for the
data. Verify with: diff the rendered PNG bytes' layout against the committed v2 PNG.

New content (did not exist in v2), styled from ``_paper_style`` so it sits beside the
frozen panels:
  (c) model-vs-Nagel per-game canonical-conformity scatter (Pearson+Spearman,
      game-clustered bootstrap CI), and
  (d) cross-model + human per-game P(canonical) alignment matrix.

Tables: f1_rationality_by_class.csv, f1_payoff_gap.csv, f1_human_corr.csv,
f1_alignment_matrix.csv.

Citations (captions/Methods, never just the nickname): Nagel = Moore, Germano &
Nagel (2026); Griffiths = Zhu, Peterson, Enke & Griffiths (2025).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


DATA = SD.DATA
OUTDIR = ROOT / "analysis" / "layer_a" / "figures"
TABLES = results_root() / "layer_a"
FIG1_PANEL = DATA / "fig1_panel.parquet"
RNG = np.random.default_rng(SD.RNG_SEED)


# ---- (a)/(b): the FROZEN v2 figure, data-source + output re-pointed ----------
def render_frozen_ab():
    import analysis.layer_a.frozen.make_rationality_figure as B
    import analysis.layer_a.frozen.fig_main_rationality_v2 as V2
    # === THE ONLY CHANGES: data source and output dir ===
    B.DATA = FIG1_PANEL
    V2.OUT = OUTDIR
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # build() is the frozen body, unchanged
    df_raw, realA, randref, ng = V2.build("raw")
    V2.build("max")
    print("[fig1 a/b] wrote fig_rationality_by_class_v2clean_{raw,max}.{png,pdf}")
    print("[fig1 a/b] N games per class:", {k: ng.get(k, 0) for k in V2.CLASSES},
          "total =", sum(ng.values()))
    return V2, B


def tables_from_frozen(V2):
    """Per (agent, class) realised/soft proximity (+CI) and payoff, both norms."""
    rows = []
    for norm in ("raw", "max"):
        df = V2.load_all()
        df["R"] = V2.proximity("d_obs", df, norm)
        df["Rs"] = V2.proximity("d_soft", df, norm)
        realA, softA = V2.agg(df, "R"), V2.agg(df, "Rs")
        for s in V2.AGENTS:
            for lk in V2.CLASSES:
                m, lo, hi = realA[(s, lk)]
                sm, slo, shi = softA[(s, lk)]
                rows.append(dict(norm=norm, agent=V2.DISPLAY.get(s, s), sample=s, cls=lk,
                                 R_realized=m, R_lo=lo, R_hi=hi, R_soft=sm))
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "f1_rationality_by_class.csv", index=False)
    print(f"[fig1] wrote {TABLES/'f1_rationality_by_class.csv'}")

    # payoff gap table (welfare): realised vs equilibrium vs achievable, per agent×class
    df = V2.load_all()
    df["R"] = V2.proximity("d_obs", df, "raw")
    w = df.dropna(subset=["R", "lo_sum"]).copy()
    w["eq_sum"], w["rd_sum"], w["re_sum"] = w.eq1 + w.eq2, w.rd1 + w.rd2, w.e1 + w.e2
    gl = w.groupby(["lk", "cid"])[["lo_sum", "hi_sum", "eq_sum", "rd_sum"]].mean().reset_index()
    refs = {lk: gl[gl.lk == lk][["lo_sum", "hi_sum", "eq_sum", "rd_sum"]].mean() for lk in V2.CLASSES}
    pa = w.groupby(["sample", "lk", "cid"])["re_sum"].mean().reset_index()
    prows = []
    for s in V2.AGENTS:
        for lk in V2.CLASSES:
            v = pa[(pa["sample"] == s) & (pa.lk == lk)].re_sum
            r = refs[lk]
            prows.append(dict(agent=V2.DISPLAY.get(s, s), sample=s, cls=lk,
                              realized_total=float(v.mean()) if len(v) else np.nan,
                              eq_total=float(r["eq_sum"]), random_total=float(r["rd_sum"]),
                              worst_total=float(r["lo_sum"]), best_total=float(r["hi_sum"])))
    pd.DataFrame(prows).to_csv(TABLES / "f1_payoff_gap.csv", index=False)
    print(f"[fig1] wrote {TABLES/'f1_payoff_gap.csv'}")


# ---- (c)/(d): new human-alignment content -----------------------------------
def _boot_corr(x, y, kind="pearson", n=SD.BOOT_N):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan, np.nan, np.nan
    from scipy.stats import pearsonr, spearmanr
    r = pearsonr(x, y)[0] if kind == "pearson" else spearmanr(x, y)[0]
    bs = []
    for _ in range(n):
        idx = RNG.integers(0, len(x), len(x))  # cluster = game (one row per game here)
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-9 or np.std(yb) < 1e-9:
            continue
        bs.append((pearsonr(xb, yb)[0] if kind == "pearson" else spearmanr(xb, yb)[0]))
    return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def human_alignment(gl, m):
    """P(canonical) per game: model (hard p_canon, soft p_canon_soft) vs Nagel pcanon."""
    hum = m[["game_code", "nagel_pcanon"]].rename(columns={"nagel_pcanon": "human"})
    rows, wide = [], hum.set_index("game_code")[["human"]].copy()
    for mod in SD.MODELS:
        sub = gl[gl.model == mod][["game_code", "p_canon", "p_canon_soft"]]
        d = sub.merge(hum, on="game_code", how="inner")
        wide[mod] = sub.set_index("game_code")["p_canon"]
        for kind in ("pearson", "spearman"):
            for col, lab in (("p_canon", "hard"), ("p_canon_soft", "soft")):
                r, lo, hi = _boot_corr(d[col], d["human"], kind)
                rows.append(dict(model=mod, readout=lab, corr=kind, r=r, lo=lo, hi=hi,
                                 n_games=int(d[[col, "human"]].dropna().shape[0])))
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "f1_human_corr.csv", index=False)
    print(f"[fig1] wrote {TABLES/'f1_human_corr.csv'}")
    print("[fig1 c] model-vs-Nagel P(canonical) Pearson r (hard readout):")
    hp = pd.DataFrame(rows).query("readout=='hard' and corr=='pearson'")[["model", "r", "lo", "hi", "n_games"]]
    print(hp.round(3).to_string(index=False))

    # (d) alignment matrix among models + human
    wide = wide.rename(columns={"human": "Nagel"})
    cols = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss", "Nagel"]
    cm = wide[cols].corr(method="pearson")
    cm.to_csv(TABLES / "f1_alignment_matrix.csv")
    print(f"[fig1] wrote {TABLES/'f1_alignment_matrix.csv'}")
    return pd.DataFrame(rows), cm, wide


def render_cd(corr_df, cm, wide, gl, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from strategic_anatomy import paper_style as S
    S.apply()

    hum = m[["game_code", "nagel_pcanon"]].rename(columns={"nagel_pcanon": "human"})

    # ---- panel (c): per-game scatter, small multiples ----
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharex=True, sharey=True)
    for ax, mod in zip(axes, SD.MODELS):
        d = gl[gl.model == mod][["game_code", "p_canon"]].merge(hum, on="game_code", how="inner")
        ax.scatter(d["human"], d["p_canon"], s=8, alpha=0.55, color=SD.COL[mod], edgecolor="none")
        ax.plot([0, 1], [0, 1], color=S.GREY, lw=0.8, ls="--", zorder=0)
        r = corr_df.query("model==@mod and readout=='hard' and corr=='pearson'")["r"].iloc[0]
        ax.text(0.04, 0.93, f"{SD.SHORT[mod]}\nr={r:.2f}", transform=ax.transAxes,
                fontsize=S.FS_TICK, va="top", ha="left", color=S.INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=S.FS_TICK)
    axes[0].set_ylabel("model P(canonical)", fontsize=S.FS_AXIS)
    fig.text(0.5, 0.0, "Nagel human P(canonical)  —  Moore, Germano & Nagel (2026)",
             ha="center", fontsize=S.FS_AXIS)
    S.panel_title(axes[0], "c", "Model vs human, per game", fontsize=S.FS_PANEL)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig1c_human_scatter.{ext}", dpi=600 if ext == "png" else 300,
                    bbox_inches="tight")
    plt.close(fig)

    # ---- panel (d): alignment matrix ----
    labels = ["Qwen-I", "Qwen-B", "Llama", "GPT-OSS", "Nagel"]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(cm.values, vmin=-0.2, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=S.FS_TICK)
    ax.set_yticklabels(labels, fontsize=S.FS_TICK)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=S.FS_FOOT, color=S.INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=S.FS_FOOT)
    S.panel_title(ax, "d", "Per-game P(canonical) alignment", fontsize=S.FS_PANEL)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig1d_alignment.{ext}", dpi=600 if ext == "png" else 300,
                    bbox_inches="tight")
    plt.close(fig)
    print("[fig1 c/d] wrote fig1c_human_scatter + fig1d_alignment")


def main() -> int:
    m = SD.master()
    gl = pd.read_parquet(DATA / "layerA_game_level.parquet")
    V2, B = render_frozen_ab()
    tables_from_frozen(V2)
    corr_df, cm, wide = human_alignment(gl, m)
    render_cd(corr_df, cm, wide, gl, m)
    print("\n[fig1 d] alignment matrix (Pearson):")
    print(cm.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
