#!/usr/bin/env python3
"""Figure 1 -- regime-matched strategic rationality on the one-shot substrate.

Main figure: four panels. The model-human scatter is used in Figure 3; panel (d)
shows the complexity response on the corrected generated/commit decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr


from strategic_anatomy import paper_style as S  # noqa: E402
from analysis.layer_a.build_regime_rationality import build_all  # noqa: E402
from strategic_anatomy.config import repo_root, results_root, taxonomy_dir

ROOT = repo_root()


S.apply()
mpl.rcParams.update({
    "font.size": 6.0,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "savefig.bbox": "standard",
})

LAYERA = ROOT / "analysis" / "layer_a"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
TAB_DIR = results_root() / "layer_a"
MASTER = taxonomy_dir() / "human_game_master_per_canonical.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

AGENTS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss", "nagel", "griffiths"]
MODELS = AGENTS[:4]
HUMANS = AGENTS[4:]
COMPLEXITY_AGENTS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss", "nagel"]
FAMILIES = ["DD", "OD1", "OD2", "CO1", "CO2", "MP"]
DOM_FAMILIES = ["DD", "OD1", "OD2"]
SEED = 20260520
N_BOOT = 2000

LABEL = {
    "qwen": "Qwen2.5",
    "qwen_instruct": "Qwen2.5-I",
    "llama31_instruct": "Llama",
    "gptoss": "GPT-OSS",
    "nagel": "Moore et al.",
    "griffiths": "Zhu et al.",
}
COLOR = {
    "qwen": "#1f77b4",
    "qwen_instruct": "#17becf",
    "llama31_instruct": "#2ca02c",
    "gptoss": "#9467bd",
    "nagel": "#c83b31",
    "griffiths": "#ff7f0e",
}
SEG = {"best": "#168f96", "other": "#d99a2b", "miscoord": "#b8b8b8"}
SEG_CONFLICT = {"coord": "#dbe4f0", "miscoord": "#6f86a7"}

FS_TITLE = 8.0
FS_SUBTITLE = 6.1
FS_PANEL = S.NHB_FS_PANEL
FS_AXIS = S.NHB_FS_AXIS
FS_TICK = S.NHB_FS_TICK
FS_LEGEND = S.NHB_FS_LEGEND
FS_FOOT = S.NHB_FS_FOOT


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _compact_decimal(x: float) -> str:
    if not _finite(x):
        return "NA"
    s = f"{float(x):.2f}"
    return s.replace("0.", ".").replace("-0.", "-.")


def _dodged_y_positions(targets: list[float], *, lo: float = 0.08,
                        hi: float = 0.96, min_gap: float = 0.045) -> list[float]:
    if not targets:
        return []
    order = np.argsort(targets)
    ys = [float(np.clip(targets[i], lo, hi)) for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - hi
    if overflow > 0:
        ys = [y - overflow for y in ys]
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - min_gap)
    ys = [float(np.clip(y, lo, hi)) for y in ys]
    out = [0.0] * len(targets)
    for idx, y in zip(order, ys):
        out[int(idx)] = y
    return out


def panel_title(ax, letter: str, text: str, subtitle: str | None = None) -> None:
    y_title = 1.105 if subtitle else 1.07
    ax.text(-0.12, y_title, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_PANEL, fontweight="bold", color=S.INK, clip_on=False)
    ax.text(0.055, y_title, text, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_PANEL, fontweight="bold", color=S.INK, clip_on=False)
    if subtitle:
        ax.text(-0.015, 1.055, subtitle, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=4.8, color=S.SUBTLE, clip_on=False)


def finish_axes(ax, *, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK, width=0.5)
    if grid:
        ax.grid(axis=grid, color="#dedede", linewidth=0.45, alpha=0.85)


def add_shared_legend(fig) -> None:
    handles = []
    for agent in AGENTS:
        hatch = "///" if agent in HUMANS else None
        edge = S.INK if agent in HUMANS else "none"
        handles.append(Patch(facecolor=COLOR[agent], edgecolor=edge, hatch=hatch,
                             linewidth=0.4, label=LABEL[agent]))
    handles.extend([
        Patch(facecolor="#555555", edgecolor="none", label="generated decision"),
        Patch(facecolor="#555555", alpha=0.28, edgecolor="none", label="soft-policy readout"),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, S.NHB_LEGEND_Y),
               ncol=8, frameon=False, fontsize=FS_LEGEND, handlelength=1.0,
               columnspacing=0.55, handletextpad=0.30, borderaxespad=0.0)


def plot_dominance(ax, dominance: pd.DataFrame) -> None:
    width = 0.085
    offsets = np.linspace(-0.30, 0.30, len(AGENTS))
    for i, fam in enumerate(DOM_FAMILIES):
        for off, agent in zip(offsets, AGENTS):
            row = dominance[(dominance["family"].eq(fam)) & (dominance["agent"].eq(agent))]
            if row.empty:
                continue
            r = row.iloc[0]
            x = i + off
            if agent in MODELS and _finite(r["R_soft"]):
                ax.bar(x, r["R_soft"], width=width * 1.30, color=COLOR[agent],
                       alpha=0.28, linewidth=0, zorder=1)
            hatch = "///" if agent in HUMANS else None
            edge = S.INK if agent in HUMANS else "white"
            ax.bar(x, r["R_realized"], width=width, color=COLOR[agent], alpha=0.96,
                   edgecolor=edge, linewidth=0.3, hatch=hatch, zorder=3)
            if _finite(r["R_realized_ci_lo"]) and _finite(r["R_realized_ci_hi"]):
                y = float(r["R_realized"])
                ax.errorbar([x], [y],
                            yerr=[[y - float(r["R_realized_ci_lo"])],
                                  [float(r["R_realized_ci_hi"]) - y]],
                            fmt="none", ecolor=S.INK, elinewidth=0.45,
                            capsize=1.2, capthick=0.45, zorder=4)
    ax.axhline(0, color=S.GREY, lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(len(DOM_FAMILIES)))
    ax.set_xticklabels(["DD\nn=36", "OD1\nn=36", "OD2\nn=36"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Conformity\n(0 random, 1 NE)", fontsize=FS_AXIS)
    ax.annotate("reasoning depth", xy=(2.28, 1.055), xytext=(0.12, 1.055),
                arrowprops=dict(arrowstyle="->", lw=0.55, color=S.FAINT),
                ha="left", va="center", fontsize=FS_FOOT, color=S.FAINT)
    panel_title(ax, "a", "Unique-equilibrium conformity")
    finish_axes(ax)


def plot_payoff(ax, payoff: pd.DataFrame) -> None:
    offsets = np.linspace(-0.22, 0.22, len(AGENTS))
    random_refs = {}
    for i, fam in enumerate(FAMILIES):
        sub = payoff[payoff["family"].eq(fam)]
        if sub.empty:
            continue
        ax.axvspan(i - 0.16, i + 0.16, color="#eeeeee", zorder=0)
        # The common guide is the canonical rank-scale substrate used by all
        # four LLMs and Moore et al. Zhu et al. is plotted from its original
        # cardinal matrices and has its own exact references in the CSV table.
        rank_ref = sub[sub["payoff_kind"].eq("ordinal_rank")]
        eq_ref = float(rank_ref["eq_efficiency"].mean())
        random_ref = float(rank_ref["random_efficiency"].mean())
        random_refs[fam] = random_ref
        ax.plot([i - 0.29, i + 0.29], [eq_ref, eq_ref], color=S.GOOD_GREEN,
                lw=0.9, solid_capstyle="round", zorder=2)
        ax.plot([i - 0.29, i + 0.29], [random_ref, random_ref], color=S.GREY,
                lw=0.65, ls=(0, (3, 2)), solid_capstyle="round", zorder=2)
        for off, agent in zip(offsets, AGENTS):
            row = sub[sub["agent"].eq(agent)]
            if row.empty or not _finite(row.iloc[0]["realized_efficiency"]):
                continue
            marker = "D" if agent in HUMANS else "o"
            face = "white" if agent in HUMANS else COLOR[agent]
            ax.scatter(i + off, row.iloc[0]["realized_efficiency"], s=12,
                       marker=marker, facecolor=face, edgecolor=COLOR[agent],
                       linewidth=0.7, zorder=4)
    ax.text(5.0, 0.05, "MP descriptive", ha="center", va="bottom",
            fontsize=FS_FOOT, color=S.FAINT)
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Payoff efficiency\n(0 worst, 1 best)", fontsize=FS_AXIS)
    panel_title(ax, "b", "Payoff efficiency")
    finish_axes(ax)
    ax.text(0.58, 0.90, "rank-scale equilibrium", color=S.GOOD_GREEN, fontsize=FS_FOOT,
            transform=ax.transAxes, ha="left", va="center")
    if "OD2" in random_refs:
        ax.text(2.0, random_refs["OD2"] - 0.055, "rank-scale random",
                color=S.GREY, fontsize=FS_FOOT, ha="center", va="center")


def plot_coordination(fig, slot, coord: pd.DataFrame) -> None:
    outer = fig.add_subplot(slot)
    outer.axis("off")
    outer.text(-0.12, 1.18, "(c)", transform=outer.transAxes, ha="left",
               va="bottom", fontsize=FS_PANEL, fontweight="bold", color=S.INK, clip_on=False)
    outer.text(0.055, 1.18, "Coordination structure", transform=outer.transAxes, ha="left",
               va="bottom", fontsize=FS_PANEL, fontweight="bold", color=S.INK, clip_on=False)
    sub = slot.subgridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.16)
    ax_rank = fig.add_subplot(sub[0, 0])
    ax_conf = fig.add_subplot(sub[0, 1])

    rank = coord[coord["structure"].eq("rankable")].set_index("agent")
    conflict = coord[coord["structure"].eq("conflict")].set_index("agent")
    y = np.arange(len(AGENTS), dtype=float)

    def stacked_barh(ax, ypos, parts, colors, *, hatch: str | None, alpha: float = 1.0) -> None:
        left = 0.0
        for value, color in zip(parts, colors):
            if not _finite(value) or value <= 0:
                continue
            ax.barh(ypos, value, left=left, height=0.56, color=color,
                    edgecolor="white", linewidth=0.35, hatch=hatch, alpha=alpha, zorder=3)
            left += float(value)

    for yi, agent in zip(y, AGENTS):
        hatch = "///" if agent in HUMANS else None
        r = rank.loc[agent]
        c = conflict.loc[agent]
        stacked_barh(ax_rank, yi,
                     [r["pdom_share"], r["other_share"], r["miscoord_share"]],
                     [SEG["best"], SEG["other"], SEG["miscoord"]], hatch=hatch)
        stacked_barh(ax_conf, yi,
                     [c["coord_share"], c["miscoord_share"]],
                     [SEG_CONFLICT["coord"], SEG_CONFLICT["miscoord"]],
                     hatch=hatch, alpha=0.92)

    for subax in (ax_rank, ax_conf):
        subax.axvline(0.5, color=S.GREY, lw=0.5, ls=(0, (3, 2)), zorder=0)
        subax.set_xlim(0, 1)
        subax.set_ylim(len(AGENTS) - 0.45, -0.55)
        subax.set_xticks([0, 0.5, 1.0])
        subax.set_xticklabels(["0", "50 (random)", "100"])
        finish_axes(subax, grid="x")
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels([LABEL[a] for a in AGENTS])
    ax_conf.set_yticks(y)
    ax_conf.set_yticklabels([])
    ax_rank.set_title("Pareto-rankable", fontsize=FS_TICK, color=S.INK, pad=3)
    ax_conf.set_title("Distributional conflict", fontsize=FS_TICK, color=S.INK, pad=3)
    handles = [
        Patch(facecolor=SEG["best"], label="best NE"),
        Patch(facecolor=SEG["other"], label="other NE"),
        Patch(facecolor=SEG["miscoord"], label="miscoord."),
        Patch(facecolor=SEG_CONFLICT["coord"], label="conflict coord."),
        Patch(facecolor=SEG_CONFLICT["miscoord"], label="conflict miscoord."),
    ]
    outer.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.20),
                 ncol=5, frameon=False, fontsize=5.8, handlelength=0.72,
                 handletextpad=0.25, columnspacing=0.48, labelspacing=0.18,
                 borderaxespad=0.0)


def _boot_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 5 or np.std(x) <= 1e-10 or np.std(y) <= 1e-10:
        return np.nan, np.nan, np.nan
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(x), size=len(x))
        if np.std(x[idx]) <= 1e-10 or np.std(y[idx]) <= 1e-10:
            continue
        vals.append(float(spearmanr(x[idx], y[idx]).correlation))
    if not vals:
        return rho, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return rho, float(lo), float(hi)


def complexity_response(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(MASTER)[["game_code", "complexity_score"]]
    d = panel[panel["agent"].isin(COMPLEXITY_AGENTS)].copy()
    d = d.merge(master, on="game_code", how="left", validate="many_to_one")
    d["p_canon"] = np.where(
        d["canonical_action_p1"].astype(int).eq(0),
        d["realized_p1"].astype(float),
        1.0 - d["realized_p1"].astype(float),
    )
    rows = []
    bin_rows = []
    for agent in COMPLEXITY_AGENTS:
        sub = d[d["agent"].eq(agent)].dropna(subset=["complexity_score", "p_canon"]).copy()
        x = sub["complexity_score"].to_numpy(float)
        y = sub["p_canon"].to_numpy(float)
        rho, lo, hi = _boot_spearman(x, y)
        rows.append({
            "agent": agent,
            "display": LABEL[agent],
            "n_games": int(len(sub)),
            "spearman_rho": rho,
            "spearman_ci_lo": lo,
            "spearman_ci_hi": hi,
        })
        if len(sub) == 0:
            continue
        sub["bin"] = pd.qcut(sub["complexity_score"], q=5, labels=False, duplicates="drop")
        for b, bsub in sub.groupby("bin", sort=True):
            bin_rows.append({
                "agent": agent,
                "display": LABEL[agent],
                "bin": int(b),
                "n_games": int(len(bsub)),
                "complexity_mean": float(bsub["complexity_score"].mean()),
                "p_canon_mean": float(bsub["p_canon"].mean()),
                "p_canon_se": (
                    float(bsub["p_canon"].std(ddof=1) / np.sqrt(len(bsub)))
                    if len(bsub) > 1 else np.nan
                ),
            })
    summary = pd.DataFrame(rows)
    bins = pd.DataFrame(bin_rows)
    summary.to_csv(TAB_DIR / "f1_complexity_response.csv", index=False)
    return summary, bins


def plot_complexity(ax, panel: pd.DataFrame) -> None:
    summary, bins = complexity_response(panel)
    label_items = []
    for agent in COMPLEXITY_AGENTS:
        b = bins[bins["agent"].eq(agent)].sort_values("complexity_mean")
        if b.empty:
            continue
        is_human = agent == "nagel"
        ax.plot(
            b["complexity_mean"], b["p_canon_mean"],
            color=COLOR[agent], lw=1.25 if not is_human else 1.15,
            ls=(0, (3, 2)) if is_human else "-", alpha=0.95, zorder=3,
        )
        ax.scatter(
            b["complexity_mean"], b["p_canon_mean"], s=15,
            marker="D" if is_human else "o", color=COLOR[agent],
            edgecolor="white", linewidth=0.35, zorder=4,
        )
        row = summary[summary["agent"].eq(agent)].iloc[0]
        label_items.append({
            "agent": agent,
            "x": float(b["complexity_mean"].iloc[-1]),
            "y": float(b["p_canon_mean"].iloc[-1]),
            "text": f"rho={_compact_decimal(row['spearman_rho'])}",
        })
    ax.axhline(0.5, color=S.GREY, lw=0.55, ls=(0, (3, 2)), zorder=0)
    x0 = float(bins["complexity_mean"].min()) if len(bins) else 0.0
    x1 = float(bins["complexity_mean"].max()) if len(bins) else 1.0
    label_x = x1 + 0.045
    label_ys = _dodged_y_positions([item["y"] for item in label_items],
                                   lo=0.08, hi=0.96, min_gap=0.045)
    for item, y_label in zip(label_items, label_ys):
        agent = item["agent"]
        y_end = float(np.clip(item["y"], 0.08, 0.96))
        if abs(y_label - y_end) > 0.012:
            ax.plot([item["x"], label_x - 0.010], [y_end, y_label],
                    color=COLOR[agent], lw=0.45, alpha=0.65, zorder=4,
                    clip_on=False)
        ax.text(label_x, y_label, item["text"],
                color=COLOR[agent], fontsize=6.0, ha="left", va="center",
                clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.15))
    ax.set_xlim(x0 - 0.04, x1 + 0.38)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Complexity score", fontsize=FS_AXIS)
    ax.set_ylabel("P(canonical action)", fontsize=FS_AXIS)
    ax.text(0.03, 0.50, "random", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_FOOT, color=S.FAINT)
    panel_title(ax, "d", "Game complexity")
    finish_axes(ax)


def make_main(dominance: pd.DataFrame, payoff: pd.DataFrame, coord: pd.DataFrame,
              panel: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, S.NHB_TWO_ROW_HEIGHT_IN),
                     constrained_layout=False)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    top = gs[0, 0].subgridspec(1, 2, width_ratios=[1.62, 0.98], wspace=0.34)
    bottom = gs[1, 0].subgridspec(1, 2, width_ratios=[1.62, 0.98], wspace=0.34)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_d = fig.add_subplot(bottom[0, 1])

    plot_dominance(ax_a, dominance)
    plot_payoff(ax_b, payoff)
    plot_coordination(fig, bottom[0, 0], coord)
    plot_complexity(ax_d, panel)

    add_shared_legend(fig)
    fig.subplots_adjust(top=S.NHB_FIG2_TOP, left=S.NHB_FIG2_LEFT,
                        right=S.NHB_FIG2_RIGHT, bottom=S.NHB_FIG2_BOTTOM)
    fig.savefig(FIG_DIR / "fig1_regime_rationality.pdf")
    fig.savefig(FIG_DIR / "fig1_regime_rationality.png", dpi=600)
    plt.close(fig)


def make_mp_supplement(mp: pd.DataFrame) -> None:
    summary = mp[mp["game_code"].eq("__SUMMARY__")].copy()
    summary["freq_gap"] = summary[["abs_freq_gap_p1", "abs_freq_gap_p2"]].mean(axis=1)
    summary["mp_entropy"] = summary[["pref_entropy_p1", "pref_entropy_p2"]].mean(axis=1)
    summary["dom_entropy"] = summary[["dominance_pref_entropy_p1", "dominance_pref_entropy_p2"]].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    gap = summary.dropna(subset=["freq_gap"])
    x = np.arange(len(gap))
    bars = ax.bar(x, gap["freq_gap"], color=[COLOR[a] for a in gap["agent"]], width=0.66)
    for bar, agent in zip(bars, gap["agent"]):
        if agent in HUMANS:
            bar.set_hatch("///")
            bar.set_edgecolor(S.INK)
            bar.set_linewidth(0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in gap["agent"]], rotation=25, ha="right", fontsize=FS_TICK)
    ax.set_ylabel("Mean absolute gap\nfrom mixed NE", fontsize=FS_AXIS)
    panel_title(ax, "a", "Mixed games: aggregate frequencies")
    finish_axes(ax)

    ax = axes[1]
    ent = summary[summary["agent"].isin(MODELS)].copy()
    x = np.arange(len(ent))
    w = 0.34
    ax.bar(x - w / 2, ent["mp_entropy"], width=w, color=[COLOR[a] for a in ent["agent"]],
           alpha=0.92, label="MP")
    ax.bar(x + w / 2, ent["dom_entropy"], width=w, color=[COLOR[a] for a in ent["agent"]],
           alpha=0.28, label="DD/OD")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in ent["agent"]], rotation=25, ha="right", fontsize=FS_TICK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Soft-policy entropy (bits)", fontsize=FS_AXIS)
    ax.legend(frameon=False, fontsize=FS_LEGEND)
    panel_title(ax, "b", "LLM policy flatness")
    finish_axes(ax)

    fig.text(0.5, 0.98, "Mixed-equilibrium games are descriptive only",
             ha="center", va="top", fontsize=FS_TITLE, fontweight="bold", color=S.INK)
    fig.text(0.5, 0.91,
             "Single-shot choices do not identify a mixed strategy; the supplement reports frequencies and policy entropy.",
             ha="center", va="top", fontsize=FS_SUBTITLE, color=S.SUBTLE)
    fig.subplots_adjust(top=0.72, left=0.09, right=0.98, bottom=0.28, wspace=0.38)
    fig.savefig(FIG_DIR / "figS_mp_descriptive.pdf")
    fig.savefig(FIG_DIR / "figS_mp_descriptive.png", dpi=600)
    plt.close(fig)


def main() -> int:
    tables = build_all()
    make_main(tables["dominance"], tables["payoff"], tables["coordination"], tables["panel"])
    make_mp_supplement(tables["mp"])
    print(f"wrote {FIG_DIR / 'fig1_regime_rationality.png'} (+pdf)")
    print(f"wrote {FIG_DIR / 'figS_mp_descriptive.png'} (+pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
