#!/usr/bin/env python3
"""Figure 2 — trait-cue response and attribution.

Final NHB layout:
  (a) trait-cue movement magnitude and target alignment;
  (b) TreeSHAP feature attribution for canonical conformity;
  (c) nested held-out AUC.

No in-figure master title/subtitle. Interpretive text belongs in the caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_rgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from adjustText import adjust_text

from strategic_anatomy import paper_style as S  # noqa: E402
from strategic_anatomy.config import game_features_csv, repo_root, results_root

ROOT = repo_root()

S.apply()

LAYERA = ROOT / "analysis" / "layer_a"
PER_GAME = results_root() / "layer_a" / "_data" / "oneshot_trait_steering_per_game.csv"
NESTED = results_root() / "layer_a" / "f2_nested_auc.csv"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
TAB_DIR = results_root() / "layer_a"
# phase-4: panel D read `figures/_attrib_panel_data/panelD_beeswarm.csv`, a byproduct
# build_attribution.py drops beside its own output. That file is regenerated on every rebuild
# and is byte-identical to the COMMITTED s_shap_beeswarm.csv, so a clean clone (which has the
# committed table but not the byproduct) could not build this figure. Read the committed table
# and keep the byproduct only as a fallback for a tree that has just run build_attribution.
PANEL_DIR = FIG_DIR / "_attrib_panel_data"
BEESWARM = TAB_DIR / "s_shap_beeswarm.csv"
FEATURES = game_features_csv()

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
MODEL_LABEL = {
    "qwen": "Qwen2.5",
    "qwen_instruct": "Qwen2.5-I",
    "llama31_instruct": "Llama",
    "gptoss": "GPT-OSS",
}
TRAITS = ["risk", "loss", "inequity", "selfish"]
CONTROL = "maximin"
TRAIT_LABEL = {
    "risk": "Risk",
    "loss": "Loss",
    "inequity": "Inequity",
    "selfish": "Selfish",
    "maximin": "Maximin",
}
TRAIT_COLOR = {
    "risk": S.GOOD_GREEN,
    "loss": S.GOOD_GREEN,
    "inequity": "#1f6f8b",
    "selfish": "#7a4fa3",
    "maximin": S.GOOD_GREEN,
}
POINT_DODGE = {
    "risk": (-0.035, -0.010),
    "loss": (0.035, 0.006),
    "maximin": (0.000, 0.020),
}
LABEL_DODGE = {
    "placebo": (-0.24, 0.055),
    "risk": (0.085, -0.045),
    "loss": (0.075, 0.040),
    "inequity": (0.000, 0.060),
    "selfish": (0.075, -0.035),
    "maximin": (-0.21, 0.055),
}
SEED = 20260521
N_BOOT = 4000
BLOCK_COLOR = {"structure": S.GOOD_GREEN, "trait": S.SOFT_BLUE, "model": S.GREY}


def _blockof(feature: str) -> str:
    if feature.startswith("model="):
        return "model"
    if feature.startswith("trait="):
        return "trait"
    return "structure"


def _ramp(color: str, light: float = 0.80) -> LinearSegmentedColormap:
    base = np.array(to_rgb(color))
    lo = np.array([1.0, 1.0, 1.0]) * light + base * (1 - light)
    return LinearSegmentedColormap.from_list("r", [lo, base])


BLOCK_CMAP = {k: _ramp(v) for k, v in BLOCK_COLOR.items()}


def _pretty_feature(feature: str) -> str:
    d1 = "Δ₁"
    names = {
        "model=qwen": "Model: Qwen (base)",
        "model=qwen_instruct": "Model: Qwen-Instruct",
        "model=llama31_instruct": "Model: Llama-3.1",
        "model=gptoss": "Model: GPT-OSS",
        "dInc_signed_unif": f"Incentive {d1} → canonical (uniform)",
        "complexity_score": "Game complexity",
        "maximin_to_ne_payoff_distance": "Maximin-NE payoff gap",
        "maximin_to_ne_action_distance": "Maximin-NE action gap",
        "maximin_is_ne": "Maximin is NE",
        "trait=maximin": "Cue: maximin rule",
        "trait=inequity_aversion": "Cue: inequity aversion",
        "trait=selfish_maximizer": "Cue: selfish",
        "trait=loss_aversion": "Cue: loss aversion",
        "trait=risk_aversion": "Cue: risk aversion",
        "trait=length_match_null": "Cue: procedural control",
    }
    if feature in names:
        return names[feature]
    if feature.startswith("nagel_lk_type="):
        return "Game type: " + feature.split("=", 1)[1]
    if feature.startswith("ne_pareto_rankable="):
        return "NE Pareto-rankable = " + feature.split("=", 1)[1]
    return feature


def _ci(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    draws = vals[rng.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _conflict_filter(d: pd.DataFrame, trait: str) -> pd.Series:
    astar = pd.to_numeric(d["a_star"], errors="coerce")
    if trait in {"risk", "loss", "maximin"}:
        # Security cues are identifiable only when maximin and the optimistic
        # best-outcome action diverge. ``l0_action_p1`` is
        # argmax_i max_j u_1(i,j), not a uniform-EV action; uniform EV is tied
        # on these 48 mean-preserving-spread games.
        optimistic = pd.to_numeric(d["l0_action_p1"], errors="coerce")
        return astar.notna() & optimistic.isin([0, 1]) & astar.astype(float).ne(optimistic.astype(float))
    if trait == "inequity":
        # Inequity is identifiable where the most-equal action differs from the
        # canonical/rational action (~20 games after target ties are dropped).
        rational = pd.to_numeric(d["canonical_action_p1"], errors="coerce")
        return astar.notna() & rational.isin([0, 1]) & astar.astype(float).ne(rational.astype(float))
    if trait == "selfish":
        # Manipulation check: all identifiable uniform-EV games (96 games).
        return astar.notna()
    return pd.Series(False, index=d.index)


def conflict_summary(pg: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(FEATURES)[["game_code", "canonical_action_p1", "l0_action_p1"]]
    d = pg.merge(features, left_on="game", right_on="game_code", how="left")
    rows: list[dict] = []
    for (model, round_id, trait), grp0 in d.groupby(["model", "round", "trait"]):
        keep = _conflict_filter(grp0, str(trait))
        grp = grp0[keep].copy()
        if grp.empty:
            continue
        mag, mag_lo, mag_hi = _ci(grp["abs_shift"].to_numpy(float))
        null_mag, _, _ = _ci(grp["abs_nullshift"].to_numpy(float))
        dir_shift, dir_lo, dir_hi = _ci(grp["shift_pref"].to_numpy(float))
        net, net_lo, net_hi = _ci(grp["net_pref"].to_numpy(float))
        flips = int(grp["flips"].sum())
        pairs = int(grp["n_pairs"].sum())
        aligned = int(grp["flips_aligned"].sum())
        rows.append({
            "model": model,
            "round": int(round_id),
            "trait": trait,
            "family": grp["family"].iloc[0],
            "n_games": int(grp["game"].nunique()),
            "conflict_rule": {
                "risk": "security_vs_optimistic_best",
                "loss": "security_vs_optimistic_best",
                "maximin": "security_vs_optimistic_best",
                "inequity": "equal_vs_canonical",
                "selfish": "uniform_ev_identifiable",
            }.get(str(trait), ""),
            "magnitude": mag,
            "magnitude_lo": mag_lo,
            "magnitude_hi": mag_hi,
            "magnitude_null": null_mag,
            "flip_rate": flips / pairs if pairs else np.nan,
            "flip_rate_aligned": aligned / flips if flips else np.nan,
            "dir_shift": dir_shift,
            "dir_lo": dir_lo,
            "dir_hi": dir_hi,
            "dir_net": net,
            "dir_net_lo": net_lo,
            "dir_net_hi": net_hi,
            "directionality_index": (dir_shift / mag) if np.isfinite(mag) and mag > 1e-9 else np.nan,
            "aim": (net / mag) if np.isfinite(mag) and mag > 1e-9 else np.nan,
            "base_pref_mean": float(grp["base_pref"].mean()),
            "source": "+".join(sorted(grp["source"].dropna().unique())) if "source" in grp else "",
        })
    return pd.DataFrame(rows)


def _aim(row: pd.Series) -> float:
    mag = float(row["magnitude"])
    if not np.isfinite(mag) or mag <= 1e-9:
        return 0.0
    return float(np.clip(row["dir_net"] / mag, -1.1, 1.1))


def _sig(row: pd.Series) -> bool:
    return bool((row["dir_net_lo"] > 0) or (row["dir_net_hi"] < 0))


def _panel_a(ax, rows: dict[str, pd.Series], placebo_mag: float, model: str,
             y_max: float, show_y: bool) -> None:
    ax.axvspan(0, 1.5, color="#edf6ed", zorder=0)
    ax.axvspan(-1.5, 0, color="#faf0ee", zorder=0)
    ax.axvline(0, color=S.INK, lw=0.75, zorder=2)
    texts, xs, ys = [], [], []
    lbl_bbox = dict(boxstyle="round,pad=0.11", fc="white", ec="none", alpha=0.76)
    ax.scatter(0, placebo_mag, s=27, marker="X", color=S.GREY, edgecolor="white",
               linewidth=0.45, alpha=0.9, zorder=5)
    xs.append(0.0)
    ys.append(placebo_mag)
    dx, dy = LABEL_DODGE["placebo"]
    texts.append(ax.text(dx, placebo_mag + dy, "control", fontsize=S.NHB_FS_FOOT,
                         color=S.GREY, ha="left", va="center", fontweight="medium",
                         zorder=8, bbox=lbl_bbox, clip_on=False))
    for trait in TRAITS + [CONTROL]:
        if trait not in rows:
            continue
        r = rows[trait]
        x = _aim(r)
        y = float(r["magnitude"])
        px = x + POINT_DODGE.get(trait, (0.0, 0.0))[0]
        py = y + POINT_DODGE.get(trait, (0.0, 0.0))[1]
        col = TRAIT_COLOR[trait]
        marker = "D" if trait == CONTROL else "o"
        if float(r.get("base_pref_mean", 0.0)) >= 0.8:
            ax.scatter(px, py, s=88, marker=marker, facecolor="none",
                       edgecolor="#bdbdbd", linewidth=1.25, zorder=4,
                       clip_on=False)
        ax.scatter(px, py, s=34 if trait == CONTROL else 27, marker=marker,
                   facecolor=col if _sig(r) else "white", edgecolor=col,
                   linewidth=0.95, alpha=0.92, zorder=6, clip_on=False)
        xs.append(px)
        ys.append(py)
        dx, dy = LABEL_DODGE[trait]
        ha = "right" if dx < 0 else ("center" if abs(dx) < 1e-9 else "left")
        texts.append(ax.text(px + dx, py + dy, TRAIT_LABEL[trait],
                             fontsize=S.NHB_FS_FOOT, color=col,
                             ha=ha, va="center", fontweight="medium",
                             zorder=8, bbox=lbl_bbox, clip_on=False))
    ax.set_title(MODEL_LABEL[model], fontsize=S.NHB_FS_MINI_TITLE, color=S.INK, pad=2.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.015, y_max)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel("target alignment\n(away \u2190 0 \u2192 toward)", fontsize=S.NHB_FS_TICK_SMALL)
    ax.set_box_aspect(1.0)
    if show_y:
        ax.set_ylabel("magnitude\nmean |ΔP(act0)|", fontsize=S.NHB_FS_AXIS)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=S.NHB_FS_TICK_SMALL, width=0.5, length=2)
    ax.grid(axis="y", color="#e1e1e1", linewidth=0.4, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.25, 1.55),
                force_text=(0.55, 0.80), force_static=(0.35, 0.55),
                force_pull=(0.008, 0.010), min_arrow_len=5,
                ensure_inside_axes=True, prevent_crossings=True,
                arrowprops=dict(arrowstyle="-", color="#9a9a9a",
                                lw=0.42, shrinkA=2, shrinkB=4))


def _panel_auc(ax, auc: pd.DataFrame) -> None:
    x = np.arange(4)
    xt = ["M0\nmodel", "M1\n+structure", "M2\n+cue", "M3\n+interact"]
    trait_text = None
    for set_id, color, ls, marker, label, dx in [
        ("gptoss_full", "#9467bd", (0, (3, 2)), "D", "GPT-OSS full", 0.035),
        ("dense_primary", S.INK, "-", "o", "dense primary", -0.035),
    ]:
        d = auc[auc["analysis_set"].eq(set_id)].copy()
        if d.empty:
            continue
        y = d["auc"].to_numpy(float)
        yerr = np.vstack([y - d["ci_lo"].to_numpy(float),
                          d["ci_hi"].to_numpy(float) - y])
        ax.errorbar(x + dx, y, yerr=yerr, color=color, ls=ls, marker=marker,
                    lw=1.55, ms=4.3, capsize=2.6, label=label, zorder=4)
        if len(d) >= 2:
            baseline = float(y[0])
            is_dense = set_id == "dense_primary"
            y_offset = -12 if is_dense else 9
            va = "top" if is_dense else "bottom"
            x_offset = 3
            text_box = dict(boxstyle="round,pad=0.08", fc="white",
                            ec="none", alpha=0.82)
            for xi, yi in zip(x[1:] + dx, y[1:]):
                this_x_offset = x_offset
                ha = "left"
                if (not is_dense) and xi > 2.7:
                    this_x_offset = -7
                    ha = "right"
                ax.annotate(f"{yi - baseline:+.2f}", (xi, yi),
                            xytext=(this_x_offset, y_offset), textcoords="offset points",
                            ha=ha, va=va, fontsize=S.NHB_FS_FOOT,
                            color=color, zorder=7, bbox=text_box)
        if set_id == "dense_primary" and len(d) >= 3:
            trait_text = f"+cue ΔAUC = {float(d.iloc[2]['auc'] - d.iloc[1]['auc']):+.2f}"
    if trait_text:
        ax.text(0.98, 0.08, trait_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=S.NHB_FS_FOOT, color=S.FAINT)
    ax.axhline(0.5, color=S.GREY, lw=0.75, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.35, 3.35)
    y0 = max(0.20, float(np.nanmin(auc["ci_lo"])) - 0.03)
    y1 = min(1.0, float(np.nanmax(auc["ci_hi"])) + 0.04)
    ax.set_ylim(y0, y1)
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=S.NHB_FS_TICK)
    for tick, color in zip(ax.get_xticklabels(), [S.GREY, S.GOOD_GREEN, S.SOFT_BLUE, S.INK]):
        tick.set_color(color)
    ax.set_ylabel("held-out AUC", fontsize=S.NHB_FS_AXIS)
    ax.tick_params(axis="y", labelsize=S.NHB_FS_TICK, width=0.5, length=2)
    ax.grid(axis="y", color="#e1e1e1", linewidth=0.45, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.legend(frameon=False, fontsize=S.NHB_FS_LEGEND, loc="upper left")


def _panel_shap(ax, ax_label, ax_scale, bees: pd.DataFrame) -> None:
    order = (
        bees.assign(abs_shap=bees["shap"].abs())
        .groupby("feature")["abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    rng = np.random.default_rng(SEED)
    n = len(order)
    for i, feature in enumerate(order):
        y_base = n - 1 - i
        sub = bees[bees["feature"].eq(feature)]
        if len(sub) > 1800:
            sub = sub.sample(1800, random_state=SEED)
        jitter = rng.uniform(-0.34, 0.34, len(sub))
        ax.scatter(
            sub["shap"],
            y_base + jitter,
            c=sub["fval_norm"],
            cmap=BLOCK_CMAP[_blockof(feature)],
            vmin=0,
            vmax=1,
            s=3.0,
            alpha=0.58,
            edgecolor="none",
            zorder=3,
            rasterized=True,
        )
    ax.axvline(0, color=S.GREY, lw=0.75, ls=(0, (4, 3)), zorder=1)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SHAP value (→ canonical conformity)", fontsize=S.NHB_FS_AXIS)
    ax.tick_params(axis="x", labelsize=S.NHB_FS_TICK, width=0.5, length=2)
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.4, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)

    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(ax.get_ylim())
    ax_label.axis("off")
    for ypos, feature in zip(range(n), order[::-1]):
        ax_label.text(0.02, ypos, _pretty_feature(feature), ha="left", va="center",
                      fontsize=S.NHB_FS_FOOT, color=BLOCK_COLOR[_blockof(feature)],
                      clip_on=True)

    key = LinearSegmentedColormap.from_list("feature_value_key", ["#dcdcdc", "#2b2b2b"])
    ax_scale.imshow(key(np.linspace(0, 1, 256))[:, None, :], aspect="auto",
                    origin="lower", extent=[0, 1, 0, 1])
    ax_scale.set_xticks([])
    ax_scale.set_yticks([0.04, 0.96])
    ax_scale.set_yticklabels(["low", "high"], fontsize=S.NHB_FS_FOOT)
    ax_scale.yaxis.set_ticks_position("left")
    ax_scale.tick_params(length=0, pad=1)
    for spine in ax_scale.spines.values():
        spine.set_visible(False)
    ax_scale.yaxis.set_label_position("right")
    ax_scale.set_ylabel("feature value", fontsize=S.NHB_FS_FOOT,
                        color=S.SUBTLE, labelpad=3)
    ax_scale.yaxis.set_label_coords(1.12, 0.5)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pg = pd.read_csv(PER_GAME)
    auc = pd.read_csv(NESTED)
    bees_src = BEESWARM if BEESWARM.exists() else PANEL_DIR / "panelD_beeswarm.csv"
    bees = pd.read_csv(bees_src)
    s1 = conflict_summary(pg)

    y_max = min(1.0, max(0.86, float(s1["magnitude"].max()) + 0.090))

    fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, 6.05))
    top_gs = GridSpec(1, 12, left=0.085, right=0.985,
                      top=0.900, bottom=0.615, wspace=0.70)
    bottom_gs = GridSpec(1, 12, left=0.035, right=0.985,
                         top=0.425, bottom=S.NHB_FIG2_BOTTOM, wspace=0.70)

    axes = []
    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(top_gs[0, i * 3:(i + 1) * 3])
        rows = {t: s1[(s1["model"].eq(model)) & (s1["trait"].eq(t))].iloc[0]
                for t in TRAITS + [CONTROL]
                if not s1[(s1["model"].eq(model)) & (s1["trait"].eq(t))].empty}
        prow = s1[s1["model"].eq(model)]
        placebo_mag = float(prow["magnitude_null"].mean()) if not prow.empty else 0.0
        _panel_a(ax, rows, placebo_mag, model, y_max, show_y=(i == 0))
        axes.append(ax)

    shap_gs = bottom_gs[0, :8].subgridspec(1, 3, width_ratios=[0.10, 1.72, 1.38], wspace=0.08)
    ax_shap_scale = fig.add_subplot(shap_gs[0, 0])
    ax_shap = fig.add_subplot(shap_gs[0, 1])
    ax_shap_label = fig.add_subplot(shap_gs[0, 2], sharey=ax_shap)
    _panel_shap(ax_shap, ax_shap_label, ax_shap_scale, bees)

    ax_auc = fig.add_subplot(bottom_gs[0, 8:])
    _panel_auc(ax_auc, auc)

    S.nhb_panel_title(axes[0], "a", "Response to fixed decision cues", title_x=0.17, y=1.14)
    S.nhb_panel_title(ax_shap_scale, "b", "Feature attribution",
                      letter_x=0.0, title_x=2.25, y=1.075)
    S.nhb_panel_title(ax_auc, "c", "Incremental cue value", y=1.075)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=S.INK,
               markeredgecolor=S.INK, ms=5.5, label="sig."),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=S.INK, ms=5.5, label="n.s."),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=S.GOOD_GREEN,
               markeredgecolor=S.GOOD_GREEN, ms=5.8, label="maximin"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=S.GREY,
               markeredgecolor=S.GREY, ms=6.2, label="procedural control"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#bdbdbd", markeredgewidth=1.6, ms=8.5, label="ceiling"),
        Line2D([0], [0], color=S.INK, lw=1.5, marker="o", ms=4, label="dense"),
        Line2D([0], [0], color="#9467bd", lw=1.5, ls=(0, (3, 2)), marker="D",
               ms=4, label="GPT-OSS"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, S.NHB_LEGEND_Y),
               ncol=7, frameon=False, fontsize=S.NHB_FS_LEGEND,
               handlelength=0.95, handletextpad=0.24, columnspacing=0.50)

    with matplotlib.rc_context({"savefig.bbox": None}):
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"fig2_trait_steering.{ext}",
                        dpi=600 if ext == "png" else 300, bbox_inches=None)
    plt.close(fig)

    s1.to_csv(TAB_DIR / "f2_trait_aim.csv", index=False)
    print(f"wrote {FIG_DIR/'fig2_trait_steering.png'} (+pdf)")
    print(f"wrote {TAB_DIR/'f2_trait_aim.csv'}, {TAB_DIR/'f2_nested_auc.csv'}")

    report = s1.copy()
    report["target_alignment"] = report.apply(_aim, axis=1)
    report["directed_sig"] = report.apply(_sig, axis=1)
    cols = [
        "model", "trait", "n_games", "conflict_rule",
        "magnitude", "magnitude_lo", "magnitude_hi", "magnitude_null",
        "dir_net", "dir_net_lo", "dir_net_hi", "target_alignment",
        "flip_rate", "flip_rate_aligned", "base_pref_mean", "directed_sig",
    ]
    print("\n=== panel A: cue movement and target alignment ===")
    print(report[cols].round(3).to_string(index=False))

    print("\n=== panel C: nested held-out AUC ===")
    auc_cols = ["analysis_set", "step", "auc", "ci_lo", "ci_hi", "delta_prev", "n_games", "n_rows"]
    print(auc[auc_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
