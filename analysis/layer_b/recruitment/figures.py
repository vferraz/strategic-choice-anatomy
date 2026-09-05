#!/usr/bin/env python3
"""Render the compact Layer B recruitment-final figure."""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.layer_b.recruitment import shared as S


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _panel_label(ax, label: str, title: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(title, loc="left", pad=8, fontweight="bold")


def _short(model: str) -> str:
    return S.SHORT.get(model, model)


def _read_panel(name: str) -> pd.DataFrame:
    parquet = S.DATA / f"{name}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(S.TABLES / f"{name}.csv")


def _ci_bars(ax, x, y, lo, hi, color):
    err = np.vstack([np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)])
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color, elinewidth=1.0, capsize=2, zorder=4)


def panel_inventory(ax) -> None:
    df = _read_panel("representation_inventory")
    concepts = [
        "realised canonical action",
        "sign(Delta1c q=0.5)",
        "5-way disposition cue shifts",
        "rank of P1 payoff at cell 00",
    ]
    labels = ["Decision", "Incentive", "Cue axes", "Stimulus"]
    width = 0.18
    x = np.arange(len(concepts))
    for i, model in enumerate(S.MODELS):
        sub = df[df["model"].eq(model)].set_index("concept").reindex(concepts)
        xpos = x + (i - 1.5) * width
        vals = sub["value"].to_numpy()
        ax.bar(xpos, vals, width=width, color=S.COLORS[model], label=_short(model), alpha=0.88)
        _ci_bars(ax, xpos, vals, sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(), "#222222")
    ax.axhline(0.5, color="#b0b0b0", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.18, 1.03)
    ax.set_ylabel("Held-out AUC / accuracy")
    ax.grid(axis="y", color="#e7e7e7", lw=0.6)
    _panel_label(ax, "A", "Minimal representation inventory")


def panel_crystallization(ax) -> None:
    df = _read_panel("decision_crystallization")
    ax.set_axis_off()
    left = ax.inset_axes([0.00, 0.07, 0.64, 0.82])
    right = ax.inset_axes([0.73, 0.07, 0.27, 0.82])
    for model in [m for m in S.MODELS if m != "gptoss"]:
        sub = df[df["model"].eq(model)].sort_values("layer")
        left.plot(sub["layer"], sub["auc"], color=S.COLORS[model], lw=2, marker="o", ms=2.4)
    sub = df[df["model"].eq("gptoss")].sort_values("layer")
    right.plot(sub["layer"], sub["auc"], color=S.COLORS["gptoss"], lw=2, marker="o", ms=2.4)
    for aa, title in [(left, "dense residual layers"), (right, "GPT-OSS commit layers")]:
        aa.axhline(0.5, color="#a8a8a8", lw=0.8, ls=":")
        aa.set_ylim(0.35, 0.94)
        aa.set_title(title, fontsize=7, pad=3)
        aa.grid(color="#e7e7e7", lw=0.6)
        aa.set_xlabel("Layer", fontsize=7)
    left.set_ylabel("Decision AUC", fontsize=8)
    right.set_yticklabels([])
    _panel_label(ax, "B", "Decision crystallization, absolute")


def panel_geometry(ax) -> None:
    df = _read_panel("recruitment_geometry_depth")
    ax.set_axis_off()
    left = ax.inset_axes([0.00, 0.07, 0.64, 0.82])
    right = ax.inset_axes([0.73, 0.07, 0.27, 0.82])
    for model in [m for m in S.MODELS if m != "gptoss"]:
        sub = df[df["model"].eq(model)].sort_values("layer")
        left.plot(
            sub["layer"],
            sub["angle_decision_incentive_deg"],
            color=S.COLORS[model],
            lw=2,
            marker="o",
            ms=2.4,
        )
    sub = df[df["model"].eq("gptoss")].sort_values("layer")
    right.plot(
        sub["layer"],
        sub["angle_decision_incentive_deg"],
        color=S.COLORS["gptoss"],
        lw=2,
        marker="o",
        ms=2.4,
    )
    for aa, title in [(left, "dense residual layers"), (right, "GPT-OSS commit layers")]:
        aa.axhline(90, color="#999999", lw=0.8, ls=":")
        aa.set_ylim(20, 95)
        aa.set_title(title, fontsize=7, pad=3)
        aa.grid(color="#e7e7e7", lw=0.6)
        aa.set_xlabel("Layer", fontsize=7)
    left.set_ylabel("Angle(d_dec, d_inc) degrees", fontsize=8)
    right.set_yticklabels([])
    _panel_label(ax, "C", "Decision-incentive geometry")


def panel_bridge(ax) -> None:
    df = _read_panel("final_geometry_bridge").set_index("model").reindex(S.MODELS).reset_index()
    x = df["decision_coordinate_delta1c_r"].to_numpy()
    y = df["slope_pcanonical_per_sd_neural_inc"].to_numpy()
    yerr = np.vstack([y - df["ci_lo"].to_numpy(), df["ci_hi"].to_numpy() - y])
    for i, row in df.iterrows():
        model = row["model"]
        ax.errorbar(
            row["decision_coordinate_delta1c_r"],
            row["slope_pcanonical_per_sd_neural_inc"],
            yerr=[[row["slope_pcanonical_per_sd_neural_inc"] - row["ci_lo"]], [row["ci_hi"] - row["slope_pcanonical_per_sd_neural_inc"]]],
            fmt="o",
            color=S.COLORS[model],
            ms=8,
            mec="#222222",
            mew=0.5,
            capsize=3,
            zorder=3,
        )
        ax.annotate(
            _short(model),
            (row["decision_coordinate_delta1c_r"], row["slope_pcanonical_per_sd_neural_inc"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )
    if np.isfinite(x).all() and np.isfinite(y).all() and len(x) > 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(min(x) - 0.02, max(x) + 0.02, 50)
        ax.plot(xs, np.polyval(coef, xs), ls="--", color="#666666", lw=1.0, zorder=1)
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.text(0.04, 0.94, f"descriptive rho={rho:.2f}, n=4", transform=ax.transAxes, va="top", fontsize=7)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("Final corr(decision coordinate, Delta1c)")
    ax.set_ylabel("Bridge slope: P(canonical) per SD d_inc")
    ax.grid(color="#e7e7e7", lw=0.6)
    _panel_label(ax, "D", "Final geometry tracks recruitment")


def panel_causal(ax) -> None:
    df = _read_panel("causal_status")
    complete = df[df["status"].eq("complete") & df["effect_mean"].notna()]
    if complete.empty:
        ax.axis("off")
        statuses = ", ".join(sorted(df["status"].unique()))
        msg = (
            "A5 steering slot is reserved.\n\n"
            "No causal claim is plotted because the expected run is not complete in this checkout.\n"
            f"Status: {statuses}"
        )
        ax.text(
            0.5,
            0.52,
            msg,
            ha="center",
            va="center",
            fontsize=9,
            linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.55", fc="#f7f7f7", ec="#cfcfcf", lw=0.8),
        )
    else:
        plot = complete.groupby(["mode", "model"], as_index=False)["effect_mean"].mean()
        modes = sorted(plot["mode"].unique())
        width = 0.18
        x = np.arange(len(modes))
        for i, model in enumerate(S.MODELS):
            sub = plot[plot["model"].eq(model)].set_index("mode").reindex(modes)
            ax.bar(x + (i - 1.5) * width, sub["effect_mean"], width, color=S.COLORS[model], alpha=0.9)
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=20, ha="right")
        ax.set_ylabel("Mean causal effect")
        ax.grid(axis="y", color="#e7e7e7", lw=0.6)
    _panel_label(ax, "E", "Causal steering status")


def panel_disposition(ax) -> None:
    df = _read_panel("disposition_dissociation")
    markers = {
        "risk_aversion": "o",
        "loss_aversion": "s",
        "inequity_aversion": "^",
        "maximin": "D",
        "selfish_maximizer": "v",
    }
    for model in S.MODELS:
        sub = df[df["model"].eq(model)]
        for cue, cue_sub in sub.groupby("cue"):
            size = 36 + 155 * cue_sub["behavior_magnitude"].clip(0, 0.9) / 0.9
            ax.scatter(
                cue_sub["angle_to_decision_deg"],
                cue_sub["behavior_aim"],
                s=size,
                color=S.COLORS[model],
                marker=markers.get(cue, "o"),
                edgecolor="white",
                linewidth=0.6,
                alpha=0.86,
            )
    ax.axvline(90, color="#b0b0b0", lw=0.8, ls=":")
    ax.axhline(1.0, color="#b0b0b0", lw=0.8, ls=":")
    ax.text(
        0.03,
        0.04,
        "size = behavioural movement\ncue identity acc is ceiling (~0.99-1.00)",
        transform=ax.transAxes,
        fontsize=6.6,
        va="bottom",
        ha="left",
        color="#444444",
    )
    ax.set_xlabel("Angle(cue shift, decision axis) degrees")
    ax.set_ylabel("Behavioural cue aim")
    ax.grid(color="#e7e7e7", lw=0.6)
    _panel_label(ax, "F", "Disposition recruitment, not identity")


def render() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2), constrained_layout=True)
    panel_inventory(axes[0, 0])
    panel_crystallization(axes[0, 1])
    panel_geometry(axes[0, 2])
    panel_bridge(axes[1, 0])
    panel_causal(axes[1, 1])
    panel_disposition(axes[1, 2])

    handles = [
        plt.Line2D([0], [0], marker="o", color=S.COLORS[m], lw=2, label=_short(m), markersize=5)
        for m in S.MODELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Layer B: Recruitment, Not Representation", y=1.055, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        textwrap.fill(
            "All behavioural targets are realised actions aligned to canonical_action_p1. "
            "The primary incentive axis is q=0.5 canonical-signed Delta1c. "
            "Panel E remains pending unless A5 steering outputs are present.",
            width=155,
        ),
        ha="center",
        va="top",
        fontsize=7,
        color="#444444",
    )
    png = S.FIGURES / "fig_layerB_recruitment_final.png"
    pdf = S.FIGURES / "fig_layerB_recruitment_final.pdf"
    png_v2 = S.FIGURES / "fig_layerB_recruitment_final_v2.png"
    pdf_v2 = S.FIGURES / "fig_layerB_recruitment_final_v2.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png_v2, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_v2, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
