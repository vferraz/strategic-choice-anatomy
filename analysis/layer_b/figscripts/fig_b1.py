#!/usr/bin/env python3
"""Figure B1 — the game is represented everywhere; the decision forms across depth.

(a) deepest-layer decodability of strategic variables + a balanced STIMULUS CONTROL floor
    (cell(0,0) rank) -> decodability does not separate competence (representation is universal).
(b) the decision crystallizes across depth (canonical-action AUC vs normalized depth), with the
    embedding (layer-0) floor marked -> the attributable signal is the gain over the embedding;
    Llama's representation erodes below its embedding by the deepest layer.
(c) decision x incentive state space (leak-free OOF projections) per model with the oriented angle
    -> a small angle = the decodable decision is INCENTIVE-ALIGNED (recruitment), not better decoded.
(d) equilibrium-structure (num_pure_ne 3-class) decodability vs the stimulus floor.

Style frozen via analysis/_shared/_paper_style; data fully recomputed on the one-shot substrate.
GPT-OSS residual nulls are router-mediated (see Fig S-router).

  .venv/bin/python analysis/layer_b/figscripts/fig_b1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

PS = lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

ORDER = list(lib.MODELS)
COL = lib.COL
SHORT = lib.SHORT
NICE = lib.NICE
COL_CHOICE = {0: PS.REF_RED, 1: PS.GOOD_GREEN}
TAB = lib.TAB_DIR
FIG = lib.FIG_DIR


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}")


def _ellipse(ax, x, y, col, n_sigma=1.5):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    ev, evec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(*evec[:, 1][::-1]))
    w, h = 2 * np.sqrt(np.maximum(ev, 0)) * n_sigma
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, fc=col, alpha=0.12, ec=col, lw=1.0))


def main():
    dec = pd.read_csv(TAB / "b1_decodability.csv")
    eq = pd.read_csv(TAB / "b1_equilibrium_structure.csv")
    cry = pd.read_csv(TAB / "b1_crystallization.csv")
    ss = pd.read_csv(TAB / "b1_state_space.csv")
    geo = pd.read_csv(TAB / "b2_geometry_lambda.csv").set_index("model")

    fig = plt.figure(figsize=(11.0, 11.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 0.95, 0.95],
                          hspace=0.55, wspace=0.55, top=0.90, bottom=0.07, left=0.08, right=0.97)

    # ── (a) decodability + stimulus control floor ─────────────────────────
    axa = fig.add_subplot(gs[0, 0:2])
    probes = [("canonical_action", "Canonical\naction"), ("sign_delta1c", "Incentive\nsign (Δ1c)"),
              ("dominant_action_p1", "Dominant\naction"), ("sign_delta2c", "Opponent\nincentive (Δ2c)")]
    labels = [p[1] for p in probes] + ["Equilibrium\nstructure", "Stimulus\ncontrol"]
    x = np.arange(len(labels)); w = 0.2
    for i, m in enumerate(ORDER):
        vals, los, his = [], [], []
        for key, _ in probes:
            r = dec[(dec.model == m) & (dec.probe == key)]
            vals.append(r["auc"].values[0]); los.append(r["lo"].values[0]); his.append(r["hi"].values[0])
        e = eq[eq.model == m]
        vals.append(e["macro_auc"].values[0]); los.append(e["macro_lo"].values[0]); his.append(e["macro_hi"].values[0])
        sc = dec[(dec.model == m) & (dec.probe == "stim_control_cell00")]
        vals.append(sc["auc"].values[0]); los.append(sc["lo"].values[0]); his.append(sc["hi"].values[0])
        vals, los, his = map(np.array, (vals, los, his))
        axa.bar(x + i * w, vals, w, color=COL[m], edgecolor="white", lw=0.4, zorder=3, label=SHORT[m])
        axa.errorbar(x + i * w, vals, yerr=[vals - los, his - vals], fmt="none",
                     ecolor=PS.INK, elinewidth=0.5, capsize=1.5, zorder=5)
    axa.axhline(0.5, color=PS.GREY, lw=0.7, ls=":")
    axa.set_xticks(x + 1.5 * w); axa.set_xticklabels(labels, fontsize=PS.FS_TICK)
    axa.set_ylabel("AUC / macro-OvR AUC (out-of-fold)", fontsize=PS.FS_AXIS)
    axa.set_ylim(0.35, 1.02); axa.tick_params(labelsize=PS.FS_TICK)
    axa.legend(fontsize=PS.FS_LEGEND, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    axa.text(2.0, 0.37, "dominant: n=72 games", fontsize=PS.FS_FOOT, color=PS.FAINT)
    PS.panel_title(axa, "a", "Strategic variables are decodable everywhere (incl. control)")

    # ── (b) crystallization across depth ──────────────────────────────────
    axb = fig.add_subplot(gs[0, 2:4])
    for m in ORDER:
        s = cry[cry.model == m].sort_values("depth_frac")
        axb.plot(s.depth_frac, s.auc, "-", color=COL[m], lw=1.4, label=SHORT[m], zorder=3)
        e0 = s[s.layer == 0]["auc"]
        if len(e0):
            axb.scatter([0], [e0.values[0]], s=22, facecolor="white", edgecolor=COL[m], lw=1.0, zorder=4)
    axb.axhline(0.5, color=PS.GREY, lw=0.7, ls=":")
    axb.set_xlabel("normalized depth (0=embedding, 1=final)", fontsize=PS.FS_AXIS)
    axb.set_ylabel("canonical-action AUC", fontsize=PS.FS_AXIS)
    axb.tick_params(labelsize=PS.FS_TICK); axb.legend(fontsize=PS.FS_LEGEND, frameon=False, loc="lower right")
    axb.text(0.03, 0.52, "○ = embedding floor (the stimulus is already encoded)",
             fontsize=PS.FS_FOOT, color=PS.FAINT, transform=axb.get_yaxis_transform() if False else axb.transData)
    PS.panel_title(axb, "b", "The decision crystallizes across depth (gain over embedding)")

    # ── (c) decision × incentive state space, 4 models ────────────────────
    sub = gs[1, 0:4].subgridspec(1, 4, wspace=0.18)
    for k, m in enumerate(ORDER):
        ax = fig.add_subplot(sub[0, k])
        d = ss[ss.model == m]
        for c in (0, 1):
            q = d[d.canonical_action == c]
            ax.scatter(q.x_decision_oof, q.y_incentive_oof, s=8, alpha=0.45,
                       color=COL_CHOICE[c], edgecolors="none", zorder=3)
            _ellipse(ax, q.x_decision_oof.to_numpy(), q.y_incentive_oof.to_numpy(), COL_CHOICE[c])
        ax.axhline(0, color=PS.GREY, lw=0.4, ls=":"); ax.axvline(0, color=PS.GREY, lw=0.4, ls=":")
        ax.set_xticks([]); ax.set_yticks([])
        ang = geo.loc[m, "angle_deg"]; lam = geo.loc[m, "lambda_L1"]
        ax.set_title(f"{SHORT[m]}\nangle={ang:.0f}°  λ={lam:.2f}", fontsize=PS.FS_TICK, color=PS.INK)
        if k == 0:
            ax.set_xlabel("decision axis →", fontsize=PS.FS_TICK)
            ax.set_ylabel("incentive axis (Δ1ᶜ) →", fontsize=PS.FS_TICK)
    fig.text(0.08, 0.355, "(c)  Decision × incentive geometry — small angle = decision wired to incentive",
             fontsize=PS.FS_PANEL, fontweight="bold", color=PS.INK)

    # ── (d) equilibrium-structure decodability vs stim floor ──────────────
    axd = fig.add_subplot(gs[2, 0:2])
    xm = np.arange(len(ORDER))
    macro = [eq[eq.model == m]["macro_auc"].values[0] for m in ORDER]
    mlo = [eq[eq.model == m]["macro_lo"].values[0] for m in ORDER]
    mhi = [eq[eq.model == m]["macro_hi"].values[0] for m in ORDER]
    floor = [eq[eq.model == m]["stim_floor_macro_auc"].values[0] for m in ORDER]
    macro, mlo, mhi = map(np.array, (macro, mlo, mhi))
    axd.bar(xm, macro, 0.6, color=[COL[m] for m in ORDER], edgecolor="white", lw=0.4, zorder=3)
    axd.errorbar(xm, macro, yerr=[macro - mlo, mhi - macro], fmt="none", ecolor=PS.INK,
                 elinewidth=0.6, capsize=2.5, zorder=5)
    axd.scatter(xm, floor, marker="_", s=420, color=PS.INK, lw=1.6, zorder=6, label="stimulus floor")
    axd.axhline(0.5, color=PS.GREY, lw=0.7, ls=":")
    axd.set_xticks(xm); axd.set_xticklabels([SHORT[m] for m in ORDER], fontsize=PS.FS_TICK)
    axd.set_ylabel("equilibrium-type macro-OvR AUC", fontsize=PS.FS_AXIS)
    axd.set_ylim(0.4, 1.0); axd.tick_params(labelsize=PS.FS_TICK)
    axd.legend(fontsize=PS.FS_LEGEND, frameon=False, loc="upper right")
    PS.panel_title(axd, "d", "Equilibrium type (MP/OD/CO) — Qwen above floor, Llama at it")

    # notes box (d-right): the organizing principle + GPT-OSS caveat
    axn = fig.add_subplot(gs[2, 2:4]); axn.axis("off")
    axn.text(0.0, 1.0, "Representation is universal and cheap; recruitment is selective.",
             fontsize=PS.FS_AXIS, fontweight="bold", color=PS.INK, va="top")
    axn.text(0.0, 0.80,
             "• Every strategic variable is a deterministic function of the payoff matrix in the\n"
             "  prompt, so it is linearly decodable from any competent encoder — decodability\n"
             "  alone (panel a) proves the game is encoded, never competence.\n"
             "• The load-bearing signal is geometry (panel c): only high-λ models wire the\n"
             "  decision axis to the incentive axis (small angle).\n"
             "• GPT-OSS residual nulls are router-mediated (Fig S-router): the dense residual\n"
             "  under-reads an MoE model; its strategic structure lives in expert routing.\n"
             "• AUC magnitudes are inflated by the 16-cell counterbalance density — read the\n"
             "  per-model pattern and cross-model rank, not absolute AUC.",
             fontsize=PS.FS_FOOT, color=PS.SUBTLE, va="top", linespacing=1.5)

    PS.figure_titles(fig, "The game is represented everywhere; the decision forms across depth",
                     f"one-shot substrate{PS.SEP}144 games × 16 counterbalance cells × 4 models"
                     f"{PS.SEP}decision-slot residuals, game-grouped CV, bootstrap-by-game CIs")
    _save(fig, "fig_b1_representation")


if __name__ == "__main__":
    main()
