#!/usr/bin/env python3
"""Legacy-style disposition state-space figure, regenerated on Layer B Final.

This is the corrected-root version of the old block_b
``fig_disposition_state_space`` panel: deepest-layer P1 decision-slot residuals,
baseline canonical-choice axis on x, common disposition cue-shift axis on y
(orthogonalized to the choice axis). Cue points are absolute cued states, matching
the original block_b visual shape.

  .venv/bin/python analysis/layer_b/figscripts/fig_disposition_state_space.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

PS = lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ORDER = list(lib.MODELS)
SHORT = lib.SHORT
FIG = lib.FIG_DIR

TRAIT_LABEL = {
    "risk_aversion": "risk-averse",
    "loss_aversion": "loss-averse",
    "inequity_aversion": "inequity-averse",
    "selfish_maximizer": "selfish",
    "maximin": "maximin",
}
TRAIT_COLOR = {
    "risk_aversion": "#4c6ef5",
    "loss_aversion": "#7048e8",
    "inequity_aversion": "#2f9e44",
    "selfish_maximizer": "#e8590c",
    "maximin": "#9c36b5",
}


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}")


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    keep = np.isfinite(scores)
    scores, y = scores[keep], y[keep]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _kde_1d(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.zeros_like(grid)
    bw = max(float(bw), 1e-6)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def _zscore_to_baseline(model: str):
    meta, X = lib.load_cues(model)
    meta = meta.reset_index(drop=True)
    X = np.asarray(X, dtype=np.float32)
    base = (meta["condition"] == "baseline").to_numpy()
    mu = X[base].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (X[base].std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
    Z = (X - mu) / sd
    return meta, Z, base


def _common_disposition_axis(meta: pd.DataFrame, Z: np.ndarray, d_dec: np.ndarray) -> np.ndarray:
    base = (meta["condition"] == "baseline").to_numpy()
    bidx = {
        (g, int(cb)): i
        for g, cb, i in zip(meta.loc[base, "game_code"], meta.loc[base, "counterbalance_id"], np.where(base)[0])
    }
    acc = np.zeros(Z.shape[1], dtype=np.float64)
    n = 0
    for t in lib.TRAITS:
        cond = f"cue_{t}"
        sub = meta.index[meta["condition"] == cond].to_numpy()
        for i in sub:
            r = meta.iloc[i]
            j = bidx.get((r["game_code"], int(r["counterbalance_id"])))
            if j is None:
                continue
            acc += Z[i].astype(np.float64) - Z[j].astype(np.float64)
            n += 1
    if n == 0:
        return np.zeros(Z.shape[1], dtype=np.float64)
    axis = acc / n
    axis = axis - float(axis @ d_dec) * d_dec
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 0 else np.zeros_like(axis)


def main() -> None:
    lam = pd.read_csv(lib.TAB_DIR / "b2_geometry_lambda.csv").set_index("model")

    fig = plt.figure(figsize=(7.5, 8.5))
    outer = GridSpec(
        2, 1, height_ratios=[1, 1], hspace=0.38,
        left=0.09, right=0.985, top=0.84, bottom=0.10,
    )
    gs_top = outer[0].subgridspec(2, 2, height_ratios=[0.17, 1], hspace=0.08, wspace=0.25)
    gs_bot = outer[1].subgridspec(2, 2, height_ratios=[0.17, 1], hspace=0.08, wspace=0.25)
    cellgs = {ORDER[0]: (gs_top, 0), ORDER[1]: (gs_top, 1), ORDER[2]: (gs_bot, 0), ORDER[3]: (gs_bot, 1)}
    letters = dict(zip(ORDER, "abcd"))

    for m in ORDER:
        meta, Z, base = _zscore_to_baseline(m)
        aligned = (
            meta.loc[base, "decoded_action"].to_numpy()
            == meta.loc[base, "canonical_action_p1"].to_numpy()
        ).astype(int)

        d_dec = lib.decision_axis_diffmeans(Z[base], aligned)
        d_disp = _common_disposition_axis(meta, Z, d_dec)
        x_dec = Z @ d_dec
        y_disp = Z @ d_disp
        choice_auc = _auc(x_dec[base], aligned)
        lval = float(lam.loc[m, "lambda_L1"]) if m in lam.index and "lambda_L1" in lam else float("nan")

        panel_gs, col = cellgs[m]
        ax_marg = fig.add_subplot(panel_gs[0, col])
        ax = fig.add_subplot(panel_gs[1, col], sharex=ax_marg)

        xb = x_dec[base]
        yb = y_disp[base]
        p_can = float(aligned.mean())
        grid = np.linspace(np.nanmin(xb) - 1.0, np.nanmax(xb) + 1.0, 220)
        bw = max(np.nanstd(xb) * 0.18, 0.6)
        # Weight class-conditional KDEs by empirical class share; otherwise the marginals
        # misleadingly make a 73/27 split look like two equal-sized distributions.
        ax_marg.fill_between(
            grid, 0, (1.0 - p_can) * _kde_1d(xb[aligned == 0], grid, bw),
            color=PS.REF_RED, alpha=0.55, lw=0,
        )
        ax_marg.fill_between(
            grid, 0, p_can * _kde_1d(xb[aligned == 1], grid, bw),
            color=PS.GOOD_GREEN, alpha=0.55, lw=0,
        )
        ax_marg.axis("off")
        PS.panel_title(
            ax_marg,
            letters[m],
            f"{SHORT[m]}  L{lib.deepest_layer(m)}\naxis AUC={choice_auc:.2f}  p(can)={p_can:.2f}  lambda={lval:+.2f}",
        )

        ax.scatter(xb[aligned == 0], yb[aligned == 0], s=5, color=PS.REF_RED, alpha=0.48, lw=0, zorder=3)
        ax.scatter(xb[aligned == 1], yb[aligned == 1], s=5, color=PS.GOOD_GREEN, alpha=0.48, lw=0, zorder=3)
        for t in lib.TRAITS:
            sel = (meta["condition"] == f"cue_{t}").to_numpy()
            ax.scatter(x_dec[sel], y_disp[sel], s=5, color=TRAIT_COLOR[t], alpha=0.30, lw=0, zorder=2)

        ax.axhline(0, color=PS.GREY, lw=0.6, ls=(0, (4, 3)))
        ax.tick_params(labelsize=PS.FS_TICK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if col == 0:
            ax.set_ylabel("disposition-shift axis\n(orthogonal to choice)", fontsize=PS.FS_AXIS)
        if panel_gs is gs_bot:
            ax.set_xlabel("choice axis  (chose canonical ->)", fontsize=PS.FS_AXIS)

    handles = [
        Line2D([0], [0], marker="o", ls="", mfc=PS.GOOD_GREEN, mec="none", ms=5, label="baseline - chose canonical"),
        Line2D([0], [0], marker="o", ls="", mfc=PS.REF_RED, mec="none", ms=5, label="baseline - chose other"),
    ] + [
        Line2D([0], [0], marker="o", ls="", mfc=TRAIT_COLOR[t], mec="none", ms=5, label=TRAIT_LABEL[t])
        for t in lib.TRAITS
    ]
    fig.legend(
        handles=handles, fontsize=PS.FS_LEGEND, frameon=False,
        loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.010),
        handletextpad=0.3, columnspacing=1.0,
    )
    PS.figure_titles(
        fig,
        "Disposition-shift residual state spaces",
        f"corrected Akata one-shot{PS.SEP}deepest-layer P1 decision-slot residuals"
        f"{PS.SEP}baseline + five disposition cues",
    )
    _save(fig, "fig_disposition_state_space")


if __name__ == "__main__":
    main()
