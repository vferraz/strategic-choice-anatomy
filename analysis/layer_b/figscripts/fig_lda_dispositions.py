#!/usr/bin/env python3
"""Legacy-style held-out LDA disposition map, regenerated on Layer B Final.

This is the corrected-root version of the old block_b ``fig_lda_dispositions``
panel. It fits PCA->Fisher LDA on paired deepest-layer cue-shift vectors from
train games, projects held-out games, and overlays the length-matched placebo
as a null prompt-perturbation control. Quantitative B3 values remain in
``tables/b3_lda.csv``; this script is the visual state-space supplement.

  .venv/bin/python analysis/layer_b/figscripts/fig_lda_dispositions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

PS = lib.style()
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FIG = lib.FIG_DIR
PCA_K = 30
SPLIT_SEED = 0

TRAITS = [
    ("risk_aversion", "risk-averse", "#4c6ef5"),
    ("loss_aversion", "loss-averse", "#7048e8"),
    ("inequity_aversion", "inequity-averse", "#2f9e44"),
    ("selfish_maximizer", "selfish", "#e8590c"),
    ("maximin", "maximin", "#9c36b5"),
]
PLACEBO = (lib.PLACEBO, "placebo (null)", "#868e96")


def _save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "pdf" else 600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{pdf,png}}")


def _zscore_to_baseline(model: str):
    meta, X = lib.load_cues(model)
    meta = meta.reset_index(drop=True)
    X = np.asarray(X, dtype=np.float32)
    base = (meta["condition"] == "baseline").to_numpy()
    mu = X[base].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (X[base].std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
    Z = (X - mu) / sd
    return meta, Z, base


def _paired_shifts(model: str):
    meta, Z, base = _zscore_to_baseline(model)
    bidx = {
        (g, int(cb)): i
        for g, cb, i in zip(meta.loc[base, "game_code"], meta.loc[base, "counterbalance_id"], np.where(base)[0])
    }
    rows = []
    for trait in lib.TRAITS + [lib.PLACEBO]:
        cond = f"cue_{trait}"
        for i in meta.index[meta["condition"] == cond].to_numpy():
            r = meta.iloc[i]
            j = bidx.get((r["game_code"], int(r["counterbalance_id"])))
            if j is None:
                continue
            rows.append((Z[i] - Z[j], trait, r["game_code"]))
    return (
        np.asarray([x for x, _, _ in rows], dtype=np.float32),
        np.asarray([c for _, c, _ in rows]),
        np.asarray([g for _, _, g in rows]),
    )


def _density(points: np.ndarray, gx: np.ndarray, gy: np.ndarray, bw: float) -> np.ndarray:
    out = np.zeros(gx.shape, dtype=np.float64)
    if len(points) == 0:
        return out
    bw2 = max(float(bw) ** 2, 1e-12)
    for px, py in points:
        out += np.exp(-0.5 * ((gx - px) ** 2 + (gy - py) ** 2) / bw2)
    return out


def _cmap(hex_color: str, peak_alpha: float = 0.50):
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.LinearSegmentedColormap.from_list(
        f"fade_{hex_color}", [(r, g, b, 0.0), (r, g, b, peak_alpha)], N=256
    )


def _place_labels(positions, names, colors, ax, xlim, ylim) -> None:
    placed = []
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    candidates = [
        (0.00, 0.00), (0.00, 0.10), (0.00, -0.10), (0.00, 0.20), (0.00, -0.20),
        (0.12, 0.00), (-0.12, 0.00), (0.12, 0.12), (-0.12, 0.12),
        (0.12, -0.12), (-0.12, -0.12), (0.22, 0.00), (-0.22, 0.00),
        (0.22, 0.12), (-0.22, 0.12), (0.22, -0.12), (-0.22, -0.12),
    ]
    for (cx, cy), name, color in zip(positions, names, colors):
        best = (cx, cy)
        for dx, dy in candidates:
            tx = np.clip(cx + dx * xr, xlim[0] + 0.04 * xr, xlim[1] - 0.04 * xr)
            ty = np.clip(cy + dy * yr, ylim[0] + 0.05 * yr, ylim[1] - 0.05 * yr)
            if all(abs(tx - px) >= 0.18 * xr or abs(ty - py) >= 0.095 * yr for px, py in placed):
                best = (tx, ty)
                break
        placed.append(best)
        ax.text(
            best[0], best[1], name, ha="center", va="center",
            fontsize=PS.FS_LEGEND, color=color, fontweight="bold", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
        )


def main() -> None:
    table_acc = pd.read_csv(lib.TAB_DIR / "b3_lda.csv")
    rng = np.random.default_rng(SPLIT_SEED)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    for idx, (ax, model) in enumerate(zip(axes.ravel(), lib.MODELS)):
        shifts, cue, game = _paired_shifts(model)
        trait_names = [t[0] for t in TRAITS]
        trait_mask = np.isin(cue, trait_names)

        ug = rng.permutation(np.unique(game))
        train_games = set(ug[:len(ug) // 2])
        is_train = np.asarray([g in train_games for g in game])

        mu, comp = lib.pca_fit(shifts[trait_mask & is_train], PCA_K)
        p_all = (shifts - mu) @ comp.T
        ylab = {name: j for j, name in enumerate(trait_names)}
        y_train = np.asarray([ylab[c] for c in cue[trait_mask & is_train]])
        W = lib.lda_fit(p_all[trait_mask & is_train], y_train)
        emb = p_all @ W

        test = ~is_train
        shown = emb[test]
        lo, hi = np.percentile(shown, [1, 99], axis=0)
        pad = np.maximum(0.25 * (hi - lo), 1e-3)
        x0, x1 = lo[0] - pad[0], hi[0] + pad[0]
        y0, y1 = lo[1] - pad[1], hi[1] + pad[1]
        gx, gy = np.meshgrid(np.linspace(x0, x1, 150), np.linspace(y0, y1, 150))
        bw = 0.12 * max(np.mean([x1 - x0, y1 - y0]), 1e-3)

        label_pos, label_name, label_color = [], [], []
        for cond, label, color in TRAITS + [PLACEBO]:
            pts = emb[test & (cue == cond)]
            if len(pts) < 4:
                continue
            dens = _density(pts, gx, gy, bw)
            dens /= dens.max() + 1e-12
            ax.imshow(
                dens, extent=[x0, x1, y0, y1], origin="lower",
                aspect="auto", cmap=_cmap(color), interpolation="bilinear", zorder=1,
            )
            ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, alpha=0.58, lw=0, zorder=4)
            label_pos.append(tuple(np.median(pts, axis=0)))
            label_name.append(label)
            label_color.append(color)

        _place_labels(label_pos, label_name, label_color, ax, (x0, x1), (y0, y1))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= 2:
            ax.set_xlabel("LD1", fontsize=PS.FS_AXIS, color=PS.SUBTLE)
        if idx % 2 == 0:
            ax.set_ylabel("LD2", fontsize=PS.FS_AXIS, color=PS.SUBTLE)

        overall = table_acc[(table_acc.model == model) & (table_acc.disposition == "OVERALL")]
        acc = float(overall["held_out_acc"].iloc[0]) if len(overall) else float("nan")
        PS.panel_title(ax, chr(ord("a") + idx), f"{lib.SHORT[model]}\nheld-out acc={acc:.2f}")

    fig.legend(
        handles=[Line2D([0], [0], marker="o", ls="", mfc=c, mec="none", ms=6, label=lab) for _, lab, c in TRAITS + [PLACEBO]],
        fontsize=PS.FS_LEGEND, frameon=False, loc="lower center", ncol=6,
        bbox_to_anchor=(0.5, 0.005), handletextpad=0.3, columnspacing=0.9,
    )
    PS.figure_titles(
        fig,
        "Disposition geometry - held-out LDA discriminant",
        f"fit on train games, projected on held-out games{PS.SEP}placebo = length-matched null control",
    )
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.07, right=0.97, wspace=0.15, hspace=0.35)
    _save(fig, "fig_lda_dispositions")


if __name__ == "__main__":
    main()
