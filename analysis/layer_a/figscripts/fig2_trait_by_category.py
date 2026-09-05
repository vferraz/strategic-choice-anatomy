#!/usr/bin/env python3
"""Figure 2 companion — trait steering by game category (one-shot substrate).

FROZEN DESIGN copy of analysis/block_a/trait_steering_by_category_v2.py. Only the
data-loading lines change: PER_GAME -> the one-shot trait table; the game-category
map (nagel_lk_type) is read from the human game master instead of the design_v2
master_behavior_long parquet. Heatmap layout / RdYlGn TwoSlopeNorm / typography are
unchanged. Round 1.
Output: analysis/layer_a/figures/fig2_trait_by_category.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from strategic_anatomy import paper_style as S  # noqa: E402
from strategic_anatomy.config import repo_root, results_root, taxonomy_dir

ROOT = repo_root()

S.apply()

LAYERA = ROOT / "analysis" / "layer_a"
PER_GAME = results_root() / "layer_a" / "_data" / "oneshot_trait_steering_per_game.csv"
MASTER_CSV = taxonomy_dir() / "human_game_master_per_canonical.csv"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
TAB_DIR = results_root() / "layer_a"

RNG = np.random.default_rng(20260521)
N_BOOT = 4000

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
MODEL_LABEL = {"qwen": "Qwen2.5-72B", "qwen_instruct": "Qwen2.5-72B-Instruct",
               "llama31_instruct": "Llama-3.1-70B-Instruct", "gptoss": "GPT-OSS-120B"}
TRAITS = ["risk", "loss", "inequity", "selfish", "maximin"]
TRAIT_LABEL = {"risk": "Risk-averse", "loss": "Loss-averse", "inequity": "Inequity-averse",
               "selfish": "Selfish-max", "maximin": "Maximin (rule)"}
CATS = ["Dominance", "Coordination", "Mixed"]
COLLAPSE = {"DD": "Dominance", "OD1": "Dominance", "OD2": "Dominance",
            "CO1": "Coordination", "CO2": "Coordination", "MP": "Mixed"}


def _ci(vals):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    draws = vals[RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load() -> pd.DataFrame:
    pg = pd.read_csv(PER_GAME)
    pg = pg[pg["round"] == 1].copy()
    cat = pd.read_csv(MASTER_CSV)[["game_code", "nagel_lk_type"]].drop_duplicates("game_code")
    cat["cat3"] = cat.nagel_lk_type.map(COLLAPSE)
    return pg.merge(cat, left_on="game", right_on="game_code", how="left")


def summarize(pg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for trait in TRAITS:
            for cat in CATS:
                s = pg[(pg.model == model) & (pg.trait == trait) & (pg.cat3 == cat)]
                if len(s) == 0:
                    continue
                net, lo, hi = _ci(s.net_pref.to_numpy())
                rows.append({"model": model, "trait": trait, "category": cat, "n": int(len(s)),
                             "magnitude": float(s.abs_shift.mean()),
                             "net_dir": net, "net_lo": lo, "net_hi": hi,
                             "base_pref": float(s.base_pref.mean())})
    return pd.DataFrame(rows)


def figure(summ: pd.DataFrame) -> None:
    models = [m for m in MODELS if m in set(summ.model.unique())]
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 3.9), squeeze=False)
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    cmap = plt.get_cmap("RdYlGn")
    for ci, model in enumerate(models):
        ax = axes[0][ci]
        grid = np.full((len(TRAITS), len(CATS)), np.nan)
        for i, trait in enumerate(TRAITS):
            for j, cat in enumerate(CATS):
                r = summ[(summ.model == model) & (summ.trait == trait) & (summ.category == cat)]
                if r.empty:
                    continue
                r = r.iloc[0]
                grid[i, j] = r.net_dir
                ax.text(j, i, f"{r.net_dir:+.2f}\n(n={r.n})", ha="center", va="center", fontsize=6.0,
                        color="#111" if abs(r.net_dir) < 0.28 else "white")
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(CATS)))
        ax.set_xticklabels(["Dom.", "Coord.", "Mixed"], fontsize=S.FS_TICK)
        ax.set_yticks(range(len(TRAITS)))
        ax.set_yticklabels([TRAIT_LABEL[t] for t in TRAITS] if ci == 0 else [], fontsize=7.2)
        ax.set_title(MODEL_LABEL[model], fontsize=8.0, fontweight="normal", color="#333", pad=4)
        ax.set_xticks(np.arange(-.5, len(CATS), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(TRAITS), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0); ax.tick_params(which="major", length=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    cbar.set_label("net Δ P(trait action) vs placebo", fontsize=S.FS_AXIS)
    cbar.ax.tick_params(labelsize=S.FS_TICK)
    fig.subplots_adjust(left=0.11, right=0.87, top=0.75, bottom=0.24)
    for ext in ("png", "pdf"):
        dpi = 600 if ext == "png" else 300
        fig.savefig(FIG_DIR / f"fig2_trait_by_category.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pg = load()
    summ = summarize(pg)
    summ.to_csv(TAB_DIR / "f2_trait_by_category.csv", index=False)
    figure(summ)
    safe = pg[pg.trait.isin(["risk", "loss", "maximin"])]
    print("Headroom mechanism (play-safe cues): corr(net steering, 1 - baseline P(a*)) per model:")
    for model in MODELS:
        s = safe[safe.model == model]
        if s.net_pref.notna().sum() > 5:
            r = np.corrcoef(s.net_pref, 1 - s.base_pref)[0, 1]
            print(f"  {MODEL_LABEL[model]:20s} r = {r:+.2f}  (n={len(s)})")
    print(f"\nwrote {FIG_DIR/'fig2_trait_by_category.png'} (+pdf)")
    print(f"wrote {TAB_DIR/'f2_trait_by_category.csv'}")


if __name__ == "__main__":
    main()
