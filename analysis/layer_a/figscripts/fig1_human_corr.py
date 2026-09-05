#!/usr/bin/env python3
"""Model-vs-human per-game agreement (expanded companion to Figure 1d).

Per model, P(canonical action) on each game vs the Moore et al. human reference
(Moore, Germano & Nagel 2026), with Pearson + Spearman r and game-clustered bootstrap
CIs. Both axes are on the canonical action axis (docs/METHODS.md HC-2); Moore et al. is already
axis-flipped to P(canonical) in the master df.

Supplementary: cross-agent alignment matrix — per-game P(canonical) correlations across
the four LLMs + the Moore et al. human reference. These correlations measure shared
game-to-game variation, so near-ceiling policies can correlate weakly despite high
equilibrium conformity.

Adopts the FROZEN palette/typography (analysis/_shared/_paper_style + the Fig-1
model-colour map) so it sits seamlessly beside panels (a)/(b).
Outputs:
  figures/fig1c_human_corr.{png,pdf}
  figures/figS_alignment_matrix.{png,pdf}   (supplementary)
  tables/f1_human_corr.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategic_anatomy import paper_style as S  # noqa: E402
S.apply()
from analysis.layer_a.build_regime_rationality import build_game_panel  # noqa: E402
from strategic_anatomy.config import repo_root, results_root, taxonomy_dir

ROOT = repo_root()


LAYERA = ROOT / "analysis" / "layer_a"
MASTER = taxonomy_dir() / "human_game_master_per_canonical.csv"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
TAB_DIR = results_root() / "layer_a"

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
DISPLAY = {"qwen": "Qwen2.5-72B", "qwen_instruct": "Qwen2.5-72B-Instruct",
           "llama31_instruct": "Llama-3.1-70B-Instruct", "gptoss": "GPT-OSS-120B",
           "nagel": "Moore et al. (human)"}
COLOR = {"qwen": "#1f77b4", "qwen_instruct": "#17becf", "llama31_instruct": "#2ca02c",
         "gptoss": "#9467bd", "nagel": "#d62728"}
RNG = np.random.default_rng(20260520)


def _boot_r(x, y, n=2000):
    """Game-clustered (here games are the unit) bootstrap CI on Pearson r."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    r = float(pearsonr(x, y)[0])
    boots = []
    for _ in range(n):
        idx = RNG.integers(0, len(x), size=len(x))
        if np.std(x[idx]) < 1e-9 or np.std(y[idx]) < 1e-9:
            continue
        boots.append(pearsonr(x[idx], y[idx])[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return r, float(lo), float(hi)


def load_wide() -> pd.DataFrame:
    panel, _ = build_game_panel()
    llm = panel[panel["agent"].isin(MODELS)].copy()
    llm["p_canon"] = np.where(
        llm["canonical_action_p1"].astype(int).eq(0),
        llm["realized_p1"],
        1.0 - llm["realized_p1"],
    )
    wide = llm.pivot(index="game_code", columns="agent", values="p_canon")
    master = pd.read_csv(MASTER).set_index("game_code")
    wide["nagel"] = master["nagel_pcanon"]
    return wide


def panel_c(wide: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.2))
    rows = []
    for ax, model in zip(axes.ravel(), MODELS):
        d = wide[[model, "nagel"]].dropna()
        x, y = d["nagel"].to_numpy(), d[model].to_numpy()
        r, lo, hi = _boot_r(x, y)
        rho = float(spearmanr(x, y)[0])
        ax.axline((0, 0), slope=1, color=S.GREY, lw=0.7, ls=":", zorder=1)
        ax.scatter(x, y, s=14, color=COLOR[model], alpha=0.7, edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(DISPLAY[model], fontsize=7.2, color="#333", pad=3)
        ax.text(0.04, 0.93, f"r = {r:.2f} [{lo:.2f}, {hi:.2f}]\n" + r"$\rho$ = " + f"{rho:.2f}   n={len(d)}",
                transform=ax.transAxes, fontsize=6.2, va="top", color=S.INK)
        ax.tick_params(labelsize=S.FS_TICK)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        rows.append(dict(model=model, agent=DISPLAY[model], n_games=len(d),
                         pearson_r_hard=r, pearson_lo=lo, pearson_hi=hi,
                         spearman_rho_hard=rho))
    fig.supxlabel("Moore et al. human  P(canonical action)", fontsize=S.FS_AXIS)
    fig.supylabel("Model  P(canonical action)", fontsize=S.FS_AXIS)
    fig.text(0.013, 0.985, "(c)  Model vs human, game by game", fontsize=S.FS_PANEL,
             fontweight="bold", color=S.INK, ha="left", va="top")
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig1c_human_corr.{ext}", dpi=600 if ext == "png" else 300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def supp_matrix(wide: pd.DataFrame) -> None:
    cols = MODELS + ["nagel"]
    C = wide[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(C.to_numpy(), vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    short = {"qwen": "Qwen", "qwen_instruct": "Qwen-I", "llama31_instruct": "Llama",
             "gptoss": "GPT-OSS", "nagel": "Moore"}
    ax.set_xticklabels([short[c] for c in cols], rotation=35, ha="right", fontsize=S.FS_TICK)
    ax.set_yticklabels([short[c] for c in cols], fontsize=S.FS_TICK)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{C.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.0, color="#222" if C.iloc[i, j] < 0.6 else "white")
    ax.set_title("Per-game P(canonical) agreement", fontsize=S.FS_PANEL, color=S.INK, pad=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=S.FS_TICK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figS_alignment_matrix.{ext}", dpi=600 if ext == "png" else 300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    wide = load_wide()
    tab = panel_c(wide)
    supp_matrix(wide)
    tab.to_csv(TAB_DIR / "f1_human_corr.csv", index=False)
    print(f"wrote {FIG_DIR/'fig1c_human_corr.png'}, {FIG_DIR/'figS_alignment_matrix.png'}")
    print(f"wrote {TAB_DIR/'f1_human_corr.csv'}")
    print("\n=== model vs Moore et al., per-game P(canonical) ===")
    print(tab[["agent", "n_games", "pearson_r_hard", "pearson_lo", "pearson_hi",
               "spearman_rho_hard"]].round(3).to_string(index=False))
    print("\n=== cross-agent alignment matrix (Pearson) ===")
    print(wide[MODELS + ["nagel"]].corr().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
