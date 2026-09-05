#!/usr/bin/env python3
"""
Aggregated token output-projection heatmap, one panel per model (house style, paper grade).

For each model the per-token impact (the order-dependent increment in canonical lens score) is averaged
at every structural position across all 144 games (cb=0, L79, baseline). The prompt template is
identical across games, so position N is the same slot everywhere; averaging exposes what
SYSTEMATICALLY moves each model's decision, and option-letter tokens wash to ~0 (echo cancels).

Output: figures/fig_token_heatmap_3panel.{png,pdf}
"""
import os, sys
import numpy as np, pandas as pd, glob
import matplotlib
from strategic_anatomy.config import layerc_root
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
from strategic_anatomy import paper_style as S  # noqa: E402
S.apply()

ROOT = str(layerc_root())
FIG = os.path.join(HERE, "figures")
MODELS = ["qwen", "qwen_instruct", "llama31_instruct"]
NAME = {"qwen": "Qwen2.5", "qwen_instruct": "Qwen2.5-I", "llama31_instruct": "Llama"}
NEED = ["game_code", "cb_id", "condition", "layer", "token_index", "token_str", "region", "score_canonical"]
MAXCOL, FS, VMAX = 100, 8.0, 1.2
CMAP = LinearSegmentedColormap.from_list("impact", ["#2166ac", "#f5f4f1", "#b2182b"])  # away(blue)-neutral-toward(red)
NORM = Normalize(-VMAX, VMAX)
INK = S.INK


def aggregate(model):
    per_game, ref_tok = [], None
    for f in sorted(glob.glob(f"{ROOT}/{model}/*/tokens.parquet")):
        d = pd.read_parquet(f, columns=NEED)
        d = d[(d.layer == 79) & (d.condition == "baseline") & (d.cb_id == 0)].sort_values("token_index")
        if d.empty:
            continue
        incr = np.diff(d["score_canonical"].values, prepend=d["score_canonical"].values[0]); incr[0] = 0.0
        per_game.append(incr)
        if ref_tok is None or len(d) > len(ref_tok):
            ref_tok = d["token_str"].tolist()
    # phase-4: without the deposit this used to die on `min() arg is an empty sequence`.
    if not per_game:
        raise FileNotFoundError(
            f"no Layer-C token tables for model '{model}' under {ROOT}/{model}/*/tokens.parquet.\n"
            "This figure needs the released layerc component. Fetch it with\n"
            "    python scripts/download_data.py --component layerc\n"
            "or point SCA_DATA_ROOT at an existing copy."
        )
    L = min(len(x) for x in per_game)
    M = np.vstack([x[:L] for x in per_game])
    return ref_tok[:L], M.mean(axis=0), M.sum(axis=1).mean(), len(per_game)


def layout(tokens, vals):
    placed, r, c = [], 0, 0
    for tok, v in zip(tokens, vals):
        parts = tok.split("\n")
        for k, p in enumerate(parts):
            if p:
                w = len(p)
                if c + w > MAXCOL and c > 0:
                    r += 1; c = 0
                placed.append((r, c, p, v)); c += w
            if k < len(parts) - 1:
                r += 1; c = 0
    return placed, r + 1


DATA = {m: aggregate(m) for m in MODELS}
layouts = {m: layout(DATA[m][0], DATA[m][1]) for m in MODELS}
heights = [layouts[m][1] for m in MODELS]

fig_h = 0.145 * sum(heights) + 0.34 * len(MODELS) + 0.80
fig = plt.figure(figsize=(S.NHB_TEXTWIDTH_IN, fig_h))
gs = fig.add_gridspec(len(MODELS), 1, height_ratios=heights,
                      hspace=0.22, left=0.015, right=0.985, top=0.965, bottom=0.095)

for i, m in enumerate(MODELS):
    ax = fig.add_subplot(gs[i]); ax.axis("off")
    placed, nr = layouts[m]
    _, _, final, ng = DATA[m]
    for (r, c, p, v) in placed:
        ax.text(c, r, p, family="monospace", fontsize=FS, va="center", ha="left", color=INK,
                bbox=dict(boxstyle="square,pad=0.04", fc=CMAP(NORM(v)), ec="none"))
    ax.set_xlim(-1, MAXCOL + 1); ax.set_ylim(nr - 0.4, -0.9)
    S.nhb_panel_title(ax, "abc"[i], NAME[m], letter_x=0.0, title_x=0.052, y=1.05, fontsize=S.NHB_FS_PANEL)
    ax.text(1.0, 1.05, f"mean net signal {final:+.2f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=S.NHB_FS_FOOT, color=S.FAINT)

cax = fig.add_axes([0.34, 0.050, 0.32, 0.016])
cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal",
                  ticks=[-VMAX, 0, VMAX])
cb.set_ticklabels(["away", "0", "toward canonical"])
cb.ax.tick_params(labelsize=S.NHB_FS_FOOT, length=1.6, width=0.5, colors=INK)
cb.outline.set_linewidth(0.4)
cb.set_label("mean token increment in canonical output projection (L79)",
             fontsize=S.NHB_FS_FOOT, color=S.FAINT, labelpad=2.5)

with plt.rc_context({"savefig.bbox": "tight"}):
    fig.savefig(os.path.join(FIG, "fig_token_heatmap_3panel.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG, "fig_token_heatmap_3panel.png"), dpi=600)
plt.close(fig)
print("wrote fig_token_heatmap_3panel; net signals:", {m: round(float(DATA[m][2]), 2) for m in MODELS})
