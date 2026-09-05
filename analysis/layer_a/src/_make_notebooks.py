#!/usr/bin/env python3
"""Generate the two thin Layer-A notebooks (nb1, nb2) from the src/ builders.

The notebooks are deliberately thin: every computation lives in the importable
``analysis.layer_a.src`` modules (cache-aware), so the notebooks import them,
print findings inline, and embed the figures. Execute with:
    .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
        analysis/layer_a/notebooks/nb1_behavior_models.ipynb
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                      "language_info": {"name": "python"}}
    return nb


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


HEADER = """import os, sys
from pathlib import Path
ROOT = Path.cwd()
os.chdir(ROOT)
from IPython.display import Image, display
FIG = ROOT / "analysis" / "layer_a" / "figures"
"""


def nb1():
    cells = [
        md("# Notebook 1 — Layer A behaviour & behavioural models (one-shot)\n\n"
           "**Paper-final, isolated.** The 144 one-shot canonical games. The realized choice is "
           "the **corrected integrated-root realized action** from "
           "`$SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet`: dense rows use "
           "`decoded_action` gated by `parse_ok`, and GPT-OSS uses `realized_action` with "
           "`commit_type != 'none'` (mixed rows retained as resolved actions). No argmax decoder, "
           "no round-based / design_v2 data, no legacy 76-game set. `pref0` (soft) is a "
           "model-internal robustness readout only.\n\n"
           "**Citations:** Nagel = Moore, Germano & Nagel (2026); Griffiths = Zhu, Peterson, "
           "Enke & Griffiths (2025).\n\n"
           "**Honesty invariants:** canonical action axis for all cross-game aggregates; "
           "generate()/commit decode (never slot argmax); human λ on MGN *individual* decisions (binary↔binary with "
           "the model's generated move); bootstrap-by-game CIs (2000, seed 20260520)."),
        code(HEADER),
        md("## Shared data layer (cached)\nBuilds `_data/layerA_game_level.parquet`, the "
           "cell-level P1 table, and the Fig-1 panel. Build-time asserts: 144 MGN vectors are "
           "{1,2,3,4} permutations; LLM `payoff_multiplier=1`."),
        code("from analysis.layer_a.src import shared_data as SD\nSD.main()"),
        md("## Figure 1 — strategic behaviour & rationality\n"
           "(a)/(b) reuse the frozen `fig_main_rationality_v2` body (data-source re-pointed only); "
           "(c) model-vs-Nagel per-game P(canonical) scatter; (d) cross-agent alignment matrix."),
        code("from analysis.layer_a.src import fig1_rationality as F1\nF1.main()"),
        code("for f in ['fig_rationality_by_class_v2clean_raw.png','fig1c_human_scatter.png','fig1d_alignment.png']:\n"
             "    display(Image(filename=str(FIG/f)))"),
        md("## Figure 2 — personality cues: magnitude vs direction\n"
           "Frozen `fig_main_trait_steering_v2` body re-pointed to one-shot trait effects; panel "
           "(c) variance partition is the ONE-SHOT number. Inequity/selfish/maximin aim toward the "
           "target; risk/loss are anti-aimed (shift toward the equality action)."),
        code("from analysis.layer_a.src import trait_effects as TE\nTE.main()\n"
             "from analysis.layer_a.src import fig2_trait_steering as F2\nF2.main()"),
        code("display(Image(filename=str(FIG/'fig_main_trait_steering_v2clean.png')))"),
        md("## Figure 3 — the behavioural model: QRE → level-k\n"
           "(a) quantal response curve + precision λ (hard/Generate, q=0.5; human overlaid); "
           "(b) bounded-rationality fingerprint (depth × precision). Model-human alignment and "
           "leave-games-out model selection live in the supplement. **Scale:** LLMs and MGN both on ranks {1,2,3,4} — λ "
           "directly comparable; Griffiths cardinal lives on a separate axis."),
        code("from analysis.layer_a.figscripts import fig3_qre_to_levelk as F3\nF3.main()"),
        code("display(Image(filename=str(FIG/'fig3_qre_to_levelk.png')))"),
        md("All numeric findings are printed above and saved under "
           "`analysis/layer_a/tables/` (f1_*, f2_*, f3_*). See README.md for the "
           "figure→table→finding map and the honesty/scale notes."),
    ]
    return _nb(cells)


def nb2():
    cells = [
        md("# Notebook 2 — attribution & variance decomposition (Supplementary)\n\n"
           "Rebuild of `layer1_conformity_attribution` on the one-shot **corrected integrated-root "
           "decisions** (GPT-OSS mixed rows retained as resolved actions). Produces the Supplementary figure and the "
           "partition the main text cites. Gradient-boosted conformity prediction with "
           "GroupKFold-by-game OOF; TreeSHAP of |Δ1ᶜ|; variance decomposition (structure vs trait "
           "vs model); GLMM confirm."),
        code(HEADER),
        code("from analysis.layer_a.src import attribution as A\nA.main()"),
        code("display(Image(filename=str(FIG/'figS_attribution.png')))"),
        md("Tables: `s_variance_partition.csv`, `s_attribution_shap.csv`, `s_glmm.csv`.\n\n"
           "**One-shot vs design_v2 (flag for the writeup):** the trait block is much larger on "
           "one-shot (ΔAUC ≈0.10 vs ≈0.01); structure > trait but not ≫; and **both selfish-max and "
           "maximin** are positive trait families (design_v2 reported only maximin positive)."),
    ]
    return _nb(cells)


def main():
    NBDIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb1(), NBDIR / "nb1_behavior_models.ipynb")
    nbf.write(nb2(), NBDIR / "nb2_attribution.ipynb")
    print("wrote", NBDIR / "nb1_behavior_models.ipynb")
    print("wrote", NBDIR / "nb2_attribution.ipynb")


if __name__ == "__main__":
    main()
