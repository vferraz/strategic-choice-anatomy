#!/usr/bin/env python3
"""Generate the two Layer-A-Final orchestration notebooks (nb1, nb2).

Each notebook runs the analysis scripts as subprocesses (so every printed finding
lands inline), then displays the rendered figures and the backing CSV tables. The
notebooks are thin orchestrators — the real, reproducible analysis lives in the
sibling *.py scripts. Execute with:

  jupyter nbconvert --to notebook --execute --inplace analysis/layer_a/notebooks/nb*.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "analysis" / "layer_a" / "notebooks"

# walk up to the repo root so relative paths work regardless of nbconvert cwd
BOOTSTRAP = (
    "import os, sys, subprocess\n"
    # phase-4: the marker was CLAUDE.md, the private repo's agent-context file, which is not
    # part of the release — every generated notebook would have walked all the way to '/'.
    "while not os.path.exists('pyproject.toml') and os.getcwd() != '/':\n"
    "    os.chdir('..')\n"
    "ROOT = os.getcwd(); print('repo root:', ROOT)\n"
    "PY = sys.executable\n"
    "def run(script):\n"
    "    r = subprocess.run([PY, script], capture_output=True, text=True)\n"
    "    print(r.stdout)\n"
    "    if r.returncode != 0:\n"
    "        print('STDERR:', r.stderr[-2000:])\n"
)
SHOW = (
    "from IPython.display import Image, display\n"
    "import pandas as pd\n"
    "def show_fig(name):\n"
    "    display(Image(filename=f'analysis/layer_a/figures/{name}'))\n"
    "def show_table(name, n=12):\n"
    "    df = pd.read_csv(f'analysis/layer_a/tables/{name}')\n"
    "    print(name, df.shape); display(df.head(n))\n"
)


def nb1() -> nbf.NotebookNode:
    c = []
    c.append(new_markdown_cell(
        "# Notebook 1 — Behavioural models (Figures 1, 2, 3)\n\n"
        "Layer A on the corrected one-shot substrate `$SCA_DATA_ROOT/substrate/`. Builds the cached data\n"
        "layer, then renders the three main figures, **printing every finding inline**.\n\n"
        "**Citations** (nicknames are code-only): `nagel` → Moore, Germano & Nagel (2026), UPF WP 1942;\n"
        "`griffiths` → Zhu, Peterson, Enke & Griffiths (2025), Nat. Hum. Behav. 9:2114.\n\n"
        "**Honesty invariants:** canonical action axis only; action-space λ headline (soft λ = calibration);\n"
        "Nagel axis-flipped to P(canonical); Griffiths cardinal on a separate axis; bootstrap-by-game CIs."))
    c.append(new_code_cell(BOOTSTRAP + SHOW))

    c.append(new_markdown_cell(
        "## Foundation — master df, data layer, trait table, validation gate\n"
        "Reproduces the `oneshot_audit_v2` matched76 λ before any figure is built."))
    c.append(new_code_cell(
        "run('analysis/layer_a/build_master_df.py')\n"
        "run('analysis/layer_a/build_data_layer.py')\n"
        "run('analysis/layer_a/build_trait_steering_oneshot.py')\n"
        "run('analysis/layer_a/build_attribution.py')        # writes s_variance_partition.csv for Fig 2c\n"
        "run('analysis/layer_a/validate_foundation.py')"))

    c.append(new_markdown_cell(
        "## Figure 1 — strategic behaviour & rationality\n"
        "(a) dominance-solvable games only: chance-normalized conformity to the unique pure equilibrium;\n"
        "(b) normalized payoff efficiency across regimes; (c) coordination games split into Pareto-rankable\n"
        "selection and distributional-conflict descriptive rates; (d) model-vs-human per-game\n"
        "P(canonical) scatter. MP remains descriptive only."))
    c.append(new_code_cell(
        "run('analysis/layer_a/figscripts/fig1_rationality.py')\n"
        "run('analysis/layer_a/figscripts/fig1_human_corr.py')"))
    c.append(new_code_cell(
        "show_fig('fig1_regime_rationality.png')\n"
        "show_fig('figS_mp_descriptive.png')\n"
        "show_fig('fig1c_human_corr.png')   # expanded standalone version of panel d\n"
        "show_fig('figS_alignment_matrix.png')\n"
        "show_table('f1_dominance_conformity.csv')\n"
        "show_table('f1_payoff_efficiency.csv')\n"
        "show_table('f1_coordination.csv')\n"
        "show_table('f1_human_corr.csv')"))

    c.append(new_markdown_cell(
        "## Figure 2 — personality cues: magnitude vs direction\n"
        "(a) magnitude vs aim per trait×model (risk/loss **anti-aimed** → shift toward the equality action);\n"
        "(b) incentive-gating; (c) one-shot variance partition. Companion: steering-by-category heatmap."))
    c.append(new_code_cell(
        "run('analysis/layer_a/figscripts/fig2_trait_steering.py')\n"
        "run('analysis/layer_a/figscripts/fig2_trait_by_category.py')"))
    c.append(new_code_cell(
        "show_fig('fig2_trait_steering.png')\n"
        "show_fig('fig2_trait_by_category.png')\n"
        "show_table('f2_trait_aim.csv')"))

    c.append(new_markdown_cell(
        "## Figure 3 — the behavioural model: QRE → level-k\n"
        "(a) action-space quantal-response curve + faded soft-readout overlay; (b) bounded-rationality\n"
        "fingerprint (depth × precision). Model-human alignment and model-selection grids are supplementary; soft λ is\n"
        "calibration/readout robustness; Griffiths (cardinal) stays on a separate robustness axis."))
    c.append(new_code_cell("run('analysis/layer_a/figscripts/fig3_qre_to_levelk.py')"))
    c.append(new_code_cell(
        "show_fig('fig3_qre_to_levelk.png')\n"
        "show_table('f3_lambda.csv')\n"
        "show_table('f3_fingerprint.csv')"))

    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def nb2() -> nbf.NotebookNode:
    c = []
    c.append(new_markdown_cell(
        "# Notebook 2 — Attribution & variance decomposition (Supplementary)\n\n"
        "One-shot rebuild of `layer1_conformity_attribution`. GroupKFold-by-game OOF\n"
        "HistGradientBoosting + shap.TreeExplainer (libomp-free stack), variance partition of\n"
        "canonical conformity (game structure vs trait vs model identity), and a cluster-robust GLM\n"
        "(|Δ1c| ↑ conformity, preference family ↓, maximin the one positive trait family)."))
    c.append(new_code_cell(BOOTSTRAP + SHOW))
    c.append(new_code_cell("run('analysis/layer_a/build_attribution.py')"))
    c.append(new_code_cell(
        "show_fig('figS_attribution.png')\n"
        "show_table('s_variance_partition.csv')\n"
        "show_table('s_glmm.csv')\n"
        "show_table('s_attribution_shap.csv')"))
    c.append(new_markdown_cell(
        "**Reading:** the variance partition restates the trait section's headline (structure and model\n"
        "identity dominate; trait prompts are a smaller, real share). The GLMM confirms the incentive\n"
        "gradient and the preference-vs-maximin asymmetry on the one-shot substrate."))
    nb = new_notebook(cells=c)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> int:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb1(), NB_DIR / "nb1_behavior_models.ipynb")
    nbf.write(nb2(), NB_DIR / "nb2_attribution.ipynb")
    print(f"wrote {NB_DIR/'nb1_behavior_models.ipynb'}")
    print(f"wrote {NB_DIR/'nb2_attribution.ipynb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
