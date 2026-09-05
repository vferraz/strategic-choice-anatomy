# Layer A Final — behaviour & rationality on the one-shot substrate

Paper-ready rebuild of Layer A (behaviour) on the corrected integrated substrate
`output/oneshot_akata_main_dl/`, isolated from the rest of the tree. Three main
figures + one supplementary, each backed by a CSV table, every headline number
printed with a game-clustered CI. Reads **only** the corrected one-shot substrate +
version-controlled metadata — no `design_v2`, `_legacy`, split one-shot roots, or
other old roots.

## How to run (deterministic, CPU only, minutes)

```bash
# 1. foundation (build once, validate before figures)
.venv/bin/python analysis/layer_a/build_master_df.py            # -> taxonomy/human_game_master_per_*.csv
.venv/bin/python analysis/layer_a/build_data_layer.py           # -> corrected generated/commit caches + audit
.venv/bin/python analysis/layer_a/build_trait_steering_oneshot.py  # -> _data/oneshot_trait_steering_*.csv
.venv/bin/python analysis/layer_a/validate_foundation.py        # GATE: corrected-source cache/lambda canaries

# 2. attribution (writes Fig 2b / appendix attribution tables)
.venv/bin/python analysis/layer_a/build_attribution.py

# 3. figures
.venv/bin/python analysis/layer_a/figscripts/fig1_rationality.py
.venv/bin/python analysis/layer_a/figscripts/fig1_human_corr.py
.venv/bin/python analysis/layer_a/figscripts/fig2_trait_steering.py
.venv/bin/python analysis/layer_a/figscripts/fig2_trait_by_category.py
.venv/bin/python analysis/layer_a/figscripts/fig3_qre_to_levelk.py

# or run the two orchestration notebooks (print every finding inline)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis/layer_a/notebooks/nb1_behavior_models.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis/layer_a/notebooks/nb2_attribution.ipynb
```

`build_data_layer.py` must run after `build_master_df.py`. It is a compatibility
wrapper around `src/shared_data.py`, which delegates to `src/corrected_substrate.py`.
Dense models read `decoded_action` gated by `parse_ok == True`; GPT-OSS reads
`realized_action` gated by `commit_type != "none"` and action in `{0,1}`. GPT-OSS
`commit_type == "mixed"` rows are included as resolved behaviour for now while
`commit_type`, `stated_p_act0`, and `prob_source` are retained as provenance/count
metadata. The legacy `_data/oneshot_unified_pairs.parquet` filename is overwritten
as a corrected alias of `fig1_panel.parquet` for old readers.

## Citations — nicknames are code-only; the manuscript must cite the sources

| code nickname | manuscript citation |
|---|---|
| `nagel` / MGN | **Moore, Germano & Nagel (2026)**, *Understanding human behavior via similarity: a geometric and behavioral rules-based approach to games*, UPF Economics WP 1942 (Barcelona School of Economics WP 1571). All 78 one-shot 2×2 games, payoffs from {1,2,3,4} without replacement, no feedback. |
| `griffiths` | **Zhu, Peterson, Enke & Griffiths (2025)**, *Capturing the complexity of human strategic decision-making with machine learning*, Nature Human Behaviour 9, 2114–2120. doi:10.1038/s41562-025-02230-5. >90,000 decisions over >2,400 procedurally generated games (initial play). |

Every figure caption and printed table label in the manuscript must use these
citations, never just "Nagel"/"Griffiths".

## Payoff-scale rule (HARD; enforced by build-time asserts)

- **LLMs (this study):** canonical rank 8-vector {1,2,3,4}, `payoff_multiplier=1`.
- **MGN:** rank 8-vector **{1,2,3,4}** — **same scale as the LLMs, fit directly, no rescaling.** (The old ×2/{2,4,6,8} note was wrong; do not reintroduce it.)
- **Griffiths:** **cardinal** payoffs → fit on a **separate axis** (rank-clean subset), never on the same λ bar as the rank agents.

`build_master_df.py` asserts at build time that every MGN `df_ros.game_vector` and
every LLM `canonical_8vec` is a permutation of {1,2,3,4} per 4-block, and that all
576 one-shot `config.json` carry `payoff_multiplier == 1`. It aborts otherwise.

## Honesty invariants (enforced in code)

1. **Canonical axis only.** Cross-game aggregates use `p_canon`/`pref_canon`/`delta1c`
   (canonical action axis), never `(decoded_action==0).mean()` pooled across games.
2. **Action-space λ is the headline, on ONE belief basis.** λ and all human comparisons use
   realized/generated canonical choices fit on a single **q=0.5** belief for every
   agent. LLM λ uses corrected-root counterbalance choices; Moore et al. λ uses
   the 451 complete sessions from 450 participants, matched by perspective. Soft-token λ is retained
   only as calibration/readout robustness.
3. **Nagel axis flip.** `nagel_frac_choose_act0_canonical` is P(act0), NOT P(canonical);
   it is flipped to `nagel_pcanon = where(canon==0, frac, 1-frac)` everywhere
   (mean P(canon) = 0.754, not the raw 0.528).
4. **Griffiths is cardinal → separate axis** (rank-clean subset only).
5. **Bootstrap-by-game CIs everywhere** (cluster = game, 2000 resamples, fixed seed 20260520);
   for the human λ, session-clustered uncertainty is also written to `f3_human_lambda.csv`.

## Foundation validation (gate; corrected integrated-root cache)

`validate_foundation.py` checks that the rebuilt caches use the corrected integrated
root and not slot argmax fallback. Current canaries:

| model | P(canonical) | λ, q=0.5 corrected-root |
|---|---:|---:|
| qwen | 0.714 | +2.04 |
| qwen_instruct | 0.726 | +1.92 |
| llama31_instruct | 0.740 | +1.89 |
| gptoss | 0.889 | +0.87 |

The baseline cell cache contains 2,304 usable P1 cells
(4 models × 144 games × 4 counterbalance cells). The corrected root has 576
`config.json` files with `payoff_multiplier == 1` and 32 rows per game-condition
file. GPT-OSS baseline has 51 mixed P1 rows included as resolved actions; no-commit
rows are dropped rather than replaced by argmax.

## Figure → table → finding map

### Figure 1 — strategic behaviour & rationality
- `figscripts/fig1_rationality.py` → `figures/fig1_regime_rationality.{png,pdf}` and
  `figures/figS_mp_descriptive.{png,pdf}`. Panel (a) is restricted to
  dominance-solvable games (DD/OD1/OD2) and reports chance-normalized proximity to the
  unique pure equilibrium; shaded bars are the soft policy readout. Panel (b) reports
  normalized payoff efficiency across all six families. Panel (c) shows both
  coordination structures: Pareto-rankable games as Pareto-best NE / other NE /
  miscoordination, and distributional-conflict games as coordinate / miscoordinate
  descriptive rates.
  MP games are descriptive only. Tables: `f1_dominance_conformity.csv`,
  `f1_payoff_efficiency.csv`, `f1_coordination.csv`, `f1_mp_descriptive.csv`.
- `figscripts/fig1_human_corr.py` → `figures/fig1c_human_corr.{png,pdf}` (expanded
  appendix version of the removed model-human scatter) and `figures/figS_alignment_matrix.{png,pdf}`
  (Supplementary: cross-agent per-game agreement). Table: `f1_human_corr.csv`.
  - **Findings:** corrected model-vs-Moore et al. Pearson r — Qwen 0.63,
    Qwen-I 0.62, Llama 0.54, GPT-OSS 0.36. This panel measures shared
    game-to-game variation, not competence: GPT-OSS has the highest canonical
    choice rate but lower per-game human alignment because its choices are more
    level-rule concentrated.

### Figure 2 — trait cues: response and attribution
- `figscripts/fig2_trait_steering.py` → `figures/fig2_trait_steering.{png,pdf}`
  Panels (a) cue-shift magnitude vs target alignment on conflict/identifiability sets,
  (b) TreeSHAP feature attribution for canonical conformity, (c) nested held-out AUC.
  Tables: `f2_trait_aim.csv`, `f2_nested_auc.csv`, `s_shap_beeswarm.csv`.
  - **Findings:** on identifiable conflict sets, capable models often move toward the cue
    target; game structure supplies the large predictive jump, while cue identity adds
    a smaller but reproducible held-out increment. Dense-primary AUC moves
    0.49 → 0.68 → 0.72 → 0.75; GPT-OSS full-sample AUC moves
    0.46 → 0.79 → 0.83 → 0.91.

### Figure 3 — the behavioural model: incentive response + fingerprint
- `figscripts/fig3_qre_to_levelk.py` → `figures/fig3_qre_to_levelk.{png,pdf}`.
  Panels (a) action-space quantal-response curve + faded soft-readout overlay, (b)
  bounded-rationality fingerprint (quantal level-k depth × QLk precision, log scale).
  Complexity response now lives in Fig. 1d; model-human alignment is supplementary.
  Tables: `f3_lambda.csv`, `f3_fingerprint.csv`, `f3_complexity_response.csv`,
  `f3_complexity_bins.csv`, `f3_complexity_partial.csv`.
  - **Findings:** action-space λ is the headline revealed-choice precision on the
    shared q=0.5 belief basis: Qwen 2.04, Qwen-I 1.92, Llama 1.89,
    Moore et al. 0.91, GPT-OSS 0.87. The fingerprint is a behavioural-model summary,
    not a one-dimensional rationality score; GPT-OSS hits the QLk precision boundary
    and shifts to a deeper revealed level.

### Supplementary — attribution & variance decomposition
- `build_attribution.py` → `figures/figS_attribution.{png,pdf}`. GroupKFold-by-game OOF
  HistGradientBoosting + shap.TreeExplainer (libomp-free stack). Tables:
  `s_variance_partition.csv`, `s_attribution_shap.csv`, `s_glmm.csv`.
  - **Findings (corrected):** signed-incentive SHAP dependence is positive (ρ≈+0.945);
    GLMM confirms Δ1c strongly increases canonical conformity. Variance partitioning is
    structure 60.5%, trait 26.1%, model 13.4%. The supplementary figure now carries
    the three remaining checks after the beeswarm moved to main Fig. 2.

## Data layer (built once, cached in `_data/`)

- `layerA_game_level.parquet` — per (model, game): `p_act0, pref0, p_canon, pref_canon,
  q2, delta1c` + game structure + canonical-aligned human refs. Built from corrected
  integrated-root decisions; no slot-argmax behaviour columns.
- `layerA_cells_p1baseline.parquet` — usable P1 corrected-root baseline cells for
  λ/QRE and validation.
- `fig1_panel.parquet` — corrected unified-pairs-shaped rows for Fig. 1; the legacy
  `oneshot_unified_pairs.parquet` filename is a corrected alias of this file.
- `oneshot_trait_steering_per_game.csv` / `_summary.csv` — one-shot reproduction of
  `trait_steering_proper.py`'s effect table (counterbalance-paired, round=1, 5 cues).

## Master dataframe (shared artifact, `datasets/processed/taxonomy/`)

`human_game_master_per_canonical.csv` (144 rows) + `human_game_master_per_game.csv`
(78 rows) — the long-pending single source with explicit per-source payoff columns
(`llm_rank_8vec`, `mgn_rank_8vec`, `grif_rank_clean`, `payoff_scale_*`) and the
axis-flipped `nagel_pcanon`. Build-time asserts make the scale provenance unambiguous.
