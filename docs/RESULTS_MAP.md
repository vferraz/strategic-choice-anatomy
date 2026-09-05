# Results map

An index from **what the paper says** to **the file that holds the number** and **the script that
produced it**. This document deliberately carries **no numbers of its own** — the committed tables
under `data/results/` are the authority, so this index cannot drift away from them.

To rebuild a figure or regenerate a table, see [`../REPRODUCING.md`](../REPRODUCING.md).
For what the numbers mean and what may not be claimed from them, see [`METHODS.md`](METHODS.md).

Reference renders of all 11 paper figures are committed at `data/results/figures_reference/`, so
you can diff your rebuild against ours.

---

## Layer A — behaviour

*Do models choose like strategic agents?*

| Claim area | Table(s) under `data/results/layer_a/` | Producer |
|---|---|---|
| Alignment with the canonical action, by game class | `f1_rationality_by_class.csv`, `f1_alignment_matrix.csv` | `analysis/layer_a/build_regime_rationality.py` |
| Dominance conformity | `f1_dominance_conformity.csv` | same |
| Coordination and matching-pennies behaviour | `f1_coordination.csv`, `f1_mp_descriptive.csv` | same |
| Payoff efficiency and payoff gap | `f1_payoff_efficiency.csv`, `f1_payoff_gap.csv` | same |
| Complexity response | `f1_complexity_response.csv`, `f3_complexity_bins.csv`, `f3_complexity_partial.csv`, `f3_complexity_response.csv` | `build_regime_rationality.py`, `analysis/layer_a/src/fig3_behavioral_model.py` |
| Correlation with human choice | `f1_human_corr.csv` | `analysis/layer_a/build_data_layer.py` |
| Precision λ (QRE) per agent | `f3_lambda.csv` | `analysis/layer_a/src/fig3_behavioral_model.py` |
| Human λ (fitted on **individual** choices) | `f3_human_lambda.csv` | `analysis/layer_a/src/human_lambda_mgn.py` — needs the raw Nagel export |
| Duplicate-session robustness of the human λ | `human_dedup_robustness.csv` | `analysis/layer_a/src/human_dedup_robustness.py` — needs the raw Nagel export |
| QRE vs level-k model selection | `f3_model_selection_cv.csv`, `f3_fingerprint.csv` | `analysis/layer_a/src/model_selection_cv.py` |
| Griffiths cardinal robustness | `f3_griffiths_cardinal.csv` | `analysis/layer_a/build_data_layer.py` |
| Decision-rule classification | `b_rule_fit.csv`, `b_winning_rule.csv`, `b_rule_validation.csv`, `b_human_rule_reference.csv` | `analysis/layer_a/build_rule_classification.py` |
| Trait-cue steering, aim and gating | `f2_trait_aim.csv`, `f2_trait_gating.csv`, `f2_trait_by_category.csv`, `f2_nested_auc.csv` | `analysis/layer_a/build_trait_steering_oneshot.py` |
| Attribution: SHAP, GLMM, variance partition | `s_attribution_shap.csv`, `s_glmm.csv`, `s_variance_partition.csv`, `s_shap_beeswarm.csv`, `s_shap_delta1_dependence.csv` | `analysis/layer_a/build_attribution.py` |

Narrative: `analysis/layer_a/README.md`, `analysis/layer_a/LAYER_A_FINDINGS.md`.

---

## Layer B — representation

*Is the game encoded, and does the model recruit that encoding?*

| Claim area | Table(s) under `data/results/layer_b/` | Producer |
|---|---|---|
| Decodability of game structure from residuals | `b1_decodability.csv` | `analysis/layer_b/build_decodability.py` |
| Equilibrium-structure 3-class probe | `b1_equilibrium_structure.csv` | same |
| Crystallization (depth at which the decision becomes decodable) | `b1_crystallization.csv`, `b1_crystallization_summary.csv` | `analysis/layer_b/build_crystallization.py` |
| GPT-OSS router audit: coverage, decodability, behaviour link | `router_oneshot_coverage.csv`, `router_oneshot_decodability.csv`, `router_oneshot_behavior_link.csv`, `router_oneshot_decision_process.csv`, `router_oneshot_best_summary.csv`, `router_oneshot_audit_report.md` | `analysis/layer_b/router_oneshot_audit.py` |
| Router bottleneck | `router_bottleneck_depth.csv`, `router_bottleneck_summary.csv`, `router_bottleneck_game_level.csv` | `analysis/layer_b/router_bottleneck_analysis.py` |

**Fusion / associative geometry** — `data/results/layer_b/fusion/`:

| Claim area | Table(s) | Producer |
|---|---|---|
| Depth-resolved fusion angle | `fusion_depth_table.csv`, `fusion_depth_empirical.csv`, `fusion_summary.csv` | `analysis/layer_b/fusion_associative/build_fusion_figures.py` (GPT-OSS legs read the uniform-site cache from `recap_cache.py`) |
| GPT-OSS uniform-site cache | `$SCA_DATA_ROOT/layer_b_cache/recap_baseline/` (derived, not committed) | `analysis/layer_b/fusion_associative/recap_cache.py` |
| GPT-OSS router fusion | `oss_router_fusion.csv`, `oss_router_fusion_sensitivity.csv` | `analysis/layer_b/fusion_associative/oss_router_fusion.py` |
| **Capture-site sensitivity** (all four historical scopes) | `gptoss_capture_site_sensitivity.csv` | `analysis/layer_b/fusion_associative/build_fusion_figures.py` (uniform-site scope needs `recap_cache.py` first; see [`AMENDMENT_uniform_site_geometry.md`](AMENDMENT_uniform_site_geometry.md)) |
| Null spread vs geometry | `null_spread_vs_geometry.csv` | `analysis/layer_b/fusion_associative/confirm_null_spread.py` |
| Pre-amendment originals (labeled sensitivities) | `_pre_uniform_site_20260713/` | — |

**Recruitment bridge** — `data/results/layer_b/recruitment/` and `.../rebuild/`:

| Claim area | Table(s) | Producer |
|---|---|---|
| Representation → behaviour bridge, per model | `rebuild/bridge_{model}.csv`, `rebuild/bridge_points_gptoss.csv` | `analysis/layer_b/rebuild/rebuild_bridge.py` |
| Bridge variants (the released specification) | `rebuild/bridge_variants_{model}.csv` | `analysis/layer_b/rebuild/rebuild_bridge_variants.py` |
| Base-vs-instruct recruitment contrast | `recruitment/bridge_contrast.csv`, `recruitment/within_model_bridge.csv`, `recruitment/within_model_bridge_points.csv` | `analysis/layer_b/recruitment/build_all.py` |
| Final geometry bridge | `recruitment/final_geometry_bridge.csv` | same |
| Depth-resolved recruitment geometry | `recruitment/recruitment_geometry_depth.csv` (+ `_data/*.parquet`) | `analysis/layer_b/fusion_associative/build_fusion_figures.py` — this table is the uniform-belief fusion table under renamed columns, for every model; `recruitment/build_all.py` reads it rather than deriving it |
| Disposition dissociation | `recruitment/disposition_dissociation.csv` | same |
| Decision crystallization (recruitment view) | `recruitment/decision_crystallization.csv` | same |
| Representation inventory | `recruitment/representation_inventory.csv` | same |
| GPT-OSS sensitivity | `recruitment/within_model_bridge_gptoss_sensitivity.csv` | same |
| Method guardrails, causal/router status | `recruitment/method_guardrails.csv`, `recruitment/causal_status.csv`, `recruitment/router_status.csv` | same |
| Round-2 geometry summary | `rebuild/round2_geometry_summary.csv` | `analysis/layer_b/rebuild/rebuild_round2.py` |

Narrative: `analysis/layer_b/README.md`, `analysis/layer_b/METHODS_AND_FINDINGS.md`,
`analysis/layer_b/fusion_associative/FUSION_AND_STORYLINE_MEMO.md`,
`analysis/layer_b/rebuild/REBUILD_ALL_SPEC.md`, `analysis/layer_b/recruitment/README.md`.

---

## Layer C — token attribution

*Where in the prompt does the strategic signal enter?*

| Claim area | Table(s) under `data/results/layer_c/` | Producer |
|---|---|---|
| Per-token logit-lens readouts | `decision_readouts.csv` | `analysis/layer_c/compute_layerc.py` |
| Per-region signed contribution | `region_contrib.csv` → `stat_region_contrib.csv` | `compute_layerc.py` → `analysis/layer_c/stats_layerc.py` |
| Neural λ vs behavioural λ (the dissociation contrast) | `stat_neural_lambda.csv` | `stats_layerc.py` |
| Own- vs opponent-payoff attention | `stat_ownopp.csv` | same |
| Depth position of the readout | `stat_depth_position.csv` | same |
| "Heard vs obeyed" | `stat_heard_obeyed.csv` | same |
| Strategic payoff cell | `stat_strategic_payoff.csv` | same |
| Placebo (length-match null) | `stat_placebo_null.csv` | same |
| Incentive Δ₁ᶜ per game | `incentive_delta1c.csv` | `compute_layerc.py` |
| Probe vs lens bridge | `bridge_probe_vs_lens.csv` | `analysis/layer_c/probe_bridge.py` |
| Token-level SHAP payload | `token_shap_payload.json` | `analysis/layer_c/fig_token_shap.py` |

Narrative: `analysis/layer_c/README.md`, `analysis/layer_c/LAYER_C_METHODS.md`,
`analysis/layer_c/LAYER_C_FINDINGS.md`.

---

## Causal steering

*Does intervening on the representation change the decision?*

The released incentive arm is the **corrected q = 0.5** one. Everything in the `_q05` tables below
is what the paper reports; the unsuffixed tables are the **pre-correction empirical-belief** arm,
kept as labelled history. See [`METHODS.md`](METHODS.md) §6.4.

**Released arm (q = 0.5)** — `data/results/steering/`:

| Claim area | Table(s) | Producer |
|---|---|---|
| **Headline small-dose results, by family** | `smalldose_summary_q05/family_canonical.csv` | `analysis/steering/smalldose_final_summary.py --in-root .../smalldose_q05 --delta-suffix _q05 --out ...` |
| Per-game slopes, per-cell ranges, flip rates | `smalldose_summary_q05/game_slopes.csv` | same |
| Overall summary | `smalldose_summary_q05/summary.csv` | same |
| h1 vs h2 cross-mode correlation | `smalldose_summary_q05/crossmode_r.csv` | same |
| **Permutation matched-control** | `smalldose_summary_q05/perm_null_results.csv` | `analysis/steering/perm_null_analysis.py --perm-root .../perm_q05 --game-slopes .../smalldose_summary_q05/game_slopes.csv --out ...` |
| **Apparatus geometry + the cos = 1.000 refit self-check** | `smalldose_summary_q05/apparatus_geometry_q05.{csv,json}` | `analysis/steering/apparatus_geometry_q05.py` |
| δ₁ᶜ moderation deltas, objective target | `summary/delta_tables_{model}_q05.csv` | **see note below** |
| Direction provenance (`incentive_belief: q05`, per-vector sha256) | `data/manifests/directions/akata_q05{,_perp,_perm}/{model}/manifest.json` | `steering/extract_directions.py` (see note) |

**Producer note.** The three `delta_tables_{model}_q05.csv` and the q05 direction sets were produced
on the GPU machine by an extractor invocation this repository does not yet expose: the released
`steering/extract_directions.py` has no `--belief` switch. Closing that is a tracked release item;
until it lands these are shipped **data with an external producer**, and the standing evidence that
they are the objective-target artifacts is `apparatus_geometry_q05.py`'s refit self-check
(cos = 1.000000, §7) plus `incentive_belief: "q05"` in the tracked manifests. Note also that those
manifests' `git_commit` field records the GPU machine's checked-out commit, which does **not**
contain the code that produced them.

**Pre-correction arm (empirical belief) — historical, retained unmodified:**

| Claim area | Table(s) | Producer |
|---|---|---|
| Small-dose results *(historical)* | `smalldose_summary/{family_canonical,game_slopes,summary,crossmode_r}.csv` | `analysis/steering/smalldose_final_summary.py` (defaults) |
| Permutation matched-control *(historical)* | `smalldose_summary/perm_null_results.csv` | `analysis/steering/perm_null_analysis.py` (defaults) |
| Group / family / contrast summaries *(historical)* | `group_summary.csv`, `family_summary.csv`, `contrast_summary.csv` | `analysis/steering/analyze_smalldose.py` |
| Per-game slopes (analysis view) *(historical)* | `per_game_slopes.csv` | same |
| Per-cell ranges and decision flips *(historical)* | `per_cell_ranges_flips.csv` | same |
| Letter-vs-strategy coherence *(historical)* | `coherence_by_game.csv`, `top_canonical_range_games.csv` | same |
| Verdicts by model × mode × layer *(historical)* | `verdict_by_model_mode_layer.csv` | same |
| Apparatus audit *(historical)* | `audit_summary.csv` | same |
| δ₁ᶜ moderation deltas *(historical)* | `summary/delta_tables_{model}.csv` | `steering/extract_directions.py` |
| Run metadata *(historical)* | `analysis_metadata.json` | `analysis/steering/analyze_smalldose.py` |

`analyze_smalldose.py` and `perm_geometry_study.py` read the unsuffixed roots only; they have **no
q05 counterpart** in this release, so every table in the historical block above describes the
pre-correction arm.

Narrative: `analysis/steering/README.md` (also pre-correction — see its header). **Before quoting
any steering number, read the reporting guards in [`METHODS.md`](METHODS.md) §6.** In particular:
family-resolved rather than pooled canonical estimates; the letter component shown or explicitly
removed; no slope-only readouts; no permutation-p-value language; and no comparison of the two arms'
permutation results, whose seed sets differ.

---

## Game metadata and human references

| Content | File | Producer |
|---|---|---|
| Canonical action axis, structural covariates (144 × 45) | `data/games/game_features.csv` | `features/compute_game_features.py` |
| Equivalence tables, human crosswalks | `data/games/taxonomy/` | `features/build_master_df.py` |
| Unified human + LLM per-game panel (7,466 × 92) | `data/human_refs/unified_pairs.parquet` | `features/build_unified_pairs.py` |
| Trait-effect tables | `data/human_refs/trait_*.csv` | `analysis/layer_a/build_trait_steering_oneshot.py` |

Column definitions: `data/human_refs/CODEBOOK_unified_pairs.md`,
`data/games/taxonomy/EQUIVALENCE_DATASET.md`.

---

## Provenance

`data/MANIFEST.json` records, for every file above: its sha256 and size, the builder that produced
it, and the locked seeds. Several table directories additionally carry a `provenance.json` written
by their builder at run time.
