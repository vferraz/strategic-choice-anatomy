# analysis/layer_c

Token-level logit-lens analysis for Layer C, on the corrected one-shot Akata substrate
(`output/oneshot_akata_layerc/`). Reframes Layer C from confirmation to the **prompt-space
localization of output projection**: a vocabulary-space λ, token-resolved payoff-region increments
(own vs opponent), a depth×position map, and absolute cue-target projections by position.

Methods: `LAYER_C_METHODS.md`.  Results: `LAYER_C_FINDINGS.md`.

## Run (from repo root)

```bash
python3 analysis/layer_c/compute_layerc.py            # tables/{decision_readouts,region_contrib,incentive_delta1c}.csv (~6s)
python3 analysis/layer_c/stats_layerc.py              # bootstrap CIs -> tables/stat_*.csv (R1-R4)
python3 analysis/layer_c/fig_layerC_main.py           # 4-panel main figure (R1-R4)
python3 analysis/layer_c/fig_token_shap.py            # direct-logit-attribution beeswarm + waterfall (R5)
python3 analysis/layer_c/fig_token_heatmap_3panel.py  # aggregated heatmap, one panel per model (R5)
python3 analysis/layer_c/fig_strategic_payoff.py      # deferred-integration / strategic payoff test (R6)
```

## What each metric is

- **Δ₁ᶜ** — signed canonical level-1 incentive (q=0.5), reconstructed from `matrix_8vec` in
  `analysis/_shared/datasets/unified_pairs.parquet`; + = toward the canonical action.
- **Decision signal** — `score_canonical` at the FINAL pre-choice token (`" Option"`, model-comparable;
  `answer_prefix`-mean kept as `*_apmean` robustness).
- **λ_lens** — OLS slope of decision signal on Δ₁ᶜ across games (a reduced-form
  vocabulary-space sensitivity, not QRE).
- **Region projection increment** — Σ per-token increments `Δs[t]=s[t]−s[t−1]` within a region;
  this additive, order-dependent decomposition telescopes to the final readout and is descriptive,
  not causal token attribution (first token → 0).
- **Cue-target projection by position** — `score_trait_target` at `cue_prefix` versus the FINAL
  token on each defined-target subset. Both are absolute within-cue margins, not cue-minus-baseline
  shifts or behavioural "heard"/"obeyed" measures.

## Constraints respected

Canonical/target axis baked per cb at capture (CLAUDE.md #2); GPT-OSS excluded; cluster-bootstrap by
game; within-model rank/sign comparisons only; 4 cb cells are the replicate. See `LAYER_C_FINDINGS.md`
for results and `analysis/block_c/SPEC_layerC_oneshot_analysis_RECOMMENDED.md` for the rationale and
SotA framing.
