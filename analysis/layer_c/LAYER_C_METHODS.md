# Layer C — Methods (one-shot Akata logit-lens token attribution)

*Self-contained methods for `analysis/layer_c/`. Pairs with `LAYER_C_FINDINGS.md`.*
*Substrate provenance: `analysis/block_c/SPEC_oneshot_layerc_collection.md` + `generate_oneshot_layerc.py`.*

## 1. Data substrate

Token-level logit-lens capture on the corrected one-shot Akata substrate
(`output/oneshot_akata_layerc/{model}/{game}/tokens.parquet`; `config.json` records
`lens = final_norm+lm_head_per_token_canonical_JP`, capture git commit `2218782a`, 2026-06-27/28).

> **Provenance gap (audit point 5, to resolve for a publication).** The shipped data is internally
> complete (3×144 games, 4 cb, 7 conditions, L40/L79) but: (i) the recorded capture commit `2218782a`
> is **not reachable in the local tree**; and (ii) the checked-in generator
> `analysis/block_c/generate_oneshot_layerc.py` still defaults to `output/oneshot_layerc` and asserts a
> **16-cell** grid, whereas the shipped data is **4-cell** `akata_4cell` under `output/oneshot_akata_layerc`
> — so the checked-in script *cannot* have produced the final data (it was superseded on Spark). The
> generator's `persist`/dry-run have been generalised to derive the cb count from the data and to point
> at the correct root, but the **exact Spark capture script / a reachable tag must still be committed**
> for Nature-grade reproducibility.

- **Models (dense only):** `qwen` (Qwen2.5-72B base), `qwen_instruct` (Qwen2.5-72B-Instruct),
  `llama31_instruct` (Llama-3.1-70B-Instruct, chat-templated). **GPT-OSS is excluded by design** —
  its MoE/harmony commit is not read by a single-layer residual lens at the prompt boundary; its
  mechanism is the router (Layer B).
- **Scope:** all **144** Bruns canonical 2×2 games; **4 counterbalance cells** per game
  (`cb_id ∈ {0,1,2,3}`, the `akata_4cell` grid — *not* 16); **layers 79 (final) and 40 (mid)**;
  **7 conditions** — `baseline`, five fixed decision cues (`risk/loss/inequity/maximin/selfish`),
  and the neutral procedural control stored under the historical condition name
  `length_match_null`. The control is not length matched.
- **Prompt:** the corrected Akata-style one-shot sentence prompt with J/P labels, ending exactly in
  `A: Option`. Inspection of the stored token rows confirms the natural-language payoff rules and
  this answer-prefix boundary; it is not the old A/B-matrix `"\nDecision: "` substrate.
- **Per-token schema:** `model, game_code, cb_id, condition, layer, token_index, token_str,
  char_start, char_end, region, is_canonical_row, score_canonical, score_trait_target,
  canonical_action_letter, prompt_hash`.
- **Regions:** `intro, rule, own_payoff, opponent_payoff, label_token, question, answer_prefix`
  (`cue_prefix` for trait/chat-header tokens; `special` for zero-width). The own/opponent split is
  **swap-invariant**: each cell renders as `(your payoff, other player's payoff)`, so the first number
  is `own_payoff`, the second `opponent_payoff`, regardless of counterbalance swaps.

## 2. The lens and the two pre-baked axes

For a chosen layer L, each token's residual is passed through the model's final norm and unembedding
(`final_norm + lm_head`) to vocabulary logits. Two scalar axes are computed at capture and **re-scored
to the canonical/target letter within each cb** (letters are randomised per cb; raw `J/P` aggregation
is invalid — CLAUDE.md #2):

- `score_canonical = logit[canonical-action letter] − logit[non-canonical letter]` (all conditions,
  incl. baseline). Positive = the lens, read at this token, favours the game-theoretic canonical action.
- `score_trait_target = logit[cue-target letter] − logit[other letter]` (substantive cued conditions
  only; **NaN when the cue's target action ties** for that game). The canonical action is
  `canonical_action_p1` from `datasets/processed/game_features.csv`; trait targets are the
  canonical-axis-validated actions from `analysis/block_a/trait_steering_proper.py`.

## 3. Derived quantities

- **Decision signal** *D* (PRIMARY) = `score_canonical` at the **final pre-choice token** — the last
  `" Option"`, identical across models (so model-comparable). The mean over the `answer_prefix` region
  (`ap_*`) is kept only as a robustness column: it dilutes Llama, whose `answer_prefix` region also
  contains chat-scaffolding tokens (assistant header, etc.). [Readout corrected 2026-06-28 per audit.]
- **Per-token output-projection increment (impact).** Within each ordered sequence (one `game × cb ×
  condition × layer`), `impact[t] = Δs[t] = score_canonical[t] − score_canonical[t−1]`, with the first
  token of each sequence set to 0 (its "increment" is the start artefact). Increments telescope:
  Σ_t impact[t] = (final-token readout − first-token), so region / own-vs-opponent contributions are an
  **additive, telescoping** decomposition. It is an order-dependent lens-increment scheme, not a
  Shapley value, causal mediation estimate or causal token attribution.
- **Signed canonical incentive Δ₁ᶜ** (the QRE/level-1 variable, q=0.5 uniform belief), reconstructed
  from the payoff matrix `matrix_8vec` in `analysis/_shared/datasets/unified_pairs.parquet`:
  for player 1, `d1 = E[own | act0] − E[own | act1]` (mean over the opponent's two actions);
  `Δ₁ᶜ = d1` if `canonical_action_p1 == 0` else `−d1` (so positive = incentive points toward
  canonical). Verified against a known level-1-indifferent game (Δ₁ = 0).

## 4. Estimators (by result)

All inferential CIs are **95% percentile cluster-bootstrap resampling whole games** (`game_code`),
`nboot = 1000` (800 for Result 6). Primary depth L79; L40 used only for the depth panel. Comparisons
are interpreted **within model by rank/sign**, not as architecture-invariant absolute magnitudes.

- **R1 Neural-λ** (`stats_layerc.py`). Per model/layer, baseline: OLS slope of *D* (final-token) on
  Δ₁ᶜ across game-cells, `λ_lens`; Pearson *r*; `P(sign = canonical) = mean(D > 0)`. `*_apmean` columns
  repeat this on the answer_prefix-mean as robustness. Behavioral λ (q=0.5 generate basis, Layer A)
  reported alongside for the dissociation.
- **R2 Region contribution** (`stats_layerc.py`). Baseline L79: per `game × cb`, sum `impact` within
  each `region`; mean across game-cells per model, bootstrap CI. Own-vs-opponent share computed two
  ways — `|increment|` share and `|score|` (mean-|`score_canonical`|) share.
- **R3 Depth × position** (`stats_layerc.py`). Per model/layer/region: slope of region-mean
  `score_canonical` on Δ₁ᶜ (`λ(region)`), bootstrap CI; the `answer_prefix` point uses the **final-token**
  λ (consistent with R1; the region-mean would dilute Llama). Trajectory over prompt regions, L40 vs L79.
- **R4 Absolute cue-target projection by position** (`stats_layerc.py`). L79: per model/cue, mean
  `score_trait_target` at `cue_prefix` versus the **final pre-choice token**, on each cue's
  **defined-target subset** (`score_trait_target` not NaN), with game-bootstrap CIs. These are
  within-cue target-letter margins, not cue-minus-baseline changes or behavioural "heard"/"obeyed"
  estimators. `answer_prefix_target_projection` is an answer-prefix-mean robustness value. The
  neutral procedural-control diagnostic is the paired final-token `score_canonical` shift versus
  baseline; its wording is not length matched and it is not subtracted from the cue estimates.
- **R5 Token-level output-projection map** (`fig_token_shap.py`, `fig_token_heatmap_3panel.py`).
  *Beeswarm:* per model, top token strings by mean `|impact|` (min 80 occurrences), excluding
  bare single-capital-letter tokens (option-letter echoes) and `special`; dots = per-occurrence impact
  coloured by region, • = mean. *Waterfall:* one representative high-incentive decision
  (`qwen_instruct`, Δ₁ᶜ ≥ 1.5, *D* > 1), top tokens by `|impact|`. *Aggregated heatmap:* the prompt
  template is identical across games (verified: 161 tokens at cb0 for the Qwen models; the only varying
  positions are payoff digits and `point/points`), so token position N is the same structural slot in
  every game; per model we average `impact` per `token_index` across all 144 games (cb0) and colour the
  template tokens by that mean. Option-letter tokens average to ≈0 because the echoed letter is
  canonical in ~half the games and non-canonical in the other half.
- **R6 Strategic payoff reading** (`fig_strategic_payoff.py`). Own-payoff digit tokens, all cb × 144
  games, baseline L79. `value = digit`, `vc = value − mean`, `sign = +1 if is_canonical_row else −1`,
  `strategic_value = vc × sign`. **Impact is demeaned within each `(cb, token_index)` slot** so the
  effect is identified purely from across-game variation at the same position. OLS
  `impact ~ vc + strategic_value`: the `vc` term absorbs pure digit-identity, the `strategic_value`
  slope is the strategic modulation net of identity and position; cluster-bootstrap by game. The
  committed-incentive comparator is `λ_lens` (R1). `is_canonical_row` mean = 0.50 (2 of 4 rule lines),
  verifying the coding.

## 5. Constraints respected

1. **Canonical/target axis per cb** is baked in at capture (CLAUDE.md #2); no raw positional `J/P`
   aggregation anywhere. Cross-game aggregation of `score_canonical`/`impact` is valid because both are
   already canonical-signed.
2. **GPT-OSS excluded** from all Layer C analyses (MoE/harmony).
3. **Correlational, not causal.** The lens shows the decision signal a token/region *carries*, not that
   the model *uses* it downstream; no `do()` language. Causation is Layer B's steering arm. The
   per-token increment is a particular attribution dominated, at content tokens, by local next-token
   prediction — interpret dissociations (e.g. R6 token vs commit), not single-token signs.
4. **Payoff multiplier = 1** (one-shot convention); align units explicitly against Bruns ×2 runs.
5. **8-bit quantisation; Llama chat-wrapped, Qwen raw** → residuals not byte-comparable across the
   Qwen/Llama family boundary; compare within model. `λ_lens < behavioral λ` in magnitude (the lens
   underreads); the ordering/dissociation is the claim, not the absolute slope.
6. **4 cb cells** are the replicate dimension. R5's aggregated heatmap uses cb0 reading order.

## 6. Reproduce

```bash
python3 analysis/layer_c/compute_layerc.py        # tables/{decision_readouts,region_contrib,incentive_delta1c}.csv
python3 analysis/layer_c/stats_layerc.py          # tables/stat_*.csv  (R1–R4)
python3 analysis/layer_c/fig_layerC_main.py        # figures/fig_layerC_main.*  (R1–R4)
python3 analysis/layer_c/fig_token_shap.py         # figures/fig_token_shap.* + token_shap_payload.json  (R5)
python3 analysis/layer_c/fig_token_heatmap_3panel.py  # figures/fig_token_heatmap_3panel.*  (R5)
python3 analysis/layer_c/fig_strategic_payoff.py   # figures/fig_strategic_payoff.* + stat_strategic_payoff.csv  (R6)
```

## 7. Layer B↔C bridge (probe vs lens) — RAN 2026-07-02

Tests whether deferred integration (R6) is *representation present but not projected* vs *representation
absent until commit*. GPU re-capture `capture_residuals_bridge.py` stored per-token **residual vectors**
(fp16) at own/opponent-payoff, answer, and `win/wins` positions (baseline, L40/L79, 4 cb, 144 games;
release `akata-layerc-residuals-20260702`, capture commit `7445b005`, on the **corrected current
prompt** — not the old A/B-matrix one). `probe_bridge.py` then decodes the own-payoff `incentive_sign`
per (region, layer) with a game-grouped CV logistic probe on **120 PCA components** (3 folds, leak-free
per fold) and compares to the lens (AUC of `score_canonical`, the output-direction projection).

**Result:** a position flip — probe > lens at the own-payoff tokens (probe 0.60–0.67 vs lens 0.47–0.52),
lens ≫ probe at the final token (lens 0.98 vs probe ≈0.6 for both instruct models). The pattern is
consistent with information moving onto the output axis at commitment, but the comparison is not
capacity-controlled.
`tables/bridge_probe_vs_lens.csv`; details + caveats in the "Synthesis" section of `LAYER_C_FINDINGS.md`.

**Probe caveat:** the probe retains at most 120 principal components and is not a feature superset of
the one-dimensional vocabulary lens. A low-variance output direction could be discarded, but that
explanation is untested. A regularised full-dimensional probe with bootstrap intervals is the required
strengthening. Reproduce: `python3 analysis/layer_c/probe_bridge.py --root
output/oneshot_akata_layerc_residuals --model {qwen,qwen_instruct,llama31_instruct}`.
