# Layer C — Findings (one-shot Akata logit-lens)

*Generated 2026-06-28 by `analysis/layer_c/` pipeline. Primary readout = FINAL pre-choice token,
L79; L40 for depth.* *3 dense models × 144 games × 4 cb × 7 conditions. GPT-OSS excluded (MoE/harmony).*
*All CIs: 95% cluster-bootstrap by game (nboot=1000).*

> **Readout correction (2026-06-28, independent audit).** The primary statistic is now the **final
> pre-choice token** (the last `" Option"`, identical across models), not the mean over the
> `answer_prefix` region. The region-mean **diluted Llama**, whose `answer_prefix` region also contains
> chat-scaffolding tokens (assistant header, etc.), and wrongly made it look like chance. The
> answer_prefix-mean is retained as a robustness column (`*_apmean`). This **changes the headline from
> "Qwen-Instruct-only recruitment" to graded recruitment** (Result 1, 4). Also fixed: first-token
> increment set to 0 (matches the figures and methods); "token-SHAP" renamed to direct logit
> attribution (it is not Shapley values). The own/opponent result (R2) and strategic-payoff result (R6)
> are unaffected.

## Thesis

The logit lens **localizes recruitment in prompt-space**: every dense model *reads* the strategic
incentive, but how strongly it is wired into the choice — read at the final pre-choice token — is
**graded**. Layer C recovers a quantal-response λ in **vocabulary space** that orders the models by
recruitment (Qwen-Instruct strongest, Llama clearly recruits, Qwen-*base* weak) — a dissociation
behavioral λ cannot see (all three dense models have behavioral λ ≈ 1.9–2.0). The weak case is the
**base** model, consistent with Layer B's "build-then-collapse" (it represents but fails to commit).

## Result 1 — Neural-λ in vocabulary space, graded recruitment (headline)

Slope of the **final-token** lens decision signal on the signed canonical incentive Δ₁ᶜ, across games.

| model | λ_lens (final, L79) [95% CI] | r | P(sign=canonical) | behavioral λ | λ_apmean / P_apmean (robustness) |
|---|---|---|---|---|---|
| **Qwen2.5-Instruct** | **1.432 [1.299, 1.577]** | 0.74 | **0.747** | 1.92 | 0.642 / 0.748 |
| **Llama-3.1-Instruct** | **0.789 [0.724, 0.859]** | 0.66 | **0.731** | 1.89 | 0.175 / 0.500 |
| Qwen2.5 (base) | 0.170 [0.137, 0.199] | 0.13 | 0.498 | 2.04 | 0.138 / 0.502 |

**Graded recruitment, with the base model as the weak case.** Both instruction-tuned models wire the
incentive into the final-token decision — Qwen-Instruct most strongly (λ=1.43), Llama clearly
(λ=0.79, non-overlapping CIs, well above the base) — while Qwen-*base* stays at chance (P(sign)=0.50).
Each instruct model's lens recovers its *own* canonical rate (Qwen-I 0.747 ≈ behavioral 0.726; Llama
0.731 ≈ behavioral 0.740). Behavioral λ is ≈ equal across all three (1.9–2.0), so the lens reveals a
graded *internal* difference behaviour hides — and pinpoints the **base** model as the one that
represents but does not commit (its `λ_apmean`-vs-final gap is small; it is simply weak everywhere).
The earlier "Llama ≈ chance" reading (`λ_apmean`=0.175) was the chat-scaffolding dilution.

## Result 2 — Token-resolved recruitment of the payoff structure

Signed contribution (Σ per-token increments toward canonical) of each region to the decision signal,
L79 baseline. Opponent-payoff contribution:

| model | opponent_payoff contribution [95% CI] | own_payoff | answer_prefix |
|---|---|---|---|
| Qwen2.5-Instruct | **−0.653 [−1.031, −0.324]** (sig.) | −0.887 | +1.214 |
| Qwen2.5 (base) | −0.068 [−0.232, +0.107] (n.s.) | −0.350 | +0.133 |
| Llama-3.1-Instruct | +0.021 [−0.133, +0.172] (n.s.) | −0.391 | +0.674 |

**Opponent payoffs move the decision only in Qwen-Instruct** (CI excludes 0); in base and Llama the
opponent-payoff contribution is statistically zero. All models *read* the numbers (|score| on payoff
tokens is comparable: opponent share of |score| 0.49–0.60), but only the recruiting model lets the
opponent's payoffs *move the choice* — a token-level theory-of-mind / level-k signature.
(Increment-based opponent share: Qwen-I 0.584 [0.571,0.596] > base 0.478 > Llama 0.500.)

## Result 3 — Depth × position

λ(region) at L40 vs L79 (answer_prefix uses the final-token λ). The incentive concentrates at the
decision point, but the **depth profile is graded and model-specific**: final-token λ goes
Qwen-Instruct **−0.00 (L40) → 1.43 (L79)** (crystallizes late), Llama **0.59 → 0.79** (already
recruiting by mid-depth, then sustains), Qwen-base **0.01 → 0.17** (flat throughout). So the
vocabulary-space depth story maps onto Layer B as: Qwen-Instruct late-crystallize, Qwen-base
build-then-collapse (weak commit), and Llama an earlier/sustained recruiter — not "only Qwen-Instruct."

## Result 4 — Absolute cue-target projection by position

`score_trait_target` at the cue tokens versus the **final pre-choice token**, L79, per cue's
defined-target subset. These are absolute target-letter margins within the cued prompts. They are
not cue-minus-baseline movements, do not use the behavioural identifiability sets, and therefore
do not estimate whether a cue was "heard" or "obeyed."

| final target projection | risk | loss | maximin | selfish | inequity |
|---|---|---|---|---|---|
| Qwen2.5-Instruct | 1.98 | 0.62 | 2.82 | 3.94 | 4.44 |
| Llama-3.1-Instruct | **−0.05** | 0.22 | 1.52 | 2.42 | 2.11 |
| Qwen2.5 (base) | 0.36 | 0.20 | 0.28 | 0.96 | 0.48 |

The cue-token target projections are approximately zero for every model/cue, whereas final-position
projections vary by model and wording. This is a descriptive localisation of absolute output
projection, not evidence that the cue was read, converted or followed. In particular, Llama's
risk-wording final projection is near zero (−0.05 [−0.257, 0.145]); this does not by itself establish
behavioural non-compliance because the statistic lacks a paired baseline subtraction.
n_games: risk/loss/maximin 76, selfish 67, inequity 30.

## Result 5 — Token-level output-projection map

Per-token **output-projection increment**: impact of token *t* = Δ(lens canonical
log-odds) = s[t] − s[t−1]; increments telescope to the final readout. This is an order-dependent
descriptive decomposition, not a Shapley value or causal token attribution. Aggregated across all 144 games on the fixed prompt template
(cb0). This panel is descriptive (*which* tokens carry decision-aligned signal); the recruitment
*magnitude* is Result 1.

- **The answer-prefix region carries the largest signed contribution in every model**, and its size /
  cleanliness is graded (strongest, most concentrated in Qwen-Instruct, whose single strongest
  toward-canonical token across games is "Option" at the answer prefix, +1.11). Region-mean impact on
  the answer-prefix tokens (cb0) understates Llama (chat scaffolding) — use the final-token statistic
  (R1) for magnitude.
- *Caveat on levels.* The average net decision movement (cb0) is intercept/level-dominated, not the
  recruitment metric: because Δ₁ᶜ is ~zero-mean across games, an incentive-tracking signal averages
  near zero. So a low average level (e.g. Llama) is **not** evidence against recruitment — R1's slope
  is the recruitment measure.
- **Aggregation removes the letter-echo for free:** option-letter tokens (J/P) dominate a *single*
  prompt but average to ≈0 across games (the echoed letter is canonical in ~half the games and
  non-canonical in the other half). Confirms those single-game spikes are lens echo, not strategy.
- Figures: `fig_token_shap` (attribution beeswarm + single-decision waterfall),
  `fig_token_heatmap_3panel` (aggregated heatmap, one panel per model).

## Result 6 — Deferred integration: payoffs are read but not integrated on contact

Does an own-payoff digit move the decision the strategically correct way?
`strategic_value = (digit − 2.5) × (+1 if canonical-action row else −1)`; regress per-token impact
~ `value + strategic_value`, position-demeaned (within cb×slot, so identified from across-game
variation at the same slot), all cb × 144 games, cluster-bootstrap by game. `is_canonical_row` = 0.50
exactly (2 of 4 rule lines), coding verified.

| model | own-payoff strategic slope at the TOKEN [95% CI] | same incentive at COMMIT (final-token neural-λ) |
|---|---|---|
| Qwen2.5-Instruct | **−0.108 [−0.170, −0.044]** | **+1.432** |
| Qwen2.5 (base) | −0.070 [−0.113, −0.029] | +0.170 |
| Llama-3.1-Instruct | −0.030 [−0.095, +0.027] (n.s.) | +0.789 |

At the moment a payoff number is read, the lens-readable decision does **not** move toward the
strategically correct action (faintly the opposite for the Qwen models); yet the **committed** decision
is strongly incentive-aligned (neural-λ, Result 1). **Strategic integration of payoffs is deferred to
the decision point, not performed incrementally as each number is read** — reading and deciding are
separated in time. Both recruiting models show the gap (token faint/negative vs commit strongly
positive: Qwen-Instruct −0.11 vs +1.43; Llama −0.03 n.s. vs +0.79); the base model is weak at both.
The defensible claim is this token↔commit *dissociation*; we do not over-interpret the small negative
sign (the per-token lens at a digit is dominated by local next-token prediction). Figure: `fig_strategic_payoff`.

## Synthesis — relation to Layer B (the same recruitment, on two axes)

*The bridge test below has now been run (2026-07-02) and supports the unification.* Layer B finds
canonical action / incentive sign become decodable only in the **late layers** at the decision token
(depth crystallization). Layer C's "late" is that same phenomenon plus a second axis:

- **Depth axis — partial match.** Result 3 reproduces B's late crystallization for **Qwen-Instruct**
  (final-token λ −0.00 at L40 → 1.43 at L79) and B's build-then-collapse for **Qwen-base** (weak at
  both). **Tension to reconcile:** the corrected (final-token) Layer C shows **Llama recruits**
  (λ=0.79, P(sign)=0.73), whereas Layer B's recruitment bridge classed Llama as a non-recruiter. The
  most likely cause is that the Layer B readout has the *same* chat-scaffolding issue this audit fixed
  in C — i.e. **Layer B's Llama result should be re-checked at the final pre-choice token.** Until then,
  treat the "Llama doesn't recruit" claim as unsettled.
- **Position axis — new (Result 6).** Deferred integration adds sequence position: the incentive is
  not projected onto the choice axis at the payoff tokens, only at the commit.

**Why they are consistent (represent vs recruit, in temporal form).** The probe (B) finds the incentive
in *any* linear direction; the lens (C) sees only the component aligned with the output direction
(`score_canonical` is a projection onto a difference of unembedding rows). So "B decodes it" and "C's
lens reads ≈0 at the payoff tokens" jointly say: the payoff incentive is **represented** as the model
reads (probe-decodable, sharpening across depth) but **recruited onto the choice axis** only at the
output corner — late layer *and* answer position. Reading stores the incentive; deciding projects it,
in one late operation rather than a running tally.

**Discriminating test — RESULT (2026-07-02, run on the GPU residual re-capture, 3 models × 144 games,
L79, game-grouped CV probe on 120 PCs vs the lens, decoding the own-payoff incentive sign;
`tables/bridge_probe_vs_lens.csv`).**

AUC [95% game-cluster bootstrap CI]:

| L79 | own-payoff: probe / lens | final: probe / lens |
|---|---|---|
| Qwen-Instruct | **0.67 [.56,.77] / 0.47 [.43,.52]** (non-overlap) | 0.68 [.59,.78] / **0.98 [.96,1.0]** (non-overlap) |
| Llama-Instruct | 0.60 [.48,.70] / 0.48 [.45,.51] (overlap) | 0.58 [.40,.74] / **0.99 [.97,1.0]** (non-overlap) |
| Qwen-base | 0.60 [.52,.70] / 0.52 [.48,.58] (overlap) | 0.72 [.61,.81] / 0.70 [.67,.72] |

**A position-dependent flip — probe > lens at the payoff tokens, lens ≫ probe at commit — which is the
"represent → project" signature.** At the own-payoff tokens the incentive is decodable in a *free*
direction (probe 0.60–0.67, above chance) that the choice-axis **lens does not read** (0.47–0.52,
chance): the incentive is **represented off the decision axis while the payoff is read**. At the final
token the incentive is concentrated **on** the decision axis (lens 0.98 for both instruct models; the
probe can't match a single low-variance direction). So **reading stores the incentive off-axis;
committing projects it onto the choice axis — Layer B (representation) and Layer C (deferred
projection) are the same mechanism.** The **base** model shows the represent-*without*-recruit half: it
represents the incentive off-axis at the payoff tokens (probe 0.60 > lens 0.52) but only weakly
projects it at commit (lens 0.70, not 0.98) — matching its weak R1 λ. Opponent-payoff tokens are near
chance for both readouts (the own-canonical incentive isn't carried there; expected).

*Significance (game-cluster bootstrap):* the **final-token lens ≫ probe is significant for both
instruct models** (CIs non-overlapping; incentive strongly on the choice axis at commit). The
**own-payoff probe > lens is significant for Qwen-Instruct** (0.67 [.56,.77] vs 0.47 [.43,.52],
non-overlapping) but only *suggestive* for base/Llama (CIs overlap) — so the clean, significant
position flip is Qwen-Instruct's; the others show the commit half clearly and the payoff half
directionally. *Caveats:* payoff-token effects are modest, and the PCA probe retains at most 120
components. It is therefore not a feature superset of the one-dimensional vocabulary lens: its
commitment-site disadvantage may reflect removal of a low-variance output direction, but that
explanation has not been tested. A regularised full-dimensional probe is the required strengthening.
That Llama shows lens 0.99 at commit reinforces that Layer B's "Llama doesn't recruit" needs the
final-token re-check.

## Honest limits

- **Correlational** (lens projection), not causal — causation is Layer B's steering arm.
- **GPT-OSS excluded** (harmony/MoE; its mechanism is the router).
- 8-bit, per-cb, Llama chat-wrapped → compare **within model by rank/sign**, not absolute scale.
  λ_lens < behavioral λ in magnitude (lens underreads); the *ordering/dissociation* is the claim.
- **Readout definition.** Primary = final pre-choice token (`" Option"`), model-comparable; the
  answer_prefix-mean (`*_apmean`) diluted Llama and is kept only as robustness. Any cross-layer
  comparison (esp. the Llama-vs-B reconciliation) must use the same final-token readout.
- **The historical `length_match_null` condition is a neutral procedural control, not a
  length-matched placebo.** At the final token it shifts the canonical projection by Qwen-I
  **+0.68**, base +0.13 and Llama +0.34. Result 4 reports absolute within-cue target margins and does
  not subtract this control; it is therefore not a control-corrected cue-effect estimate.
- inequity has the smallest defined-target subset (n=30); FDR within the trait family recommended.
- **Result 5** aggregated heatmap uses cb0 reading order; its average levels are intercept-dominated
  (use R1 slope for magnitude). **Result 6**'s token-level slope is L79-only and its small negative
  sign is not mechanistically interpreted — the claim is the token↔commit dissociation.

## Files

Methods: `LAYER_C_METHODS.md`.  Pipeline (run from repo root):
`compute_layerc.py` → `stats_layerc.py` → figure scripts.

| script | output |
|---|---|
| `compute_layerc.py` | `tables/{decision_readouts,region_contrib,incentive_delta1c}.csv` |
| `stats_layerc.py` | `tables/stat_*.csv` (Results 1–4) |
| `fig_layerC_main.py` | `figures/fig_layerC_main.{png,pdf}` (Results 1–4) |
| `fig_token_shap.py` | `figures/fig_token_shap.*` + `tables/token_shap_payload.json` (Result 5) |
| `fig_token_heatmap_3panel.py` | `figures/fig_token_heatmap_3panel.*` (Result 5) |
| `fig_strategic_payoff.py` | `figures/fig_strategic_payoff.*` + `tables/stat_strategic_payoff.csv` (Result 6) |

`plot_layerc_candidates.py` and `tables/c1*/c2*/c3*`, `figures/figC*` are the earlier candidate
exploration (policy-bridge / semantic-evidence / trait-redirection), kept for reference.
