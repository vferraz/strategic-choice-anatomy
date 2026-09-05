# Methods And Findings — Layer B Redesign Panels D/F

Last updated: 2026-06-28.

This note documents the two redesigned panels in the current Layer B main figure:

- **Panel D:** GPT-OSS MoE bottleneck (`gate logits` vs actual `top-k` route).
- **Panel F:** cue-identity encoding and generated-choice uptake.

No causal steering data are used here. The dense causal sweep is in progress and
is not part of this figure. GPT-OSS router evidence is read-only/descriptive
because `openai/gpt-oss-120b` uses an MXFP4 fused routing path that is not
editable through a patchable `router.forward`.

Current figure:

- `analysis/layer_b/figures/fig_layerB_main_v2.png`
- `analysis/layer_b/figures/fig_layerB_main_v2.pdf`

Main script:

- `analysis/layer_b/fig_layerB_main_v2.py`

## Proofed Storyline

The corrected one-shot substrate changed the Layer B story. The old 10-round
visual language suggested a cleaner split between models with and without
representations. The current one-shot result is different:

> Strategic structure is represented broadly and cheaply. The Layer B finding is
> whether that represented structure is recruited into choice.

The current main figure is therefore organized as:

1. **Represented, but cheaply**: incentive, canonical choice, and an inert
   stimulus control are decodable. Decodability sets up the problem; it is not
   the competence claim and not a cross-model rationality ranking.
2. **Forms across depth**: canonical-choice information develops across layers.
3. **Dense late alignment**: decision and incentive axes cross the game-block
   null late in all three dense models. GPT-OSS pure-complete residual/router
   geometry remains near orthogonal; its former all-row alignment is a
   heterogeneous capture-site sensitivity.
4. **MoE bottleneck**: GPT-OSS dense router gate logits score strategic
   incentive more strongly than sparse selected top-k route summaries.
5. **Recruitment bridge**: the load-bearing result. Internal incentive
   representation predicts canonical play in some models; the partial bridge
   beyond objective `Delta1c` is positive for Qwen-Instruct, but not Qwen-Base,
   Llama or pure-commitment GPT-OSS.
6. **Cue effects are gated**: cue identity is nearly perfectly represented, but
   generated-choice uptake varies by cue and model. These are encoding and
   behavioural-shift estimates, not literal ``heard'' or ``obeyed'' scores.

What this supports:

> Models do not fail because they cannot represent games. They broadly encode
> strategic structure. They differ in recruitment: whether the represented
> incentive becomes choice-relevant.

What this does **not** support:

- no causal steering claim yet;
- no GPT-OSS router-edit claim;
- no claim that actual routed expert selection causally implements strategic
  choice;
- no residual-only interpretation of GPT-OSS as a dense-model null.

## Panel D — GPT-OSS MoE Bottleneck

### Provenance

Primary GPT-OSS root:

- `output/oneshot_akata_main_dl/gptoss`

Audit/config provenance from `analysis/layer_b/tables/router_oneshot_audit_report.md`:

- substrate: `akata_oneshot`
- decoder: `generate_parse_JP_per_cell_pure_mixed_none`
- feeding: `harmony_chat_template_pure_greedy`
- capture: `all_layers_incl_l0`
- counterbalance: `akata_4cell`
- games: 144
- rows: 4,608
- router files: 144
- router layers: `1,3,6,9,12,15,18,21,22,24,27,30,33,35`
- P1 baseline rows: 576
- P1 baseline pure commits: 525
- P1 baseline mixed commits: 51

Relevant code:

- `analysis/layer_b/router_oneshot_audit.py`
- `analysis/layer_b/router_bottleneck_analysis.py`
- `analysis/layer_b/fig_layerB_main_v2.py`

Relevant tables:

- `analysis/layer_b/tables/router_oneshot_decodability.csv`
- `analysis/layer_b/tables/router_oneshot_best_summary.csv`
- `analysis/layer_b/tables/router_oneshot_behavior_link.csv`
- `analysis/layer_b/tables/router_bottleneck_depth.csv`
- `analysis/layer_b/tables/router_bottleneck_summary.csv`
- `analysis/layer_b/tables/router_bottleneck_game_level.csv`

### Method

Panel D uses the corrected GPT-OSS commit-token router captures. The target is
the canonical-signed objective incentive sign at uniform belief:

```text
sign_delta1c_q05
```

The plotted quantities are held-out game-grouped probe AUCs across GPT-OSS router
layers for:

- `router_gate`: full pre-softmax router gate-logit vector;
- `router_topk_weight`: 128-dimensional sparse vector containing the selected
  top-k expert weights;
- `router_topk_binary`: 128-dimensional sparse binary vector containing only
  the selected top-k expert identities.

The final main figure deliberately omits the residual line from Panel D, even
though residual results are computed, because the panel's purpose is MoE-specific:
latent gate scoring versus enacted selected route.

Important comparison caveat: the gate-logit feature set is dense, whereas the
top-k feature sets are sparse selected-route summaries. The gate-versus-route gap
therefore should be read as a descriptive difference between exposed gate state
and the selected expert route representation, not as proof that routing discards
strategic information.

The analysis is descriptive. It does not edit router state, does not regenerate
under intervention, and does not claim causality.

### Results

At layer 18, the panel annotation is:

```text
gate 0.89 vs top-k set 0.61
```

Exact layer-18 values from `router_oneshot_decodability.csv`:

| feature set | AUC | 95% CI | n | games |
|---|---:|---:|---:|---:|
| router gate logits | 0.891 | [0.851, 0.925] | 576 | 144 |
| top-k weights | 0.640 | [0.585, 0.693] | 576 | 144 |
| top-k set | 0.609 | [0.554, 0.661] | 576 | 144 |
| residual, not plotted in Panel D | 0.900 | [0.860, 0.937] | 576 | 144 |

Best incentive-sign AUCs across layers:

| feature set | best layer | AUC | 95% CI |
|---|---:|---:|---:|
| router gate logits | 18 | 0.891 | [0.851, 0.925] |
| top-k weights | 3 | 0.714 | [0.657, 0.766] |
| top-k set | 12 | 0.660 | [0.595, 0.717] |
| residual | 21 | 0.915 | [0.878, 0.948] |

The bottleneck pattern is stable across layers: gate-logit incentive AUC remains
well above the selected top-k route. From `router_bottleneck_depth.csv`, the
gate-minus-top-k-set AUC gap ranges from 0.135 at layer 1 to 0.345 at layer 15;
at layer 18 it is 0.283.

Choice-readout guardrail: literal final J/P choice is nearly perfectly readable
from gate and residual state, and from top-k weights, but this is not the same as
strategic canonical choice. For primary pure canonical choice, top-k route
readout is weak (`router_topk_weight` AUC 0.569; `router_topk_binary` AUC 0.531),
while literal J/P choice is high (`router_topk_weight` AUC 0.986).

Process-link analysis, used for the appendix bottleneck figure but not shown in
the final main Panel D:

- gate strategic evidence vs log median new tokens: Spearman `rho = -0.693`,
  `p = 6.83e-22`;
- after objective controls (`|Delta1c|`, complexity, dominance profile,
  number of pure Nash equilibria): partial Pearson `r = -0.351`,
  `p = 1.6e-05`;
- objective controls alone explain `R2 = 0.596` for log token length;
- adding gate evidence gives `R2 = 0.646`, `Delta R2 = 0.050`;
- adding top-k-set evidence gives `R2 = 0.606`, `Delta R2 = 0.010`.

Interpretation:

> GPT-OSS gate logits contain a strong strategic incentive signal at the
> final-channel commit token. Sparse selected top-k route summaries carry less
> of that signal. This is evidence for an MoE bottleneck/readout distinction,
> not evidence that expert selection causally implements the decision or that
> routing discards strategic information.

## Panel F — Cue Effects Are Gated In Use

### Provenance

Primary table:

- `analysis/layer_b/recruitment/tables/disposition_dissociation.csv`

Builder provenance:

- `analysis/layer_b/recruitment/src/build_all.py`
- `analysis/layer_b/recruitment/src/shared.py`

The table joins cue-shift neural separability with Layer A trait/cue behavioral
effects:

- cue identity / LDA side: residual cue-shift activations at the final available
  layer for each model;
- behavioral side: `analysis/layer_a/tables/f2_trait_aim.csv`;
- data root: corrected integrated one-shot root, `output/oneshot_akata_main_dl/`;
- behavior aligned to `canonical_action_p1` / cue target, not raw action labels.

### Method

The previous version plotted:

```text
x = cue/disposition decodability
y = behavioral movement
```

This was visually bad because cue identity is near ceiling for nearly every
point, so all observations pile up at the right edge of the panel.

The redesigned Panel F keeps the same finding and same source table, but changes
only the visual encoding:

```text
y-axis = behavioral trait/cue
x-axis = behavioral movement toward the cue target
points = models
horizontal error bars = bootstrapped CI in the source table
```

Cue identity decodability is retained as text because it is a setup fact, not
the visual axis:

```text
LDA accuracy range: 0.97-1.00
```

Traits/cues shown:

- inequity;
- risk;
- loss;
- maximin;
- selfish.

The behavioral movement statistic is not an all-games average. It uses the
Layer A conflict/identifiability filters in `fig2_trait_steering.py`: inequity
is scored where the most-equal action differs from the canonical action;
risk/loss/maximin are scored where the security target differs from the
uniform-EV action; selfish is an identifiable uniform-EV manipulation check
rather than a rational-conflict subset.

### Results

Cue identity is near ceiling:

- minimum held-out LDA trait accuracy: 0.969;
- maximum held-out LDA trait accuracy: 1.000.

Mean behavioral movement by cue, across the four models:

| cue | mean movement | min | max | LDA range |
|---|---:|---:|---:|---:|
| inequity | 0.643 | 0.515 | 0.868 | 0.981-1.000 |
| risk | 0.311 | 0.224 | 0.427 | 0.995-1.000 |
| loss | 0.309 | 0.188 | 0.427 | 0.969-1.000 |
| maximin | 0.331 | 0.276 | 0.427 | 1.000-1.000 |
| selfish | 0.093 | 0.065 | 0.115 | 0.998-1.000 |

Per-model inequity movement, the largest consistent shift:

| model | inequity movement | 95% CI |
|---|---:|---:|
| Qwen-Instruct | 0.559 | [0.426, 0.691] |
| Qwen-Base | 0.515 | [0.426, 0.603] |
| Llama-3.1 | 0.632 | [0.471, 0.779] |
| GPT-OSS | 0.868 | [0.706, 1.000] |

Selfish cue movement is consistently small:

| model | selfish movement | 95% CI |
|---|---:|---:|
| Qwen-Instruct | 0.087 | [0.056, 0.121] |
| Qwen-Base | 0.065 | [0.039, 0.094] |
| Llama-3.1 | 0.107 | [0.073, 0.143] |
| GPT-OSS | 0.115 | [0.063, 0.172] |

Interpretation:

> The models encode cue identity almost perfectly, but generated-choice uptake
> is cue-specific and model-constrained. Inequity wording produces the largest
> consistent behavioural shift; selfish cueing remains small. This establishes
> cue-identity encoding and cue-dependent movement, not psychological
> comprehension or compliance.

## Panel-Level Claims For The Caption

Paste-ready main-text caption:

> **Layer B: strategic structure is represented broadly, but recruited
> selectively.** **a**, Incentive sign, canonical choice and an inert stimulus
> control are all decodable from residual activations, showing that decodability
> alone is not the competence claim or a cross-model rationality ranking. **b**,
> Canonical-action information crystallises across depth. **c**, Decision and incentive axes become
> non-orthogonal in all models. For GPT-OSS, panels a-c are read at the
> final-channel commit token rather than the dense-model answer slot. **d**, For
> GPT-OSS, dense MoE gate logits decode incentive sign more strongly than sparse
> selected top-k route summaries, a descriptive gate-versus-route bottleneck
> rather than a router-causal claim. **e**, The recruitment bridge is
> selective: internal incentive projections predict canonical play beyond the
> objective payoff gap for Qwen2.5-I and GPT-OSS, but not reliably for Qwen2.5 or
> Llama. **f**, Cue identity is near-ceiling decodable, but behavioral uptake is
> cue- and model-constrained on cue-identifiable/conflict subsets. No causal
> steering or router-edit result is shown.

Paste-ready appendix caption:

> **GPT-OSS router diagnostic.** **a**, On the corrected Akata one-shot
> substrate, incentive sign is more strongly decodable from router gate logits
> than from sparse selected top-k expert identities or weights; residual results
> are shown only as a reference. This gap compares dense gate logits with sparse
> route summaries and is descriptive, not evidence that routing discards
> strategy. **b**, Across games, stronger gate-level strategic
> evidence is associated with shorter final-channel commitment, with mixed-rate
> variation shown by color. **c**, Gate evidence adds more objective-controlled process
> information about log token length than selected-expert identity. The analysis
> is descriptive and read-only: the fused MXFP4 router path is not edited, so no
> router-causal claim is made.

Safe caption language:

> GPT-OSS is read at its final-channel commit token. In the MoE router, incentive
> sign is strongly decodable from gate logits, while the actual selected top-k
> expert route summaries carry less of that signal. Because this compares a
> dense gate-logit vector with sparse selected-route summaries, it supports a
> descriptive gate-versus-route bottleneck, not a claim that routing discards
> strategy.

> Disposition cues are linearly separable from neural cue-shift vectors at
> near-ceiling accuracy, but their behavioral effects are selective on the
> relevant conflict/identifiability subsets: inequity aversion produces the
> largest consistent movement toward the cue target, while selfish cueing remains
> small.

Unsafe caption language:

- "GPT-OSS router causes strategic choice."
- "Routed expert selection carries the strategy."
- "Routing discards strategic information."
- "Selected experts decode strategic canonical choice near-perfectly."
- "Personality cues are ignored."
- "Cue decodability proves cue use."

## Folder Outputs To Keep

Final main figure/code:

- `analysis/layer_b/fig_layerB_main_v2.py`
- `analysis/layer_b/figures/fig_layerB_main_v2.png`
- `analysis/layer_b/figures/fig_layerB_main_v2.pdf`

MoE appendix/support:

- `analysis/layer_b/router_oneshot_audit.py`
- `analysis/layer_b/router_bottleneck_analysis.py`
- `analysis/layer_b/figures/fig_router_bottleneck.png`
- `analysis/layer_b/figures/fig_router_bottleneck.pdf`
- `analysis/layer_b/tables/router_oneshot_*.csv`
- `analysis/layer_b/tables/router_bottleneck_*.csv`
