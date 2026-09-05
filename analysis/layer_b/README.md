# Layer B Redesign — One-Shot Main Figure

Additive redesign for the Layer B paper figure on the corrected integrated
one-shot substrate. Dense causal steering data are intentionally excluded here.
GPT-OSS is excluded from dense H0-H3 steering because its decision is the later
harmony final-channel commit and its MXFP4 fused router path is readable but not
editable.

## Regenerate

```bash
.venv/bin/python analysis/layer_b/router_oneshot_audit.py
.venv/bin/python analysis/layer_b/fig_layerB_main_v2.py
```

Outputs:

- `figures/fig_layerB_main_v2.{png,pdf}`
- `tables/router_oneshot_*.csv`
- `tables/router_oneshot_audit_report.md`

Panel D/F methods, findings, provenance, and safe storyline are documented in:

- `METHODS_AND_FINDINGS.md`

## Current Figure Contract

Main claim: **strategic competence is recruitment, not representation**.

The figure should establish a descriptive ladder, not a causal claim:

1. Strategic variables are broadly represented.
2. Choice-related representations form across depth.
3. Decision and incentive axes align late in the dense models; GPT-OSS
   pure-complete geometry remains near orthogonal.
4. Recruitment is selective: represented incentive predicts choice strongly in
   some models, but only Qwen-Instruct shows a reliable partial bridge beyond
   objective Delta1c in the population-corrected current table. GPT-OSS's former
   bridge was driven by mixing pure and mixed capture sites.
5. Cue axes are represented, but cue effects are trait/model dependent rather
   than a blanket "heard, not obeyed" result.
6. GPT-OSS is MoE-native: router state must be audited, but router decodability
   is not router use.

## Router Correction

The old v2 wording overclaimed that GPT-OSS strategy was "carried by routed
expert selection." The corrected audit does not support that statement.

On `output/oneshot_akata_main_dl/gptoss`:

- `router_gate` decodes incentive sign well: best AUC 0.891
  [0.851, 0.925].
- Commit residual also decodes incentive sign well: best AUC 0.915
  [0.878, 0.948].
- Actual top-k expert load is weaker for incentive sign: best AUC 0.714
  [0.657, 0.766].
- For primary pure final-channel choices, `router_gate` is moderate
  (AUC 0.724 [0.646, 0.792]), while actual top-k expert selection is weak
  (`router_topk_binary` AUC 0.531 [0.410, 0.652]).
- Literal J/P choice is almost perfectly decodable from gate/residual state, and
  from top-k weights, so choice readout must be interpreted as final-token state
  evidence, not as a strategic mechanism by itself. This near-perfect readout
  does not hold for primary pure canonical choice.
- Pure-choice log-loss improvement beyond objective Delta1c/counterbalance
  covariates is not reliable for gate or top-k. Therefore router evidence is
  descriptive and should not be used as a causal mechanism claim.

Interpretation: GPT-OSS router gate logits contain strategic information, but
the gate-versus-route gap compares a dense gate-logit vector with sparse
selected-route summaries. The current one-shot data do not show that selected
experts independently drive pure choices beyond objective covariates, and they
do not establish that routing discards strategic information. Router analysis
belongs in the main figure only as an MoE guardrail/audit, or in supplement if
the main figure needs to be tighter.

## GPT-OSS Steering Constraint

GPT-OSS should not appear in dense H0-H3 causal steering panels:

- The dense decision-slot intervention site is not the GPT-OSS decision. GPT-OSS
  commits later in the harmony final channel.
- The faithful MoE intervention would edit router state, but the MXFP4 fused
  kernel bypasses a patchable `router.forward`; router state is readable, not
  editable.
- Greedy regeneration can be unstable over long chains under near-tie MXFP4
  perturbations, so regenerate-and-compare steering is confounded.

The honest causal statement is therefore pending/non-applicable for GPT-OSS-120B:
current evidence is descriptive residual/router state at commit time. A future
non-fused bf16 MoE track, e.g. Qwen3.6, is the right substrate for router-causal
experiments.

## GPT-OSS Decision Process Metric

There are no stored discrete "reasoning rounds." The available process measure is
`n_new_tokens` before final-channel commitment. Verified on all 4,608 corrected
GPT-OSS cells:

- pure commits: median 221 new tokens, median generation time 9.6 s;
- mixed commits: median 951 new tokens, median generation time 41.35 s;
- P1 baseline per-game median ranges from 180 to 1,232.5 new tokens.

This is useful as a descriptive process supplement. It should not be confused
with level-k iteration count, because the stored `final_text` is truncated and
does not preserve the full reasoning trace needed for that annotation.

## Data Guardrails

- Primary root: `output/oneshot_akata_main_dl/`
- GPT-OSS router root: `output/oneshot_akata_main_dl/gptoss`
- Behaviour target: dense realised/generated P1 action; GPT-OSS choice-dependent
  headlines use literal pure commitments, aligned to `canonical_action_p1`
- GPT-OSS primary choice target for router audit: pure final-channel commits.
  Mixed rows are robustness/comparability only because their realized action is
  seeded resolution, not a pure action commitment.
- No raw `act0`, raw `decoded_action == 0`, or A/B label aggregation.
- No GPT-OSS causal or router-edit claim for the MXFP4 fused path.
