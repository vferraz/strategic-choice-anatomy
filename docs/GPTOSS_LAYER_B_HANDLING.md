# GPT-OSS handling in Layer B

This is the rule for handling `gptoss` / `openai/gpt-oss-120b` in Layer-B neural analyses. GPT-OSS
is a mixture-of-experts model and must not be interpreted as if it were simply a smaller dense
residual model. This document is HC-11 in [`METHODS.md`](METHODS.md).

---

## 1. Core rule

For dense models, the primary Layer-B object is the decision-slot residual stream.

For GPT-OSS, the primary Layer-B object is:

```text
residual stream + MoE router state
```

A weak or misaligned dense-style residual geometry in GPT-OSS is **not** evidence that the model
lacks strategic computation. It may mean the relevant state is expressed partly through the MoE
routing path rather than through a dense residual axis that is homologous to Qwen/Llama.

Do not describe GPT-OSS as "stupid", "inert", or "not representing the game" from residual probes
alone. The correct formulation is architecture-specific:

> GPT-OSS expresses strategic sensitivity through MoE routing rather than the dense residual
> geometry used as the main readout for dense models.

---

## 2. Which root, which environment

- **Residual and router state:** `$SCA_DATA_ROOT/substrate/gptoss/` — accessor
  `config.substrate_root()`.
- **Cross-row geometry:** `$SCA_DATA_ROOT/gptoss_recap/` instead. See
  [`AMENDMENT_uniform_site_geometry.md`](AMENDMENT_uniform_site_geometry.md) and
  [`DATA.md`](DATA.md) §4. This matters — it is the difference between an identifiable and an
  unidentifiable estimand.
- Earlier substrates from this project are **not released** and must not be used for one-shot
  Layer-B claims.

Any run that **instantiates** GPT-OSS (forward passes, router capture) must use the gpt-oss
environment — see [`ENVIRONMENTS.md`](ENVIRONMENTS.md). A generic analysis environment lacks the
required `kernels` path and can trigger the wrong MXFP4 fallback behaviour, which changes the router
code path. CPU-only analysis of already-saved arrays may use the analysis environment, as long as it
does not import or instantiate the model.

The shipped router audit is `analysis/layer_b/router_oneshot_audit.py`; its tables are in
`data/results/layer_b/` (`router_*.csv`, `router_oneshot_audit_report.md`).

---

## 3. Valid Layer-B analyses for GPT-OSS

### 3.1 Residual probes

Valid, but interpret narrowly. Residual probes answer:

> Is this variable linearly decodable from the GPT-OSS residual stream at this layer?

They do **not** answer:

> Is GPT-OSS strategically inert?

Residual-probe results may be compared across GPT-OSS layers. Cross-model comparisons to dense
models require caveats, because:

- GPT-OSS is MoE, not dense;
- the GPT-OSS hidden size differs from the dense models (2880 vs 8192);
- GPT-OSS uses a different precision and execution path;
- strategic structure may be routed rather than residual-local.

### 3.2 Router analysis

Router analysis is **mandatory** for any serious GPT-OSS Layer-B interpretation. The audit on the
released substrate finds:

- collection integrity passes: 144 games, 4,608 rows, 14 router layers;
- P1 baseline has 576 rows, including 525 pure final-channel commitments;
- router gate logits **and** residual state both decode objective incentive sign;
- actual top-k expert selection is **weaker** than gate logits, especially for pure final-channel
  choices;
- pure-choice router behaviour-link beyond objective covariates is **not reliable**.

So GPT-OSS should be discussed as a model with **readable MoE router state** — not as a
dense-residual null model, and not as an established router-causal result.

### 3.3 Causal tests

Residual-only steering is not the correct final word for GPT-OSS, and direct router editing is
blocked for `openai/gpt-oss-120b`. GPT-OSS is excluded from the dense H0–H3 steering because:

- Dense models are steered at the prompt decision slot and regenerated. GPT-OSS commits hundreds to
  thousands of generated tokens later, in the harmony final channel, so **the prompt slot is not the
  decision**.
- The MoE-native causal target would be router patching/editing, but the **MXFP4 fused kernel
  bypasses a patchable `router.forward`**; hooks can fire without changing the routed state.
- **GPT-OSS greedy generation is not bit-identical** under batching or near-tie perturbations over
  long chains, so regenerate-and-compare steering is confounded by model-instability effects.

GPT-OSS router evidence is therefore **read-only / descriptive**. The honest causal framing is
residual/router state → final-channel commitment, with direct router causality deferred to a
non-fused router model.

---

## 4. What not to claim

Forbidden:

- "GPT-OSS lacks strategic representations" from residual AUC alone.
- "GPT-OSS decision forms only at the last layer."
- "GPT-OSS is behaviorally dumb" as a mechanism statement.
- "Router decodability proves causality."
- "GPT-OSS can be included in dense H0–H3 steering."
- "Trait-cue router shifts exist" from the current router evidence.
- "Opponent modeling" for static `delta2_c` / opponent-incentive directions.

The released root contains baseline **and** cue rows, but the router audit is scoped to baseline
choice and objective structure. Trait-router causality would require a separate MoE-native causal
substrate.

---

## 5. How to include GPT-OSS in figures

**Include GPT-OSS in:**

- Layer-A behaviour;
- residual decodability, with the architecture caveat;
- router-vs-residual decodability;
- router gate / top-k audit;
- router–behaviour-link panels;
- MoE causal panels only for a non-fused / editable router substrate.

**Do not pool GPT-OSS naively into:**

- dense-only neural-gain claims;
- dense residual geometry conclusions;
- claims where hidden-dimension units are compared directly to the dense 8192-d models;
- final causal claims based only on residual steering;
- dense H0–H3 causal panels.

If a pooled figure includes GPT-OSS, label it **cross-architecture descriptive**, not a
dense-homologous estimate.

---

## 6. Recommended wording

Safe:

> GPT-OSS departs from the dense-model pattern. Its residual geometry alone does not settle whether
> strategic computation is absent, because GPT-OSS is a reasoning MoE whose decision is made at a
> later harmony final-channel commit. On the corrected one-shot substrate, router gate logits and
> residual state contain strategic information, while actual top-k expert selection and
> behaviour-link evidence are weaker. The result is an MoE-native descriptive audit, not a
> router-causal claim.

If a non-fused MoE causal protocol passes in a future model:

> Dense and MoE models can be tested with architecture-appropriate interventions: residual-stream
> steering in dense models and editable-router patching in an MoE. GPT-OSS-120B itself remains
> read-only under the MXFP4 fused path.

---

## 7. Checklist before any GPT-OSS Layer-B claim

1. Am I using the released one-shot roots, and the **recap** root if this is cross-row geometry?
2. Is the analysis in action / canonical space, not raw letters or raw `act0`?
3. If I ran a forward pass, was it under the gpt-oss environment?
4. Am I separating residual evidence from router evidence?
5. Am I avoiding dense-model language such as "the decision forms at the final layer"?
6. Am I treating router decodability as descriptive / read-only for GPT-OSS-120B?
7. Am I clear that dense H0–H3 steering excludes GPT-OSS?

If any answer is no, fix the analysis or the wording before proceeding.
