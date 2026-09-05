# GPT-OSS uniform-site geometry — why `gptoss_recap/` exists

The released deposit carries **two** GPT-OSS roots. This document explains the difference, which
numbers it superseded, and which still stand. If you are doing cross-row GPT-OSS geometry, read
this before using `substrate/gptoss`.

---

## 1. The ruling chain

**1. The defect.** The original GPT-OSS capture anchored the read-out position to *the model's own
output*: pure answers were captured at the commitment letter, mixed answers at the analysis→final
transition. Cross-row geometry on that substrate is **unidentifiable**:

> the pooled 37.5° angle was **capture-site confounded** (apparent alignment already at layer 0),
> and the pure-only 91.1° "fix" conditioned on the model's own strategy type (**endogenous
> selection, different estimand**). **Neither is a valid headline.**

**2. The fix.** A uniform capture site for every row — the analysis→final transition, which is
present in all streams. All 576 P1-baseline rows are kept. The neural target is the model's
**stated policy** mapped to P(canonical): pure → 0/1, mixed → the stated probability. **35
harness-imputed `default_uniform` rows are flagged out and never imputed → n = 541.** The seeded
binary draw stays behavioural-only (the mixed-strategy rule of `METHODS.md` §4b is unchanged).

**3. Run and audit.** The recapture completed 144/144 games; every stream reproduced bit-identically
(576/576, `val_cos = 1.000`). The repaired release replaced a crash-corrupted game (the fsync defect
that motivated the crash-safe persist path).

**4. Independent reproduction.** The substrate was re-verified array-by-array — **46,080 arrays,
0 corrupt, the layer-0 anchor exactly invariant**; parquet gates 576/576. Geometry was recomputed
from scratch with the locked fusion estimator (policy axis = unit centered OLS slope on
P(canonical); incentive axis = unit centered OLS slope on Δ₁ᶜ; joint game-block permutation null,
`n_perm = 200`, `p = (k+1)/(n+1)`): **final angle 39.03°**, matching an external audit exactly.

---

## 2. Frozen results

Scope `uniform_transition_policy`, n = 541, 144 games.

| variant | final angle | final p | below-null layers |
|---|---|---|---|
| residual · objective (**primary, audited**) | 39.03° | 0.28 | L1–L2 only (no 3-layer onset) |
| residual · own belief | 33.60° | 0.16 | L1–L17, L19 (sustained early–mid) |
| router gate · objective | 41.89° | 0.39 | none |
| router gate · own belief | 40.73° | 0.38 | 6 of 14 early–mid layers |

> **Interpretation bound:** this is a post-reasoning read-out of the *formed* policy — **not a
> localisation of decision formation**; the **CoT-echo caveat applies**. The own-belief variants
> were computed with the same locked machinery but were **not part of the external audit — treat
> them as descriptive pending review.**

Tables: the gptoss blocks of `fusion_depth_table.csv`, `fusion_depth_empirical.csv` and
`oss_router_fusion.csv` (in `data/results/layer_b/fusion/`), and
`recruitment_geometry_depth.csv` (in `data/results/layer_b/recruitment/`). The pre-amendment
originals are preserved alongside them in
`data/results/layer_b/fusion/_pre_uniform_site_20260713/`, with provenance in
`provenance_uniform_site_20260713.json`. A new scope block was appended to
`gptoss_capture_site_sensitivity.csv`, so all four historical scopes are co-located.

---

## 3. What is superseded, what stands

**Superseded:**

- the all-row **37.5°** geometry (site-confounded);
- the pure-only **85.8° / 91.1°** figures (endogenous selection).

Both are retained only as **labeled sensitivities**.

**Stands:**

- all behavioural results (mixed resolved by the seeded draw, all rows);
- game-derived neural targets on 576 rows;
- **all dense-model results** (bit-identical — the amendment touches GPT-OSS only);
- letter-site pure-labeled decodability 0.802, crystallisation, and bridge +0.006, **explicitly
  labeled as letter-site**; a uniform-site respecification of those is a separate open question, not
  part of this release.

---

## 4. Practical consequence

| Doing this | Use |
|---|---|
| Behavioural analysis, any model | `substrate/` |
| Dense-model geometry | `substrate/` |
| GPT-OSS residual/router decodability at a fixed layer | `substrate/gptoss/` |
| **GPT-OSS cross-row geometry / angles / fusion depth** | **`gptoss_recap/`** |

See [`DATA.md`](DATA.md) §2 for the recap schema, including the `capture_site`,
`use_in_neural_target` and `val_*` columns that encode the rules above, and
[`GPTOSS_LAYER_B_HANDLING.md`](GPTOSS_LAYER_B_HANDLING.md) for what may be claimed from either.
