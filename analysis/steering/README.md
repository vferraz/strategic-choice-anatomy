# Akata Small-Dose Steering Analysis

> **Historical snapshot — this describes the PRE-CORRECTION arm.** Generated 2026-07-05, before the
> q = 0.5 construct-identity correction. Every number below was computed on the empirical-belief
> incentive axis, which the release supersedes; the released arm and its numbers are in
> [`docs/METHODS.md`](../../docs/METHODS.md) §6.2–§6.4 and
> `data/results/steering/smalldose_summary_q05/`. This file is kept unrevised as the record of what
> the pre-correction analysis said.

Generated 2026-07-05 from the pre-correction small-dose arm
(`$SCA_DATA_ROOT/steering/smalldose/`).

## Headline

The small-dose steering package is complete and the apparatus clearly moves the models, but the dominant effect is still letter/presentation control rather than a stable cross-model canonical strategic intervention.

The cleanest canonical strategic signal is localized to **Qwen at L65**, especially `h2_choice`:

| model | mode | layer | contrast | canonical slope diff vs random | 95% bootstrap CI |
|---|---:|---:|---|---:|---:|
| qwen | h2_choice | 65 | main-random | +0.0427 | [+0.0191, +0.0667] |
| qwen | h2_choice | 65 | main_perp-random | +0.0473 | [+0.0241, +0.0725] |
| qwen | h1_dinc | 65 | main-random | +0.0150 | [+0.0007, +0.0292] |
| qwen | h1_dinc | 65 | main_perp-random | +0.0171 | [+0.0048, +0.0312] |
| qwen_instruct | h1_dinc | 65 | main_perp-random | +0.0506 | [+0.0011, +0.0983] |
| llama31_instruct | h2_choice | 65 | main-random | -0.0360 | [-0.0651, -0.0049] |
| llama31_instruct | h2_choice | 65 | main_perp-random | -0.0373 | [-0.0701, -0.0021] |

Everything else is either near zero after counterbalance, unstable across games, opposite-signed, or collapses after letter-orthogonalization.

## Audit

| item | value |
|---|---:|
| rows | 54,432 |
| models | qwen, qwen_instruct, llama31_instruct |
| modes | h1_dinc, h2_choice |
| layers | 65, 79 |
| variants | main, main_perp, random |
| doses | -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25 |
| games | 54 |
| status values | OK |
| parse-failure rows | 66 |

The 66 parse failures are all `qwen_instruct`, concentrated at high positive L79 doses: `h1_dinc/main` has all 54 games failing at `dose=+0.25` plus one at `+0.10`; `h2_choice/main` has 5 at `+0.25`; `h2_choice/main_perp` has 6 at `+0.25`. Slot-probability readouts remain available, but regenerated-decision behavior at those cells should be treated cautiously.

## Interpretation

The key decomposition is counterbalance-based:

- `d_letter`: change in `pref(J)`.
- `d_act0`: change in action-space `pref(act0)`.
- `d_canon`: change in canonical strategic `pref(canonical_action_p1)`.
- Behavior columns repeat the same logic for regenerated decisions, with parse failures treated as missing.

The letter slopes are often huge. Examples: `llama31_instruct h1_dinc L79 main` has letter slope `+2.5934` with a full `1.0000` letter range; `qwen_instruct h1_dinc L65 main` has `+1.3606`; `qwen h1_dinc L65 main` has `+1.0231`. This confirms the small-dose arm still strongly controls the next-option letter.

But canonical dose coherence is near 50% in the main high-movement cells. That is the giveaway: after the four counterbalance cells are averaged, the signed strategic effect mostly cancels. Per-cell flips can be very high while the counterbalanced canonical slope remains small, so raw flip rates are not evidence of strategic steering by themselves.

## Model Notes

**Qwen.** Best evidence for a real canonical effect. `h2_choice L65` is the most convincing: main and main_perp both beat random with paired game-cluster CIs away from zero, and the effect survives letter orthogonalization. `h1_dinc L65` is smaller but positive. L79 is mostly letter/presentation movement with weak canonical change.

**Qwen-Instruct.** Strong letter movement and large canonical ranges in `main`, but signed canonical slopes are game-heterogeneous and mostly CI-overlapping zero. The only paired canonical contrast away from zero is `h1_dinc L65 main_perp-random`, and it is borderline. At L79, `main_perp` largely collapses both letter and canonical movement.

**Llama.** No positive strategic recruitment result. It shows very strong letter movement, especially L79, but canonical slopes are near zero or negative. `h2_choice L65` is significantly negative relative to random in the paired contrast.

## Files

- `analyze_akata_smalldose.py` - reproducible analysis script.
- `tables/audit_summary.csv` - package completeness and parse-failure audit.
- `tables/group_summary.csv` - per model/mode/layer/variant slopes, ranges, flip rates, coherence.
- `tables/contrast_summary.csv` - paired game-level contrasts against random.
- `tables/per_game_slopes.csv` - game-level slopes and ranges.
- `tables/per_cell_ranges_flips.csv` - per-cell ranges and decision flips.
- `tables/family_summary.csv` - summary by game family.
- `figures/slope_letter_vs_canonical.png` - letter slope vs canonical slope.
- `figures/canonical_range_and_flips.png` - counterbalance-mean canonical ranges and per-cell flips.
- `figures/dose_curves_l79.png` - L79 dose curves, with dashed letter and solid canonical components.

Re-run:

```bash
uv run python analysis/steering/analyze_akata_smalldose.py
```

## Limitation

At the time this report was written the `delta_tables_{model}.csv` moderation inputs were not
available, so the optional `delta1_c` moderation block was not run; the report above uses only the
steering package plus versioned game metadata. Those tables **do** ship in this release, at
`data/results/steering/summary/`, so `analysis/steering/analyze_smalldose.py` now runs its
moderation block — but against the pre-correction arm, which is the only arm it reads.
