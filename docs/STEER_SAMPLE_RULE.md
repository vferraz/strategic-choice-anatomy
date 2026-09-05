# Steering sample and sequential stopping rule

**Pre-registered: locked before any steering result was seen.**

**Purpose.** One fixed, stratified game sample for ALL steering (every mode × dose × direction ×
layer × cb × model), with a precision-based early stop so weeks of GPU are not spent past the point
of diminishing returns. Locked **before** any steering outcome was observed — the stop criterion is
**precision, never significance** (independent of the effect's sign → no p-hacking, no forking
paths).

## Sample

`data/manifests/steer_sample_54.csv` — **54 games, 9 per family** (MP, DD, OD1, OD2, CO1, CO2 = the
`nagel_lk_type` structural archetypes; family ≈ `num_pure_ne` ≈ dominance/iesds, so one axis
stratifies the structure). Within family the draw is deterministic and balanced on
`canonical_action_p1` (the CO families are structurally lopsided → the action imbalance is
**reported, not fixable**). The **same 54** are used for all models, so cross-model and
cross-condition contrasts are paired within game.

## Wave order (every boundary is stratified)

Ordered into **9 waves of 6 games (1 per family)**, so **any prefix is itself balanced**. Stopping
early still leaves a valid stratified subset.

- after wave 3 = 18 games · wave 5 = 30 · wave 7 = 42 · wave 9 = 54.

## Sequential stopping rule (PRECISION, not significance)

Run waves in order. Starting at **wave 5 (30 games, above the ~30-cluster floor for cluster-robust
and mixed-model inference)**, after each wave compute the **95% CI half-width of the primary
standardized steering estimate** (the canonical-pref dose-response slope, game as random effect /
game-clustered SE).

- **STOP** when the CI half-width ≤ **0.10 SD**.
- Otherwise continue to the next wave, up to wave 9 (54).
- Report the wave at which it stopped plus the achieved precision; state results as
  "powered for |d| ≥ X."

The criterion is the **CI width**, which does not depend on whether the effect is non-zero — so
stopping early cannot inflate false positives, unlike stop-at-p<0.05.

## Expansion reserve (also pre-declared)

If wave 9 (54) does **not** meet the precision target, expand in **stratified increments to a
90-game tier** (+6 per family where available; MP capped at 18, CO capped at 9 → the big families
absorb the rest), same protocol, poolable with the 54. Expansion is triggered by the **precision
miss declared above**, never by peeking at the result. (The substrate covers all 144 games already,
so expansion is steering-compute only — no re-collection.)

## Why this is defensible

1. Locked a priori; stratified on game **structure**, not outcome.
2. The game is the inference unit (random effect / clustered SE); cb × dose × layer are within-game
   precision, not replication.
3. Stop on **precision**, expand on a **pre-declared precision miss** — both independent of
   significance.
4. Equal-per-family protects the rare and interesting strata (MP mixed, CO coordination); reweight
   to the 144 marginals (36/36/36/18/9/9) for population-level estimates.

Cost is cut by **scope** (this sample), never by changing what is measured — the causal test
regenerates the decision, exactly as the substrate collection does.

See [`METHODS.md`](METHODS.md) §5.4 for the realized composition of the sample and §6.2 for the
results it produced.
