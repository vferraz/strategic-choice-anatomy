# SPEC — Regenerate ALL missing Layer-B paper tables (evidence freeze, end-to-end)

PI-approved 2026-07-09. Execute top to bottom. This spec is self-contained:
do not rely on chat context. Status of each item is tracked in §9 — update it
as you go.

## 0. Hard rules (violations invalidate the run)

1. **Substrate root, fixed:** `output/oneshot_akata_main_dl/` (corrected Akata
   one-shot collection; per-game `config.json` says `substrate: akata_oneshot`,
   spec `docs/AKATA_ONESHOT_RECOLLECTION.md`). No other root, ever. Do not
   infer roots from older docs (`oneshot_main`, `oneshot_commit*` are legacy).
2. **Never read `analysis/_archive/`, `_legacy/`, or `docs/legacy/` for values
   or code.** Archived Layer-B tables are confirmed STALE (old 16-cell
   substrate: llama incentive AUC 0.545 vs paper 0.767; n_obs=2304 not 576).
3. **Decision label = realized generated move.** Dense: `decoded_action`
   gated by `parse_ok`. GPT-OSS: `realized_action`, excluding
   `commit_type == "none"`. Never a slot-probability argmax.
4. **Canonical axis:** `canonical_action_p1` from
   `datasets/processed/game_features.csv`; per-row aligned label =
   `(action == canonical_action_p1)`. Incentive Δ1c from the committed
   `analysis/layer_c/tables/incentive_delta1c.csv`
   (canonical-signed, q=0.5).
5. **Probe protocol:** logistic regression (sklearn defaults, lbfgs,
   max_iter≥1000, standardized features), GroupKFold(5) grouped by game,
   out-of-fold predictions only. CIs: bootstrap-by-game, n=1000,
   **seed 20260627**.
6. **Read positions:** dense models at the answer-slot capture
   (`p{player}_{cond}_cb{c}_l{L}` keys in `acts.npz`, L0–L80); GPT-OSS at its
   commit capture (same key format, L0–L36) and `router.npz`
   (`..._l{L}_gate/_idx/_w`, 14 layers). Never `_seq` keys.
7. **Strategic probes use P1 baseline rows only** (4 counterbalance cells per
   game, 576 rows/model/layer). Cue analyses use the cue conditions.
8. **Acceptance rule:** every regenerated headline value must match the tex
   target (§ per table) within rounding/CI. On material mismatch: STOP that
   table, write the mismatch into §9, do NOT touch `paper/paper_NHB/*.tex`,
   do not "fix" by trying unlisted estimator variants. (One bounded variant
   round per table is pre-authorized where noted, nothing beyond it.)
9. **Git:** `.gitignore` has a blanket `*.csv` — every table must be
   `git add -f`-ed. Commit tables + a `provenance.json` per output dir
   (schema: copy `analysis/layer_b/recruitment/tables/provenance.json`).

## 1. Proven infrastructure (reuse, don't reinvent)

- Loaders/estimator helpers: `analysis/layer_b/rebuild/rebuild_lib.py`
  (`games`, `meta`, `load_game_rows`, `realized_aligned`, `boot_ci`).
- Reference implementations: `rebuild_bridge.py`, `rebuild_bridge_variants.py`
  — the bridge was reconstructed and VERIFIED with these (all tex values incl.
  CIs). The conventions they embody are the house style for everything below:
  OOF projections, game-mean then **z-score across games** for game-level
  regressors, final-layer read where a single layer is reported.
- Consumer schemas: `analysis/layer_b/fig_layerB_main_v2.py` defines the
  exact columns each table must have — read its loaders before writing CSVs:
  L109 (`b1_decodability.csv`), L160 (`b1_crystallization.csv`),
  L196 (`recruitment_geometry_depth.csv`), L240 (`router_bottleneck_depth.csv`),
  L288 (`within_model_bridge.csv`, already done), L368
  (`disposition_dissociation.csv`).
- Environment: repo `.venv` (numpy/pandas/pyarrow/sklearn present).

## 2. Table A — `analysis/layer_b/tables/b1_decodability.csv`

Final-layer (L80 dense / L36 gptoss) held-out AUC per model × concept:
- `incentive sign` = sign(Δ1c), P1-baseline rows; drop Δ1c==0 games.
- `canonical action` = realized aligned choice for dense models; for GPT-OSS,
  literal pure commitments only (525 rows across 144 games). Mixed rows remain
  eligible for game-target probes because those labels do not use the seeded
  binary realization.
- `stimulus control` = rank of P1's payoff in canonical cell (0,0), 3-class
  {1,2,3} (balanced 48/48/48; compute from the canonical payoff vectors via
  `datasets/processed/taxonomy/equivalence_per_canonical.csv` `canonical_p1`
  string "u00,u01,u10,u11"), macro-AUC (ovr).
Schema: match fig loader L109-ff (model, concept, auc, lo, hi, …).

**Acceptance (tex, Results "Strategic variables are broadly decodable"):**
incentive sign 0.869 / 0.847 / 0.767 / 0.894 (qwen-I / qwen / llama / gptoss);
canonical action 0.866 / 0.827 / 0.787 / 0.802;
control 0.834 / 0.802 / 0.829 dense, gptoss 0.567 [.532,.601].
Pre-authorized variant round if mismatch: peak layer instead of final;
row-level vs game-level bootstrap.

## 3. Table B — `analysis/layer_b/tables/b1_crystallization.csv`

Canonical-choice probe (Table-A target 2) at EVERY layer (dense 0–80, gptoss
0–36), per model: held-out AUC + CI + gain-over-embedding(L0).
Schema: fig loader L160-ff.

**Acceptance:** L0 floors 0.441 / 0.429 / 0.418 (dense);
peaks 0.917@L62 / 0.898@L63 / 0.841@L60; final-layer 0.866 / 0.827 / 0.787;
gptoss pure-commitment floor 0.336, peak 0.845@L21, final 0.802.
This is the heavy job (~4×81 probe fits × 5 folds): cache per-layer matrices
(float32), log progress per layer, expect ≥1 h.

## 4. Table C — `analysis/layer_b/recruitment/tables/recruitment_geometry_depth.csv`

Per model × layer: oriented angle between
`d_dec` (diff of class means, canonical vs non-canonical realized choice) and
`d_inc` (centred covariance with continuous Δ1c), both on P1-baseline rows,
plus a joint game-block permutation null (move the choice and Δ1c target block
together, all four forms per game, ≥200 draws) → null median + 95% band and a
below-null flag. GPT-OSS primary geometry uses the 108 games whose four cells
are all literal pure commitments; all-row and 525-pure-row sensitivities are
reported separately.
Schema: fig loader L196-ff.

**Acceptance (qualitative, from the tex fusion section + committed
`analysis/layer_b/fusion_associative/FUSION_AND_STORYLINE_MEMO.md`):**
all three dense models cross below the null only late (Qwen-I L56/depth 0.70,
Qwen base L56/0.70, Llama L37/0.46). GPT-OSS pure-complete geometry has no
three-layer onset and ends near orthogonal (91.10°); its former low-angle trace
is a heterogeneous capture-site sensitivity, not a headline result.

## 5. Table D — `analysis/layer_b/recruitment/tables/disposition_dissociation.csv`

Per model × cue (risk/loss/inequity/maximin/selfish + placebo):
- representation axis: cue-shift vectors (cue − matched-baseline residual,
  same game & cb, final layer), PCA (fit on train games) → Fisher LDA on the
  5 cue labels, GroupKFold-by-game held-out accuracy; placebo projected,
  reported separately.
- behaviour axis: conflict-set uptake — reuse the committed Layer-A pipeline
  outputs (look in `analysis/layer_a/`; do NOT recompute the
  behavioural definitions from scratch; if the Layer-A table with per-cue
  uptake is absent, STOP and report).
Schema: fig loader L368-ff (incl. `behavior_magnitude`, `_lo`, `_hi`).

**Acceptance:** held-out LDA accuracy range 0.969–1.000 across displayed
cue×model points; uptake means ≈ 0.643 (inequity), 0.311/0.309/0.331
(risk/loss/maximin), 0.093 (selfish).

## 6. Table E — `analysis/layer_b/tables/router_bottleneck_depth.csv`

Use the COMMITTED generator `analysis/layer_b/router_bottleneck_analysis.py`
(read it first; point its root at `output/oneshot_akata_main_dl/gptoss`).
Targets: sign(Δ1c) decoded from gate logits vs top-k weights vs top-k expert
set vs residual, per captured router layer.

**Acceptance:** gate 0.891 [.851,.925] @L18; top-k weights 0.640 [.585,.693];
top-k set 0.609 [.554,.661]; best residual 0.915; best-per-substrate across
layers as in tex panel d.

## 7. Fusion tables — `analysis/layer_b/fusion_associative/tables/`

The fusion figure's input tables are also missing. Regenerate via the
COMMITTED scripts in `analysis/layer_b/fusion_associative/`
(`build_fusion_figures.py`, `oss_router_fusion.py`) against the fixed root;
then rebuild `fig_fusion_combined.pdf` and compare to the committed PDF.
Acceptance: memo values (§4) + figure reproduces.

## 8. Finish line (in order)

1. Rebuild `fig_layerB_main_v2.pdf` from the restored tables; visually compare
   with the committed PDF (smoke test — CI whiskers may differ by bootstrap
   noise; bars/points must not).
2. `git add -f` every new CSV + provenance.json; commit with a message citing
   this spec; keep generators committed too.
3. Update §9 below and `docs/PIPELINE_STATE.md` (short block: Layer-B tables
   regenerated + verified, evidence freeze item 2 closed).
4. Do NOT edit the tex. If any acceptance failed, the report in §9 is the
   deliverable for that table.

## 9. STATUS (update as you go)

- [x] within_model_bridge.csv — GPT-OSS population corrected 2026-07-12.
      Dense estimates are unchanged. GPT-OSS pure-only raw=.0261
      [.0015,.0538], partial=.0055 [-.0292,.0368]; the former heterogeneous
      partial=.0716 is retained only as a sensitivity.
- [x] b1_decodability.csv — regenerated 2026-07-10 with the recovered
      corrected-era generator. All 12 headline AUCs match the TeX to rounding:
      incentive sign 0.869 / 0.847 / 0.767 / 0.894; canonical action
      0.866 / 0.827 / 0.787 / 0.802; stimulus control
      0.834 / 0.802 / 0.829 / 0.567 (qwen-I / qwen / llama / gptoss).
- [x] b1_crystallization.csv — GPT-OSS pure-commitment curve corrected
      2026-07-12: floor 0.336, peak 0.845@L21, final 0.802. Dense values are
      unchanged.
- [x] recruitment_geometry_depth.csv — shipped 2026-07-10 from the passing
      fusion joint-null implementation (§10.1/§11-approved alternative), in
      the main-figure schema and corrected to four-form game blocks on
      2026-07-12. Onsets: Qwen-I 0.70, Qwen base 0.70, Llama 0.46; GPT-OSS
      pure-complete has no onset and final angle 91.10°.
- [x] disposition_dissociation.csv — regenerated 2026-07-10 with the recovered
      original protocol. Displayed cue×model LDA accuracies range from
      0.96875 to 1.000 (0.969–1.000 to TeX precision); behavioral uptake means
      match exactly (inequity 0.643, risk 0.311, loss 0.309, maximin 0.331,
      selfish 0.093).
- [x] router_bottleneck_depth.csv — regenerated 2026-07-10 from the committed
      generator and fixed root. Headline points/layers match: gate 0.891@L18,
      top-k weights 0.640@L18, top-k set 0.609@L18, best residual 0.915@L21.
      Seed-20260627 gate CI is [0.852,0.929] vs TeX [0.851,0.925]; recorded in
      provenance as a small endpoint difference.
- [x] fusion_associative/tables/ — regenerated 2026-07-10 from committed
      scripts and corrected 2026-07-12 for game blocks and GPT-OSS capture
      sites. Pure-complete GPT-OSS has 0/36 objective residual and 0/14 router
      layers below null (also 0/36 and 0/14 under empirical belief).
- [x] fig_layerB_main_v2.pdf rebuilt + compared 2026-07-10 — all bars, traces,
      points and intervals reproduce the accepted evidence; the rebuilt PDF
      uses an explicit white page instead of the committed transparent page.
- [x] committed + PIPELINE_STATE updated — all Layer-B tables, provenance,
      generators and the rebuilt main figure are committed; evidence-freeze
      item 2 is CLOSED. No TeX was edited and no TEX_DELTA_REPORT is needed.

---

## 10. ROUND 2 (PI-authorized 2026-07-10) — targeted fixes for A–D

Execute after reading §9. Same hard rules (§0). Update §9 when done.

### 10.1 Table C — rebuild from the PASSING fusion implementation

Diagnosis: §4's null was mis-specified by the spec author. The paper's null
(tex fusion section; FUSION_AND_STORYLINE_MEMO.md) is a **joint game-level
permutation that PRESERVES the choice–incentive correlation** — not a
Δ1c-only permutation. The committed fusion pipeline passed §7 with a
byte-identical figure, so it IS the paper's implementation.
Action: derive `recruitment_geometry_depth.csv` from the fusion pipeline's
angle/null machinery (`build_fusion_figures.py`), reshaped to the fig-loader
L196 schema. Acceptance after the 2026-07-12 estimator audit: dense onsets
0.70/0.70/0.46 (Qwen-I/base/Llama); GPT-OSS pure-complete has no onset. The
former all-row GPT-OSS alignment is a capture-site sensitivity only.

### 10.2 Table D — one inclusion variant

Test with `commit_type=none` cue cells INCLUDED (the likely original,
pre-dating the exclusion insight). If gptoss LDA ≥0.969 → original
identified; commit with a provenance note that the CORRECT protocol
(exclusion) gives 0.965. If not → adopt 0.965 with the same note. Either
way the table ships this round.

### 10.3 Tables A/B — staged protocol grid (bounded, early-exit)

Search ONE protocol that matches ALL Table-A targets simultaneously
(tolerance ±0.005 per AUC; model ordering must reproduce). Stages, defaults
held elsewhere; stop the moment a full match is found:
1. rows × layer: {P1 baseline; P1 baseline+cues; P1+P2 baseline} ×
   {final; per-target-peak}.
2. folds × standardization: {GroupKFold 5, 10, leave-one-game-out} ×
   {per-fold, global, none}.
3. C ∈ {0.01, 0.1, 1.0} × class_weight {None, balanced} ×
   AUC {pooled-OOF, per-fold mean} × control-variant {3-class rank00
   macro-ovr; binary cell(0,1)==4}.
Log every cell to `out/gridA_{model}.csv`. Declare IDENTIFIED only if a
single cell matches ≥11/12 Table-A headline values within tolerance; then
rebuild Table B (crystallization, all layers) under that exact protocol and
check its targets (§3).

### 10.4 Fallback (pre-authorized — do NOT ask again, do NOT edit tex)

If no grid cell qualifies: adopt the §0-protocol regenerated values as the
paper's new source. Commit the §0-default A and B tables (and 10.2's D
outcome), then write `analysis/layer_b/rebuild/TEX_DELTA_REPORT.md`
listing every tex sentence/number affected (old → new, tex line refs,
including the panel-a AUCs, crystallization floors/peaks/finals, LDA range,
and any caption edits). The PI + writing session apply the tex changes;
the agent never touches `paper/`.

### 10.5 Finish

Rebuild `fig_layerB_main_v2.pdf` from whatever tables shipped; commit
(`git add -f`) tables + provenance + grid logs; update §9 and
`docs/PIPELINE_STATE.md`; state clearly whether freeze item 2 is CLOSED
(all tables committed) and whether the tex needs the delta report.

---

## 11. ROUND 3 (2026-07-10) — ORIGINAL GENERATORS RECOVERED. §10.3's grid is CANCELLED.

The corrected-era generator code was found in `_archive` (read/restore
exception PI-granted 2026-07-10) and RESTORED to active paths:

- `analysis/layer_b/` — top-level `lib.py` (SUBSTRATE already =
  `output/oneshot_akata_main_dl`) + `build_decodability.py` (→
  `tables/b1_decodability.csv`) + `build_crystallization.py` (→
  `tables/b1_crystallization.csv`). NOTE: this dir's `src/` subpackage and
  the leftover `_data/*.parquet` caches are the OLD-substrate
  implementation (llama incentive 0.545, n_obs=2304) — do NOT run `src/`,
  and DELETE the stale `_data/b1_*.parquet` caches before running if the
  build scripts read them (check `lib.py` cache logic first). The old
  tracked tables were quarantined to
  `analysis/_archive/stale_layer_b_tables_20260710/`.
- `analysis/layer_b/recruitment/src/` — `build_all.py::main()` builds
  disposition (D), depth+geometry (C: `recruitment_geometry_depth.csv`),
  the bridge (already verified — its output must agree with the committed
  `tables/within_model_bridge.csv`; if it differs, STOP and report), and
  status tables, then renders its figure. Original run's provenance kept at
  `tables/provenance_original_20260627.json`.
- `analysis/oneshot/{__init__.py,_oneshot_common.py}` — probe_auc
  dependency restored (minimal).
- Import fix applied (cleanup rename): `analysis.layer_a.*` →
  `analysis.layer_a.*` in `layer_b/lib.py`,
  `layer_b/build_geometry.py`, `layer_b/recruitment/src/shared.py`.

**Run (repo root, .venv):**
```bash
python3 analysis/layer_b/build_decodability.py
python3 analysis/layer_b/build_crystallization.py
python3 -m analysis.layer_b/recruitment.src.build_all
```

**Acceptance:** §2/§3 targets for A/B; §4 memo pattern for C; §5 for D
(the committed-fusion-derived C from §10.1 remains a valid alternative if
build_all's C output differs — prefer whichever matches the memo pattern
AND the fig schema). Bridge output must reproduce the committed table.
Then §10.5 finish line. §10.4 fallback still stands if — unexpectedly —
the original generators do not reproduce the tex.

### §11 STATUS (updated 2026-07-12 after estimator/population audit)

- ✅ **B `b1_crystallization.csv` — population-corrected:** dense values
  unchanged; GPT-OSS pure commitments floor 0.3360, peak 0.8451@L21,
  final 0.8024.
- ✅ **D `disposition_dissociation.csv` — DONE + VERIFIED:** displayed
  cue×model LDA range 0.96875–1.000 (0.969–1.000 to TeX precision; round-1's
  0.965 was its reconstruction, not the original protocol); uptake means
  exact 5/5 (0.643/0.311/0.309/0.331/0.093).
- ✅ **C `recruitment_geometry_depth.csv` — DONE + VERIFIED:** the original
  generator's angle trace was regenerated, then the §10.1/§11-approved fusion
  reshape was selected because it adds the memo-consistent joint-null fields
  in the main-figure schema (onsets 0.70/0.70/0.46; GPT-OSS pure-complete no onset).
- ✅ **Bridge population-corrected:** GPT-OSS pure-only raw .026, partial
  +.006 (CI spans zero); heterogeneous old estimate is sensitivity.
- ✅ **A `b1_decodability.csv` — population-corrected:** incentive
  sign 0.869/0.847/0.767/0.894; canonical action
  0.866/0.827/0.787/0.802; stimulus control 0.834/0.802/0.829/0.567.
- ✅ **§10.5 finish line — DONE:** main figure rebuilt and visually checked,
  §9 and PIPELINE_STATE updated, evidence-freeze item 2 CLOSED. The cancelled
  §10.3 grid was not used. No TeX delta report is required. Substrate untouched
  throughout (all `output/oneshot_akata_main_dl` mtimes = Jun 24–25).
