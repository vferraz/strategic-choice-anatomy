# datasets/processed/

Derived datasets used as canonical references throughout the project. Built deterministically by scripts in `../../scripts/`.

## Files

### `game_features.csv` / `game_features.parquet` — game-theoretic predictions + structural features

- **144 rows × 45 cols.** Built by `../../scripts/compute_game_features.py`.
- Covers exactly the 144 Bruns / Robinson-Goforth canonical games. Do not add
  `Nag_*` or `Grif_*` reference rows here; human matching lives in the taxonomy
  tables below.
- **Per-game action predictions** (deterministic from payoff matrix):
  - `canonical_action_p1`, `canonical_action_p2` — **always-defined single binary** via precedence: dominant > unique pure NE > payoff-dominant NE > risk-dominant NE > maximin. Use this for any cross-game alignment metric.
  - `dominant_action_p1/p2`, `nash_action_p1/p2`, `payoff_dominant_ne_p1/p2`, `risk_dominant_ne_p1/p2`, `l0_action_p1/p2`, `l1_action_p1/p2`, `l2_action_p1/p2`, `p1_maximin_action`, `p2_maximin_action`, `cooperative_action_p1/p2` — per-axis predictors (some null where axis doesn't apply).
- **Game-level covariates** (continuous; for correlation / regression / stratification):
  - `complexity_score` (always defined; 132 rows use 3 components, 72 use 4 — see `complexity_score_n_components`).
  - `payoff_variance`, `payoff_conflict`, `iesds_depth`, `num_pure_ne`, `dominance_profile`, `ne_distributional_conflict`, `ne_pareto_rankable`.
- The legacy human-reference columns (`frac_choose_A`, `entropy_binary`,
  `lk_type`, `emp_class`, `up_choice`, `topology`, `delta_norm`) are preserved
  for schema compatibility but are null in this Bruns-only table. For human
  data and class labels, use the taxonomy/unified-pairs tables below.

### `taxonomy/` — Bruns ↔ Nagel ↔ Griffiths equivalence hub

See `taxonomy/EQUIVALENCE_DATASET.md` for the full narrative. Quick reference:

- `taxonomy/equivalence_per_canonical.csv` (144 rows) — one row per Bruns canonical, with:
  - `nagel_lk_type`, `nagel_emp_class`, `nagel_rgg_class` — Nagel class labels.
  - `nagel_frac_choose_act0_canonical` — Nagel/Rosemarie human rate on the
    canonical-form **act0** axis.
  - `clean_grif_p1_mean`, `clean_grif_p2_mean`, `clean_grif_n_games`,
    `clean_grif_p1_std`, `clean_grif_p2_std` — **strict** Griffiths robustness
    centroids (rank-clean plus cardinal-L1 agrees with ordinal-L1; 35 of 78
    paired games covered). This is not the main full-coverage Griffiths
    behavioral reference.
  - `canon_l1_p1`, `canon_l1_p2` — L1 predictions per canonical.
- `taxonomy/equivalence_per_game.csv` (78 rows) — player-swap-collapsed view.
- `taxonomy/equivalence_clean_griffiths.csv` (279 rows) — long-form per clean Griffiths game.
- `taxonomy/canonical_master.csv`, `taxonomy/bruns_to_canonical.csv`, `taxonomy/nagel_to_canonical.csv`, `taxonomy/griffiths_to_canonical.csv` — the underlying hub-and-spoke tables.

### `analysis/_shared/datasets/unified_pairs.parquet` — main behavioral reference

This is currently the closest thing to a final long-form human/model dataframe:

- 7,466 rows × 92 columns.
- One row per `(sample, condition, game)` for LLMs, Nagel/Rosemarie, and
  Griffiths.
- Includes all 144 Bruns canonicals as the game spine.
- Includes Nagel/Rosemarie for all 144 perspectives.
- Includes Griffiths after the **rank-clean** filter only: drop every Griffiths
  `game_id` with tied cardinal ranks; keep L1-shifted games and compute
  theory on the re-oriented Griffiths **cardinal** matrix actually shown to
  subjects.
- Current Griffiths rank-clean coverage: 832 Griffiths games / 1,664 rows,
  reaching 126 / 144 canonicals and 69 / 78 paired Bruns games.

Use this table for the main human behavioral benchmark. Use the strict
`clean_grif_*` columns in the equivalence tables only as a robustness subset.

### Missing final master artifact

There is **not yet** one clean 144-row dataframe that contains all mapping,
human data, matrix features, and both Griffiths inclusion policies. The pieces
exist, but they are split across `game_features.csv`,
`taxonomy/equivalence_per_canonical.csv`, `taxonomy/equivalence_per_game.csv`,
`taxonomy/griffiths_to_canonical.csv`, and
`analysis/_shared/datasets/unified_pairs.parquet`.

The desired final products are:

- `taxonomy/human_game_master_per_canonical.csv` — 144 rows; Bruns/canonical IDs,
  full `game_features` columns, Nagel/Rosemarie perspective data, full
  rank-clean Griffiths aggregate, and strict-clean Griffiths robustness fields.
- `taxonomy/human_game_master_per_game.csv` — 78 rows; player-swap-collapsed
  paired-game view with both Nagel/Rosemarie player coordinates and Griffiths
  aggregates in the same frame.

### Other files

- `griffiths_clean_symmetric.csv`, `griffiths_games.json`, `griffiths_selected_reference.csv` — Griffiths source data (clean cardinal games).
- `nagel_games.json`, `nagel_symmetric.csv` — Nagel source data.

## Canonical action axis — required reading

`A` and `B` are positional labels. Their **strategic meaning changes across games**. In PdPd act0 is cooperate (dominated); in BaBa act0 is one of two coordination points; in ChCh act0 is swerve. Aggregating `(move1 == 0).mean()` across games is methodologically invalid — real trait effects cancel because they push toward act0 in one game and act1 in another.

For any cross-game analysis, use `canonical_action_p1` / `canonical_action_p2` from `game_features.csv`:

```python
features = pd.read_csv('datasets/processed/game_features.csv')
equiv    = pd.read_csv('datasets/processed/taxonomy/equivalence_per_canonical.csv')

df = df.merge(features, on='game_code', how='left') \
       .merge(equiv, left_on='game_code', right_on='bruns_name', how='left')

df['aligned_canonical_p1'] = (df['move1'] == df['canonical_action_p1']).astype(int)
```

`nagel_frac_choose_act0_canonical`, `nagel_p1`, `nagel_p2`, and Griffiths
`up_choice_canonical` / `clean_grif_*` are on the canonical-form **act0**
axis. That is an orientation convention, not the same thing as the
`canonical_action_p1` prediction. If an analysis needs "probability of the
canonical predicted action", convert via `canonical_action_p1` / `p2`.

**Naming warning:** `equivalence_per_canonical.csv` has `canonical_p1` and `canonical_p2` columns that are **payoff vector strings** (e.g., `"1,4,3,2"`). The **action prediction** columns are `canonical_action_p1` / `canonical_action_p2` in `game_features.csv`. They are not the same. Do not conflate.

See `../../CLAUDE.md` hard constraint #2 for the full statement.

## Reproducibility

```bash
# Rebuild game_features.csv + parquet
python scripts/compute_game_features.py

# Rebuild equivalence dataset (taxonomy/*.csv)
python scripts/build_taxonomy_tables.py
python scripts/build_equivalence_dataset.py
```

Both pipelines are deterministic — re-running produces byte-identical outputs.
