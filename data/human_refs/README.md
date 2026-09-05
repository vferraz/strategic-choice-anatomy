# Human reference data

This directory ships **derived per-game aggregates only**. No participant-level record is included.

**Two paper figures do need the raw participant-level export**, because the human λ is fitted on
individual choices rather than on aggregate frequencies — the aggregates cannot stand in for it.
Those two are `fig_behaviour_merged_2x3` and `figS_model_selection`; `REPRODUCING.md` marks them,
and `raw/README.md` explains what to obtain and how to verify your copy against our sha256.
Everything else in the paper rebuilds from the aggregates below.

## What is here

| File(s) | Content |
|---|---|
| `unified_pairs.parquet` | 7,466 rows × 92 cols — one row per (game, agent, condition). Game-level choice frequencies for both human samples and the four LLMs, joined to the structural game features. |
| `unified_pairs.head.csv` | First 100 rows, for eyeballing the schema without a parquet reader. |
| `CODEBOOK_unified_pairs.md` | Column-by-column definitions. |
| `trait_*.csv` | Trait-cue effect tables (LLM-side; per game or per trait × model). |
| `../games/taxonomy/human_game_master_per_canonical.csv` | 144 rows, one per Robinson–Goforth/Bruns canonical game. |
| `../games/taxonomy/human_game_master_per_game.csv` | 78 rows, one per player-swap-collapsed game. |

**On the `p_id` columns.** `nagel_p_id`, `p_id_x` and `p_id_y` are **game-perspective
identifiers** in the 144-perspective Robinson–Goforth catalogue — not participant
identifiers. Each takes 144 (resp. 78) distinct values across 144 (resp. 78) rows, one per
game. Verified at release time.

## What is NOT here, and how to obtain it

The two source studies' **raw participant-level exports** are not redistributed here. They belong
to their authors and are available from the original publications. The Stage-F builders under
`features/` need them, and so does the human-λ estimator
(`analysis/layer_a/src/human_lambda_mgn.py`) and everything downstream of it.

Place the raw exports at:

```
data/human_refs/raw/nagel/       df_ros.csv, normalized_data_20241612.csv, perspective_info.csv
data/human_refs/raw/griffiths/   the released choice data
```

`raw/README.md` lists every consumer, and `data/MANIFEST.json` (`human_raw_sources`) records the
sha256 of both files so you can confirm your copy is byte-identical to the one used here.

### Sources

- **Moore, Germano & Nagel (2026)**, working paper — the 144-perspective 2×2 catalogue and
  the associated human choice frequencies (`df_ros.csv`, `normalized_data_20241612.csv`).
- **Zhu, J.-Q., Peterson, J. C., Enke, B. & Griffiths, T. L. (2025).** Capturing the
  complexity of human strategic decision-making with machine learning.
  *Nature Human Behaviour* **9**, 2114–2120.

### Inclusion rules applied to the derived tables

- **Griffiths** is *rank-clean* in `unified_pairs.parquet`: tied-rank `game_id`s dropped,
  L1-shifted games kept, theory recomputed on the re-oriented cardinal matrix subjects
  actually saw — 832 games / 1,664 rows, covering 126/144 canonicals and 69/78 paired games.
  The stricter `clean_grif_*` fields in `equivalence_*` are a robustness subset (35 of 78
  paired games), not the main reference.
- **Nagel/Rosemarie** covers all 144 perspectives; `p_t` gives the paired perspective for
  the other role, which is how the 78 player-swap-collapsed games are formed.

Both are re-expressed on the canonical action axis
(`canonical_action_p1` / `canonical_action_p2` in `data/games/game_features.csv`) before any
cross-game aggregation.
