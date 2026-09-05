# Raw human data — not redistributed

This directory is **empty in the distributed repository** and is git-ignored apart from this
file. The two source studies' participant-level exports belong to their authors; we do not
redistribute them. Nothing here is needed for the committed tables, and most of the paper's
figures rebuild without it — see `REPRODUCING.md` for the exact per-figure requirements.

## What goes where

```
data/human_refs/raw/nagel/       df_ros.csv
                                 normalized_data_20241612.csv
                                 perspective_info.csv
data/human_refs/raw/griffiths/   games2p2k_main griffith emke.csv
```

Override the location by placing the files here, or point the code elsewhere with
`SCA_REPO_ROOT` (see `strategic_anatomy/config.py:human_refs_root`).

## Verifying you have the same file we used

`data/MANIFEST.json` records the SHA-256 of both raw exports under `human_raw_sources`. We
cannot ship the files, but we can let you prove your copy is byte-identical to ours — which is
what makes the human λ fit reproducible:

```bash
shasum -a 256 data/human_refs/raw/nagel/normalized_data_20241612.csv
# must equal human_raw_sources.moore_nagel.raw_export.sha256 in data/MANIFEST.json
```

## What needs these files

| Consumer | Needs |
|---|---|
| `analysis/layer_a/src/human_lambda_mgn.py` (`individual_choices`) | `nagel/normalized_data_20241612.csv` |
| `analysis/layer_a/src/model_selection_cv.py` | via `human_lambda_mgn` |
| `analysis/layer_a/src/fig3_behavioral_model.py` | via `human_lambda_mgn` |
| `analysis/layer_a/src/human_dedup_robustness.py` | `nagel/normalized_data_20241612.csv` |
| `analysis/layer_a/build_rule_classification.py` | `nagel/normalized_data_20241612.csv` |
| `analysis/layer_a/figscripts/fig3_qre_to_levelk.py` → `figS_model_selection.pdf` | via `human_lambda_mgn` |
| `analysis/layer_a/figscripts/fig_behaviour_merged_2x3.py` → `fig_behaviour_merged_2x3.pdf` | via `human_lambda_mgn` |
| `analysis/layer_a/figscripts/fig3b_v2_bounded_rationality_map.py` | `nagel/normalized_data_20241612.csv` |
| `features/build_unified_pairs.py` (Stage F) | both directories |

The human λ is fitted on **individual choices**, not on aggregate frequencies, which is why the
committed per-game aggregates cannot stand in for the raw export here.

## Sources

- **Moore, Germano & Nagel (2026)**, working paper — the 144-perspective 2×2 catalogue and the
  associated human choice data (`df_ros.csv`, `normalized_data_20241612.csv`,
  `perspective_info.csv`). Request from the authors.
- **Zhu, J.-Q., Peterson, J. C., Enke, B. & Griffiths, T. L. (2025).** Capturing the complexity
  of human strategic decision-making with machine learning. *Nature Human Behaviour* **9**,
  2114–2120. The choice data is distributed with that article, subject to its own redistribution
  terms.

Derived per-game aggregates for both studies **are** shipped, in the parent directory — see
`../README.md`.
