# Codebook — `unified_pairs.parquet`

**File:** `analysis/_shared/datasets/unified_pairs.parquet` (7,466 rows × 92 columns)
**Built by:** `analysis/block_a/build_unified_pairs.py`
**Purpose:** one long table putting three samples — four LLMs and two human references — on the *same* footing for every 2×2 game: the realized decision pair, the game's matrix, the equilibria/rule predictions computed from that matrix, and the structural features.

This is the main long-form behavioral reference table, not the final 144-row
metadata master. A future `datasets/processed/taxonomy/human_game_master_*`
artifact should join this behavioral table back to `game_features.csv` and the
taxonomy equivalence tables.

---

## Row grain & samples

One row = one **(sample, condition, game)**.

| sample | agent_kind | rows | grain detail |
|---|---|---|---|
| `qwen`, `qwen_instruct`, `llama31_instruct`, `gptoss` | `llm` | 1444 / 1444 / 1444 / 1326 | one row per (model, game, condition); 76 games × 19 conditions (gpt-oss partial) |
| `nagel` | `human` | 144 | one row per Nagel perspective (game) |
| `griffiths` | `human` | 1664 | one row per Griffiths cardinal game-id × perspective; collapse to per-`canonical_id` for analysis |

`in_model_set = True` marks the 76 games all agents share — use it for cross-agent comparison.

Griffiths inclusion rule: this table drops every Griffiths `game_id` with tied
cardinal ranks (`has_ties=True`) and keeps the remaining L1-shifted games.
Theory for Griffiths rows is computed on the re-oriented **cardinal** matrix
subjects actually saw (`payoff_kind="cardinal"`). This is the main Griffiths
human benchmark. The stricter `clean_grif_*` fields in
`datasets/processed/taxonomy/equivalence_*.csv` are robustness-only fields
that additionally require cardinal L1 to match ordinal L1.

---

## Conventions (read before using)

- **Action axis.** `act0` = the **first row** (player 1) / **first column** (player 2) of the *canonical* matrix. Every `*_p1` / `*_p2` probability column is **P(play act0)**.
- **Behavior columns are P(act0):** `realized_p1/p2` (choice frequency, the analysis object) and `pref0_p1/p2` (softmax, LLM only).
- **Theoretical columns are P(act0) too**, in {0, 0.5, 1} for deterministic rules (0 = plays act1, 1 = plays act0, 0.5 = indifferent/tie) and interior for mixed equilibria (`ne*`).
- **P2 orientation fix:** human `realized_p2` is `1 − partner-perspective rate` (the partner's own canonical axis is the complement of this game's column axis). Validated: Nagel↔Griffiths agree r ≈ 0.96 on both coordinates.
- **Two parallel predictor families.** The eq-engine columns (`dom_*`, `l1_*`, `l2_*`, `ne*`, …) are in **P(act0)** space. The columns inherited from `game_features.csv` (`dominant_action_*`, `l1_action_*`, `nash_action_*`, …) are **action indices** (0 = act0, 1 = act1, `-1` = undefined/sentinel). They are the *complement* of each other; do not mix them.
- **No game exclusions.** This file carries all games; the v3 figure (`make_rationality_figure_v3`) scores all 76 shared games under the `raw`/`max` normalizations (MP N=9). The retired `1 − d/d_chance` scale used to drop the 4 MP games whose mixed NE is uniform play (÷0) — that exclusion no longer applies.

---

## 1. Identity & provenance

| column | type | coverage | values / example | meaning |
|---|---|---|---|---|
| `sample` | str | all | qwen, qwen_instruct, llama31_instruct, gptoss, nagel, griffiths | data source / agent |
| `agent_kind` | str | all | `llm`, `human` | agent family |
| `condition` | str | all | `baseline`, `{trait}_typeA/B`, `length_match_null_*`, `observed` | LLM prompt cell; `observed` for humans |
| `trait_p1`, `trait_p2` | str | all | baseline, level0_naive, level1, level2, maximin, selfish_maximizer, risk_aversion, inequity_aversion, loss_aversion, length_match_null, observed | per-player prompt |
| `game_code`, `bruns_name` | str | all | `AsBa`, `PdPd`, … (144) | Robinson-Goforth/Bruns game id (identical columns) |
| `canonical_id` | int | all | 1–144 | canonical perspective id (join key to taxonomy) |
| `nagel_lk_type` | str | all | DD, OD1, OD2, CO1, CO2, MP | level-k(α) similarity class |
| `griffiths_game_id` | float | griffiths | 1–1170 | Griffiths cardinal game id |
| `griffiths_unique_id` | float | griffiths | 1–2340 | Griffiths (game × role) row id |
| `griffiths_role` | str | griffiths | rowplayer, colplayer | which side this Griffiths row is |
| `n_pooled` | int | all | 1, 2, 3 | observations pooled (LLM = #seeds at round 1; humans = 1) |
| `in_model_set` | bool | all | True/False | game is in the 76 shared with the LLM sweep |
| `source` | str | all | `bruns` | taxonomy source tag |

## 2. Game matrix

| column | type | coverage | values / example | meaning |
|---|---|---|---|---|
| `matrix_8vec` | str (JSON) | all | `[1,4,3,2,2,4,3,1]` | payoff vector `[u1(00),u1(01),u1(10),u1(11), u2(00),u2(01),u2(10),u2(11)]`, act0=row0/col0 |
| `payoff_kind` | str | all | `ordinal`, `cardinal` | ordinal ranks for LLM/Nagel; cardinal for Griffiths |

## 3. Observed behavior — the analysis objects

| column | type | coverage | values | meaning |
|---|---|---|---|---|
| `realized_p1`, `realized_p2` | float | all | [0,1] | **P(play act0)** from realized choices (the headline behavioral object) |
| `pref0_p1`, `pref0_p2` | float | LLM only (5658) | [0,1] | **P(act0)** from the renormalized softmax over the two option tokens (the model's soft preference) |

## 4. Equilibria & rule predictions — computed from `matrix_8vec`, in P(act0) space

| column | type | coverage | values | meaning |
|---|---|---|---|---|
| `ne_count` | int | all | 1, 3 | number of Nash equilibria (nashpy enumeration) |
| `ne_kind` | str | all | unique_pure, unique_mixed, two_pure_plus_mixed | NE structure |
| `ne1_p1/p2` | float | all | corners / interior | NE #1 (unique NE; or first pure NE for CO) as P(act0) |
| `ne2_p1/p2` | float | CO only (1444) | corners | second pure NE (CO) |
| `ne3_p1/p2` | float | CO only (1444) | interior | the mixed NE of a CO game |
| `dom_p1/p2` | float | where a dominant action exists | {0,1} | dominant strategy |
| `l1_p1/p2`, `l2_p1/p2`, `l3_p1/p2` | float | all | {0,0.5,1} | naive level-1/2/3 (best response to uniform / Lk−1) |
| `l1alpha_p1/p2` | float | all | {0,0.5,1} | level-1(α) (Fudenberg-Liang risk-adjusted; breaks payoff-sum-5 ties) |
| `maximin_p1/p2` | float | all | {0,1} | pure-strategy maximin action |
| `equalsplit_p1/p2` | float | all | {0,0.5,1} | near-equal-split (fair/efficient) action |

## 5. Selected-rule targets (derived; see note)

| column | type | coverage | values | meaning |
|---|---|---|---|---|
| `target_p1/p2` | float | all | {0,0.25,0.5,1} | per-class selected-rule target (built by `agg_ref`/`load_rows`) |
| `target_paydom_p1/p2` | float | CO subset (755) | {0,1} | payoff-dominant NE where it exists |
| `target_riskdom_p1/p2` | float | CO subset (984) | {0,1} | risk-dominant NE where it exists |

> **Note.** The equilibrium-conformity figure does **not** use `target_*`; it scores `realized_*` directly against the `ne*` columns (point-to-set for CO). The `target_*` columns carry a known caveat (OD1 used naive L1 with ties; the CO selection was unresolved) and are kept only for reference.

## 6. Structural features (joined from `game_features.csv`)

| column | type | coverage | values | meaning |
|---|---|---|---|---|
| `dominance_profile` | int | all | 0,1,2 | # players with a dominant strategy |
| `iesds_depth` | int | all | 0,1,2 | iterated-dominance solution depth |
| `num_pure_ne` | int | all | 0,1,2 | # pure Nash equilibria |
| `pure_ne_cells` | str | where ≥1 pure NE | `(0,1);(1,0)` | pure-NE cell(s) |
| `ne_pareto_rankable` | bool | CO (1444) | True/False | are the two pure NE Pareto-rankable |
| `ne_distributional_conflict` | bool | CO (1444) | True/False | do the players prefer different equilibria |
| `payoff_variance` | float | all | 1.25 | per-game payoff variance |
| `payoff_conflict` | float | all | 0–2 | conflict-of-interest index |
| `p1_maximin_action`, `p2_maximin_action` | int | all | 0,1 | maximin action (index) |
| `joint_maximin_cell` | str | all | `(1,0)` | joint maximin cell |
| `maximin_is_ne` | bool | all | True/False | is the maximin profile a NE |
| `maximin_to_ne_action_distance` | float | where pure NE | 0,1 | action distance maximin→NE |
| `maximin_to_ne_payoff_distance` | float | where pure NE | 0,2,3,4 | payoff distance maximin→NE |
| `near_equal_split_cell` | str | all | `(0,1)` | near-equal-split cell |
| `pareto_efficient_cells` | str | all | `(0,0);(1,1)` | Pareto-efficient cells |
| `complexity_score` | float | all | −0.77 … 0.86 | game complexity (PCA composite) |
| `complexity_score_n_components` | int | all | 3,4 | components in the complexity score |

## 7. Action-index predictors (from `game_features.csv`) — index space, NOT P(act0)

`dominant_action_p1/p2`, `nash_action_p1/p2`, `payoff_dominant_ne_p1/p2`, `risk_dominant_ne_p1/p2`,
`l0_action_p1/p2`, `l1_action_p1/p2`, `l2_action_p1/p2`, `cooperative_action_p1/p2`,
`canonical_action_p1/p2` — all **action indices** (`0` = act0, `1` = act1, `-1` = undefined for level-k on the dominant side). These are the *complement* of the eq-engine P(act0) columns in §4; `canonical_action_*` is the precedence-rule single binary used elsewhere in the project.

## 8. Vestigial / empty columns

`frac_choose_A`, `entropy_binary`, `lk_type`, `emp_class`, `up_choice`, `topology`, `delta_norm` — legacy schema slots inherited from older `game_features.csv` builds. Current `game_features.csv` is Bruns-only, so these columns are **entirely NaN here**. Ignore them; use `nagel_lk_type` / taxonomy columns for human class labels.

---

## Quick-start

```python
import pandas as pd
df = pd.read_parquet("analysis/_shared/datasets/unified_pairs.parquet")
df = df.loc[:, ~df.columns.duplicated()]
shared = df[df.in_model_set]                       # 76 games all agents share
base = shared[(shared.agent_kind=="llm") & (shared.condition=="baseline")]
# observed pair vs the game's equilibrium (point-to-set for CO via ne1/ne2)
```
