# 2×2 game equivalence dataset — Bruns/Robinson–Goforth + Nagel + clean Griffiths

A single consolidated mapping between three independent 2×2 game taxonomies that all describe the same underlying Robinson–Goforth space.

**Built**: 2026-05-04
**Scripts**: `scripts/build_taxonomy_tables.py`, `scripts/build_equivalence_dataset.py`
**Outputs (this folder)**:
- `equivalence_per_canonical.csv` — 144 rows
- `equivalence_per_game.csv` — 78 rows
- `equivalence_clean_griffiths.csv` — 279 rows
- `canonical_master.csv`, `bruns_to_canonical.csv`, `nagel_to_canonical.csv`, `griffiths_to_canonical.csv` (the underlying hub-and-spoke tables this dataset is built on)

**Important scope note (2026-06-06):** this equivalence dataset stores the
**strict Griffiths robustness subset** in the `clean_grif_*` fields. It is not
the main full-scope Griffiths behavioral reference. The main paper benchmark
should use `analysis/_shared/datasets/unified_pairs.parquet`, where Griffiths is
filtered only for tied ranks: drop every Griffiths `game_id` with
`has_ties=True`, keep L1-shifted cardinal games, and compute theory on the
re-oriented cardinal matrix the subjects actually saw. That main rank-clean
Griffiths set has 832 games / 1,664 rows, reaches 126 / 144 canonicals and 69 /
78 paired Bruns games. The stricter `clean_grif_*` subset has 279 games, reaches
61 / 144 canonicals and 35 / 78 paired Bruns games, and is useful as a robustness
check for cases where cardinal payoffs preserve ordinal L1.

---

## 1. Why this dataset exists

Three independent sources describe ordinal 2×2 games with different conventions, and we needed to ask cross-source questions in one join:

| Source | Granularity | Encoding | Empirical data |
|---|---|---|---|
| **Bruns / Robinson–Goforth** (`src/games.py`) | 144 canonical classes (orbit reps under row × col swap) | Ordinal 1–4, fixed (P1, P2) layout, named (`PdPd`, `BaHr`, …) | none |
| **Nagel/Rosemarie** (`datasets/nagel/df_ros.csv`, `perspective_info.csv`) | 144 perspectives (= 12 symmetric + 66 asymmetric × 2 roles), paired into 78 games by `df_ros.csv::p_t` / `perspective_info.csv::paired_perspective` | Ordinal 1–4, *interleaved* column order in `perspective_info.csv` (`p1_AA, p2_AA, p1_AB, …`) | `frac_choose_A` per perspective + Lk-class metadata; paired-game centroids via both perspectives |
| **Griffiths** | 1,208 cardinal games × 2 roles (2,416 rows) | Real-valued cardinal payoffs, `rowplayer` / `colplayer` rows | `up_choice` per row + topology label |

Before this dataset existed, answering *"which Nagel perspective is Bruns `PdPd`?"* or *"which Griffiths games map to the same canonical class as Nag p_id 57?"* required hand-aligning swaps. Now it's two joins.

---

## 2. The design: canonical hub + spokes

The 144 Robinson–Goforth canonical classes are the **only piece that is intrinsic to the games themselves** (not to a particular dataset's naming). So we made them the hub.

```
                       canonical_master.csv (144 classes)
                                |
                                | canonical_id
            ___________________ |_____________________
           |                    |                     |
  bruns_to_canonical    nagel_to_canonical    griffiths_to_canonical
       (144)                  (144)                 (2,416)
```

Every Bruns name *is* a canonical class. Every Nagel perspective and every untied Griffiths row resolves to one canonical via a `(swap_sr, swap_sc)` row/column swap. Two joins answer every cross-source question, with no consistency drift between pairwise tables.

---

## 3. How we got there — the key insights

### 3a. The 8-vector layout

We fixed one convention everywhere:

```
vec = (p1_AA, p1_AB, p1_BA, p1_BB,  p2_AA, p2_AB, p2_BA, p2_BB)
       └────── P1 (row player) ──┘  └────── P2 (col player) ──┘
```

This is the order in `src/games.py`. The matching procedure for any input 8-vector tries all four `(sr, sc) ∈ {F, T}²` swaps against the 144-canonical lookup table and returns the unique match. Strict-ordinal inputs always match exactly one canonical via the direct phase.

### 3b. Nagel's interleaved columns (the #1 footgun)

`perspective_info.csv` columns are listed as `p1_AA, p2_AA, p1_AB, p2_AB, p1_BA, p2_BA, p1_BB, p2_BB` — **not** the Bruns layout. The script extracts by named column and reorders to the canonical 8-vector layout before any swap matching. Without this, the canary fails silently.

### 3c. The PdPd canary

Nagel perspective `P57` (PdPd) was used as the load-bearing alignment check:

| Quantity | Value |
|---|---|
| Raw `frac_choose_A` | 0.6785 |
| Matched canonical | 131 (PdPd) |
| `(swap_sr, swap_sc)` | `(True, True)` |
| Aligned `frac_choose_act0_canonical` | `1 − 0.6785 = 0.3215` |

The build script asserts this with a hard `assert`; if it ever breaks, the script exits non-zero.

### 3d. Robinson–Goforth's 144 perspectives pair into 78 Nagel/Rosemarie games

Bruns' 144 enumerate every ordinal 2×2 game *up to (row swap, col swap)* — not up to player swap. Under additional player swap:

- 12 strategically-symmetric canonicals are self-paired.
- 132 asymmetric canonicals form 66 player-swap pairs (e.g., `PdHr` ↔ `HrPd`).
- **Total Nagel/Rosemarie games = 12 + 66 = 78**, but each asymmetric game shows up as two perspectives/two Bruns canonicals.

The raw pairing source is `datasets/nagel/df_ros.csv`: `p` is the perspective ID and
`p_t` is the paired perspective for the other role. The processed
`perspective_info.csv` carries the same relationship as `p_id` /
`paired_perspective`. Use this pairing whenever a two-player human centroid is
needed; do not treat the 144 perspective rows as 144 independent two-player games.

So both granularities are useful:
- `equivalence_per_canonical.csv` (144 rows) is the **Robinson–Goforth / perspective spine** — best when you need each ordinal orbit treated separately.
- `equivalence_per_game.csv` (78 rows) is the **player-swap-collapsed paired-game view** — best when you want one row per Nagel/Rosemarie game (game number 1–78 = `min(p_id, p_paired)`) with both human player perspectives (`nagel_p1`, `nagel_p2`).

### 3e. Cardinal payoffs can damage the canonical class

This was the central correctness insight. Griffiths' games carry *cardinal* payoffs (real numbers like 21, 10, 24, 17). Two ways cardinal payoffs can break the canonical mapping:

**(i) Rank ties** — if a player's cardinal vector has repeated values (e.g., `(5, 8, 5, 12)`), there's no strict ordinal 1–4 ranking, so no canonical class.
- **376 of 1,208 Griffiths games (31 %)** have rank ties on at least one player and are excluded.

**(ii) Shifting strategic concepts** — even when ordinal ranks are clean, cardinal magnitudes can flip the **Level-1 best response** (which uses expected values over a uniform opponent, so magnitudes matter).

A worked example: ordinal P1 = `(1, 4, 2, 3)` has row sums `5` and `5` — L1 is *tied* at the ordinal level. Cardinal P1 = `(1, 100, 50, 51)` gives row sums `101` vs `101` — also tied. But cardinal `(1, 99, 50, 52)` gives row sums `100` vs `102` — cardinal picks row 1, breaking the ordinal tie.

The math fact: when canonical-ordinal L1 is *well-defined* (sum_row0 ≠ sum_row1 at the rank level), cardinal L1 is *forced* to agree — the ordinal inequality is preserved by any cardinal payoffs that ordinalize to those ranks. So cardinal-L1 only disagrees with canonical-L1 in the *tied-ordinal* (MP-style) cases. We filter those out.

**(iii) Things that are not damaged**: pure-strategy Nash equilibria, iterated strict dominance, and existence of mixed NE all depend only on strict ordinal inequalities and are preserved by any rank-preserving cardinal payoffs. We don't filter on these.

### 3f. The strict "clean Griffiths" robustness filter

A Griffiths game is **strict-clean** when *both* of these hold:

1. `has_ties == False` (no rank ties on either player's cardinal payoff vector).
2. `L1(cardinal) == L1(canonical)` for both players, with both well-defined (no L1 ties on either side, ordinal or cardinal).

Equivalently: the strategic skeleton (pure NE, IESDS, L1 best response) is identical between the game's cardinal Griffiths form and its canonical Bruns ordinal form.

Funnel:

```
1,208 Griffiths games
   │   drop has_ties=True (cardinal rank-tied)
   ▼
   832 games
   │   drop L1(cardinal) tied or different from L1(canonical)
   ▼
   279 STRICT-CLEAN games   ← what equivalence_clean_griffiths.csv contains
```

For the main behavioral comparison, do **not** apply the L1 filter. Keep the
832 rank-clean Griffiths games and score each row on its own cardinal matrix
(`payoff_kind="cardinal"` in `unified_pairs.parquet`). This preserves coverage
while keeping realized behavior and theoretical targets on the same payoff
surface.

### 3g. Validation against Nagel categories

Per Rosemarie Nagel's request the figures use her own `lk_type` classification (DD, OD1, OD2, CO1, CO2, MP) and her `emp_class` (A1, …, B4). The clean filter lines up with these in a satisfying way:

- The **A1 cluster** (DD + OD1 + the lone CO1 = game 60) — these are "Lk-α at k=1, near-pure-strategy" games. After clean-Griffiths filtering, the mean Nagel ↔ Griffiths Euclidean gap drops to **0.050**.
- The **MP cluster** (mixed-pure / cyclic) is where the L1 filter drops the most games — exactly the "cardinal magnitudes flip the L1 prediction" cases.

---

## 4. The three CSVs

### `equivalence_per_canonical.csv` (144 rows, 39 cols)

One row per Bruns/Robinson–Goforth canonical class.

**Bruns canonical features**:
`canonical_id`, `bruns_name`, `bruns_long_name`, `canonical_p1`, `canonical_p2`, `canonical_8vec`, `nash_count`, `dominance_level` (= iterated-strict-dominance depth, 0–2), `is_symmetric`, `canon_l1_p1`, `canon_l1_p2`.

**Nagel perspective** (the unique perspective that maps to this canonical):
`nagel_p_id`, `nagel_p_paired`, `nagel_perspective`, `nagel_matrix_code`, `nagel_lk_type` (+ description), `nagel_emp_class` (+ description), `nagel_rgg_class` (+ description), `nagel_swap_sr/sc`, `nagel_role_relative_to_canonical`, `nagel_frac_choose_A`, `nagel_frac_choose_act0_canonical`, `nagel_entropy_binary/4outcome`, `nagel_utility_efficiency`, `nagel_quality_level`, `nagel_n_pure_NE`, `nagel_p1_has_dominant`, `nagel_p2_has_dominant`.

**Clean Griffiths aggregate** (rowplayer-side matches this canonical):
`clean_grif_n_games`, `clean_grif_p1_mean`, `clean_grif_p2_mean`, `clean_grif_p1_std`, `clean_grif_p2_std`, `clean_grif_game_ids` (semicolon-joined list).

### `equivalence_per_game.csv` (78 rows, 39 cols)

Player-swap-collapsed view. `game_id = min(p_id, p_paired)` so game numbers run 1–78.

**Game identity**: `game_id`, `p_id_x`, `p_id_y`, `is_symmetric`.

**The two paired canonicals**: `canonical_x`, `canonical_y`, `bruns_name_x/y`, `bruns_long_name_x/y`, `canonical_p1_x/y`, `canonical_p2_x/y`, `nash_count`, `dominance_level`, `canon_l1_p1/p2`.

**Both Nagel perspectives**: `nagel_perspective_x/y`, `lk_type_x/y`, `emp_class_x/y`, `rgg_class_x/y`, `frac_choose_A_x/y`, `nagel_p1`, `nagel_p2` (canonical-aligned), `entropy_binary_x/y`.

**Clean Griffiths aggregate** combining BOTH directions (rowplayer matches `canonical_x` contributes `(rp, cp)`; rowplayer matches `canonical_y` contributes reflected `(cp, rp)`):
`clean_grif_n`, `clean_grif_p1/p2_mean`, `clean_grif_p1/p2_std`, `clean_grif_game_ids`, `nagel_grif_gap` (Euclidean distance between Nagel and Griffiths centroid in X frame).

### `equivalence_clean_griffiths.csv` (279 rows, 32 cols)

Long-form. One row per clean Griffiths game.

**Griffiths identity**: `game_id`, `unique_id_rowplayer`, `unique_id_colplayer`, `topology_raw`.

**Cardinal & ordinal payoffs (rowplayer side)**: `cardinal_p1_rowplayer`, `cardinal_p2_rowplayer`, `ordinal_p1_rowplayer`, `ordinal_p2_rowplayer`.

**Empirical rates (both seats)**: `up_choice_rowplayer`, `up_choice_canonical_rowplayer`, `up_choice_colplayer`, `up_choice_canonical_colplayer`.

**Canonical identification**: `canonical_id_rowplayer`, `bruns_name_rowplayer_side`, `canonical_id_colplayer`, `bruns_name_colplayer_side`, `swap_sr_rowplayer/colplayer`, `swap_sc_rowplayer/colplayer`, plus structural features (`nash_count`, `dominance_level`, `is_symmetric`).

**Nagel correspondence** (joined via `canonical_id_rowplayer`): `nagel_p_id`, `nagel_p_paired`, `nagel_perspective`, `nagel_lk_type`, `nagel_emp_class`, `nagel_rgg_class`, `nagel_frac_choose_A`, `nagel_frac_choose_act0_canonical`, `nagel_entropy_binary`.

---

## 5. Coverage

| | Games | Canonicals reached |
|---|---|---|
| All Bruns canonicals | — | 144 |
| All Nagel perspectives → canonicals | — | 144 / 144 (every Bruns canonical has exactly one Nagel perspective) |
| Griffiths total | 1,208 | 126 / 144 (rowplayer-side, before any filter) |
| Griffiths rank-clean (`has_ties=False`) — **main behavioral reference** | 832 | 125 / 144 rowplayer-side; 126 / 144 either-side; 69 / 78 paired games |
| Griffiths **STRICT-CLEAN** (also L1-matches) — robustness only | **279** | **61 / 144** rowplayer-side; 35 / 78 paired games |

**18 Bruns canonicals are not represented in Griffiths at all** (e.g., `AsBa, BaCo, ChSh, …`) because no Griffiths cardinal game ordinalizes to them.

Nagel/Rosemarie coverage is complete at both useful granularities: all 144
perspectives in `df_ros.csv`, and all 78 paired games after matching each
perspective to `p_t`. Griffiths should be joined to that same paired-game frame.
Use the 832-game rank-clean Griffiths set for the main behavioral benchmark;
use the 279-game strict-clean set only when you explicitly need cardinal L1 to
preserve the ordinal L1 prediction. In particular, strict-clean Griffiths has
no MP/mixed-game paired-game coverage; those games are still covered by
Nagel/Rosemarie and by the broader rank-clean Griffiths mapping where available.

**Mean Nagel ↔ clean Griffiths Euclidean gap in canonical action-0 space**: **0.073** across the 35 covered games (was 0.46 in raw unaligned frame, 0.117 with canonical alignment but no L1 filter; dropping the L1-mismatch cases halves it again).

---

## 6. Worked examples

### Bruns → Nagel: which Nagel perspective is `PdPd`?
```python
c = pd.read_csv("equivalence_per_canonical.csv")
row = c.loc[c.bruns_name == "PdPd"].iloc[0]
# row.nagel_p_id == 57, row.nagel_lk_type == "DD",
# row.nagel_frac_choose_A == 0.6785, row.nagel_frac_choose_act0_canonical == 0.3215
```

### Nagel game → Griffiths centroid: what does the per-game view show for game 60 (CoCo)?
```python
g = pd.read_csv("equivalence_per_game.csv")
row = g.loc[g.game_id == 60].iloc[0]
# row.is_symmetric == True, row.lk_type_x == "CO1", row.emp_class_x == "A1"
# row.nagel_p1 == row.nagel_p2 == 0.849
# row.clean_grif_n == 20, row.clean_grif_p1_mean ≈ 0.869, row.clean_grif_p2_mean ≈ 0.860
# row.nagel_grif_gap ≈ 0.022  (excellent agreement)
```

### Clean Griffiths → Bruns + Nagel: what canonical does Griffiths game 1 belong to?
```python
l = pd.read_csv("equivalence_clean_griffiths.csv")
row = l.loc[l.game_id == 1].iloc[0]
# row.bruns_name_rowplayer_side == "PdHr"   (asymmetric)
# row.bruns_name_colplayer_side == "HrPd"   (its player-swap pair)
# row.nagel_p_id == 49, row.nagel_lk_type == "OD2", row.nagel_emp_class == "A2"
# row.up_choice_rowplayer == 0.30, row.up_choice_colplayer == 0.27
```

---

## 7. Conventions and caveats

- **`role_relative_to_canonical`** is `P1` for every Nagel row and every clean Griffiths row in this build, because the Bruns 144-canonical enumeration is exhaustive under `(sr, sc)` for strict-ordinal inputs. The `P2` fallback in the matching procedure is defensive and should never fire for valid inputs.
- **`dominance_level`** is `iesds_depth` from `scripts/compute_game_features.py` — the number of rounds of iterated elimination of strictly dominated strategies (0, 1, or 2 for 2×2). This is distinct from `dominance_profile` in `datasets/processed/game_features.csv`, which counts players with a dominant strategy and uses an inverted 0/1/2 encoding.
- **`frac_choose_act0_canonical` and `up_choice_canonical`** are *canonical-aligned*: they apply the row's `swap_sr` inversion so all 144 canonicals are on a common action-0/action-1 axis. The raw `frac_choose_A` and `up_choice` are in each row's own (Nagel/Griffiths) label convention and should not be averaged across games without alignment.
- **For asymmetric games**, the rowplayer and colplayer rows of a single Griffiths `game_id` land on *different* canonicals — the player-swap pair. The per-game view combines both directions; the per-canonical view treats them as separate orbits.
- **What we deliberately did NOT filter on**: pure-NE count, dominance level, mixed-NE existence. These are preserved by rank-preserving cardinal payoffs, so they don't add to the cleanliness criterion.

---

## 8. Reproducibility

```bash
# Step 1 — build the hub-and-spoke tables (canonical_master + 3 mapping CSVs)
python scripts/build_taxonomy_tables.py

# Step 2 — build the equivalence dataset (this directory's three CSVs)
python scripts/build_equivalence_dataset.py
```

Both scripts are deterministic and produce byte-identical CSVs across runs. The first ends with 13 sanity checks including the `Nag_057 → PdPd → 0.3215` canary as a hard assert.

---

## 9. Pipeline at a glance

```
src/games.py (Bruns 144)
        │
        │  + datasets/nagel/perspective_info.csv
        │  + datasets/griffiths/games2p2k_main griffith emke.csv
        ▼
build_taxonomy_tables.py
        │
        ├── canonical_master.csv          (144 canonicals, hub)
        ├── bruns_to_canonical.csv        (144 trivial)
        ├── nagel_to_canonical.csv        (144 perspectives → canonicals)
        └── griffiths_to_canonical.csv    (2,416 rows, 1,664 matched, 752 has_ties)

        │  + datasets/nagel/game_metadata_complete.csv  (Nagel's lk_type, emp_class, …)
        ▼
build_equivalence_dataset.py
        │
        │   apply: drop has_ties, then drop L1(cardinal) != L1(canonical)
        │
        ├── equivalence_per_canonical.csv   (144 rows, Bruns spine)
        ├── equivalence_per_game.csv        (78 rows, Nagel-game spine)
        ├── equivalence_clean_griffiths.csv (279 rows, long-form clean set)
        └── equivalence_README.md           (schema reference)
```
