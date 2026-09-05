#!/usr/bin/env python3
"""Compute game-theoretic features for the 144 Bruns canonical games.

Output: data/games/game_features.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from strategic_anatomy.config import games_root
from strategic_anatomy.games import bruns_games

# ---------------------------------------------------------------------------
# Payoff matrix helpers
# ---------------------------------------------------------------------------

def unpack_payoffs(vec):
    """Unpack 8-element payoff vector into P1 and P2 2x2 matrices.

    vec = [p1_AA, p1_AB, p1_BA, p1_BB, p2_AA, p2_AB, p2_BA, p2_BB]
    P1 chooses row, P2 chooses column.
    """
    p1 = np.array([[vec[0], vec[1]],
                    [vec[2], vec[3]]])
    p2 = np.array([[vec[4], vec[5]],
                    [vec[6], vec[7]]])
    return p1, p2


# ---------------------------------------------------------------------------
# Feature computation functions
# ---------------------------------------------------------------------------

def strictly_dominant(matrix):
    """Return the strictly dominant strategy index (0 or 1) for a row player,
    or None if neither strategy strictly dominates."""
    # Row 0 dominates row 1 if matrix[0,j] > matrix[1,j] for all j
    if all(matrix[0, j] > matrix[1, j] for j in range(2)):
        return 0
    if all(matrix[1, j] > matrix[0, j] for j in range(2)):
        return 1
    return None


def strictly_dominated(matrix):
    """Return the strictly dominated strategy index (0 or 1) for a row player,
    or None if neither is strictly dominated."""
    dom = strictly_dominant(matrix)
    if dom is not None:
        return 1 - dom
    return None


def dominance_profile(p1, p2):
    """0 = both have dominant, 1 = exactly one, 2 = neither."""
    d1 = strictly_dominant(p1) is not None
    d2 = strictly_dominant(p2.T) is not None  # P2 chooses column -> transpose
    count = int(d1) + int(d2)
    if count == 2:
        return 0
    elif count == 1:
        return 1
    else:
        return 2


def iesds_depth(p1, p2):
    """Iterated elimination of strictly dominated strategies.
    Returns number of elimination rounds. Max 2 for 2x2."""
    # Work with mutable sets of remaining strategies
    p1_rows = [0, 1]
    p2_cols = [0, 1]
    rounds = 0

    for _ in range(2):  # max 2 rounds for 2x2
        eliminated = False

        # Check P1's rows for strict domination given remaining P2 columns
        if len(p1_rows) == 2:
            sub_p1 = p1[np.ix_(p1_rows, p2_cols)]
            # row 0 dominates row 1?
            if all(sub_p1[0, j] > sub_p1[1, j] for j in range(len(p2_cols))):
                p1_rows = [p1_rows[0]]
                eliminated = True
            elif all(sub_p1[1, j] > sub_p1[0, j] for j in range(len(p2_cols))):
                p1_rows = [p1_rows[1]]
                eliminated = True

        # Check P2's columns for strict domination given remaining P1 rows
        if len(p2_cols) == 2:
            sub_p2 = p2[np.ix_(p1_rows, p2_cols)]
            # col 0 dominates col 1?
            if all(sub_p2[i, 0] > sub_p2[i, 1] for i in range(len(p1_rows))):
                p2_cols = [p2_cols[0]]
                eliminated = True
            elif all(sub_p2[i, 1] > sub_p2[i, 0] for i in range(len(p1_rows))):
                p2_cols = [p2_cols[1]]
                eliminated = True

        if eliminated:
            rounds += 1
        else:
            break

    return rounds


def pure_nash_equilibria(p1, p2):
    """Return list of (i, j) cells that are pure NE."""
    ne_cells = []
    for i in range(2):
        for j in range(2):
            other_i = 1 - i
            other_j = 1 - j
            # P1 has no incentive to deviate: p1[i,j] >= p1[other_i, j]
            # P2 has no incentive to deviate: p2[i,j] >= p2[i, other_j]
            if p1[i, j] >= p1[other_i, j] and p2[i, j] >= p2[i, other_j]:
                ne_cells.append((i, j))
    return ne_cells


def ne_pareto_rankable(ne_cells, p1, p2):
    """If multiple NE, can they be Pareto-ranked?"""
    if len(ne_cells) < 2:
        return np.nan
    for a in range(len(ne_cells)):
        for b in range(len(ne_cells)):
            if a == b:
                continue
            ia, ja = ne_cells[a]
            ib, jb = ne_cells[b]
            # a Pareto-dominates b if both at least as good and one strictly better
            p1_ge = p1[ia, ja] >= p1[ib, jb]
            p2_ge = p2[ia, ja] >= p2[ib, jb]
            p1_gt = p1[ia, ja] > p1[ib, jb]
            p2_gt = p2[ia, ja] > p2[ib, jb]
            if p1_ge and p2_ge and (p1_gt or p2_gt):
                return True
    return False


def ne_distributional_conflict(ne_cells, p1, p2):
    """If multiple NE, do P1 and P2 prefer different NE?"""
    if len(ne_cells) < 2:
        return np.nan
    # P1's preferred NE (highest P1 payoff)
    p1_best_idx = max(range(len(ne_cells)), key=lambda k: p1[ne_cells[k]])
    # P2's preferred NE (highest P2 payoff)
    p2_best_idx = max(range(len(ne_cells)), key=lambda k: p2[ne_cells[k]])
    return p1_best_idx != p2_best_idx


def payoff_variance(vec):
    """Variance of all 8 payoff values."""
    return float(np.var(vec))


def payoff_conflict(vec):
    """1 - pearson_correlation(p1_flat, p2_flat). 0.5 if undefined."""
    p1_flat = np.array(vec[:4], dtype=float)
    p2_flat = np.array(vec[4:], dtype=float)
    if np.std(p1_flat) == 0 or np.std(p2_flat) == 0:
        return 0.5
    r = np.corrcoef(p1_flat, p2_flat)[0, 1]
    if np.isnan(r):
        return 0.5
    return float(1.0 - r)


def maximin_action(matrix):
    """Maximin action for a row player: argmax_i min_j matrix[i,j]."""
    mins = [min(matrix[i, j] for j in range(2)) for i in range(2)]
    return int(np.argmax(mins))


def _best_response_or_tie(evs):
    """Return argmax of expected values, or -1 if tied."""
    if len(evs) == 2 and evs[0] == evs[1]:
        return -1
    return int(np.argmax(evs))


def level_k_actions(p1, p2):
    """Compute L1 and L2 best responses.
    L0: uniform random over opponent's actions.
    L1: best response to L0 opponent.
    L2: best response to L1 opponent.
    Returns -1 for any level where the expected values are exactly tied.
    """
    # L1 for P1: best response to P2 playing uniform (0.5, 0.5) over columns
    p1_ev = [0.5 * p1[i, 0] + 0.5 * p1[i, 1] for i in range(2)]
    l1_p1 = _best_response_or_tie(p1_ev)

    # L1 for P2: best response to P1 playing uniform (0.5, 0.5) over rows
    p2_ev = [0.5 * p2[0, j] + 0.5 * p2[1, j] for j in range(2)]
    l1_p2 = _best_response_or_tie(p2_ev)

    # L2 for P1: best response to P2 playing L1 (pure strategy l1_p2)
    if l1_p2 == -1:
        l2_p1 = -1  # can't compute L2 if L1 is tied
    else:
        l2_p1 = _best_response_or_tie([p1[i, l1_p2] for i in range(2)])

    # L2 for P2: best response to P1 playing L1 (pure strategy l1_p1)
    if l1_p1 == -1:
        l2_p2 = -1
    else:
        l2_p2 = _best_response_or_tie([p2[l1_p1, j] for j in range(2)])

    return l1_p1, l1_p2, l2_p1, l2_p2


def near_equal_split_cell(p1, p2):
    """Cell (i,j) with minimum |p1[i,j] - p2[i,j]|."""
    best_cell = (0, 0)
    best_diff = abs(p1[0, 0] - p2[0, 0])
    for i in range(2):
        for j in range(2):
            d = abs(p1[i, j] - p2[i, j])
            if d < best_diff:
                best_diff = d
                best_cell = (i, j)
    return best_cell


def pareto_efficient_cells(p1, p2):
    """Cells that are Pareto-efficient (not Pareto-dominated by any other cell)."""
    cells = [(i, j) for i in range(2) for j in range(2)]
    efficient = []
    for ci, cj in cells:
        dominated = False
        for di, dj in cells:
            if (di, dj) == (ci, cj):
                continue
            # (di,dj) Pareto-dominates (ci,cj) if both >= and at least one >
            p1_ge = p1[di, dj] >= p1[ci, cj]
            p2_ge = p2[di, dj] >= p2[ci, cj]
            p1_gt = p1[di, dj] > p1[ci, cj]
            p2_gt = p2[di, dj] > p2[ci, cj]
            if p1_ge and p2_ge and (p1_gt or p2_gt):
                dominated = True
                break
        if not dominated:
            efficient.append((ci, cj))
    return efficient


# ---------------------------------------------------------------------------
# Action-prediction primitives (per-player single-action outputs)
# ---------------------------------------------------------------------------

def payoff_dominant_ne(ne_cells, p1, p2):
    """When num_pure_ne == 2 and one NE Pareto-dominates the other in both
    players' own payoffs (>= both, > at least one), return that cell. Else None."""
    if len(ne_cells) != 2:
        return None
    a, b = ne_cells
    if (p1[a] >= p1[b] and p2[a] >= p2[b]
            and (p1[a] > p1[b] or p2[a] > p2[b])):
        return a
    if (p1[b] >= p1[a] and p2[b] >= p2[a]
            and (p1[b] > p1[a] or p2[b] > p2[a])):
        return b
    return None


def risk_dominant_ne(ne_cells, p1, p2):
    """When num_pure_ne == 2, return the Harsanyi-Selten risk-dominant cell
    via product of deviation losses; None on tie or |ne_cells| != 2.

    This is a descriptive diagnostic on the instantiated numerical payoff
    scale. It is not invariant to ordinal transformations and is therefore not
    used in the canonical-action precedence rule.

    Deviation loss at NE x given alternative NE y:
      P1 loss = p1[x] - p1[(y[0], x[1])]  (P1 switches row to match y)
      P2 loss = p2[x] - p2[(x[0], y[1])]  (P2 switches column to match y)
    NE x risk-dominates y iff product(loss_x) > product(loss_y).
    """
    if len(ne_cells) != 2:
        return None
    a, b = ne_cells

    def prod(x, y):
        p1_loss = p1[x] - p1[(y[0], x[1])]
        p2_loss = p2[x] - p2[(x[0], y[1])]
        return p1_loss * p2_loss

    pa = prod(a, b)
    pb = prod(b, a)
    if pa > pb:
        return a
    if pb > pa:
        return b
    return None


def level_0_action_p1(p1):
    """argmax_i max_j p1[i,j]; ties broken by lower row index."""
    row_max = [max(p1[i, j] for j in range(2)) for i in range(2)]
    return int(np.argmax(row_max))


def level_0_action_p2(p2):
    """argmax_j max_i p2[i,j]; ties broken by lower column index."""
    col_max = [max(p2[i, j] for i in range(2)) for j in range(2)]
    return int(np.argmax(col_max))


def cooperative_cell(p1, p2):
    """Return the (i, j) cell with strictly maximum mutual payoff
    (p1[i,j] + p2[i,j]), if both players have strictly dominant actions
    AND that cell strictly Pareto-dominates the dominance corner.
    Else None. On a tie for max mutual payoff -> None.
    """
    dom_p1 = strictly_dominant(p1)
    dom_p2 = strictly_dominant(p2.T)
    if dom_p1 is None or dom_p2 is None:
        return None
    dom_corner = (dom_p1, dom_p2)

    sums = [(i, j, p1[i, j] + p2[i, j]) for i in range(2) for j in range(2)]
    max_sum = max(s for _, _, s in sums)
    top = [(i, j) for i, j, s in sums if s == max_sum]
    if len(top) > 1:
        return None
    coop = top[0]
    if coop == dom_corner:
        return None
    if p1[coop] > p1[dom_corner] and p2[coop] > p2[dom_corner]:
        return coop
    return None


def canonical_action_pair(dom_p1, dom_p2, ne_cells, payoff_dom, mm_p1, mm_p2):
    """Apply the 4-step precedence chain per player:
      1. dominant_action
      2. nash_action (only when num_pure_ne == 1)
      3. payoff_dominant_ne
      4. maximin_action (always defined)
    Returns (canonical_p1, canonical_p2); both always integers in {0, 1}.
    """
    nash_cell = ne_cells[0] if len(ne_cells) == 1 else None

    def pick(side):
        # side: 0 for p1, 1 for p2
        dom = dom_p1 if side == 0 else dom_p2
        if dom is not None:
            return int(dom)
        if nash_cell is not None:
            return int(nash_cell[side])
        if payoff_dom is not None:
            return int(payoff_dom[side])
        return int(mm_p1 if side == 0 else mm_p2)

    return pick(0), pick(1)


def cell_str(cell):
    return f"({cell[0]},{cell[1]})"


def cells_str(cells):
    return ";".join(cell_str(c) for c in cells)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_features(game_code, vec):
    """Compute all features for a single game."""
    p1, p2 = unpack_payoffs(vec)
    ne_cells = pure_nash_equilibria(p1, p2)

    p1_mm = maximin_action(p1)
    p2_mm = maximin_action(p2.T)  # P2 chooses column -> transpose for row-player logic
    mm_cell = (p1_mm, p2_mm)

    l1_p1, l1_p2, l2_p1, l2_p2 = level_k_actions(p1, p2)

    nes_cell = near_equal_split_cell(p1, p2)
    pe_cells = pareto_efficient_cells(p1, p2)

    # Maximin to NE distances
    mm_is_ne = mm_cell in ne_cells
    if len(ne_cells) > 0:
        hamming_dists = [abs(mm_cell[0] - ne[0]) + abs(mm_cell[1] - ne[1])
                         for ne in ne_cells]
        mm_to_ne_action_dist = min(hamming_dists)

        payoff_dists = [abs(p1[mm_cell] - p1[ne]) + abs(p2[mm_cell] - p2[ne])
                        for ne in ne_cells]
        mm_to_ne_payoff_dist = float(min(payoff_dists))
    else:
        mm_to_ne_action_dist = np.nan
        mm_to_ne_payoff_dist = np.nan

    # Action-prediction columns
    dom_p1 = strictly_dominant(p1)
    dom_p2 = strictly_dominant(p2.T)  # P2 picks columns -> transpose
    payoff_dom_cell = payoff_dominant_ne(ne_cells, p1, p2)
    risk_dom_cell = risk_dominant_ne(ne_cells, p1, p2)
    nash_cell = ne_cells[0] if len(ne_cells) == 1 else None
    coop_cell = cooperative_cell(p1, p2)
    l0_p1 = level_0_action_p1(p1)
    l0_p2 = level_0_action_p2(p2)
    canon_p1, canon_p2 = canonical_action_pair(
        dom_p1, dom_p2, ne_cells, payoff_dom_cell, p1_mm, p2_mm
    )

    return {
        "game_code": game_code,
        "dominance_profile": dominance_profile(p1, p2),
        "iesds_depth": iesds_depth(p1, p2),
        "num_pure_ne": len(ne_cells),
        "pure_ne_cells": cells_str(ne_cells) if ne_cells else "",
        "ne_pareto_rankable": ne_pareto_rankable(ne_cells, p1, p2),
        "ne_distributional_conflict": ne_distributional_conflict(ne_cells, p1, p2),
        "payoff_variance": payoff_variance(vec),
        "payoff_conflict": payoff_conflict(vec),
        "p1_maximin_action": p1_mm,
        "p2_maximin_action": p2_mm,
        "joint_maximin_cell": cell_str(mm_cell),
        "maximin_is_ne": mm_is_ne,
        "maximin_to_ne_action_distance": mm_to_ne_action_dist,
        "maximin_to_ne_payoff_distance": mm_to_ne_payoff_dist,
        "l1_action_p1": l1_p1,
        "l1_action_p2": l1_p2,
        "l2_action_p1": l2_p1,
        "l2_action_p2": l2_p2,
        "near_equal_split_cell": cell_str(nes_cell),
        "pareto_efficient_cells": cells_str(pe_cells),
        # New action-prediction columns (appended at end of CSV via reorder in main).
        "dominant_action_p1": dom_p1,
        "dominant_action_p2": dom_p2,
        "nash_action_p1": nash_cell[0] if nash_cell is not None else None,
        "nash_action_p2": nash_cell[1] if nash_cell is not None else None,
        "payoff_dominant_ne_p1": payoff_dom_cell[0] if payoff_dom_cell is not None else None,
        "payoff_dominant_ne_p2": payoff_dom_cell[1] if payoff_dom_cell is not None else None,
        "risk_dominant_ne_p1": risk_dom_cell[0] if risk_dom_cell is not None else None,
        "risk_dominant_ne_p2": risk_dom_cell[1] if risk_dom_cell is not None else None,
        "l0_action_p1": l0_p1,
        "l0_action_p2": l0_p2,
        "cooperative_action_p1": coop_cell[0] if coop_cell is not None else None,
        "cooperative_action_p2": coop_cell[1] if coop_cell is not None else None,
        "canonical_action_p1": canon_p1,
        "canonical_action_p2": canon_p2,
    }


def main():
    # ------------------------------------------------------------------
    # 1. Collect games
    # ------------------------------------------------------------------
    # This file is the Bruns metadata table. Human matching / cross-dataset
    # equivalence lives in data/games/taxonomy/*.csv; do not add
    # Nagel_* or Grif_* reference rows here, or the one-shot universe silently
    # stops being the 144-game Akata/Bruns universe.
    all_games = {
        code: {"vec": vec, "source": "bruns"}
        for code, (vec, _name, _desc) in bruns_games.items()
    }

    print(f"Total games: {len(all_games)} (bruns={len(all_games)}, reference=0)")

    # ------------------------------------------------------------------
    # 2. Compute features
    # ------------------------------------------------------------------
    rows = []
    for code, info in all_games.items():
        feats = compute_features(code, info["vec"])
        feats["source"] = info["source"]
        rows.append(feats)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Preserve legacy schema slots for human references
    # ------------------------------------------------------------------
    # Phase-3 path-indirection miss, found by the widened check_phase2 gate: this was still
    # the PRIVATE layout (<repo>/datasets/processed), so the Stage-F rebuild wrote
    # game_features.csv somewhere the release never reads -- while docs/RESULTS_MAP.md
    # names this script as the producer of data/games/game_features.csv. Same defect class
    # as rebuild_lib.py's FEATURES.
    processed_dir = str(games_root())
    for col in [
        "frac_choose_A", "entropy_binary", "lk_type", "emp_class",
        "up_choice", "topology", "delta_norm",
    ]:
        df[col] = np.nan
    print("Human reference joins are stored in data/games/taxonomy/, not game_features.csv")

    # ------------------------------------------------------------------
    # 3b. Composite complexity_score
    # ------------------------------------------------------------------
    # Z-scored mean of 5 structural components, normalized using the 144-Bruns
    # subset mean/std. NaN where any input is NaN for that row (strict literal
    # of the spec; ne_distributional_conflict is NaN when num_pure_ne < 2 so
    # most rows will be NaN here -- by design).
    bruns_mask = df["source"] == "bruns"
    df["_ne_complexity"] = df["num_pure_ne"].map({0: 1, 1: 0, 2: 1})
    components = [
        "iesds_depth", "_ne_complexity", "ne_distributional_conflict",
        "payoff_variance", "payoff_conflict",
    ]
    z_cols = []
    skipped = []
    for c in components:
        col = pd.to_numeric(df[c], errors="coerce")
        bruns_vals = col[bruns_mask]
        mu = float(np.nanmean(bruns_vals))
        sd = float(np.nanstd(bruns_vals, ddof=0))
        if sd == 0 or np.isnan(sd):
            # Degenerate within Bruns subset (e.g. payoff_variance is constant
            # 1.25 across all permutations of {1,2,3,4}); skip from composite.
            skipped.append(c)
            continue
        z_name = f"_z_{c}"
        df[z_name] = (col - mu) / sd
        z_cols.append(z_name)
    if skipped:
        print(f"complexity_score: skipped degenerate (sd=0 in Bruns) components: {skipped}")
    df["complexity_score"] = df[z_cols].mean(axis=1, skipna=True)
    df["complexity_score_n_components"] = df[z_cols].notna().sum(axis=1)
    df = df.drop(columns=z_cols + ["_ne_complexity"])

    # ------------------------------------------------------------------
    # 3c. Column order: preserve pre-extension order; append new columns
    # ------------------------------------------------------------------
    pre_extension_columns = [
        "game_code", "dominance_profile", "iesds_depth",
        "num_pure_ne", "pure_ne_cells", "ne_pareto_rankable",
        "ne_distributional_conflict", "payoff_variance", "payoff_conflict",
        "p1_maximin_action", "p2_maximin_action", "joint_maximin_cell",
        "maximin_is_ne", "maximin_to_ne_action_distance",
        "maximin_to_ne_payoff_distance",
        "l1_action_p1", "l1_action_p2", "l2_action_p1", "l2_action_p2",
        "near_equal_split_cell", "pareto_efficient_cells", "source",
        "frac_choose_A", "entropy_binary", "lk_type", "emp_class",
        "up_choice", "topology", "delta_norm",
    ]
    new_columns = [
        "dominant_action_p1", "dominant_action_p2",
        "nash_action_p1", "nash_action_p2",
        "payoff_dominant_ne_p1", "payoff_dominant_ne_p2",
        "risk_dominant_ne_p1", "risk_dominant_ne_p2",
        "l0_action_p1", "l0_action_p2",
        "cooperative_action_p1", "cooperative_action_p2",
        "canonical_action_p1", "canonical_action_p2",
        "complexity_score",
        "complexity_score_n_components",
    ]
    df = df[pre_extension_columns + new_columns]
    if len(df) != 144:
        raise RuntimeError(f"game_features.csv must have exactly 144 Bruns rows, got {len(df)}")
    if set(df["game_code"]) != set(bruns_games):
        missing = sorted(set(bruns_games) - set(df["game_code"]))
        extra = sorted(set(df["game_code"]) - set(bruns_games))
        raise RuntimeError(f"game_features Bruns mismatch; missing={missing}, extra={extra}")

    # ------------------------------------------------------------------
    # 3d. Pin dtypes for determinism (nullable int for sparse, plain int
    #     for always-defined, float for the composite).
    # ------------------------------------------------------------------
    nullable_int_cols = [
        "dominant_action_p1", "dominant_action_p2",
        "nash_action_p1", "nash_action_p2",
        "payoff_dominant_ne_p1", "payoff_dominant_ne_p2",
        "risk_dominant_ne_p1", "risk_dominant_ne_p2",
        "cooperative_action_p1", "cooperative_action_p2",
    ]
    for c in nullable_int_cols:
        df[c] = df[c].astype("Int64")
    df["l0_action_p1"] = df["l0_action_p1"].astype("int64")
    df["l0_action_p2"] = df["l0_action_p2"].astype("int64")
    df["canonical_action_p1"] = df["canonical_action_p1"].astype("int64")
    df["canonical_action_p2"] = df["canonical_action_p2"].astype("int64")
    df["complexity_score"] = df["complexity_score"].astype("float64")
    df["complexity_score_n_components"] = df["complexity_score_n_components"].astype("int64")

    # ------------------------------------------------------------------
    # 4. Save (CSV + Parquet)
    # ------------------------------------------------------------------
    out_path = os.path.join(processed_dir, "game_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    out_pq = os.path.join(processed_dir, "game_features.parquet")
    df.to_parquet(out_pq, index=False)
    print(f"Saved to {out_pq}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total games:  {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nFeature columns ({len(df.columns)}):")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col:<35s} {non_null:>4d} non-null")

    print(f"\n--- dominance_profile distribution ---")
    print(df["dominance_profile"].value_counts().sort_index().to_string())

    print(f"\n--- num_pure_ne distribution ---")
    print(df["num_pure_ne"].value_counts().sort_index().to_string())

    print(f"\n--- iesds_depth distribution ---")
    print(df["iesds_depth"].value_counts().sort_index().to_string())

    print(f"\n--- Sample rows (first 5) ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
