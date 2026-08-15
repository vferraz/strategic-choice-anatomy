"""Unit tests for steering.causal_common.

Pure CPU tests. Run via: pytest tests/test_causal_v2_common.py -q
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from steering.causal_common import (
    PROBE_PREFIX,
    MOVE_LABELS,
    counterbalance_grid,
    build_counterbalanced_prompt,
    readout_action_probs,
    canonical_pref,
    target_pref,
    best_response_action,
    unique_defect_actions,
    signed_slope,
    bootstrap_games,
    assert_counterbalance_balance,
    assert_dose_zero,
    assert_hook_integrity,
    sha256_prompt,
    deterministic_random_unit_vector,
)


def test_probe_prefix_exact():
    assert PROBE_PREFIX == "\nDecision: ", "probe prefix must be exactly \\nDecision: with trailing space"


def test_counterbalance_grid_count_16_and_unique():
    grid = counterbalance_grid()
    assert len(grid) == 16
    # All counterbalance_ids 0..15
    ids = sorted([c["counterbalance_id"] for c in grid])
    assert ids == list(range(16))
    # Each (label_map_id, row_swap, col_swap, valid_moves_order) unique
    sigs = {(c["label_map_id"], c["row_swap"], c["col_swap"],
             c["valid_moves_order"]) for c in grid}
    assert len(sigs) == 16


def test_counterbalance_grid_payload_shape():
    for c in counterbalance_grid():
        assert set(c["action_to_label"].keys()) == {0, 1}
        assert set(c["label_to_action"].keys()) == {"A", "B"}
        # action_to_label ∘ label_to_action == identity
        for a in (0, 1):
            assert c["label_to_action"][c["action_to_label"][a]] == a


def test_sha256_prompt_deterministic():
    a = sha256_prompt("foo")
    b = sha256_prompt("foo")
    c = sha256_prompt("bar")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_unique_defect_actions_tie_returns_none():
    # Symmetric 50/50 joint welfare → tie.
    vec = [1, 1, 1, 1, 1, 1, 1, 1]
    p1, p2 = unique_defect_actions(vec)
    assert p1 is None and p2 is None


def test_unique_defect_actions_prisoners_dilemma():
    # PdPd-style: action 1 strictly defects for P1, action 0 for P2.
    # Bruns PdPd vector: [1,3,2,4, 4,3,2,1]
    p1, p2 = unique_defect_actions([1, 3, 2, 4, 4, 3, 2, 1])
    assert p1 is not None and p2 is not None


def test_canonical_pref_inverts_correctly():
    assert canonical_pref(0.7, 0) == 0.7
    assert canonical_pref(0.7, 1) == pytest.approx(0.3)


def test_target_pref_inverts_correctly():
    assert target_pref(0.7, 0) == 0.7
    assert target_pref(0.7, 1) == pytest.approx(0.3)


def test_best_response_action_basic():
    # PdPd: P1 best response to P2=cooperate(0) is defect(1) because A[1,0]=2 > A[0,0]=1
    vec = [1, 3, 2, 4, 4, 3, 2, 1]
    assert best_response_action(vec, opponent_action=0, player=1) == 1


def test_readout_action_probs_action_space():
    label_probs = {"A": 0.7, "B": 0.2}
    atl = {0: "A", 1: "B"}
    ra = readout_action_probs(label_probs, atl)
    assert ra["prob_act0"] == pytest.approx(0.7)
    assert ra["prob_act1"] == pytest.approx(0.2)
    assert ra["pref0"] == pytest.approx(0.7 / 0.9)
    # Swapped mapping
    atl2 = {0: "B", 1: "A"}
    ra2 = readout_action_probs(label_probs, atl2)
    assert ra2["prob_act0"] == pytest.approx(0.2)


def test_signed_slope_returns_expected_sign():
    df = pd.DataFrame({
        "game_code": ["g1", "g1", "g1", "g2", "g2", "g2"],
        "context_id": ["c"] * 6,
        "x": [-1, 0, 1, -1, 0, 1],
        "y": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0],
    })
    s = signed_slope(df, "x", "y", group_cols=["game_code", "context_id"])
    assert s == pytest.approx(1.5)


def test_bootstrap_games_basic():
    df = pd.DataFrame({
        "game_code": ["g1"] * 10 + ["g2"] * 10,
        "context_id": ["c"] * 20,
        "x": list(range(-5, 5)) * 2,
        "y": list(range(-5, 5)) * 2,
    })
    fn = lambda d: signed_slope(d, "x", "y", group_cols=["game_code", "context_id"])
    res = bootstrap_games(df, fn, n_boot=200, seed=0)
    assert res["n_games"] == 2
    assert res["mean"] == pytest.approx(1.0, abs=1e-6)


def test_assert_dose_zero_passes_clean():
    df = pd.DataFrame({
        "dose": [0.0, 0.5, -0.5],
        "pref0_delta": [0.0, 0.1, -0.1],
        "target_pref_delta": [0.0, 0.1, -0.1],
    })
    assert_dose_zero(df, eps=1e-9)


def test_assert_dose_zero_fires_on_violation():
    df = pd.DataFrame({
        "dose": [0.0, 0.5],
        "pref0_delta": [0.01, 0.1],
        "target_pref_delta": [0.0, 0.1],
    })
    with pytest.raises(AssertionError):
        assert_dose_zero(df, eps=1e-9)


def test_assert_hook_integrity_passes_when_clean():
    df = pd.DataFrame({
        "dose": [0.5, -0.5],
        "model": ["m"] * 2, "layer": [78] * 2,
        "requested_delta_norm": [10.0, 10.0],
        "actual_delta_norm": [10.0, 10.0],
        "actual_delta_cosine_with_requested_delta": [1.0, 1.0],
    })
    assert_hook_integrity(df)


def test_assert_hook_integrity_fires_on_bad_ratio():
    df = pd.DataFrame({
        "dose": [0.5, -0.5],
        "model": ["m"] * 2, "layer": [78] * 2,
        "requested_delta_norm": [10.0, 10.0],
        "actual_delta_norm": [5.0, 5.0],
        "actual_delta_cosine_with_requested_delta": [1.0, 1.0],
    })
    with pytest.raises(AssertionError):
        assert_hook_integrity(df)


def test_assert_counterbalance_balance_passes_full_grid():
    # Synthesize one game, one layer, one dose: 16 unique cells
    grid = counterbalance_grid()
    rows = []
    for cb in grid:
        rows.append({
            "model": "m", "mode": "h0", "game_code": "g", "context_id": "c",
            "layer": 78, "dose": 0.1, "direction": "d",
            "label_map_id": cb["label_map_id"],
            "row_swap_p1": int(cb["row_swap"]),
            "col_swap_p1": int(cb["col_swap"]),
            "valid_moves_order_p1": cb["valid_moves_order"],
        })
    df = pd.DataFrame(rows)
    assert_counterbalance_balance(df)


def test_assert_counterbalance_balance_fires_on_missing_cell():
    grid = counterbalance_grid()[:-1]  # drop one cell
    rows = []
    for cb in grid:
        rows.append({
            "model": "m", "mode": "h0", "game_code": "g", "context_id": "c",
            "layer": 78, "dose": 0.1, "direction": "d",
            "label_map_id": cb["label_map_id"],
            "row_swap_p1": int(cb["row_swap"]),
            "col_swap_p1": int(cb["col_swap"]),
            "valid_moves_order_p1": cb["valid_moves_order"],
        })
    df = pd.DataFrame(rows)
    with pytest.raises(AssertionError):
        assert_counterbalance_balance(df)


def test_deterministic_random_unit_vector_repeatable():
    v1 = deterministic_random_unit_vector(64, seed_keys=("a", "b", 1))
    v2 = deterministic_random_unit_vector(64, seed_keys=("a", "b", 1))
    np.testing.assert_array_equal(v1, v2)
    assert v1.shape == (64,)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5


def test_build_counterbalanced_prompt_contains_decision_after_swap():
    grid = counterbalance_grid()
    # Pick a PdPd-like vector
    vec = [1, 3, 2, 4, 4, 3, 2, 1]
    prompts = [build_counterbalanced_prompt(vec, [], False, cb) for cb in grid]
    assert all(isinstance(p, str) and len(p) > 0 for p in prompts)
    # Sanity: row/col swap actually changes prompt text
    assert prompts[0] != prompts[3] or prompts[0] != prompts[5]
