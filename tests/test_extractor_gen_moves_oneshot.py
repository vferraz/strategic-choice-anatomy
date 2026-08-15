"""Unit tests for the corrected-decoder override in steering.extract_directions.

These cover the SPEC_oneshot_FINAL §9 re-pointing of q̂ and d_choice from the slot-argmax
`decoded_action` to the committed generation move:
  - q̂ is the act0 rate of the PARSED move (parse_ok rows only);
  - the per-row action map excludes unparsed moves;
  - d_choice drops rows whose move did not parse (decoded_action < 0), so an unparsed move
    is never silently scored as "chose the non-canonical action".
"""
import numpy as np
import pandas as pd

from steering.extract_directions import (
    gen_qhat_series, gen_action_map, fit_d_choice,
)


def _moves_df():
    """G1: P1-baseline cells -> act 0,1 (qhat_p1=0.5); P2-baseline parsed -> 0,0 (qhat_p2=1.0);
    one P2 cell failed to parse and must be ignored by both q̂ and the action map."""
    rows = [
        dict(game_code="G1", cb_id=0, player=1, condition="baseline", parsed_action=0, parse_ok=True),
        dict(game_code="G1", cb_id=1, player=1, condition="baseline", parsed_action=1, parse_ok=True),
        dict(game_code="G1", cb_id=0, player=2, condition="baseline", parsed_action=0, parse_ok=True),
        dict(game_code="G1", cb_id=1, player=2, condition="baseline", parsed_action=0, parse_ok=True),
        dict(game_code="G1", cb_id=2, player=2, condition="baseline", parsed_action=1, parse_ok=False),
    ]
    return pd.DataFrame(rows)


def test_gen_qhat_uses_parsed_move_and_parse_ok_only():
    mv = _moves_df()
    assert gen_qhat_series(mv, 1)["G1"] == 0.5      # one act0 of two parsed P1-baseline
    assert gen_qhat_series(mv, 2)["G1"] == 1.0      # both PARSED P2-baseline are act0; failed row excluded


def test_gen_action_map_excludes_unparsed():
    amap = gen_action_map(_moves_df())
    assert amap[("G1", 0, 1, "baseline")] == 0
    assert amap[("G1", 1, 1, "baseline")] == 1
    assert ("G1", 2, 2, "baseline") not in amap     # parse_ok=False dropped


def test_fit_d_choice_drops_unparsed_rows():
    # rows 2,3 are unparsed (decoded_action=-1) and must not enter the d_choice fit.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    meta = pd.DataFrame({
        "game_code": ["A", "B", "C", "D"],
        "canonical_action_p1": [0, 1, 0, 1],
        "decoded_action": [0, 1, -1, -1],
    })
    _v, diag = fit_d_choice(X, meta, min_games_per_class=10, min_auc=0.55, seed=0)
    assert diag["n_rows"] == 2                       # the two decoded_action=-1 rows excluded
