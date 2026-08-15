"""One-shot single-shot prompt + shared helpers (additive; no core file edited).

This is the ONLY place the prompt text differs from causal V2:
``build_oneshot_prompt`` reproduces ``strategic_anatomy.prompting.build_prompt`` MINUS the
cumulative-payoff line and the history block (SPRINT_oneshot_redesign.md §3).
Everything else is re-exported by identity from ``causal_v2_common`` so the
one-shot pipeline shares the single source of truth for the 16-cell counterbalance
grid, action-space readouts, structural asserts, cluster bootstrap, etc.

It also defines the pure incentive-gap math (``u_mats``/``delta1``/``delta2``/
``canonical_sign``) re-implemented HERE so that no old behavioral data
(``c6_brain_core`` / ``unified_pairs.parquet``) can leak into the new substrate
(§1, §6). The belief ``q`` is supplied by the caller (player-specific: a player's
q-hat is its opponent's act0 rate).

CLAUDE.md hard constraints respected:
  #1 action space — readouts only via ``readout_action_probs`` (re-exported).
  #2 canonical axis — ``canonical_pref`` / ``canonical_sign`` require the per-row
     canonical action and never collapse positional labels across games.
  #3 probe prefix — ``PROBE_PREFIX`` is exactly ``"\\nDecision: "``.
  #4 counterbalancing — ``counterbalance_grid`` is the replicate dimension.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Allow imports from the project root and src/.

# Section helpers (read-only import; never edited). build_prompt and
# build_counterbalanced_prompt are deliberately NOT imported — they emit the
# cumulative-payoff line the one-shot substrate drops.
from strategic_anatomy.prompting import (  # noqa: E402
    payoff_rules_symmetric,
    valid_moves_line,
    format_instruction,
    build_move_token_map,
)

# Re-export the V2 single-source-of-truth helpers by identity.
from steering.causal_common import (  # noqa: E402
    PROBE_PREFIX,
    MOVE_LABELS,
    ACTION_LABELS,
    counterbalance_grid,
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
    load_game_vec,
    load_game_universe,
)

__all__ = [
    "build_oneshot_prompt",
    "oneshot_prompt_template_hash",
    "u_mats", "delta1", "delta2", "canonical_sign",
    # re-exports (so callers import everything from one place)
    "PROBE_PREFIX", "MOVE_LABELS", "ACTION_LABELS", "counterbalance_grid",
    "readout_action_probs", "canonical_pref", "target_pref",
    "best_response_action", "unique_defect_actions", "signed_slope",
    "bootstrap_games", "assert_counterbalance_balance", "assert_dose_zero",
    "assert_hook_integrity", "sha256_prompt", "deterministic_random_unit_vector",
    "load_game_vec", "load_game_universe", "build_move_token_map",
    "payoff_rules_symmetric", "valid_moves_line", "format_instruction",
]

# Static prompt strings copied verbatim from build_prompt (NOT the dropped chunks).
_INTRO_LINE = "You are about to play one round of a two-player game."
_VALID_MOVES_NOTE = (
    "Note: The order of 'Valid moves' below is arbitrary; "
    "do not infer preference from it."
)
_CHOOSE_LINE = "Choose your move based on your preferences and the game rules above."


def build_oneshot_prompt(
    game_vec: Sequence[int],
    cb: dict,
    *,
    player: int = 1,
    iv_prefix: str = "",
) -> str:
    """Single-shot prompt = ``build_prompt`` MINUS cumulative-payoff line + history (§3).

    Consumes the same counterbalance cell dict as ``build_counterbalanced_prompt``
    (``action_to_label`` / ``row_swap`` / ``col_swap`` / ``valid_moves``).
    ``shuffle`` is hard-False: the cell already fixes the move order and
    deterministic one-shot uses no extra RNG.
    """
    chunks: list[str] = []
    if iv_prefix:
        chunks.append(iv_prefix)
    chunks.append(_INTRO_LINE)
    chunks.append(payoff_rules_symmetric(
        list(game_vec), int(player), cb["action_to_label"],
        swap_rows=bool(cb["row_swap"]), swap_cols=bool(cb["col_swap"]),
    ))
    # DROPPED vs build_prompt: "Cumulative payoff so far: ..." and the history block.
    chunks.append(_VALID_MOVES_NOTE)
    chunks.append(valid_moves_line(cb["valid_moves"], shuffle=False))
    chunks.append(_CHOOSE_LINE)
    chunks.append(format_instruction())
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Pure incentive-gap math (re-implemented here; no old behavioral data, §6).
# game_vec is the 8-element Bruns vector [a00,a01,a10,a11, b00,b01,b10,b11].
# u1[i,j] = P1 payoff when P1 plays row i, P2 plays col j; u2[i,j] likewise for P2.
# ---------------------------------------------------------------------------
def u_mats(game_vec: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (u1, u2) as 2x2 arrays indexed [p1_action, p2_action]."""
    v = np.asarray([float(x) for x in game_vec], dtype=float)
    return v[:4].reshape(2, 2), v[4:].reshape(2, 2)


def delta1(game_vec: Sequence[float], q: float) -> float:
    """EU1(act0) - EU1(act1) when P2 plays act0 with probability q (P1's belief).

    P1 is the row player: EU1(i) = q*u1[i,0] + (1-q)*u1[i,1].
    """
    u1, _ = u_mats(game_vec)
    q = float(q)
    return float((q * u1[0, 0] + (1 - q) * u1[0, 1])
                 - (q * u1[1, 0] + (1 - q) * u1[1, 1]))


def delta2(game_vec: Sequence[float], q: float) -> float:
    """EU2(act0) - EU2(act1) when P1 plays act0 with probability q (P2's belief).

    P2 is the column player: EU2(j) = q*u2[0,j] + (1-q)*u2[1,j].
    """
    _, u2 = u_mats(game_vec)
    q = float(q)
    return float((q * u2[0, 0] + (1 - q) * u2[1, 0])
                 - (q * u2[0, 1] + (1 - q) * u2[1, 1]))


def canonical_sign(delta: float, canonical_action: int) -> float:
    """Orient a positional incentive gap so positive favors the canonical action."""
    ca = int(canonical_action)
    if ca == 0:
        return float(delta)
    if ca == 1:
        return float(-delta)
    raise ValueError(f"canonical_action must be 0 or 1, got {ca}")


def oneshot_prompt_template_hash() -> str:
    """Stable fingerprint of the single-shot prompt template (static chunks only).

    Used to detect prompt drift across runs without depending on a specific game.
    """
    import hashlib
    tmpl = "\n\n".join([_INTRO_LINE, _VALID_MOVES_NOTE, _CHOOSE_LINE, format_instruction()])
    return hashlib.sha256(tmpl.encode("utf-8")).hexdigest()
