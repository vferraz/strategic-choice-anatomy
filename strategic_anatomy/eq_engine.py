"""Matrix-level solution-concept engine for 2x2 games.

All public pair outputs are on the canonical act0 axis:
``p = P(player chooses action 0)``. Internal ``*_action_*`` fields keep raw
action indices for validation against ``game_features.csv``.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable

import nashpy as nash
import numpy as np

EPS = 1e-9


def matrices_from_8vec(vec: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return row-player and column-player payoff matrices from an 8-vector."""
    vals = [float(v) for v in vec]
    if len(vals) != 8:
        raise ValueError(f"expected 8 payoff values, got {len(vals)}")
    return np.array(vals[:4], dtype=float).reshape(2, 2), np.array(
        vals[4:], dtype=float
    ).reshape(2, 2)


def action_to_act0_prob(action: float | int | None) -> float:
    """Map an action index to probability of act0.

    ``-1`` is the local sentinel for ties/undefined best responses.
    """
    if action is None:
        return math.nan
    try:
        if math.isnan(float(action)):
            return math.nan
    except (TypeError, ValueError):
        return math.nan
    if int(action) == -1:
        return 0.5
    if int(action) == 0:
        return 1.0
    if int(action) == 1:
        return 0.0
    raise ValueError(f"action must be -1, 0, or 1; got {action!r}")


def _raw_or_nan(action: int | None) -> float:
    return math.nan if action is None else float(action)


def _best_response(evs: Iterable[float]) -> int:
    vals = [float(v) for v in evs]
    if len(vals) != 2:
        raise ValueError("2x2 games require exactly two expected values")
    if abs(vals[0] - vals[1]) <= EPS:
        return -1
    return int(np.argmax(vals))


def _is_zero_one(value: float) -> bool:
    return abs(value) <= EPS or abs(value - 1.0) <= EPS


def _is_pure_pair(pair: tuple[float, float]) -> bool:
    return _is_zero_one(pair[0]) and _is_zero_one(pair[1])


def _rounded_pair(pair: tuple[float, float]) -> tuple[float, float]:
    out = []
    for value in pair:
        if abs(value) <= EPS:
            out.append(0.0)
        elif abs(value - 1.0) <= EPS:
            out.append(1.0)
        else:
            out.append(float(value))
    return tuple(out)  # type: ignore[return-value]


def strictly_dominant_action(matrix: np.ndarray) -> int | None:
    """Strictly dominant row action, or None."""
    if np.all(matrix[0, :] > matrix[1, :]):
        return 0
    if np.all(matrix[1, :] > matrix[0, :]):
        return 1
    return None


def pure_nash_equilibria(p1: np.ndarray, p2: np.ndarray) -> list[tuple[int, int]]:
    """Return pure Nash cells as ``(row_action, col_action)``."""
    cells: list[tuple[int, int]] = []
    for i in range(2):
        for j in range(2):
            if p1[i, j] >= p1[1 - i, j] and p2[i, j] >= p2[i, 1 - j]:
                cells.append((i, j))
    return cells


def payoff_dominant_ne(
    ne_cells: list[tuple[int, int]], p1: np.ndarray, p2: np.ndarray
) -> tuple[int, int] | None:
    """Return the Pareto-dominant pure NE when uniquely defined."""
    if len(ne_cells) != 2:
        return None
    a, b = ne_cells
    if p1[a] >= p1[b] and p2[a] >= p2[b] and (p1[a] > p1[b] or p2[a] > p2[b]):
        return a
    if p1[b] >= p1[a] and p2[b] >= p2[a] and (p1[b] > p1[a] or p2[b] > p2[a]):
        return b
    return None


def risk_dominant_ne(
    ne_cells: list[tuple[int, int]], p1: np.ndarray, p2: np.ndarray
) -> tuple[int, int] | None:
    """Return the Harsanyi-Selten risk-dominant pure NE when uniquely defined."""
    if len(ne_cells) != 2:
        return None
    a, b = ne_cells

    def deviation_loss_product(x: tuple[int, int], y: tuple[int, int]) -> float:
        p1_loss = p1[x] - p1[(y[0], x[1])]
        p2_loss = p2[x] - p2[(x[0], y[1])]
        return float(p1_loss * p2_loss)

    pa = deviation_loss_product(a, b)
    pb = deviation_loss_product(b, a)
    if pa > pb:
        return a
    if pb > pa:
        return b
    return None


def level_k_actions(p1: np.ndarray, p2: np.ndarray) -> dict[str, int]:
    """Compute raw L1, L2, and L3 best-response actions.

    L0 is uniform. A tied expected value returns ``-1``.
    """
    l1_p1 = _best_response([0.5 * p1[i, 0] + 0.5 * p1[i, 1] for i in range(2)])
    l1_p2 = _best_response([0.5 * p2[0, j] + 0.5 * p2[1, j] for j in range(2)])

    l2_p1 = -1 if l1_p2 == -1 else _best_response([p1[i, l1_p2] for i in range(2)])
    l2_p2 = -1 if l1_p1 == -1 else _best_response([p2[l1_p1, j] for j in range(2)])

    l3_p1 = -1 if l2_p2 == -1 else _best_response([p1[i, l2_p2] for i in range(2)])
    l3_p2 = -1 if l2_p1 == -1 else _best_response([p2[l2_p1, j] for j in range(2)])

    return {
        "l1_action_p1": l1_p1,
        "l1_action_p2": l1_p2,
        "l2_action_p1": l2_p1,
        "l2_action_p2": l2_p2,
        "l3_action_p1": l3_p1,
        "l3_action_p2": l3_p2,
    }


def l1alpha_actions(p1: np.ndarray, p2: np.ndarray, alpha: float = 0.5) -> dict[str, int]:
    """Level-1 best response with variance penalty ``EV - alpha * Var``."""

    def adjusted(values: Iterable[float]) -> float:
        arr = np.array(list(values), dtype=float)
        return float(arr.mean() - alpha * arr.var())

    p1_scores = [adjusted(p1[i, :]) for i in range(2)]
    p2_scores = [adjusted(p2[:, j]) for j in range(2)]
    return {
        "l1alpha_action_p1": _best_response(p1_scores),
        "l1alpha_action_p2": _best_response(p2_scores),
    }


def maximin_actions(p1: np.ndarray, p2: np.ndarray) -> dict[str, int]:
    """Raw maximin actions, with ``-1`` on exact maximin ties."""
    p1_mins = [float(np.min(p1[i, :])) for i in range(2)]
    p2_mins = [float(np.min(p2[:, j])) for j in range(2)]
    return {
        "maximin_action_p1": _best_response(p1_mins),
        "maximin_action_p2": _best_response(p2_mins),
    }


def equal_split_actions(p1: np.ndarray, p2: np.ndarray) -> dict[str, int]:
    """Cell minimizing absolute payoff gap; unresolved action ties become -1."""
    gaps = [
        (abs(float(p1[i, j] - p2[i, j])), i, j)
        for i in range(2)
        for j in range(2)
    ]
    best_gap = min(g[0] for g in gaps)
    best = [(i, j) for gap, i, j in gaps if abs(gap - best_gap) <= EPS]
    rows = {i for i, _ in best}
    cols = {j for _, j in best}
    return {
        "equalsplit_action_p1": next(iter(rows)) if len(rows) == 1 else -1,
        "equalsplit_action_p2": next(iter(cols)) if len(cols) == 1 else -1,
    }


def enumerate_nash(p1: np.ndarray, p2: np.ndarray) -> tuple[list[tuple[float, float]], str]:
    """Enumerate Nash equilibria as act0-probability pairs plus kind label."""
    game = nash.Game(p1, p2)
    degenerate = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = list(game.support_enumeration())
    for item in caught:
        message = str(item.message).lower()
        if "degenerate" in message:
            degenerate = True

    pairs: list[tuple[float, float]] = []
    for sigma_r, sigma_c in raw:
        pair = _rounded_pair((float(np.asarray(sigma_r)[0]), float(np.asarray(sigma_c)[0])))
        if any(math.isnan(v) for v in pair):
            continue
        if not any(abs(pair[0] - q[0]) <= EPS and abs(pair[1] - q[1]) <= EPS for q in pairs):
            pairs.append(pair)

    pure = sorted([p for p in pairs if _is_pure_pair(p)], key=lambda x: (x[0], x[1]))
    mixed = sorted([p for p in pairs if not _is_pure_pair(p)], key=lambda x: (x[0], x[1]))
    ordered = pure + mixed

    if degenerate:
        kind = "degenerate"
    elif len(ordered) == 1 and pure:
        kind = "unique_pure"
    elif len(ordered) == 1 and mixed:
        kind = "unique_mixed"
    elif len(ordered) == 3 and len(pure) == 2 and len(mixed) == 1:
        kind = "two_pure_plus_mixed"
    else:
        kind = "other"
    return ordered, kind


def compute_matrix_metrics(vec: Iterable[float]) -> dict[str, float | int | str]:
    """Compute all row-level theoretical fields for an 8-vector."""
    p1, p2 = matrices_from_8vec(vec)
    out: dict[str, float | int | str] = {}

    ne_pairs, ne_kind = enumerate_nash(p1, p2)
    out["ne_count"] = len(ne_pairs)
    out["ne_kind"] = ne_kind
    for idx in range(3):
        if idx < len(ne_pairs):
            out[f"ne{idx + 1}_p1"] = float(ne_pairs[idx][0])
            out[f"ne{idx + 1}_p2"] = float(ne_pairs[idx][1])
        else:
            out[f"ne{idx + 1}_p1"] = math.nan
            out[f"ne{idx + 1}_p2"] = math.nan

    pure_ne = pure_nash_equilibria(p1, p2)
    out["_num_pure_ne"] = float(len(pure_ne))
    if len(pure_ne) == 1:
        out["_unique_pure_ne_action_p1"] = float(pure_ne[0][0])
        out["_unique_pure_ne_action_p2"] = float(pure_ne[0][1])
    else:
        out["_unique_pure_ne_action_p1"] = math.nan
        out["_unique_pure_ne_action_p2"] = math.nan

    dom_p1 = strictly_dominant_action(p1)
    dom_p2 = strictly_dominant_action(p2.T)
    out["_dom_action_p1"] = _raw_or_nan(dom_p1)
    out["_dom_action_p2"] = _raw_or_nan(dom_p2)
    out["dom_p1"] = action_to_act0_prob(dom_p1)
    out["dom_p2"] = action_to_act0_prob(dom_p2)

    for key, action in level_k_actions(p1, p2).items():
        out[f"_{key}"] = float(action)
        public_key = key.replace("_action", "")
        out[public_key] = action_to_act0_prob(action)

    for key, action in l1alpha_actions(p1, p2).items():
        out[f"_{key}"] = float(action)
        public_key = key.replace("_action", "")
        out[public_key] = action_to_act0_prob(action)

    for key, action in maximin_actions(p1, p2).items():
        out[f"_{key}"] = float(action)
        public_key = key.replace("_action", "")
        out[public_key] = action_to_act0_prob(action)

    for key, action in equal_split_actions(p1, p2).items():
        out[f"_{key}"] = float(action)
        public_key = key.replace("_action", "")
        out[public_key] = action_to_act0_prob(action)

    paydom = payoff_dominant_ne(pure_ne, p1, p2)
    riskdom = risk_dominant_ne(pure_ne, p1, p2)
    for label, cell in (("paydom", paydom), ("riskdom", riskdom)):
        if cell is None:
            out[f"_{label}_action_p1"] = math.nan
            out[f"_{label}_action_p2"] = math.nan
            out[f"target_{label}_p1"] = math.nan
            out[f"target_{label}_p2"] = math.nan
        else:
            out[f"_{label}_action_p1"] = float(cell[0])
            out[f"_{label}_action_p2"] = float(cell[1])
            out[f"target_{label}_p1"] = action_to_act0_prob(cell[0])
            out[f"target_{label}_p2"] = action_to_act0_prob(cell[1])

    return out
