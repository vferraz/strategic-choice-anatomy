"""Shared helpers for Spark Causal Protocol V2.

This is a pure library. No file I/O at import time. All forward-pass code lives in
``run_causal_protocol_v2.py``; all direction extraction in
``extract_causal_directions_v2.py``; all summary aggregation in
``analyze_causal_protocol_v2.py``. The exported helpers here are the single
source of truth for:

  - the 16-cell factorial counterbalance grid (spec §0.2)
  - action-space readouts (spec §0.1, docs/METHODS.md HC-1)
  - target-action / best-response game-theoretic helpers
  - cluster bootstrap by game (spec §0.4)
  - structural invariant assertions (spec §0.5, §0.6, §0.2)
  - prompt SHA256 hashing
  - explicit tie-checked defect-action computation (spec §5.1)
  - the probe-prefix constant ``PROBE_PREFIX = "\\nDecision: "`` (spec §0.8, docs/METHODS.md HC-3)

Reuses, by identity, from the existing apparatus:
  - ``strategic_anatomy.prompting.build_prompt`` (prompt construction)
  - ``strategic_anatomy.prompting.build_move_token_map`` (decision-slot token IDs)
  - ``strategic_anatomy.prompting.MOVE_LABELS`` (the abstract ``["A", "B"]`` axis)
  - ``analysis.block_b.probe_program.probe_world_model.defect_actions_from_vec``
    (kept reachable so callers can compare against the tie-tolerant variant)

the hard constraints in docs/METHODS.md respected:
  #1 action space: ``readout_action_probs`` is the only conversion point.
  #2 canonical axis: ``canonical_pref`` requires the per-row canonical_action_p1
     and must never collapse positional labels across games.
  #3 probe prefix: ``PROBE_PREFIX`` is exactly ``"\\nDecision: "``.
  #4 counterbalancing: ``counterbalance_grid`` is the V2 substitute for the
     ``rounds``-driven schedules in ``strategic_anatomy.runtime``.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# Allow imports from src/ via the project root.

from strategic_anatomy.prompting import (  # noqa: E402
    MOVE_LABELS,
    build_prompt,
    build_move_token_map,
)

# Re-export for callers (without forcing them to re-import).
__all__ = [
    "PROBE_PREFIX",
    "MOVE_LABELS",
    "ACTION_LABELS",
    "build_move_token_map",
    "counterbalance_grid",
    "build_counterbalanced_prompt",
    "readout_action_probs",
    "canonical_pref",
    "best_response_action",
    "target_pref",
    "signed_slope",
    "bootstrap_games",
    "assert_counterbalance_balance",
    "assert_dose_zero",
    "assert_hook_integrity",
    "sha256_prompt",
    "unique_defect_actions",
    "load_game_universe",
    "load_game_vec",
    "deterministic_random_unit_vector",
]

# Spec §0.8 — exact probe prefix, with leading newline and trailing space.
PROBE_PREFIX: str = "\nDecision: "
ACTION_LABELS: tuple[int, int] = (0, 1)


# ---------------------------------------------------------------------------
# Counterbalance grid (spec §0.2)
# ---------------------------------------------------------------------------
def counterbalance_grid() -> list[dict]:
    """Return the exact 16-cell factorial counterbalance grid.

    Each cell is a dict ready to be passed to ``build_counterbalanced_prompt``.
    Order is deterministic: counterbalance_id 0..15 in row-major over
    (label_map_id, row_swap, col_swap, valid_moves).
    """
    rows: list[dict] = []
    counterbalance_id = 0
    for label_map_id in (0, 1):
        if label_map_id == 0:
            label_to_action = {"A": 0, "B": 1}
        else:
            label_to_action = {"A": 1, "B": 0}
        action_to_label = {v: k for k, v in label_to_action.items()}
        for row_swap in (False, True):
            for col_swap in (False, True):
                for valid_moves in (["A", "B"], ["B", "A"]):
                    rows.append({
                        "counterbalance_id": counterbalance_id,
                        "label_map_id": label_map_id,
                        "label_to_action": dict(label_to_action),
                        "action_to_label": dict(action_to_label),
                        "row_swap": bool(row_swap),
                        "col_swap": bool(col_swap),
                        "valid_moves": list(valid_moves),
                        "valid_moves_order": "".join(valid_moves),
                        "label_map_p1": action_to_label[0],
                    })
                    counterbalance_id += 1
    assert len(rows) == 16, f"V2 grid must have 16 cells, got {len(rows)}"
    return rows


def build_counterbalanced_prompt(
    game_vec: Sequence[int],
    history: list[tuple[int, int, int, int]] | None,
    include_history: bool,
    cb: dict,
    *,
    player: int = 1,
    cumulative_you: int = 0,
    cumulative_opp: int = 0,
    iv_prefix: str = "",
    max_history: int = 0,
) -> str:
    """Wrap ``build_prompt`` with a counterbalance cell.

    The cell carries label_to_action / action_to_label / row_swap / col_swap /
    valid_moves. ``shuffle_valid_moves`` is hard-coded False because the cell
    already specifies the move order; we do not want any extra randomness.
    """
    return build_prompt(
        game_vec=list(game_vec),
        history=history or [],
        include_history=bool(include_history),
        valid_moves=cb["valid_moves"],
        player=int(player),
        cumulative_you=int(cumulative_you),
        cumulative_opp=int(cumulative_opp),
        action_to_label=cb["action_to_label"],
        swap_rows=bool(cb["row_swap"]),
        swap_cols=bool(cb["col_swap"]),
        iv_prefix=iv_prefix or "",
        max_history=int(max_history),
        shuffle_valid_moves=False,
    )


# ---------------------------------------------------------------------------
# Action-space readouts (spec §0.1, docs/METHODS.md HC-1)
# ---------------------------------------------------------------------------
def readout_action_probs(
    final_probs: dict[str, float],
    action_to_label: dict[int, str],
) -> dict[str, float]:
    """Convert {"A": p, "B": p} label probabilities into action-space probs.

    Returns a dict carrying both label space (``prob_A``, ``prob_B``) and the
    canonical action-space readouts (``prob_act0``, ``prob_act1``, ``pref0``).
    ``pref0`` is the renormalized P(action0) over (act0+act1).
    """
    prob_a = float(final_probs.get("A", 0.0))
    prob_b = float(final_probs.get("B", 0.0))
    prob_act0 = float(final_probs[action_to_label[0]])
    prob_act1 = float(final_probs[action_to_label[1]])
    den = prob_act0 + prob_act1
    pref0 = float(prob_act0 / den) if den > 0 else math.nan
    return {
        "prob_A": prob_a,
        "prob_B": prob_b,
        "prob_act0": prob_act0,
        "prob_act1": prob_act1,
        "pref0": pref0,
    }


def canonical_pref(pref0: float, canonical_action_p1: int) -> float:
    """Return P(canonical action) given pref0 and the game's canonical_action_p1.

    Spec §0.1 + docs/METHODS.md HC-2 — never aggregate raw pref0 across games.
    """
    ca = int(canonical_action_p1)
    if ca == 0:
        return float(pref0)
    if ca == 1:
        return 1.0 - float(pref0)
    raise ValueError(f"canonical_action_p1 must be 0 or 1, got {ca}")


def target_pref(pref0: float, target_action: int) -> float:
    """Return P(target action). Same shape as canonical_pref but for any target."""
    ta = int(target_action)
    if ta == 0:
        return float(pref0)
    if ta == 1:
        return 1.0 - float(pref0)
    raise ValueError(f"target_action must be 0 or 1, got {ta}")


# ---------------------------------------------------------------------------
# Game-theoretic helpers
# ---------------------------------------------------------------------------
def best_response_action(
    game_vec: Sequence[float],
    opponent_action: int,
    *,
    player: int = 1,
) -> int:
    """Return the P1 (or P2) best response to ``opponent_action``.

    ``game_vec`` is the 8-element Bruns vector ``[a00,a01,a10,a11,b00,b01,b10,b11]``.
    """
    v = [float(x) for x in game_vec]
    a = np.array(v[:4]).reshape(2, 2)   # P1 payoff matrix A[p1, p2]
    b = np.array(v[4:]).reshape(2, 2)   # P2 payoff matrix B[p1, p2]
    op = int(opponent_action)
    if player == 1:
        return int(np.argmax(a[:, op]))
    if player == 2:
        return int(np.argmax(b[op, :]))
    raise ValueError(f"player must be 1 or 2, got {player}")


def unique_defect_actions(
    game_vec: Sequence[float],
    *,
    tol: float = 1e-9,
) -> tuple[int | None, int | None]:
    """Return (p1_defect, p2_defect) only when each is strictly the lower-joint-welfare action.

    Returns ``None`` for either side that is tied. Spec §5.1 requires this
    explicit tie check; the existing ``defect_actions_from_vec`` in
    ``probe_world_model.py`` tie-breaks via numpy argmin and would silently
    classify ties.
    """
    v = [float(x) for x in game_vec]
    joint = [[v[0] + v[4], v[1] + v[5]],
             [v[2] + v[6], v[3] + v[7]]]
    p1_mean = [(joint[0][0] + joint[0][1]) / 2.0,
               (joint[1][0] + joint[1][1]) / 2.0]
    p2_mean = [(joint[0][0] + joint[1][0]) / 2.0,
               (joint[0][1] + joint[1][1]) / 2.0]

    def _arg_strict_min(xs: list[float]) -> int | None:
        if abs(xs[0] - xs[1]) <= tol:
            return None
        return 0 if xs[0] < xs[1] else 1

    return _arg_strict_min(p1_mean), _arg_strict_min(p2_mean)


# ---------------------------------------------------------------------------
# Slope + cluster bootstrap (spec §0.4)
# ---------------------------------------------------------------------------
def signed_slope(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_cols: Sequence[str],
) -> float:
    """OLS slope of y on x AFTER averaging within (group_cols, x) cells.

    Spec §0.4 — average over counterbalances within (game, context) first, then
    fit the slope. Returns NaN if fewer than two distinct x values.
    """
    g = (df[list(group_cols) + [x_col, y_col]]
         .groupby(list(group_cols) + [x_col], as_index=False)[y_col]
         .mean())
    x = g[x_col].to_numpy(dtype=float)
    y = g[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(np.unique(x)) < 2:
        return float("nan")
    xm = x - x.mean()
    den = float(np.dot(xm, xm))
    if den <= 0:
        return float("nan")
    return float(np.dot(xm, y - y.mean()) / den)


def bootstrap_games(
    df: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    *,
    game_col: str = "game_code",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster bootstrap by game.

    Returns {"mean": stat-on-data, "ci_lo": 2.5pct, "ci_hi": 97.5pct,
             "n_boot_valid": n_finite_replicates, "n_games": n_distinct_games}.
    """
    games = df[game_col].unique()
    by_game = {g: df[df[game_col] == g] for g in games}
    base = float(statistic_fn(df))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    if len(games) == 0:
        return {"mean": base, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot_valid": 0, "n_games": 0}
    for _ in range(int(n_boot)):
        pick = rng.choice(games, size=len(games), replace=True)
        sub = pd.concat([by_game[g] for g in pick], ignore_index=True)
        try:
            v = float(statistic_fn(sub))
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            samples.append(v)
    if not samples:
        return {"mean": base, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot_valid": 0, "n_games": int(len(games))}
    arr = np.array(samples)
    return {
        "mean": float(base),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "n_boot_valid": int(len(arr)),
        "n_games": int(len(games)),
    }


# ---------------------------------------------------------------------------
# Structural invariants (spec §0.5, §0.6, §0.2)
# ---------------------------------------------------------------------------
_COUNTERBALANCE_COLS = ("label_map_id", "row_swap_p1", "col_swap_p1", "valid_moves_order_p1")


def assert_counterbalance_balance(df: pd.DataFrame) -> None:
    """Raise AssertionError if any cell of the 16-cell grid is missing or duplicated.

    Spec §0.2 — within each (model, game_code, context_id, direction, layer,
    dose) block (the natural unit of the runner's enumeration loop), each of
    the 16 counterbalance cells must appear exactly once. Tolerance is zero.
    """
    missing = [c for c in _COUNTERBALANCE_COLS + ("model", "mode", "game_code", "layer", "dose") if c not in df.columns]
    if missing:
        raise AssertionError(f"assert_counterbalance_balance: missing columns {missing}")
    grouped = df.groupby(
        ["model", "mode", "game_code", "layer", "dose", "direction"]
        + [c for c in ("context_id",) if c in df.columns],
        dropna=False,
    )
    for key, sub in grouped:
        # The factorial grid: 16 unique cells expected.
        cell_counts = sub.groupby(list(_COUNTERBALANCE_COLS), dropna=False).size()
        if len(cell_counts) != 16 or not (cell_counts == 1).all():
            raise AssertionError(
                f"counterbalance imbalance in block {key}: got "
                f"{len(cell_counts)} distinct cells with counts={cell_counts.tolist()[:8]}..."
            )


def assert_dose_zero(df: pd.DataFrame, *, eps: float = 1e-12) -> None:
    """Spec §0.5 — every dose-zero row must have |pref0_delta| and |target_pref_delta| < eps."""
    if "dose" not in df.columns:
        raise AssertionError("assert_dose_zero: missing 'dose' column")
    zero = df[df["dose"].astype(float) == 0.0]
    for col in ("pref0_delta", "target_pref_delta"):
        if col not in zero.columns:
            continue
        bad = zero[zero[col].abs() > eps]
        if len(bad):
            raise AssertionError(
                f"assert_dose_zero: {len(bad)} rows have dose==0 but |{col}|>{eps}; "
                f"first row: {bad.iloc[0].to_dict()}"
            )


def assert_hook_integrity(df: pd.DataFrame, *, ratio_band: tuple[float, float] = (0.95, 1.05),
                          min_cosine: float = 0.95) -> None:
    """Spec §0.6 — per (model, layer) cell, hook delta norm and cosine must be in band.

    Computed on nonzero-dose rows only. The cosine is against the SIGNED
    requested delta (so a negative dose still has positive cosine).
    """
    if "dose" not in df.columns:
        raise AssertionError("assert_hook_integrity: missing 'dose' column")
    sub = df[df["dose"].astype(float) != 0.0]
    if len(sub) == 0:
        return
    needed = ("actual_delta_norm", "requested_delta_norm",
              "actual_delta_cosine_with_requested_delta", "model", "layer")
    missing = [c for c in needed if c not in sub.columns]
    if missing:
        raise AssertionError(f"assert_hook_integrity: missing columns {missing}")
    for (m, l), grp in sub.groupby(["model", "layer"]):
        req = grp["requested_delta_norm"].astype(float)
        act = grp["actual_delta_norm"].astype(float)
        cos = grp["actual_delta_cosine_with_requested_delta"].astype(float)
        good = req > 0
        if not good.any():
            raise AssertionError(f"assert_hook_integrity: all requested_delta_norm==0 for ({m}, {l})")
        ratio = (act[good] / req[good]).replace([np.inf, -np.inf], np.nan).dropna()
        mean_ratio = float(ratio.mean()) if len(ratio) else float("nan")
        mean_cos = float(cos.dropna().mean()) if cos.notna().any() else float("nan")
        lo, hi = ratio_band
        if not (lo <= mean_ratio <= hi):
            raise AssertionError(
                f"hook integrity ratio out of band for ({m}, l={l}): "
                f"mean(actual/requested)={mean_ratio:.4f} not in [{lo}, {hi}]"
            )
        if not (mean_cos >= min_cosine):
            raise AssertionError(
                f"hook integrity cosine too low for ({m}, l={l}): "
                f"mean cos={mean_cos:.4f} < {min_cosine}"
            )


# ---------------------------------------------------------------------------
# Prompt SHA256 (spec §0.2)
# ---------------------------------------------------------------------------
def sha256_prompt(prompt: str, probe_prefix: str = PROBE_PREFIX) -> str:
    """Hexdigest of (prompt + probe_prefix). Stable across runs."""
    h = hashlib.sha256()
    h.update((str(prompt) + str(probe_prefix)).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Game-universe + payoff vector helpers
# ---------------------------------------------------------------------------
def load_game_vec(game_code: str) -> list[int]:
    """Return the raw Bruns vector for ``game_code``.

    NOTE: this does NOT apply a payoff multiplier. The V2 protocol uses
    ``payoff_multiplier=1`` (spec §1.3 config); the caller is responsible for
    multiplying if a different scale is wanted. Returns a fresh list each call.
    """
    from strategic_anatomy.games import bruns_games  # local import — avoids torch deps in tests
    if game_code not in bruns_games:
        raise KeyError(f"Unknown game_code: {game_code!r}")
    base_vec, *_ = bruns_games[game_code]
    return [int(x) for x in base_vec]


def load_game_universe(path: str | Path) -> pd.DataFrame:
    """Read game_universe.csv. Raise FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Game universe manifest missing at {p}. "
            f"The released manifest ships at data/manifests/game_universe_oneshot.csv."
        )
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Deterministic random unit vector (spec §0.7 — controls)
# ---------------------------------------------------------------------------
def deterministic_random_unit_vector(
    hidden: int,
    *,
    seed_keys: tuple,
) -> np.ndarray:
    """Generate a unit vector deterministically from ``seed_keys``.

    Spec §0.7 — for random controls, sample from
    ``hash(run_id, model, layer, control_name, game_code, counterbalance_id)``.
    Caller passes the tuple of keys; we hash it and seed numpy.
    """
    blob = "|".join(str(x) for x in seed_keys).encode("utf-8")
    seed = int.from_bytes(hashlib.blake2s(blob, digest_size=4).digest(), "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(int(hidden)).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("Random unit vector collapsed to zero — extraordinarily unlikely")
    return (v / n).astype(np.float32)
