#!/usr/bin/env python3
"""Moore--Germano--Nagel behavioural rule port for 2x2 games.

The functions mirror `datasets/nagel/tuple_comparison/*.m`. Rules return a 2x2
profile mask; downstream scoring converts that mask to the row player's action
set and applies the MGN `unique_only` hit convention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

ALPHA = 0.95


@dataclass(frozen=True)
class RuleSpec:
    name: str
    family: str


PANEL_RULES: tuple[str, ...] = (
    "Nash Equilibrium", "RDNE", "PDNE",
    "Level-1", "Level-2", "Level-3",
    "Level-1 Alpha", "Level-2 Alpha", "Level-5 Alpha",
    "Near-Equal Split", "cNES", "rNES", "Soc-Max", "Pareto Efficiency",
)

VALIDATION_RULES: tuple[str, ...] = (
    "Nash Equilibrium", "Near-Equal Split", "Soc-Max", "Pareto Efficiency",
    "Level-1", "Level-1 Alpha", "Level-2", "Level-2 Alpha", "Level-3",
    "Level-3 Alpha", "Level-4", "Level-4 Alpha", "Level-5", "Level-5 Alpha",
    "Rationalizeability", "Equal Split", "Level-1 Focal Alpha",
    "Level-2 Focal Alpha", "Level-3 Focal Alpha", "Level-4 Focal Alpha",
    "Level-5 Focal Alpha", "PDNE", "cNES", "rNES", "RDNE", "RDNE Alpha",
)

RULE_FAMILY = {
    "Nash Equilibrium": "Equilibrium", "RDNE": "Equilibrium", "PDNE": "Equilibrium",
    "Level-1": "Level-k", "Level-2": "Level-k", "Level-3": "Level-k",
    "Level-1 Alpha": "Level-k(alpha)", "Level-2 Alpha": "Level-k(alpha)",
    "Level-5 Alpha": "Level-k(alpha)",
    "Near-Equal Split": "Equity/heuristic", "cNES": "Equity/heuristic",
    "rNES": "Equity/heuristic", "Soc-Max": "Equity/heuristic",
    "Pareto Efficiency": "Equity/heuristic",
}


def reshape8(v, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return row and column payoff matrices from an MGN/Bruns 8-vector."""
    x = np.asarray(v, dtype=float) ** float(alpha)
    return x[:4].reshape(2, 2), x[4:].reshape(2, 2)


def NE(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(R, dtype=int)
    for i in range(2):
        for j in range(2):
            if R[i, j] == R[:, j].max() and C[i, j] == C[i, :].max():
                P[i, j] = 1
    return P


def LK(R: np.ndarray, C: np.ndarray, K: int) -> np.ndarray:
    P = np.ones_like(R, dtype=float)
    for _ in range(int(K)):
        rr = P.max(axis=1)
        rc = P.max(axis=0)
        u_row = R @ rc
        u_col = C.T @ rr
        rows = np.flatnonzero(u_row == u_row.max())
        cols = np.flatnonzero(u_col == u_col.max())
        P = np.zeros_like(R, dtype=int)
        P[np.ix_(rows, cols)] = 1
    return P.astype(int)


def _pareto_profiles(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(R, dtype=int)
    pay = np.column_stack([R.ravel(), C.ravel()])
    for p in range(pay.shape[0]):
        keep = True
        for q in range(pay.shape[0]):
            if p == q:
                continue
            # MATLAB condition: keep p unless q weakly dominates p and improves nobody for p.
            if not ((pay[p] > pay[q]).sum() > 0 or np.prod(pay[p] >= pay[q]) == 1):
                keep = False
                break
        P.ravel()[p] = int(keep)
    return P


def PO(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    return _pareto_profiles(R, C)


def NES(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(R, dtype=int)
    po_idx = np.flatnonzero(_pareto_profiles(R, C).ravel() == 1)
    diff = np.abs(R.ravel()[po_idx] - C.ravel()[po_idx])
    P.ravel()[po_idx[diff == diff.min()]] = 1
    return P


def cNES(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(R, dtype=int)
    po_idx = np.flatnonzero(_pareto_profiles(R, C).ravel() == 1)
    diff = np.abs(R.ravel()[po_idx] - C.ravel()[po_idx])
    near_idx = po_idx[diff == diff.min()]
    col_pay = C.ravel()[near_idx]
    P.ravel()[near_idx[col_pay == col_pay.max()]] = 1
    return P


def rNES(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(R, dtype=int)
    po_idx = np.flatnonzero(_pareto_profiles(R, C).ravel() == 1)
    diff = np.abs(R.ravel()[po_idx] - C.ravel()[po_idx])
    near_idx = po_idx[diff == diff.min()]
    row_pay = R.ravel()[near_idx]
    P.ravel()[near_idx[row_pay == row_pay.max()]] = 1
    return P


def ES(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    P = np.zeros_like(R, dtype=int)
    po_idx = np.flatnonzero(_pareto_profiles(R, C).ravel() == 1)
    diff = np.abs(R.ravel()[po_idx] - C.ravel()[po_idx])
    P.ravel()[po_idx[diff == 0]] = 1
    return P


def maxsum(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    S = R + C
    return (S == S.max()).astype(int)


def MaxMax(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    return (R == R.max()).astype(int)


def PDNE(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    ne = NE(R, C)
    P = np.zeros_like(R, dtype=int)
    idx = np.flatnonzero(ne.ravel() == 1)
    if idx.size == 0:
        return P
    pay = np.column_stack([R.ravel()[idx], C.ravel()[idx]])
    for k, profile_idx in enumerate(idx):
        if np.sum(np.prod(pay > pay[k], axis=1)) == 0:
            P.ravel()[profile_idx] = 1
    return P


def RDNE(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    ne = NE(R, C)
    P = np.zeros_like(R, dtype=int)
    rows, cols = np.where(ne == 1)
    if len(rows) == 0:
        return P
    loss = []
    for i, j in zip(rows, cols):
        loss_r = R[i, j] - R[1 - i, j]
        loss_c = C[i, j] - C[i, 1 - j]
        loss.append(loss_r * loss_c)
    loss = np.asarray(loss)
    for i, j in zip(rows[loss == loss.max()], cols[loss == loss.max()]):
        P[i, j] = 1
    return P


def rationalizeability(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    rows = np.arange(R.shape[0])
    cols = np.arange(R.shape[1])
    last_R = last_C = None
    red_R = R.copy()
    red_C = C.copy()
    while not (last_R is not None and np.array_equal(last_R, red_R) and np.array_equal(last_C, red_C)):
        last_R = red_R.copy()
        last_C = red_C.copy()
        dom_r = np.zeros(last_R.shape[0], dtype=bool)
        dom_c = np.zeros(last_R.shape[1], dtype=bool)
        for i in range(last_R.shape[0]):
            for j in range(last_R.shape[0]):
                if np.prod(last_R[i, :] < last_R[j, :]) == 1:
                    dom_r[i] = True
                    break
        for i in range(last_R.shape[1]):
            for j in range(last_R.shape[1]):
                if np.prod(last_C[:, i] < last_C[:, j]) == 1:
                    dom_c[i] = True
                    break
        keep_r = np.flatnonzero(~dom_r)
        keep_c = np.flatnonzero(~dom_c)
        red_R = last_R[np.ix_(keep_r, keep_c)]
        red_C = last_C[np.ix_(keep_r, keep_c)]
        rows = rows[keep_r]
        cols = cols[keep_c]
    P = np.zeros_like(R, dtype=int)
    P[np.ix_(rows, cols)] = 1
    return P


def LK_with_focmax(R: np.ndarray, C: np.ndarray, K: int) -> np.ndarray:
    P = LK(R, C, K)
    r_max_idx = np.flatnonzero(R.ravel() == R.max())
    c_max_idx = np.flatnonzero(C.ravel() == C.max())
    if r_max_idx.size == 1 and c_max_idx.size == 1 and r_max_idx[0] == c_max_idx[0]:
        P = np.zeros_like(R, dtype=int)
        P.ravel()[r_max_idx[0]] = 1
    return P


def LK_RA_focal(R: np.ndarray, C: np.ndarray, K: int) -> np.ndarray:
    P = LK(R, C, K)
    rows = np.flatnonzero(P.max(axis=1) == 1)
    cols = np.flatnonzero(P.max(axis=0) == 1)
    if len(rows) > 1 or len(cols) > 1:
        P = P * LK(R ** ALPHA, C ** ALPHA, 1)
    r_max_idx = np.flatnonzero(R.ravel() == R.max())
    c_max_idx = np.flatnonzero(C.ravel() == C.max())
    if r_max_idx.size == 1 and c_max_idx.size == 1 and r_max_idx[0] == c_max_idx[0]:
        P = np.zeros_like(R, dtype=int)
        P.ravel()[r_max_idx[0]] = 1
    return P


def row_actions(P: np.ndarray) -> set[int]:
    r = P.max(axis=1)
    return {int(i) for i in (0, 1) if r[i] == 1}


def rule_hit(rec_set: set[int], chosen) -> float:
    if chosen is None:
        return 0.0
    return 1.0 if (len(rec_set) == 1 and int(chosen) in rec_set) else 0.0


def _level_k(rule: str) -> int | None:
    m = re.match(r"Level-(\d)", rule)
    return int(m.group(1)) if m else None


def profile_for_rule(rule: str, vec8) -> np.ndarray:
    alpha = ALPHA if "Alpha" in rule else 1.0
    base = rule.replace(" Alpha", "")
    R, C = reshape8(vec8, alpha=alpha)
    k = _level_k(base)
    if rule == "Nash Equilibrium":
        return NE(R, C)
    if base == "Rationalizeability":
        return rationalizeability(R, C)
    if base == "cNES":
        return cNES(R, C)
    if base == "rNES":
        return rNES(R, C)
    if base == "Near-Equal Split":
        return NES(R, C)
    if base == "Equal Split":
        return ES(R, C)
    if base == "Soc-Max":
        return maxsum(R, C)
    if base == "Max-Max":
        return MaxMax(R, C)
    if base == "Pareto Efficiency":
        return PO(R, C)
    if base == "PDNE":
        return PDNE(R, C)
    if base == "RDNE":
        return RDNE(R, C)
    if k is not None and "Focal" not in base:
        return LK(R, C, k)
    if k is not None and "RA Focal" in base:
        return LK_RA_focal(R, C, k)
    if k is not None and "Focal" in base:
        return LK_with_focmax(R, C, k)
    raise ValueError(f"unknown MGN rule: {rule}")


def row_action_set_for_rule(rule: str, vec8) -> set[int]:
    return row_actions(profile_for_rule(rule, vec8))


def unique_action_for_rule(rule: str, vec8) -> int | None:
    rec = row_action_set_for_rule(rule, vec8)
    return next(iter(rec)) if len(rec) == 1 else None
