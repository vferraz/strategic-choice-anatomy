#!/usr/bin/env python3
"""Moore-panel duplicate-session robustness (citation audit 2026-07-11, row 21).

The raw export ``datasets/nagel/normalized_data_20241612.csv`` holds 451
complete sessions from 450 unique participants: participant
``665cf485dd9884a03cbb441a`` completed the task twice (ResponseIds
``R_2hKb8TLhaEccf6r``, ``R_2knsYUXDLbh9TmO``; the two sessions differ on
68/144 choices). The shipped analyses keep BOTH sessions and treat the
session (``QT.ResponseId``) as the analysis unit — see
``human_lambda_mgn.py``.

This script quantifies the sensitivity of the two headline human numbers to
that choice by refitting under each single-session exclusion rule:

    lambda (logit slope, q=0.5 belief basis, session-clustered CI)
        451 sessions : 0.9071      <- frozen f3_human_lambda.csv
        keep-first   : 0.9139  (drop R_2knsYUXDLbh9TmO)
        keep-second  : 0.9089  (drop R_2hKb8TLhaEccf6r)

    spearman rho (per-game P(canonical) recomputed from raw sessions vs
    complexity_score, 144 games; committed bootstrap table value -0.4349)
        451 sessions : -0.4356
        keep-first   : -0.4356
        keep-second  : -0.4370

Manuscript statement licensed by this check (Supplementary Methods): dropping
either duplicate session changes lambda by < 0.01 and rho by < 0.002.

Standalone on purpose: no torch-importing project modules, so it runs on any
host with pandas/numpy. Logic mirrors ``human_lambda_mgn.individual_choices``
(orientation via aggregate match against ``nagel_frac_choose_act0_canonical``)
and the master ``nagel_pcanon`` construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from strategic_anatomy.config import human_refs_root, results_root, taxonomy_dir

ROOT = Path(__file__).resolve().parents[3]
NORM = human_refs_root() / "raw" / "nagel" / "normalized_data_20241612.csv"
MASTER = taxonomy_dir() / "human_game_master_per_canonical.csv"
TABLES = results_root() / "layer_a"

DUP_SESSIONS = ("R_2hKb8TLhaEccf6r", "R_2knsYUXDLbh9TmO")  # row order in export


def _delta1(vec: list[float], q: float) -> float:
    u1 = np.asarray(vec[:4], float).reshape(2, 2)
    return float((q * u1[0, 0] + (1 - q) * u1[0, 1]) - (q * u1[1, 0] + (1 - q) * u1[1, 1]))


def _spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _long_frame(nd: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in m.iterrows():
        col = f"P{int(r['nagel_p_id'])}"
        if col not in nd.columns:
            continue
        canon = int(r["canonical_action_p1"])
        vec = [float(x) for x in str(r["canonical_8vec"]).split(",")]
        d1c = _delta1(vec, 0.5) * (1 if canon == 0 else -1)
        ch = nd[["QT.ResponseId", col]].dropna(subset=[col])
        raw_a = (ch[col] == "A").astype(int).to_numpy()
        targ = float(r["nagel_frac_choose_act0_canonical"])
        flip = abs((1 - raw_a).mean() - targ) < abs(raw_a.mean() - targ)
        act0 = (1 - raw_a) if flip else raw_a
        y = act0 if canon == 0 else (1 - act0)
        for sid, yy in zip(ch["QT.ResponseId"].to_numpy(), y):
            rows.append((sid, r["game_code"], int(yy), d1c))
    return pd.DataFrame(rows, columns=["subject", "game_code", "y", "d1c"])


def _logit_clustered(df: pd.DataFrame) -> tuple[float, float, float]:
    """Newton-fit logit slope with session-clustered sandwich SE (no statsmodels)."""
    x = np.column_stack([np.ones(len(df)), df["d1c"].to_numpy(float)])
    y = df["y"].to_numpy(float)
    b = np.zeros(2)
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-x @ b))
        step = np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]), x.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-12:
            break
    p = 1.0 / (1.0 + np.exp(-x @ b))
    hinv = np.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    scores = pd.DataFrame(x * (y - p)[:, None]).groupby(df["subject"].to_numpy()).sum().to_numpy()
    v = hinv @ (scores.T @ scores) @ hinv
    se = float(np.sqrt(v[1, 1]))
    return float(b[1]), float(b[1] - 1.96 * se), float(b[1] + 1.96 * se)


def main() -> int:
    nd = pd.read_csv(NORM)
    m = pd.read_csv(MASTER)
    assert nd["QT.ResponseId"].nunique() == 451 and nd["Name"].nunique() == 450

    out = []
    variants = [
        ("sessions_451", None),
        ("dedup_keep_first", DUP_SESSIONS[1]),
        ("dedup_keep_second", DUP_SESSIONS[0]),
    ]
    for tag, drop in variants:
        sub = nd if drop is None else nd[nd["QT.ResponseId"] != drop]
        df = _long_frame(sub, m)
        lam, lo, hi = _logit_clustered(df)
        pg = df.groupby("game_code").y.mean().reset_index().merge(
            m[["game_code", "complexity_score"]], on="game_code")
        rho = _spearman(pg["y"], pg["complexity_score"])
        out.append({"variant": tag, "n_sessions": df["subject"].nunique(),
                    "n_obs": len(df), "lambda": lam,
                    "lambda_clust_lo": lo, "lambda_clust_hi": hi,
                    "spearman_rho_complexity": rho,
                    "dropped_session": drop or ""})
        print(f"[{tag}] sessions={df['subject'].nunique()} lambda={lam:.4f} "
              f"[{lo:.3f},{hi:.3f}] rho={rho:.4f}")

    TABLES.mkdir(parents=True, exist_ok=True)
    dest = TABLES / "human_dedup_robustness.csv"
    pd.DataFrame(out).to_csv(dest, index=False)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
