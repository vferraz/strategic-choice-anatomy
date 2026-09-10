"""Out-of-fold supervised decision-state helpers.

Extracted verbatim from ``analysis/block_b/fig_decision_state_supervised.py`` (private repo,
commit 8d8370e): ``RNG_SEED`` (L54), ``oof_dir`` (L62), ``auc`` (L88) and ``_ellipse`` (L97).
The originating module is an A/B-era figure script excluded by plan §5; these four symbols are
the only ones ``analysis/probe_common.py`` imports from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


RNG_SEED = 0


def oof_dir(Z: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
            n_folds: int = 5, seed: int = RNG_SEED) -> np.ndarray:
    """Out-of-fold projection of Z onto the diff-of-means direction for binary y.

    Folds are GROUPED by ``groups`` (game) when provided, so all rows of a group are
    held out together. Required here: canonical_action_p1 / sign_delta1c are game-level
    labels (identical across a game's seeds); random folds leak near-duplicate seeds
    train<->test and inflate the separation (~+0.04-0.08 AUC).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    proj = np.zeros(n)
    if groups is None:
        splits = np.array_split(rng.permutation(n), n_folds)
    else:
        gid = pd.factorize(groups)[0]
        ug = rng.permutation(np.unique(gid))
        splits = [np.where(np.isin(gid, gg))[0] for gg in np.array_split(ug, n_folds)]
    for te in splits:
        tr = np.setdiff1d(np.arange(n), te)
        d = Z[tr][y[tr] == 1].mean(0) - Z[tr][y[tr] == 0].mean(0)
        d /= (np.linalg.norm(d) + 1e-9)
        proj[te] = Z[te] @ d
    return proj


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank (Mann-Whitney) AUC, no sklearn dependency."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ellipse(ax, x, y, col, n_sigma=1.5):
    mx, my = x.mean(), y.mean()
    cov = np.cov(x, y)
    ev, evec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(*evec[:, 1][::-1]))
    w, h = 2 * np.sqrt(ev) * n_sigma
    ax.add_patch(Ellipse((mx, my), w, h, angle=ang, fc=col, alpha=0.13, ec=col, lw=1.2))
