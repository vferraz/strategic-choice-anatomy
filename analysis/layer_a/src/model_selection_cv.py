#!/usr/bin/env python3
"""Layer A final — behavioural model selection (Figure 3b).

Leave-games-out (GroupKFold-by-game) predictive log-loss for a nested family of
strategic-choice models, per agent (4 LLMs + Nagel human). Each model predicts P1's
probability of playing the CANONICAL action per game; CV log-loss is scored against the
realized binary choices (model = corrected integrated-root realized action; human =
451 sessions from 450 participants). Bootstrap-by-game CIs (cluster = game).

Predictors are game-structure quantal gaps, canonical-signed and model-independent:
  gc_L1 = gap at q=0.5 (best-respond to uniform/L0)
  gc_L2 = gap when P2 plays its L1 action
  gc_L3 = gap when P2 plays its L2 action
  gc_NE = gap when P2 plays its Nash mixing prob (equilibrium belief)
  plays_canon_k = 1[gc_Lk > 0]  (deterministic level-k canonical action)
  ne_canon_prob = NE probability of the canonical action (avg over pure/mixed NE)

Models (free params):
  L0   uniform 0.5                          (0)   — no-skill baseline
  NE   Nash action prob (clipped)           (0)   — predicts 0/1 -> catastrophic when wrong
  QRE  sigma(a + lam*gc_NE)                  (2)   — quantal response to equilibrium
  Lk   best single deterministic level      (level) — Stahl-Wilson dominant level
  QLk  best level + sigma(a + lam*gc_Lk)     (level,2)
  CH   Poisson(tau) mixture, deterministic   (1)
  QCH  Poisson(tau) mixture + precision lam  (2)

CANONICAL AXIS throughout (docs/METHODS.md HC-2): outcome = played-canonical; gaps canonical-signed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from analysis.layer_a.src import human_lambda_mgn as HL  # noqa: E402
from collection.oneshot_common import delta1, canonical_sign  # noqa: E402
from analysis.probe_common import ne_pairs_per_game  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


TABLES = results_root() / "layer_a"
MODELS_GRID = ["L0", "NE", "QRE", "Lk", "QLk", "CH", "QCH"]
LEVELS = [1, 2, 3]
EPS = 1e-3
KMAX = 3


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---- per-game structural predictors (model-independent) ---------------------
def game_predictors() -> pd.DataFrame:
    m = SD.master()
    ne = ne_pairs_per_game()   # game_code -> ([(p1_act0, p2_act0), ...], kind)
    rows = []
    for _, r in m.iterrows():
        g = r["game_code"]; vec = SD.parse_8vec(r["canonical_8vec"]); canon = int(r["canonical_action_p1"])
        qL1 = 0.5
        qL2 = _q_from_action(r.get("l1_action_p2"))
        qL3 = _q_from_action(r.get("l2_action_p2"))
        pairs, _ = ne.get(g, ([], ""))
        # NE: average opponent (P2) act0 prob and canonical-action prob over the NE set
        if pairs:
            qNE = float(np.mean([p2 for _, p2 in pairs]))
            p1_canon_probs = [(p1 if canon == 0 else 1 - p1) for p1, _ in pairs]
            ne_canon = float(np.mean(p1_canon_probs))
        else:
            qNE, ne_canon = 0.5, 0.5
        rec = {"game_code": g,
               "gc_L1": canonical_sign(delta1(vec, qL1), canon),
               "gc_L2": canonical_sign(delta1(vec, qL2), canon),
               "gc_L3": canonical_sign(delta1(vec, qL3), canon),
               "gc_NE": canonical_sign(delta1(vec, qNE), canon),
               "ne_canon_prob": ne_canon}
        rec["plays_canon_L1"] = 1.0 if rec["gc_L1"] > 0 else 0.0
        rec["plays_canon_L2"] = 1.0 if rec["gc_L2"] > 0 else 0.0
        rec["plays_canon_L3"] = 1.0 if rec["gc_L3"] > 0 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows).set_index("game_code")


def _q_from_action(a):
    if pd.isna(a):
        return 0.5
    a = int(a)
    return 1.0 if a == 0 else (0.0 if a == 1 else 0.5)


# ---- per-agent observations: (game_code, y) ---------------------------------
def agent_observations() -> dict[str, pd.DataFrame]:
    obs = {}
    cells = pd.read_parquet(SD.DATA / "layerA_cells_p1baseline.parquet")
    for mod in SD.MODELS:
        d = cells[cells.model == mod][["game_code", "y_canon"]].rename(columns={"y_canon": "y"})
        obs[mod] = d.reset_index(drop=True)
    hdf = HL.individual_choices()[["game_code", "y_canon"]].rename(columns={"y_canon": "y"})
    obs["nagel_human"] = hdf.reset_index(drop=True)
    return obs


# ---- fit a family on training games, predict per-game P(canonical) ----------
def _fit_family(fam, gtrain: pd.DataFrame, ytrain: np.ndarray, gp: pd.DataFrame):
    """Return a callable game_code-indexed predictor dict {game_code: p_canon}."""
    from scipy.optimize import minimize
    from analysis.layer_a.layer1_lambda_delta1 import fit_lambda

    # training arrays joined to predictors
    tr = gtrain.join(gp, on="game_code")
    feats = {c: tr[c].to_numpy(float) for c in gp.columns}
    y = ytrain

    def predict_all(p_by_feat):
        return {g: p_by_feat[g] for g in gp.index}

    if fam == "L0":
        pmap = {g: 0.5 for g in gp.index}
    elif fam == "NE":
        pmap = {g: float(gp.loc[g, "ne_canon_prob"]) for g in gp.index}
    elif fam == "QRE":
        lam, a = fit_lambda(feats["gc_NE"], y, "hard")
        if not np.isfinite(lam):
            lam, a = 0.0, 0.0
        pmap = {g: _sig(a + lam * gp.loc[g, "gc_NE"]) for g in gp.index}
    elif fam == "Lk":
        best, bestll = 1, -np.inf
        for k in LEVELS:
            p = np.clip(feats[f"plays_canon_L{k}"], EPS, 1 - EPS)
            ll = _ll(y, p)
            if ll > bestll:
                bestll, best = ll, k
        pmap = {g: float(gp.loc[g, f"plays_canon_L{best}"]) for g in gp.index}
    elif fam == "QLk":
        best, bestll, bestp = 1, -np.inf, (0.0, 0.0)
        for k in LEVELS:
            lam, a = fit_lambda(feats[f"gc_L{k}"], y, "hard")
            if not np.isfinite(lam):
                continue
            p = _sig(a + lam * feats[f"gc_L{k}"])
            ll = _ll(y, p)
            if ll > bestll:
                bestll, best, bestp = ll, k, (a, lam)
        a, lam = bestp
        pmap = {g: _sig(a + lam * gp.loc[g, f"gc_L{best}"]) for g in gp.index}
    elif fam in ("CH", "QCH"):
        from scipy.stats import poisson
        G = np.vstack([feats["gc_L1"], feats["gc_L2"], feats["gc_L3"]]).T  # (n,3)
        det = (G > 0).astype(float)

        def mix_pred(tau, lam, Gmat, detmat):
            w = poisson.pmf(np.arange(KMAX + 1), tau).astype(float)
            w = w / w.sum()
            base = w[0] * 0.5
            if lam is None:
                comp = detmat
            else:
                comp = _sig(lam * Gmat)
            return base + (w[1:] * comp).sum(axis=1)

        if fam == "CH":
            def nll(theta):
                tau = max(1e-3, theta[0])
                p = mix_pred(tau, None, G, det)
                return -_ll(y, p)
            res = minimize(nll, [1.0], method="Nelder-Mead")
            tau = max(1e-3, res.x[0]); lam = None
        else:
            def nll(theta):
                tau = max(1e-3, theta[0]); lam = max(0.0, theta[1])
                p = mix_pred(tau, lam, G, det)
                return -_ll(y, p)
            res = minimize(nll, [1.0, 1.0], method="Nelder-Mead")
            tau = max(1e-3, res.x[0]); lam = max(0.0, res.x[1])
        Gall = gp[["gc_L1", "gc_L2", "gc_L3"]].to_numpy(float)
        detall = (Gall > 0).astype(float)
        pall = mix_pred(tau, lam, Gall, detall)
        pmap = {g: float(pall[i]) for i, g in enumerate(gp.index)}
    else:
        raise ValueError(fam)
    return pmap


# ---- CV ---------------------------------------------------------------------
def cv_logloss(agent_obs: pd.DataFrame, gp: pd.DataFrame, n_splits=5, seed=SD.RNG_SEED):
    from sklearn.model_selection import GroupKFold
    games = agent_obs["game_code"].to_numpy()
    y = agent_obs["y"].to_numpy(int)
    # OOF per-observation predicted prob for each family
    oof = {fam: np.full(len(y), np.nan) for fam in MODELS_GRID}
    gk = GroupKFold(n_splits)
    for tr, te in gk.split(agent_obs, y, games):
        gtr = agent_obs.iloc[tr][["game_code"]]
        for fam in MODELS_GRID:
            pmap = _fit_family(fam, gtr, y[tr], gp)
            te_games = agent_obs.iloc[te]["game_code"].to_numpy()
            oof[fam][te] = np.array([pmap[g] for g in te_games])
    # pooled OOF mean log-loss (per observation) + bootstrap-by-game CI
    rng = np.random.default_rng(seed)
    ug = np.unique(games)
    idx_by_g = {g: np.where(games == g)[0] for g in ug}
    out = {}
    for fam in MODELS_GRID:
        p = np.clip(oof[fam], EPS, 1 - EPS)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        mean_ll = float(np.mean(nll))
        boots = []
        for _ in range(SD.BOOT_N):
            pick = rng.choice(ug, size=len(ug), replace=True)
            ii = np.concatenate([idx_by_g[g] for g in pick])
            boots.append(float(np.mean(nll[ii])))
        out[fam] = (mean_ll, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def run():
    gp = game_predictors()
    obs = agent_observations()
    rows = []
    for agent, d in obs.items():
        res = cv_logloss(d, gp)
        for fam in MODELS_GRID:
            ll, lo, hi = res[fam]
            rows.append({"agent": agent, "model": fam, "cv_logloss": ll,
                         "cv_logloss_lo": lo, "cv_logloss_hi": hi,
                         "n_obs": int(len(d)), "n_games": int(d["game_code"].nunique())})
        best = min(MODELS_GRID, key=lambda f: res[f][0])
        print(f"[{agent:16s}] best={best:4s} | " +
              " ".join(f"{f}={res[f][0]:.3f}" for f in MODELS_GRID))
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "f3_model_selection_cv.csv", index=False)
    print(f"wrote {TABLES/'f3_model_selection_cv.csv'}")
    return df


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
