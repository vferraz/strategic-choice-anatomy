#!/usr/bin/env python3
"""B1c (decision x incentive state space) and B2b (cross-model geometry <-> lambda summary).

The geometry is the load-bearing DESCRIPTIVE claim: is the decodable decision axis ALIGNED with
the incentive axis (recruitment), or orthogonal to it? A stimulus encoder can decode the game
but cannot wire the decision to the incentive, so a small angle carries the thesis.

Geometry recipe (copied from c6_brain_core.representation_map; EMPIRICAL-belief Delta1c, the same
canonical-signed incentive the behavioural lambda recipe uses):
  - d_dec = diff-of-means(aligned canonical vs not)   [raw residual space]
  - d_inc = centered columnwise OLS slope on continuous Delta1c   [same raw space -> angle valid]
  - angle(d_inc, d_dec); neural gain = OLS slope of d_dec projection on Delta1c (game-clustered CI)
For the DISPLAYED map we use leak-free OOF projections (axes fit on train games, project test),
so the scatter is not in-sample-circular.

B2b is labelled DESCRIPTIVE / non-confirmatory (n<=4). The confirmatory recruitment test is the
within-model n=144 bridge (build_recruitment.py). lambda is the corrected Layer A headline
(``lam_action`` at q=0.5) from analysis/layer_a/tables/f3_lambda.csv.

  .venv/bin/python analysis/layer_b/build_geometry.py
Outputs: tables/b1_state_space.csv (per row), tables/b2_geometry_lambda.csv (per model)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


LAMBDA_FIT = results_root() / "layer_a" / "f3_lambda.csv"


def _lambda_table() -> pd.DataFrame:
    lf = pd.read_csv(LAMBDA_FIT)
    return lf[lf["agent"].isin(lib.MODELS)].set_index("agent")


def build() -> None:
    lamtab = _lambda_table()
    pts_rows, summ_rows = [], []
    for m in lib.MODELS:
        L = lib.deepest_layer(m)
        meta, Xd = lib.load_baseline(m, [L])
        X = np.asarray(Xd[L], dtype=np.float64)
        d1c = lib.empirical_delta1c(m).set_index("game_code")["delta1_c"]
        meta = meta.assign(
            aligned=(meta["decoded_action"] == meta["canonical_action_p1"]).astype(int),
            delta1_c=meta["game_code"].map(d1c).astype(float),
        )
        games = meta["game_code"].to_numpy()
        aligned = meta["aligned"].to_numpy()
        dvec = meta["delta1_c"].to_numpy(float)

        # in-sample axes (for angle + neural gain), raw residual space (c6 convention)
        d_dec = lib.decision_axis_diffmeans(X, aligned)
        d_inc = lib.incentive_axis_cov(X, dvec)
        ang = lib.angle_deg(d_inc, d_dec)
        proj = (X - X.mean(0)) @ d_dec
        gain = lib.neural_gain(proj, dvec, games, n_boot=lib.N_BOOT)

        # leak-free OOF projections for the displayed map
        x_dec = lib.oof_axis_projection(X, aligned, games, kind="diffmeans")
        y_inc = lib.oof_axis_projection(X, dvec, games, kind="cov")
        for i in range(len(meta)):
            pts_rows.append(dict(model=m, game_code=games[i], canonical_action=int(aligned[i]),
                                 x_decision_oof=float(x_dec[i]), y_incentive_oof=float(y_inc[i])))

        lam = float(lamtab.loc[m, "lam_action"]) if m in lamtab.index else np.nan
        summ_rows.append(dict(
            model=m, angle_deg=ang, gain_slope=gain["slope"], gain_lo=gain["ci_lo"],
            gain_hi=gain["ci_hi"], gain_r=gain["r"], n=gain["n"], n_games=gain["n_games"],
            lambda_L1=lam,
            lambda_L1_lo=float(lamtab.loc[m, "lam_action_ci_lo"]) if m in lamtab.index else np.nan,
            lambda_L1_hi=float(lamtab.loc[m, "lam_action_ci_hi"]) if m in lamtab.index else np.nan,
            lambda_EQ=np.nan,
            lambda_source=str(LAMBDA_FIT)))
        print(f"[{lib.SHORT[m]}] angle(d_inc,d_dec)={ang:.1f}deg  gain={gain['slope']:+.3f} "
              f"[{gain['ci_lo']:+.3f},{gain['ci_hi']:+.3f}] (r={gain['r']:+.2f})  lambda_L1={lam:.2f}")

    pts = pd.DataFrame(pts_rows)
    summ = pd.DataFrame(summ_rows)
    pts.to_csv(lib.TAB_DIR / "b1_state_space.csv", index=False)
    summ.to_csv(lib.TAB_DIR / "b2_geometry_lambda.csv", index=False)

    # cross-model descriptive correlations (n<=4)
    from scipy.stats import spearmanr
    a = summ.dropna(subset=["angle_deg", "lambda_L1"])
    rho_ang = spearmanr(a["angle_deg"], a["lambda_L1"]).correlation if len(a) >= 3 else np.nan
    dense = summ[summ["model"].isin(lib.DENSE)].dropna(subset=["gain_slope", "lambda_L1"])
    rho_gain = spearmanr(dense["gain_slope"], dense["lambda_L1"]).correlation if len(dense) >= 3 else np.nan
    print(f"\n[DESCRIPTIVE, n<=4] rho(angle, lambda)={rho_ang:+.2f} (smaller angle<->higher lambda); "
          f"rho(gain, lambda | dense)={rho_gain:+.2f}")
    print(f"wrote {lib.TAB_DIR/'b1_state_space.csv'} and {lib.TAB_DIR/'b2_geometry_lambda.csv'}")


if __name__ == "__main__":
    build()
