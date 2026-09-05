#!/usr/bin/env python3
"""B2a (within-model n=144 recruitment bridge -- the CONFIRMATORY test) and B2c (option dynamics).

B2a is the load-bearing, confound-free, powered claim. Inside each model, across the 144 games:
does the STRENGTH of the represented incentive predict the canonical-play rate?
  - represented incentive p_inc(g) = per-game mean of the leak-free OOF projection of the deepest
    residual onto the incentive axis d_inc (a NEURAL quantity, not the raw stimulus Delta1c), then
    z-scored within model so the slope is comparable across models (units = per-SD).
  - behaviour P(canonical)(g) = per-game mean of aligned (chose canonical).
  - slope of P(canonical) ~ p_inc over the 144 games, bootstrap-by-game CI.
This links REPRESENTATION STRENGTH to BEHAVIOURAL OUTCOME (not decodability of a stimulus). Both
Llama and GPT-OSS still *represent* the incentive (d_inc decodes it), but if it is not recruited
the slope is ~0; Qwen-I/Qwen-B should be steeply positive. n=144 (not n=4).

B2c (the neural quantal parameter): reuse levelk_geometry_oneshot.run_n6 -- per-layer decision-axis
projection proxy, trajectory SPLIT BY the model's actual choice, so the margin reverses (goes
negative / P(canonical)<0.5) when the model picked non-canonical (proves it tracks the decision,
not the normative answer). CPU-only proxy; faithful logit-lens is an optional upgrade only.

  .venv/bin/python analysis/layer_b/build_recruitment.py
Outputs: tables/b2_within_model_bridge.csv, tables/b2_option_dynamics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402
from analysis.layer_b.levelk_geometry_oneshot import attach_manifest, run_n6  # noqa: E402


def build_b2a() -> pd.DataFrame:
    rows = []
    for m in lib.MODELS:
        L = lib.deepest_layer(m)
        meta, Xd = lib.load_baseline(m, [L])
        X = np.asarray(Xd[L], dtype=np.float64)
        d1c = lib.empirical_delta1c(m).set_index("game_code")["delta1_c"]
        aligned = (meta["decoded_action"] == meta["canonical_action_p1"]).astype(int).to_numpy()
        dvec = meta["game_code"].map(d1c).to_numpy(float)
        games = meta["game_code"].to_numpy()
        # leak-free OOF represented-incentive projection (neural quantity)
        y_inc = lib.oof_axis_projection(X, dvec, games, kind="cov")
        df = pd.DataFrame(dict(game_code=games, p_inc=y_inc, aligned=aligned)).dropna()
        per_game = df.groupby("game_code").agg(p_inc=("p_inc", "mean"),
                                               pcanon=("aligned", "mean")).reset_index()
        x = per_game["p_inc"].to_numpy()
        x = (x - x.mean()) / (x.std() + 1e-12)            # z-score within model -> per-SD slope
        yv = per_game["pcanon"].to_numpy()
        gids = per_game["game_code"].to_numpy()
        slope = float(np.polyfit(x, yv, 1)[0]) if x.std() > 0 else np.nan
        lo, hi = lib._slope_boot_by_game(x, yv, gids, n_boot=lib.N_BOOT)
        r = float(np.corrcoef(x, yv)[0, 1]) if x.std() > 0 else np.nan
        rows.append(dict(model=m, slope_per_sd=slope, slope_lo=lo, slope_hi=hi,
                         pearson_r=r, n_games=int(len(per_game))))
        print(f"[{lib.SHORT[m]}] B2a within-model bridge: slope/SD={slope:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"(r={r:+.2f}, n={len(per_game)} games)")
    out = pd.DataFrame(rows)
    out.to_csv(lib.TAB_DIR / "b2_within_model_bridge.csv", index=False)
    return out


def build_b2c() -> pd.DataFrame:
    frames = []
    for m in lib.MODELS:
        layers = sorted(lib.capture_layers(m))
        meta, X = lib.load_baseline(m, layers)
        em = attach_manifest(meta, m)                     # adds aligned/level_bin; rows still aligned to X
        df = run_n6(m, layers, faithful=False, pre=(em, X))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(lib.TAB_DIR / "b2_option_dynamics.csv", index=False)
    print(f"wrote {lib.TAB_DIR/'b2_option_dynamics.csv'}")
    return out


if __name__ == "__main__":
    build_b2a()
    build_b2c()
    print(f"wrote {lib.TAB_DIR/'b2_within_model_bridge.csv'}")
