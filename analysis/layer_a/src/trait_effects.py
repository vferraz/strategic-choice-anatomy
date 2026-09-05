#!/usr/bin/env python3
"""Layer A final — one-shot trait-steering effect tables (feeds Figure 2).

Builds drop-in replacements for the design_v2 ``trait_steering_proper_*`` CSVs that
the FROZEN ``fig_main_trait_steering_v2.py`` body reads, computed on the corrected
integrated one-shot substrate over realized actions (dense=parse-ok decoded action;
GPT-OSS=resolved realized action, mixed included, no-commit dropped):

Per (model, trait, game) the cue effect on the canonical-axis trait target action
``a_trait`` (from the locked ``trait_targets``):
  abs_shift = |p_act0(cue) - p_act0(baseline)|                 (magnitude)
  net_pref  = sign(a) * (p_act0(cue) - p_act0(baseline))       (toward the target;
              sign=+1 if a==0 else -1; NaN target -> NaN net)

Summary per (model, trait): magnitude (mean |shift|), dir_net (mean net + boot-by-game
CI), base_pref_mean (baseline pref for the target action -> ceiling flag), and the
trait-independent placebo magnitude (length_match_null) as magnitude_null.

Outputs (analysis/layer_a/_data/):
  trait_oneshot_per_game.csv, trait_oneshot_summary.csv, trait_unified_8vec.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from analysis.probe_common import trait_targets  # noqa: E402

DATA = SD.DATA
RNG = np.random.default_rng(SD.RNG_SEED)
N_BOOT = SD.BOOT_N

# cue (oneshot condition, no cue_ prefix)  ->  Fig-2 trait short name
CUE2TRAIT = {"risk_aversion": "risk", "loss_aversion": "loss",
             "inequity_aversion": "inequity", "selfish_maximizer": "selfish",
             "maximin": "maximin"}
PLACEBO = "length_match_null"


def _ci(vals):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    draws = vals[RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def build():
    gl = pd.read_parquet(DATA / "layerA_game_level.parquet")
    tt = trait_targets()  # game_code, a_<cue>/d_<cue> on canonical 8-vec
    m = SD.master()[["game_code", "canonical_8vec"]]

    per_rows = []
    for mod in SD.MODELS:
        g = gl[gl.model == mod].merge(tt, on="game_code", how="left")
        for cue, trait in CUE2TRAIT.items():
            base = g["p1_p_act0"].to_numpy(float)
            cuep = g[f"cue_{cue}_p_act0"].to_numpy(float)
            a = g[f"a_{cue}"].to_numpy(float)          # target action (0/1/NaN)
            shift = cuep - base
            sign = np.where(a == 0, 1.0, np.where(a == 1, -1.0, np.nan))
            net = sign * shift
            for gc, ab, nv, av in zip(g["game_code"], np.abs(shift), net, a):
                per_rows.append({"model": mod, "trait": trait, "game": gc, "round": 1,
                                 "abs_shift": float(ab),
                                 "net_pref": float(nv) if np.isfinite(nv) else np.nan,
                                 "target_action": (int(av) if np.isfinite(av) else -1)})
    per = pd.DataFrame(per_rows)

    # placebo magnitude per model (trait-independent)
    plac = {}
    for mod in SD.MODELS:
        g = gl[gl.model == mod]
        plac[mod] = float(np.nanmean(np.abs(g[f"cue_{PLACEBO}_p_act0"] - g["p1_p_act0"])))

    # summary per (model, trait)
    sum_rows = []
    for mod in SD.MODELS:
        g = gl[gl.model == mod].merge(tt, on="game_code", how="left")
        for cue, trait in CUE2TRAIT.items():
            d = per[(per.model == mod) & (per.trait == trait)]
            # magnitude on the SAME games as net (defined target) so aim=net/mag in [-1,1]
            d_def = d[d.net_pref.notna()]
            mag, mlo, mhi = _ci(d_def.abs_shift.to_numpy())
            net, nlo, nhi = _ci(d_def.net_pref.to_numpy())
            a = g[f"a_{cue}"].to_numpy(float)
            base = g["p1_p_act0"].to_numpy(float)
            # baseline preference for the TARGET action
            base_pref = np.where(a == 0, base, np.where(a == 1, 1 - base, np.nan))
            sum_rows.append({
                "model": mod, "trait": trait, "round": 1,
                "magnitude": mag, "dir_net": net, "dir_net_lo": nlo, "dir_net_hi": nhi,
                "magnitude_lo": mlo, "magnitude_hi": mhi,
                "base_pref_mean": float(np.nanmean(base_pref)),
                "magnitude_null": plac[mod],
                "n_games": int(d["abs_shift"].notna().sum()),
            })
    summ = pd.DataFrame(sum_rows)

    # unified-like 8vec parquet so the frozen incentive_gap() works verbatim (JSON matrix_8vec)
    uni = m.copy()
    uni["matrix_8vec"] = uni["canonical_8vec"].apply(lambda s: json.dumps(SD.parse_8vec(s)))
    uni["agent_kind"] = "llm"; uni["condition"] = "baseline"
    uni = uni[["game_code", "matrix_8vec", "agent_kind", "condition"]]

    DATA.mkdir(parents=True, exist_ok=True)
    per.to_csv(DATA / "trait_oneshot_per_game.csv", index=False)
    summ.to_csv(DATA / "trait_oneshot_summary.csv", index=False)
    uni.to_parquet(DATA / "trait_unified_8vec.parquet", index=False)
    return per, summ


def main() -> int:
    per, summ = build()
    print(f"[trait_effects] per_game {len(per)} rows; summary {len(summ)} rows")
    print("\n[trait_effects] summary (round 1): magnitude | aim=dir_net/mag | sig | base_pref")
    for mod in SD.MODELS:
        for trait in ["risk", "loss", "inequity", "selfish", "maximin"]:
            r = summ[(summ.model == mod) & (summ.trait == trait)]
            if r.empty:
                continue
            r = r.iloc[0]
            aim = r.dir_net / r.magnitude if r.magnitude > 1e-9 else 0.0
            sig = (r.dir_net_lo > 0) or (r.dir_net_hi < 0)
            print(f"  {SD.SHORT[mod]:8s} {trait:9s} mag={r.magnitude:.3f} "
                  f"aim={aim:+.2f} sig={int(sig)} base_pref={r.base_pref_mean:.2f} "
                  f"placebo_mag={r.magnitude_null:.3f}")
    print(f"\nwrote {DATA/'trait_oneshot_per_game.csv'} ; {DATA/'trait_oneshot_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
