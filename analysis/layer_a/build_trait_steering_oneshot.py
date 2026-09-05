#!/usr/bin/env python3
"""One-shot trait-steering effect table (SPEC Phase 1b; feeds Figure 2).

Reproduces analysis/layer_a/trait_steering_proper.py's build_per_game / summarize
formulas on the ONE-SHOT substrate. The design_v2 module reads round-based moves
(seed-paired across baseline and every cue); the one-shot substrate has a single
decision per (player, condition, counterbalance), so:
  * the within-rendering pairing key is ``counterbalance_id`` (was ``seed``);
  * ``round`` is set to 1 (the frozen Figure-2 script filters round==1);
  * the unilateral cue is structural: P1 carries the trait prefix, and the ONLY P2
    rows that exist are baseline — so the uncued-P2 null is exact by construction
    (no apparatus check needed; there is no P2 row to drift).

Cues present in one-shot (5): risk/loss/inequity/selfish (dispositional) + maximin
(rule). No level1/level2 cues — exactly the trait set Figure 2 uses.

Trait targets (preferred action a*, signed dose d) are computed on the canonical
8-vector via analysis/oneshot/_oneshot_common.trait_targets() — no design_v2 root.
Corrected-data rule for the NHB figure:
  * all models read ``$SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet``;
  * dense/chat models use ``decoded_action`` gated by ``parse_ok``;
  * GPT-OSS uses ``realized_action`` gated by ``commit_type != 'none'``. Mixed rows
    are already resolved to 0/1 and are included as behaviour.
Outputs (analysis/layer_a/_data/):
  oneshot_trait_steering_per_game.csv   one row per (model, round=1, trait, game)
  oneshot_trait_steering_summary.csv    one row per (model, round=1, trait)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.probe_common import trait_targets as _oneshot_targets  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


DATA_DIR = results_root() / "layer_a" / "_data"
OUT_PER_GAME = DATA_DIR / "oneshot_trait_steering_per_game.csv"
OUT_SUMMARY = DATA_DIR / "oneshot_trait_steering_summary.csv"
MODELS = CS.MODELS

RNG = np.random.default_rng(20260521)
N_BOOT = 4000

TRAITS = ["risk", "loss", "inequity", "selfish", "maximin"]
TRAIT_FAMILY = {"risk": "trait", "loss": "trait", "inequity": "trait",
                "selfish": "trait", "maximin": "rule"}
# short trait name -> one-shot condition string
CUE = {"risk": "cue_risk_aversion", "loss": "cue_loss_aversion",
       "inequity": "cue_inequity_aversion", "selfish": "cue_selfish_maximizer",
       "maximin": "cue_maximin"}
PLACEBO = "cue_length_match_null"
# short trait name -> (_oneshot_common.trait_targets() column for a*, dose)
TGT = {"risk": "risk_aversion", "loss": "loss_aversion", "inequity": "inequity_aversion",
       "selfish": "selfish_maximizer", "maximin": "maximin"}


# --------------------------------------------------------------------------- #
def load_p1(model: str) -> pd.DataFrame:
    """P1 corrected decisions across all available conditions."""
    p1 = CS.p1_decisions(model).rename(columns={"cb": "counterbalance_id"})
    p1["source"] = "corrected_integrated_root"
    return p1[["model", "game_code", "counterbalance_id", "condition",
               "decoded_action", "source"]].copy()


def per_game_freq(p1: pd.DataFrame, cond: str) -> pd.Series:
    sub = p1[p1["condition"] == cond]
    return sub.groupby("game_code")["decoded_action"].apply(lambda s: float((s == 0).mean()))


def per_game_cb_move(p1: pd.DataFrame, cond: str) -> pd.Series:
    sub = p1[p1["condition"] == cond]
    return sub.groupby(["game_code", "counterbalance_id"])["decoded_action"].first()


def build_per_game(targets: pd.DataFrame) -> pd.DataFrame:
    """Per (model, trait, game) placebo-corrected signed shift toward a* — exact
    reproduction of trait_steering_proper.build_per_game on one-shot (round=1)."""
    rows: list[dict] = []
    for model in MODELS:
        p1 = load_p1(model)
        if p1.empty:
            continue
        source_label = "+".join(sorted(p1["source"].dropna().unique()))
        base0 = per_game_freq(p1, "baseline")
        null0 = per_game_freq(p1, PLACEBO)
        base_cb = per_game_cb_move(p1, "baseline")
        null_cb = per_game_cb_move(p1, PLACEBO)  # noqa: F841 (parity with design_v2)
        for trait in TRAITS:
            cue0 = per_game_freq(p1, CUE[trait])
            cue_cb = per_game_cb_move(p1, CUE[trait])
            a = targets.set_index("game_code")[f"a_{TGT[trait]}"]
            d = targets.set_index("game_code")[f"d_{TGT[trait]}"]
            for game in base0.index:
                if game not in cue0.index or game not in null0.index:
                    continue
                astar = a.get(game, np.nan)
                if pd.isna(astar):           # trait does not differentiate this game
                    continue
                astar = int(astar)
                b, c, n = float(base0[game]), float(cue0[game]), float(null0[game])
                shift0, nullshift0 = c - b, n - b
                sgn = 1.0 if astar == 0 else -1.0
                # per-rendering flip stats (paired counterbalance cells)
                flips = aligned = n_pairs = 0
                try:
                    bs, cs = base_cb.loc[game], cue_cb.loc[game]
                    for cb in bs.index.intersection(cs.index):
                        n_pairs += 1
                        if int(bs[cb]) != int(cs[cb]):
                            flips += 1
                            if int(cs[cb]) == astar:
                                aligned += 1
                except KeyError:
                    pass
                rows.append({
                    "model": model, "round": 1, "trait": trait,
                    "family": TRAIT_FAMILY[trait], "game": game,
                    "a_star": astar, "dose": float(d[game]),
                    "base0": b, "cue0": c, "null0": n,
                    "shift0": shift0, "nullshift0": nullshift0,
                    "abs_shift": abs(shift0), "abs_nullshift": abs(nullshift0),
                    "shift_pref": sgn * shift0, "nullshift_pref": sgn * nullshift0,
                    "net_pref": sgn * (shift0 - nullshift0),       # placebo-corrected
                    "base_pref": b if astar == 0 else 1 - b,       # baseline P(a*) -> ceiling
                    "n_pairs": n_pairs, "flips": flips, "flips_aligned": aligned,
                    "source": source_label,
                })
    return pd.DataFrame(rows)


def _ci(vals) -> tuple[float, float, float]:
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    draws = vals[RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _slope_ci(d, y) -> tuple[float, float, float, float]:
    d, y = np.asarray(d, float), np.asarray(y, float)
    m = (~np.isnan(d)) & (~np.isnan(y))
    d, y = d[m], y[m]
    if len(d) < 5 or np.std(d) < 1e-9:
        return np.nan, np.nan, np.nan, np.nan
    slope = float(np.polyfit(d, y, 1)[0])
    r = float(np.corrcoef(d, y)[0, 1])
    boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(d), size=len(d))
        if np.std(d[idx]) < 1e-9:
            continue
        boots.append(np.polyfit(d[idx], y[idx], 1)[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return slope, float(lo), float(hi), r


def summarize(pg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, round_id, trait), grp in pg.groupby(["model", "round", "trait"]):
        mag, mag_lo, mag_hi = _ci(grp.abs_shift.to_numpy())
        magn, _, _ = _ci(grp.abs_nullshift.to_numpy())
        s, s_lo, s_hi = _ci(grp.shift_pref.to_numpy())
        net, net_lo, net_hi = _ci(grp.net_pref.to_numpy())
        slope, sl_lo, sl_hi, r = _slope_ci(grp.dose.to_numpy(), grp.shift0.to_numpy())
        slope_n, _, _, _ = _slope_ci(grp.dose.to_numpy(), grp.nullshift0.to_numpy())
        flips, pairs, al = int(grp.flips.sum()), int(grp.n_pairs.sum()), int(grp.flips_aligned.sum())
        mv = grp[(grp.base_pref >= 0.2) & (grp.base_pref <= 0.8)]
        net_mv, net_mv_lo, net_mv_hi = _ci(mv.net_pref.to_numpy()) if len(mv) else (np.nan, np.nan, np.nan)
        rows.append({
            "model": model, "round": int(round_id), "trait": trait,
            "family": TRAIT_FAMILY[trait], "n_games": int(grp.game.nunique()),
            "magnitude": mag, "magnitude_lo": mag_lo, "magnitude_hi": mag_hi, "magnitude_null": magn,
            "flip_rate": flips / pairs if pairs else np.nan,
            "flip_rate_aligned": al / flips if flips else np.nan,
            "dir_shift": s, "dir_lo": s_lo, "dir_hi": s_hi,
            "dir_net": net, "dir_net_lo": net_lo, "dir_net_hi": net_hi,
            "directionality_index": (s / mag) if mag and not np.isnan(mag) and mag > 1e-9 else np.nan,
            "base_pref_mean": float(grp.base_pref.mean()),
            "dose_slope": slope, "dose_slope_lo": sl_lo, "dose_slope_hi": sl_hi,
            "dose_r": r, "dose_slope_null": slope_n,
            "n_movable": int(mv.game.nunique()),
            "dir_net_movable": net_mv, "dir_net_movable_lo": net_mv_lo, "dir_net_movable_hi": net_mv_hi,
            "source": "+".join(sorted(grp["source"].dropna().unique())) if "source" in grp else "",
        })
    return pd.DataFrame(rows)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets = _oneshot_targets()
    pg = build_per_game(targets)
    summ = summarize(pg)
    pg.to_csv(OUT_PER_GAME, index=False)
    summ.to_csv(OUT_SUMMARY, index=False)
    print(f"wrote {OUT_PER_GAME}  ({len(pg)} rows)")
    print(f"wrote {OUT_SUMMARY}  ({len(summ)} rows)")

    print("\n=== corrected sources ===")
    for model in MODELS:
        d = load_p1(model)
        if d.empty:
            continue
        print(f"  {model:16s} games={d.game_code.nunique():3d} rows={len(d):5d} "
              f"conditions={d.condition.nunique():2d} "
              f"source={','.join(sorted(d.source.unique()))}")

    print("\n=== round-1 trait steering (one-shot): magnitude | net (aim) ===")
    s1 = summ[summ["round"] == 1]
    for model in MODELS:
        sm = s1[s1.model == model]
        if sm.empty:
            continue
        print(f"  {model}")
        for t in TRAITS:
            r = sm[sm.trait == t]
            if r.empty:
                continue
            r = r.iloc[0]
            aim = r.dir_net / r.magnitude if r.magnitude > 1e-9 else 0.0
            sig = (r.dir_net_lo > 0) or (r.dir_net_hi < 0)
            print(f"    {t:9s} mag={r.magnitude:.3f}  net={r.dir_net:+.3f} "
                  f"[{r.dir_net_lo:+.3f},{r.dir_net_hi:+.3f}]  aim={aim:+.2f}  sig={sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
