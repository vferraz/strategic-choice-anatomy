#!/usr/bin/env python3
"""GATE for the Layer B Final build. Run before (and after) the heavy build.

Asserts the data-root identity, substrate completeness, the canonical-axis hard constraint,
and the verified probe-target distributions. Prints the DATA_ROOT_REGISTRY provenance block
(docs/METHODS.md HC-10). Exits non-zero on any hard failure; per-model empirical incentive-sign balance
is reported (informational, not a gate).

  .venv/bin/python analysis/layer_b/validate_foundation.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import json

import pandas as pd


from analysis.layer_b import lib  # noqa: E402

EXPECT = {
    "canonical_action_p1": {0: 78, 1: 66},
    "dominant_action_p1": {1.0: 36, 0.0: 36},           # + 72 NaN
    "num_pure_ne": {1: 108, 2: 18, 0: 18},
    "dominance_profile": {1: 72, 2: 36, 0: 36},
    "stim_cell00": {1: 48, 2: 48, 3: 48},
    "stim_cell01_ismax": {1: 72, 0: 72},
}


def main() -> int:
    fails: list[str] = []

    print("=" * 72)
    print("DATA_ROOT_REGISTRY")
    print(f"  DATA_ROOT_REGISTRY_VERSION         = {lib.DATA_ROOT_REGISTRY_VERSION}")
    print(f"  PRIMARY_ROOT_USED                  = {lib.SUBSTRATE}")
    print(f"  COMPARATOR_ROOT_USED               = NONE")
    print(f"  COMPARATOR_ROOT_WAS_USER_SUPPLIED  = False")
    print(f"  AUTODETECT_USED                    = False")
    print("=" * 72)

    # --- substrate completeness and Akata 4-cell identity ---
    for m in lib.MODELS:
        base = lib.SUBSTRATE / m
        if not base.exists():
            fails.append(f"missing substrate model dir: {base}")
            continue
        gdirs = lib.game_dirs(m)
        done = [d for d in gdirs if (d / "_DONE").exists()]
        if len(gdirs) != 144:
            fails.append(f"{m}: {len(gdirs)} game dirs (expected 144)")
        if len(done) != len(gdirs):
            fails.append(f"{m}: {len(gdirs) - len(done)} games missing _DONE")
        bad_cfg = []
        for d in gdirs:
            cfg = json.loads((d / "config.json").read_text())
            if int(cfg.get("payoff_multiplier", -1)) != 1:
                bad_cfg.append(f"{d.name}: payoff_multiplier={cfg.get('payoff_multiplier')}")
            if cfg.get("cb") != "akata_4cell":
                bad_cfg.append(f"{d.name}: cb={cfg.get('cb')}")
            if int(cfg.get("n_rows", -1)) != 32:
                bad_cfg.append(f"{d.name}: n_rows={cfg.get('n_rows')}")
        if bad_cfg:
            fails.append(f"{m}: bad config identity examples: {bad_cfg[:5]}")
        layers = lib.capture_layers(m)
        deep = lib.deepest_layer(m)
        expected_deep = 36 if m == "gptoss" else 80
        if deep != expected_deep:
            fails.append(f"{m}: deepest layer {deep} != expected {expected_deep}")
        print(f"  {m:18s} games={len(gdirs)}  _DONE={len(done)}  layers={len(layers)}  deepest={deep}")

        dec = lib.decision_rows(m)
        p1b = dec[(dec["player"] == 1) & (dec["condition"] == "baseline")]
        ok_p1b = p1b[p1b["ok"].fillna(False)]
        if len(ok_p1b) != 576:
            fails.append(f"{m}: usable P1 baseline rows {len(ok_p1b)} != 576")
        cb_counts = ok_p1b.groupby("game_code")["cb"].nunique()
        if not cb_counts.eq(4).all():
            fails.append(f"{m}: non-4-cell baseline games: {cb_counts[~cb_counts.eq(4)].to_dict()}")
        if m == "gptoss":
            raw_ok = ok_p1b.dropna(subset=["raw_decoded_action", "raw_realized_action"])
            if not (raw_ok["raw_decoded_action"].astype(int) == raw_ok["raw_realized_action"].astype(int)).all():
                fails.append("gptoss: raw decoded_action != realized_action on usable baseline rows")
            if not ok_p1b["action_source"].astype(str).str.contains("realized_action").all():
                fails.append("gptoss: action_source does not identify realized_action")
            if int(dec["commit_type"].eq("mixed").sum()) == 0:
                fails.append("gptoss: expected mixed rows retained in provenance")
            none_ok = dec["commit_type"].eq("none") & dec["ok"].fillna(False)
            if none_ok.any():
                fails.append(f"gptoss: {int(none_ok.sum())} commit_type none rows marked ok")
        else:
            if not ok_p1b["action_source"].astype(str).str.contains("decoded_action").all():
                fails.append(f"{m}: action_source does not identify dense decoded_action")
        print(f"    corrected decisions: P1 baseline ok={len(ok_p1b)}  cb/game={sorted(cb_counts.unique().tolist())}")

    # --- canonical axis + target distributions ---
    tt = lib.target_table()
    if len(tt) != 144:
        fails.append(f"target_table has {len(tt)} rows (expected 144)")
    bad = tt["canonical_action_p1"].isna() | ~tt["canonical_action_p1"].isin([0, 1])
    if bad.any():
        fails.append(f"canonical_action_p1 not in {{0,1}} for {int(bad.sum())} games")

    print("-" * 72)
    print("probe-target distributions (verified):")
    for col, exp in EXPECT.items():
        got = tt[col].value_counts(dropna=True).to_dict()
        got = {int(k) if float(k).is_integer() else k: int(v) for k, v in got.items()}
        ok = got == exp
        print(f"  {col:20s} {got}  {'OK' if ok else 'MISMATCH expected ' + str(exp)}")
        if not ok:
            fails.append(f"distribution mismatch {col}: {got} != {exp}")
    if tt["nagel_lk_type"].notna().sum() != 144:
        fails.append("nagel_lk_type not populated for all 144 games")

    # --- per-model EMPIRICAL incentive-sign balance (informational) ---
    print("-" * 72)
    print("per-model empirical canonical-signed incentive balance (informational):")
    for m in lib.MODELS:
        try:
            dt = lib.empirical_delta1c(m)
            s1 = (dt["delta1_c"] > 0).mean()
            s2 = (dt["delta2_c"] > 0).mean() if "delta2_c" in dt else float("nan")
            print(f"  {m:18s} P(sign delta1c>0)={s1:.2f}  P(sign delta2c>0)={s2:.2f}  n={len(dt)}")
        except Exception as exc:
            print(f"  {m:18s} delta tables unavailable: {type(exc).__name__}: {exc}")

    print("=" * 72)
    if fails:
        print("GATE FAILED:")
        for f in fails:
            print("  - " + f)
        return 1
    print("GATE PASSED — substrate, canonical axis, and target distributions verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
