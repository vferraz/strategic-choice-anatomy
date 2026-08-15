"""Deterministic gate for the small-dose smoke (1 game, qwen). Exit 0 = PASS -> chain may start.

Checks (all hard):
  1. exactly one parquet per mode (h1_dinc, h2_choice), 168 rows each
     (1 game x 4 cb x 2 layers x 3 variants (main/random/main_perp) x 7 doses)
  2. full cell structure: every (mode, layer, variant, cb) has all 7 doses
  3. status == OK everywhere
  4. dose-0 rows: slot_pref_J identical across (layer, variant, mode) within cb AND equal (4dp)
     to the substrate capture baseline pref_J/(pref_J+pref_P); realized_letter == substrate
     move_letter  -> harness injects nothing at dose 0, bit-consistent with the capture run
  5. injection alive: per mode, max |slot_pref_J(dose!=0) - slot_pref_J(dose=0)| > 1e-3 somewhere
  6. parse rate >= 0.90 over all rows
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import pandas as pd
from strategic_anatomy.config import repo_root, substrate_root

ROOT = repo_root()
FAILS = []


def check(cond, msg):
    print(("PASS: " if cond else "FAIL: ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-root", required=True)
    p.add_argument("--model", default="qwen")
    a = p.parse_args()
    root = Path(a.smoke_root)

    for mode in ("h1_dinc", "h2_choice"):
        fps = glob.glob(str(root / f"steer_{a.model}_{mode}_*.parquet"))
        check(len(fps) == 1, f"{mode}: exactly one parquet (found {len(fps)})")
        if not fps:
            continue
        df = pd.read_parquet(fps[0])
        check(len(df) == 168, f"{mode}: 168 rows (found {len(df)})")
        check(set(df["variant"].unique()) == {"main", "random", "main_perp"},
              f"{mode}: variants main/random/main_perp (found {sorted(df['variant'].unique())})")
        cell = df.groupby(["steer_layer", "variant", "counterbalance_id"])["dose"].nunique()
        check(len(cell) == 24 and (cell == 7).all(), f"{mode}: 24 cells x 7 doses (found {len(cell)} cells, doses {sorted(cell.unique())})")
        check((df["status"] == "OK").all(), f"{mode}: all status OK")
        check(float((df["realized_letter"] != "").mean()) >= 0.90,
              f"{mode}: parse rate >= 0.90 (got {(df['realized_letter'] != '').mean():.3f})")

        g = df["game_code"].iloc[0]
        sub = pd.read_parquet(substrate_root() / a.model / g / "results.parquet")
        sub = sub[(sub.player == 1) & (sub.condition == "baseline")].set_index("counterbalance_id")
        z0 = df[df.dose == 0]
        nu = z0.groupby("counterbalance_id")["slot_pref_J"].nunique()
        check((nu == 1).all(), f"{mode}: dose-0 identical across layers/variants (nunique per cb: {nu.to_dict()})")
        ok_pref = ok_let = True
        for cb, grp in z0.groupby("counterbalance_id"):
            sp = grp.slot_pref_J.iloc[0]
            ref = sub.loc[cb, "pref_J"] / (sub.loc[cb, "pref_J"] + sub.loc[cb, "pref_P"])
            ok_pref &= abs(sp - round(ref, 4)) <= 1e-4
            ok_let &= (grp.realized_letter == sub.loc[cb, "move_letter"]).all()
        check(ok_pref, f"{mode}: dose-0 slot pref == substrate capture (4dp)")
        check(ok_let, f"{mode}: dose-0 realized letter == substrate move_letter")

        moved = 0.0
        for (L, v, cb), grp in df.groupby(["steer_layer", "variant", "counterbalance_id"]):
            base = grp[grp.dose == 0].slot_pref_J.iloc[0]
            moved = max(moved, float((grp[grp.dose != 0].slot_pref_J - base).abs().max()))
        check(moved > 1e-3, f"{mode}: injection alive (max |Δpref| at dose!=0 = {moved:.4f})")

    print(("SMOKE_VALIDATION_PASS" if not FAILS else f"SMOKE_VALIDATION_FAIL ({len(FAILS)} checks)"), flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
