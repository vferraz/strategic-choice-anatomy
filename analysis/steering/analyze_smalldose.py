"""Small-dose steering readout — letter/strategy decomposition (the mandated non-slope analysis).

The 4-cell counterbalance separates the two causal components of an injected direction exactly:
per (game, layer, variant, dose),
  letter effect    = cb-mean of ΔprefJ      (canonical content cancels: canon maps to J in 2 cbs, P in 2)
  strategic effect = cb-mean of Δpref_canon (letter content cancels the same way)
where Δ is vs the same cell's dose-0 row. A direction that merely overlaps the J−P unembedding
axis shows letter≠0, strategic≈0. Causal recruitment of the incentive/choice representation shows
strategic≠0 (main ≫ random), ideally moderated by the game's |delta1_c|.

Reports per (model, mode, layer): dose-response slope of BOTH components (game-clustered bootstrap
CI), per-game strategic ranges + behavioral flips (no slope-only readout), and delta1_c moderation.
CPU-only. Usage: analyze_smalldose.py --root $SCA_DATA_ROOT/steering/smalldose [--model qwen]
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from strategic_anatomy.config import results_root, steering_root

ROOT = Path(__file__).resolve().parents[2]
from collection.akata_common import akata_cb_grid  # noqa: E402

GRID = {c["counterbalance_id"]: c for c in akata_cb_grid()}
RNG = np.random.default_rng(0)


def load(root: str, model: str) -> pd.DataFrame:
    fps = sorted(glob.glob(f"{root}/steer_{model}_*.parquet"))
    if not fps:
        raise SystemExit(f"no parquets for {model} under {root}")
    df = pd.concat([pd.read_parquet(f) for f in fps], ignore_index=True)
    df["j_is_act0"] = df.counterbalance_id.map(lambda c: GRID[c]["letter_to_action"]["J"] == 0)
    df["pref_act0"] = np.where(df.j_is_act0, df.slot_pref_J, 1 - df.slot_pref_J)
    df["pref_canon"] = np.where(df.canonical_action_p1 == 0, df.pref_act0, 1 - df.pref_act0)
    df["beh_canon"] = df.realized_canonical.astype(float)
    return df


def per_game_components(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (mode, steer_layer, variant, game, dose): cb-mean Δ on letter/canon axes."""
    rows = []
    for (mode, L, v, g), grp in df.groupby(["mode", "steer_layer", "variant", "game_code"]):
        b = grp[grp.dose == 0].set_index("counterbalance_id")
        for dose, gd in grp.groupby("dose"):
            gd = gd.set_index("counterbalance_id")
            cbs = [c for c in gd.index if c in b.index]
            dJ = np.mean([gd.loc[c, "slot_pref_J"] - b.loc[c, "slot_pref_J"] for c in cbs])
            dC = np.mean([gd.loc[c, "pref_canon"] - b.loc[c, "pref_canon"] for c in cbs])
            bC = np.nanmean([gd.loc[c, "beh_canon"] - b.loc[c, "beh_canon"] for c in cbs])
            rows.append({"mode": mode, "steer_layer": L, "variant": v, "game_code": g,
                         "dose": dose, "d_letter": dJ, "d_canon": dC, "d_beh_canon": bC})
    return pd.DataFrame(rows)


def game_slopes(pg: pd.DataFrame, ycol: str) -> pd.DataFrame:
    rows = []
    for (mode, L, v, g), grp in pg.groupby(["mode", "steer_layer", "variant", "game_code"]):
        grp = grp.dropna(subset=[ycol])
        if grp.dose.nunique() < 3:
            continue
        rows.append({"mode": mode, "steer_layer": L, "variant": v, "game_code": g,
                     "slope": np.polyfit(grp.dose, grp[ycol], 1)[0],
                     "range": grp[ycol].max() - grp[ycol].min()})
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(steering_root() / "smalldose"))
    p.add_argument("--model", default="qwen")
    a = p.parse_args()

    df = load(a.root, a.model)
    deltas = pd.read_csv(results_root() / "steering" / "summary" / f"delta_tables_{a.model}.csv")
    d1c = dict(zip(deltas.game_code, deltas.delta1_c))
    pg = per_game_components(df)
    n_games = df.game_code.nunique()
    print(f"=== {a.model}: {len(df)} rows, {n_games} games, modes {sorted(df['mode'].unique())} ===\n")

    for comp, ycol in (("LETTER (semantically empty; expected ≠0 from unembed overlap)", "d_letter"),
                       ("STRATEGIC canonical (THE recruitment test)", "d_canon"),
                       ("STRATEGIC canonical — realized BEHAVIOR", "d_beh_canon")):
        print(f"--- {comp} ---")
        gs = game_slopes(pg, ycol)
        for (mode, L), sub in gs.groupby(["mode", "steer_layer"]):
            line = []
            for v in ("main", "main_perp", "random"):
                s = sub[sub.variant == v]
                if not len(s):
                    continue
                lo, hi = boot_ci(s.slope.to_numpy())
                line.append(f"{v}: slope {s.slope.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
                            f"range {s['range'].mean():.3f}")
            print(f"  {mode} L{L}:  " + "  |  ".join(line))
        print()

    print("--- delta1_c moderation of the strategic component ---")
    for variant in ("main", "main_perp"):
        gs = game_slopes(pg[pg.variant == variant], "d_canon")
        gs["d1c"] = gs.game_code.map(d1c)
        for (mode, L), sub in gs.groupby(["mode", "steer_layer"]):
            sub = sub.dropna(subset=["d1c"])
            if len(sub) > 3 and sub.slope.std() > 0 and sub.d1c.std() > 0:
                r_signed = np.corrcoef(sub.slope, sub.d1c)[0, 1]
                r_abs = np.corrcoef(sub.slope.abs(), sub.d1c.abs())[0, 1]
                print(f"  {variant:>9} {mode} L{L}: corr(canon_slope, δ1c) = {r_signed:+.3f};  corr(|slope|, |δ1c|) = {r_abs:+.3f}  (n={len(sub)})")

    print("\n--- behavioral flips across the small-dose sweep (per cell, main vs random) ---")
    for (mode, L, v), grp in df.groupby(["mode", "steer_layer", "variant"]):
        flips = (grp[grp.realized_letter != ""].groupby(["game_code", "counterbalance_id"])
                 .realized_letter.nunique() > 1).mean()
        print(f"  {mode} L{L} {v:>7}: cells with a decision flip: {100*flips:.1f}%")


if __name__ == "__main__":
    main()
