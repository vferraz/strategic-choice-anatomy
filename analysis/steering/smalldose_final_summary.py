"""FINAL cross-model summary of the small-dose steering arm (3 models x h1/h2 x main/random/main_perp).

Produces the committed result tables for the linear-regime causal test (letter/canonical/behavior
decomposition; family breakdown; cross-mode common-channel correlation; pooled behavioral test).
Method: per (model, mode, layer, variant, game), the cb-mean Δ vs the same cell's dose-0 row is
regressed on dose — the 4-cell counterbalance makes the cb-mean of ΔprefJ the LETTER component
(canonical cancels) and the cb-mean of Δpref_canon the STRATEGIC component (letter cancels).
Game-clustered bootstrap CIs (deterministic seed). No slope-only readouts elsewhere: flip rates and
ranges are in game_slopes.csv for any downstream check.

Writes data/results/steering/smalldose_summary/{game_slopes.csv, summary.csv,
family_canonical.csv, crossmode_r.csv} and prints the headline report. CPU-only.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from strategic_anatomy.config import manifests_root, results_root, steering_root

ROOT = Path(__file__).resolve().parents[2]
from collection.akata_common import akata_cb_grid  # noqa: E402

GRID = {c["counterbalance_id"]: c for c in akata_cb_grid()}
RNG = np.random.default_rng(0)
MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
MODES = ("h1_dinc", "h2_choice")
IN_ROOT = steering_root() / "smalldose"
OUT = results_root() / "steering" / "smalldose_summary"
DELTA_SUFFIX = ""                 # "" = empirical belief; "_q05" = uniform belief (q=0.5)
DEFAULT_IN_ROOT, DEFAULT_OUT = IN_ROOT, OUT
SAMPLE = pd.read_csv(manifests_root() / "steer_sample_54.csv")
FAM = dict(zip(SAMPLE.game_code, SAMPLE.family))
FAMILIES = ("DD", "OD1", "OD2", "CO1", "CO2", "MP")


def boot_ci(x, n=3000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def load_game_slopes(model: str) -> pd.DataFrame:
    d1_path = results_root() / "steering" / "summary" / f"delta_tables_{model}{DELTA_SUFFIX}.csv"
    if d1_path.exists():
        d1 = pd.read_csv(d1_path)
        d1c = dict(zip(d1.game_code, d1.delta1_c))
    else:
        print(f"[smalldose] missing {d1_path}; delta1_c column will be NaN", file=sys.stderr)
        d1c = {}
    rows = []
    for mode in MODES:
        fps = glob.glob(str(IN_ROOT / f"steer_{model}_{mode}_*.parquet"))
        if not fps:
            continue
        df = pd.read_parquet(fps[0])
        df["j_is_act0"] = df.counterbalance_id.map(lambda c: GRID[c]["letter_to_action"]["J"] == 0)
        df["pref_act0"] = np.where(df.j_is_act0, df.slot_pref_J, 1 - df.slot_pref_J)
        df["pref_canon"] = np.where(df.canonical_action_p1 == 0, df.pref_act0, 1 - df.pref_act0)
        df["beh_canon"] = df.realized_canonical.astype(float)
        for (L, v, g), grp in df.groupby(["steer_layer", "variant", "game_code"]):
            b = grp[grp.dose == 0].set_index("counterbalance_id")
            rec = {"model": model, "mode": mode, "layer": L, "variant": v, "game_code": g,
                   "family": FAM.get(g), "delta1_c": d1c.get(g)}
            for comp, col in (("letter", "slot_pref_J"), ("canon", "pref_canon"), ("beh", "beh_canon")):
                pts = []
                for dose, gd in grp.groupby("dose"):
                    if dose == 0:
                        continue
                    deltas = [gd[gd.counterbalance_id == cb][col].iloc[0] - b.loc[cb, col]
                              for cb in gd.counterbalance_id.unique() if cb in b.index]
                    deltas = [x for x in deltas if np.isfinite(x)]
                    if deltas:
                        pts.append((dose, np.mean(deltas)))
                rec[f"slope_{comp}"] = (np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]
                                        if len(pts) >= 3 else float("nan"))
                rec[f"range_{comp}"] = (max(p[1] for p in pts) - min(p[1] for p in pts)) if pts else float("nan")
            lets = grp[grp.realized_letter != ""].realized_letter
            rec["beh_flip"] = int(lets.nunique() > 1)
            rows.append(rec)
    return pd.DataFrame(rows)


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="small-dose steering summary (additive path overrides)")
    p.add_argument("--in-root", default=str(DEFAULT_IN_ROOT),
                   help="steering parquet root (default: the empirical-belief small-dose arm)")
    p.add_argument("--delta-suffix", default="",
                   help='suffix on delta_tables_{model}{suffix}.csv; "_q05" for the uniform-belief target')
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output table directory")
    return p.parse_args(argv)


def main(argv=None):
    global IN_ROOT, OUT, DELTA_SUFFIX
    a = _parse_args(argv)
    IN_ROOT, OUT, DELTA_SUFFIX = Path(a.in_root), Path(a.out), a.delta_suffix
    # Construct-identity guard: a non-default input construct must never be written
    # under the committed empirical filename (the "same filename, different construct" defect).
    if (IN_ROOT != DEFAULT_IN_ROOT or DELTA_SUFFIX) and OUT == DEFAULT_OUT:
        sys.exit(f"[smalldose] REFUSING to write construct (in_root={IN_ROOT.name}, "
                 f"delta_suffix={DELTA_SUFFIX!r}) into the committed empirical dir {DEFAULT_OUT}. "
                 f"Pass an explicit --out.")
    print(f"[smalldose] in_root={IN_ROOT}\n[smalldose] delta_suffix={DELTA_SUFFIX!r}\n[smalldose] out={OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    gs = pd.concat([load_game_slopes(m) for m in MODELS], ignore_index=True)
    gs.to_csv(OUT / "game_slopes.csv", index=False)

    summary = []
    for (m, mode, L, v), sub in gs.groupby(["model", "mode", "layer", "variant"]):
        for comp in ("letter", "canon", "beh"):
            x = sub[f"slope_{comp}"].to_numpy()
            lo, hi = boot_ci(x)
            summary.append({"model": m, "mode": mode, "layer": L, "variant": v, "component": comp,
                            "mean_slope": np.nanmean(x), "ci_lo": lo, "ci_hi": hi,
                            "n_games": int(np.isfinite(x).sum()),
                            "pct_beh_flip": 100 * sub.beh_flip.mean()})
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)

    famrows = []
    for (m, mode), sub in gs[(gs.layer == 65) & (gs.variant == "main_perp")].groupby(["model", "mode"]):
        for f in FAMILIES:
            x = sub[sub.family == f].slope_canon.to_numpy()
            lo, hi = boot_ci(x)
            famrows.append({"model": m, "mode": mode, "family": f, "mean_canon_slope": np.nanmean(x),
                            "ci_lo": lo, "ci_hi": hi, "n_pos": int((x > 0).sum()), "n": len(x)})
    fam = pd.DataFrame(famrows)
    fam.to_csv(OUT / "family_canonical.csv", index=False)

    xr = []
    for m in MODELS:
        for L in sorted(gs.layer.unique()):
            p = gs[(gs.model == m) & (gs.layer == L) & (gs.variant == "main_perp")]
            h1 = p[p["mode"] == "h1_dinc"].set_index("game_code").slope_canon
            h2 = p[p["mode"] == "h2_choice"].set_index("game_code").slope_canon
            g = sorted(set(h1.index) & set(h2.index))
            if len(g) > 3:
                xr.append({"model": m, "layer": L, "r_h1_h2": np.corrcoef(h1[g], h2[g])[0, 1], "n": len(g)})
    pd.DataFrame(xr).to_csv(OUT / "crossmode_r.csv", index=False)

    print("=== HEADLINES (main_perp L65, canonical) ===")
    for (m, mode), sub in gs[(gs.layer == 65) & (gs.variant == "main_perp")].groupby(["model", "mode"]):
        x = sub.slope_canon.to_numpy(); lo, hi = boot_ci(x)
        dd = sub[sub.family == "DD"].slope_canon.to_numpy(); dlo, dhi = boot_ci(dd)
        print(f"{m:<17} {mode:<9} pooled {np.nanmean(x):+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"  DD {np.nanmean(dd):+.4f} [{dlo:+.4f},{dhi:+.4f}] ({(dd>0).sum()}/9 pos)")
    print("\n=== cross-mode r (main_perp, canonical) ===")
    print(pd.DataFrame(xr).round(3).to_string(index=False))
    print("\n=== POOLED BEHAVIOR across models (main_perp L65) vs random ===")
    for mode in MODES:
        for v in ("main_perp", "random"):
            x = gs[(gs.layer == 65) & (gs.variant == v) & (gs["mode"] == mode)].slope_beh.to_numpy()
            lo, hi = boot_ci(x)
            print(f"  {mode} {v:>9}: {np.nanmean(x):+.4f} [{lo:+.4f},{hi:+.4f}]  (n={np.isfinite(x).sum()})")
    print(f"\nwrote 4 tables -> {OUT}")


if __name__ == "__main__":
    main()
