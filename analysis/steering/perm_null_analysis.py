"""Permutation-null (matched-control) analysis — PRE-REGISTERED (perm_geometry/PERM_GEOMETRY_NOTE.md).

Inference = matched control, NOT a permutation p-value (3 seeds cannot support one).
PRIMARY: per-game paired difference on the DD family, L65 —
  (true small-dose canonical slope, variant main_perp, mode h1_dinc, from
   smalldose_summary/game_slopes.csv) − (perm-arm canonical slope), computed vs
  (a) the seed-MEAN and (b) the WORST-of-3 seed (the seed most favorable to the null,
  i.e. with the LARGEST DD effect). Game-clustered bootstrap 95% CI (deterministic).
CLAIM CRITERION (pre-registered): specificity iff the paired DD difference CI excludes 0
  in qwen AND qwen_instruct (qwen is the low-power supporting case; qwen_instruct powered).
SECONDARY: each seed's own family table (expected: no DD gradient); perm letter slopes
  (expected ~0: directions are letter-orthogonalized); pooled-54 paired difference.

Writes data/results/steering/smalldose_summary/perm_null_results.csv
and prints the verdict. CPU-only.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from strategic_anatomy.config import results_root, steering_root

ROOT = Path(__file__).resolve().parents[2]
from collection.akata_common import akata_cb_grid  # noqa: E402

GRID = {c["counterbalance_id"]: c for c in akata_cb_grid()}
RNG = np.random.default_rng(0)
MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
SEEDS = ("perm0", "perm1", "perm2")
PERM_ROOT = steering_root() / "perm"
GS_PATH = results_root() / "steering" / "smalldose_summary" / "game_slopes.csv"
OUT = results_root() / "steering" / "smalldose_summary" / "perm_null_results.csv"
DEFAULT_PERM_ROOT, DEFAULT_GS_PATH, DEFAULT_OUT = PERM_ROOT, GS_PATH, OUT
GS = None                          # loaded in main() so the paths stay overridable


def boot_ci(x, n=3000):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m = np.array([RNG.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def perm_game_slopes(model: str) -> pd.DataFrame:
    df = pd.read_parquet(glob.glob(str(PERM_ROOT / f"steer_{model}_h1_dinc_*.parquet"))[0])
    df["j_is_act0"] = df.counterbalance_id.map(lambda c: GRID[c]["letter_to_action"]["J"] == 0)
    df["pref_act0"] = np.where(df.j_is_act0, df.slot_pref_J, 1 - df.slot_pref_J)
    df["pref_canon"] = np.where(df.canonical_action_p1 == 0, df.pref_act0, 1 - df.pref_act0)
    rows = []
    for (L, v, g), grp in df.groupby(["steer_layer", "variant", "game_code"]):
        b = grp[grp.dose == 0].set_index("counterbalance_id")
        rec = {"model": model, "layer": L, "variant": v, "game_code": g}
        for comp, col in (("canon", "pref_canon"), ("letter", "slot_pref_J")):
            pts = []
            for dose, gd in grp[grp.dose != 0].groupby("dose"):
                deltas = [gd[gd.counterbalance_id == cb][col].iloc[0] - b.loc[cb, col]
                          for cb in gd.counterbalance_id.unique() if cb in b.index]
                pts.append((dose, np.mean(deltas)))
            rec[f"slope_{comp}"] = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def _parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="permutation matched-control analysis (additive path overrides)")
    p.add_argument("--perm-root", default=str(DEFAULT_PERM_ROOT), help="permutation-arm parquet root")
    p.add_argument("--game-slopes", default=str(DEFAULT_GS_PATH),
                   help="true-arm game_slopes.csv the perm arm is paired against")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output CSV path")
    return p.parse_args(argv)


def main(argv=None):
    global PERM_ROOT, GS_PATH, OUT, GS
    a = _parse_args(argv)
    PERM_ROOT, GS_PATH, OUT = Path(a.perm_root), Path(a.game_slopes), Path(a.out)
    # Construct-identity guard: the perm arm and the true arm it is paired against must
    # come from the same belief construct, and must not overwrite the committed empirical CSV.
    if (PERM_ROOT != DEFAULT_PERM_ROOT or GS_PATH != DEFAULT_GS_PATH) and OUT == DEFAULT_OUT:
        sys.exit(f"[perm] REFUSING to write non-default construct into {DEFAULT_OUT}. Pass an explicit --out.")
    if ("q05" in PERM_ROOT.name) != ("q05" in str(GS_PATH)):
        sys.exit(f"[perm] CONSTRUCT MISMATCH: perm_root={PERM_ROOT} vs game_slopes={GS_PATH}")
    print(f"[perm] perm_root={PERM_ROOT}\n[perm] game_slopes={GS_PATH}\n[perm] out={OUT}")
    GS = pd.read_csv(GS_PATH)
    fam = dict(zip(GS.game_code, GS.family))
    out = []
    print("=== PERM NULL — pre-registered matched-control analysis ===\n")
    for m in MODELS:
        pg = perm_game_slopes(m)
        pg["family"] = pg.game_code.map(fam)
        true = GS[(GS.model == m) & (GS["mode"] == "h1_dinc") & (GS.variant == "main_perp")]
        for L in (65, 79):
            tl = true[true.layer == L].set_index("game_code").slope_canon
            pl = pg[pg.layer == L]
            wide = pl.pivot_table(index="game_code", columns="variant", values="slope_canon")
            games = [g for g in wide.index if g in tl.index]
            wide = wide.loc[games]; tv = tl[games]
            fams = pd.Series({g: fam[g] for g in games})
            seed_dd = {s: wide.loc[fams == "DD", s].mean() for s in SEEDS}
            worst = max(seed_dd, key=seed_dd.get)   # seed most favorable to the null
            for label, ref in (("seed_mean", wide[list(SEEDS)].mean(axis=1)), ("worst_of_3", wide[worst])):
                for scope, mask in (("DD", fams == "DD"), ("pooled54", fams.notna())):
                    d = (tv - ref)[mask].to_numpy()
                    lo, hi = boot_ci(d)
                    out.append({"model": m, "layer": L, "ref": label, "scope": scope,
                                "true_mean": tv[mask].mean(), "perm_mean": ref[mask].mean(),
                                "diff_mean": np.nanmean(d), "ci_lo": lo, "ci_hi": hi,
                                "n_games": int(mask.sum()),
                                "worst_seed": worst if label == "worst_of_3" else ""})
            if L == 65:
                print(f"--- {m} L65 ---")
                for s in SEEDS:
                    dd = wide.loc[fams == "DD", s]
                    lo, hi = boot_ci(dd.to_numpy())
                    lt = pl[(pl.variant == s)].slope_letter
                    print(f"  {s}: own DD slope {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}] ({int((dd>0).sum())}/9 pos); "
                          f"letter slope {lt.mean():+.4f}")
                for r in [o for o in out if o["model"] == m and o["layer"] == 65 and o["scope"] == "DD"]:
                    print(f"  PAIRED DIFF ({r['ref']}{' vs '+r['worst_seed'] if r['worst_seed'] else ''}): "
                          f"true {r['true_mean']:+.4f} − perm {r['perm_mean']:+.4f} = "
                          f"{r['diff_mean']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
                print()
    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False)
    print("=== VERDICT (pre-registered: DD paired diff CI>0 in qwen AND qwen_instruct, L65) ===")
    for m in ("qwen", "qwen_instruct", "llama31_instruct"):
        for ref in ("seed_mean", "worst_of_3"):
            r = res[(res.model == m) & (res.layer == 65) & (res.scope == "DD") & (res.ref == ref)].iloc[0]
            print(f"  {m:<17} vs {ref:<10}: {r.diff_mean:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  "
                  f"{'EXCLUDES 0 ✓' if r.ci_lo > 0 else ('includes 0' if r.ci_hi > 0 else 'NEGATIVE')}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
