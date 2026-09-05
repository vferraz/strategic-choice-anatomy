"""Apparatus-geometry diagnostics for the q=0.5 (uniform-belief) steering axis.

Recomputes, on the corrected-Akata substrate, the direction-geometry quantities the
Supplementary "Apparatus validity" and "Causal interventions" paragraphs report, all of
which were previously measured on the pre-correction empirical-belief axis:

  * cos(d_inc_L, mean residual at L)            -> non-degeneracy of the covariance axis
  * cos(axis fitted at residual L, axis fitted at the injection site L+1)
                                                -> injected-vs-locally-fitted alignment
  * |cos(d_random_L, every other direction)|    -> random-direction orthogonality
  * per-seed cos(d_inc_perm_k, d_inc) [letter-orthogonalised set]
                                                -> permutation-control alignment.
    The q05 perm seed set is 0/1/2 and is NOT matched to the empirical arm's 1/2/3, and
    some seeds are anti-aligned with the true axis, so per-seed values are reported and
    no pooled cosine is emitted.
  * dose-zero identity check across variants/layers/modes
  * max |letter-axis dose-slope| of the permuted directions

Self-check: the axis refitted from the substrate at layer L must reproduce the shipped
d_inc_l{L} vector (cos ~ 1.000). The script aborts if it does not.

Writes data/results/steering/smalldose_summary_q05/apparatus_geometry_q05.{csv,json}.
CPU-only.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from strategic_anatomy.config import results_root, steering_root, substrate_root

ROOT = Path(__file__).resolve().parents[2]

MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
LAYERS = (65, 79)
SEEDS = ("perm0", "perm1", "perm2")
CB = (0, 1, 2, 3)

SUBSTRATE = substrate_root()
DIRS = steering_root() / "directions" / "akata_q05"
DIRS_PERP = steering_root() / "directions" / "akata_q05_perp"
DIRS_PERM = steering_root() / "directions" / "akata_q05_perm"
DELTA = results_root() / "steering" / "summary"
SD_Q05 = steering_root() / "smalldose_q05"
PERM_Q05 = steering_root() / "perm_q05"
OUT = results_root() / "steering" / "smalldose_summary_q05"


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def load_substrate(model: str, layers) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X[n_rows, d] stacked over games x cb, y[n_rows], games) per layer key."""
    d1 = pd.read_csv(DELTA / f"delta_tables_{model}_q05.csv")
    d1c = dict(zip(d1.game_code, d1.delta1_c))
    games = sorted(p.name for p in (SUBSTRATE / model).iterdir()
                   if p.is_dir() and (p / "acts.npz").exists())
    X = {L: [] for L in layers}
    y = []
    for g in games:
        if g not in d1c:
            continue
        z = np.load(SUBSTRATE / model / g / "acts.npz")
        for c in CB:
            for L in layers:
                X[L].append(np.asarray(z[f"p1_baseline_cb{c}_l{L}"], dtype=np.float64).ravel())
            y.append(float(d1c[g]))
        z.close()
    return {L: np.vstack(X[L]) for L in layers}, np.asarray(y), games


def fit_d_inc(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centred covariance axis, sign-aligned to a positive projection-incentive correlation.

    Mirrors extract_causal_directions_oneshot.py:
    'centered_covariance_Cov(X,delta1c)_sign_aligned_p1_baseline'.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    d = _unit((Xc * yc[:, None]).mean(axis=0))
    if np.corrcoef(X @ d, y)[0, 1] < 0:
        d = -d
    return d


def dose_zero_identity() -> dict:
    """Dose-zero rows must be identical across variants within each (model, mode, layer, game, cb)."""
    cells = mism = 0
    for fp in sorted(glob.glob(str(SD_Q05 / "*.parquet"))):
        df = pd.read_parquet(fp)
        z = df[df.dose == 0]
        for _, grp in z.groupby(["steer_layer", "game_code", "counterbalance_id"]):
            cells += 1
            if grp.slot_pref_J.nunique() > 1 or grp.realized_letter.nunique() > 1:
                mism += 1
    return {"dose_zero_cells": cells, "dose_zero_mismatches": mism}


def perm_letter_slopes() -> dict:
    """Max |letter-axis dose-slope| of the permuted directions (the paper's '+/-1.7' claim)."""
    from collection.akata_common import akata_cb_grid
    grid = {c["counterbalance_id"]: c for c in akata_cb_grid()}
    worst, per = 0.0, {}
    for fp in sorted(glob.glob(str(PERM_Q05 / "*.parquet"))):
        df = pd.read_parquet(fp)
        model = df.model.iloc[0]
        for (L, v, g), grp in df.groupby(["steer_layer", "variant", "game_code"]):
            b = grp[grp.dose == 0].set_index("counterbalance_id")
            pts = []
            for dose, gd in grp[grp.dose != 0].groupby("dose"):
                deltas = [gd[gd.counterbalance_id == cb].slot_pref_J.iloc[0] - b.loc[cb, "slot_pref_J"]
                          for cb in gd.counterbalance_id.unique() if cb in b.index]
                pts.append((dose, float(np.mean(deltas))))
            s = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])
            k = f"{model}_l{L}_{v}"
            per[k] = max(per.get(k, 0.0), abs(s))
            worst = max(worst, abs(s))
    return {"max_abs_perm_letter_slope": worst, "per_cell_max": per}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, self_checks = [], []
    for model in MODELS:
        need = sorted({L for L in LAYERS} | {L + 1 for L in LAYERS})
        print(f"[apparatus] loading substrate residuals for {model} at layers {need} ...", flush=True)
        X, y, games = load_substrate(model, need)
        base = np.load(DIRS / model / "directions.npz")
        perp = np.load(DIRS_PERP / model / "directions.npz")
        perm = np.load(DIRS_PERM / model / "directions.npz")
        print(f"[apparatus]   n_rows={X[LAYERS[0]].shape[0]} n_games={len(games)} d={X[LAYERS[0]].shape[1]}")
        for L in LAYERS:
            d_fit = fit_d_inc(X[L], y)
            d_ship = np.asarray(base[f"d_inc_l{L}"], dtype=np.float64)
            c_self = abs(cos(d_fit, d_ship))
            self_checks.append({"model": model, "layer": L, "cos_refit_vs_shipped": c_self})
            if c_self < 0.999:
                sys.exit(f"[apparatus] SELF-CHECK FAILED {model} L{L}: refit vs shipped cos={c_self:.6f}")
            d_hook = fit_d_inc(X[L + 1], y)          # axis fitted at the injection site
            rows.append({
                "model": model, "layer": L,
                "cos_dinc_vs_mean_residual": cos(d_ship, X[L].mean(axis=0)),
                "cos_injected_vs_locally_fitted": abs(cos(d_ship, d_hook)),
                "cos_random_vs_dinc": abs(cos(base[f"d_random_l{L}"], d_ship)),
                "cos_random_vs_dchoice": abs(cos(base[f"d_random_l{L}"], base[f"d_choice_l{L}"])),
                "cos_random_vs_mean_residual": abs(cos(base[f"d_random_l{L}"], X[L].mean(axis=0))),
                "cos_refit_vs_shipped": c_self,
                **{f"cos_{s}_vs_true_perp": cos(perm[f"d_inc_{s}_l{L}"], perp[f"d_inc_l{L}"])
                   for s in SEEDS},
            })
            print(f"[apparatus]   {model} L{L}: self-check cos={c_self:.6f}", flush=True)
        base.close(); perp.close(); perm.close()
        del X

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "apparatus_geometry_q05.csv", index=False)
    extra = {**dose_zero_identity(), **perm_letter_slopes()}
    (OUT / "apparatus_geometry_q05.json").write_text(json.dumps(
        {"self_checks": self_checks, "extra": extra}, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== APPARATUS GEOMETRY (q=0.5 axis) ===")
    print(df.round(4).to_string(index=False))
    for L in LAYERS:
        s = df[df.layer == L]
        print(f"\nL{L}: cos(d_inc, mean residual) range "
              f"[{s.cos_dinc_vs_mean_residual.min():+.3f}, {s.cos_dinc_vs_mean_residual.max():+.3f}]"
              f" | injected-vs-locally-fitted {s.cos_injected_vs_locally_fitted.min():.3f}"
              f"-{s.cos_injected_vs_locally_fitted.max():.3f}")
    rc = df[[c for c in df.columns if c.startswith("cos_random")]].abs().to_numpy().max()
    print(f"\nmax |cos(random, anything)| = {rc:.4f}")
    pc = df[[f"cos_{s}_vs_true_perp" for s in SEEDS]]
    print(f"per-seed cos(perm, true perp axis): min {pc.to_numpy().min():+.3f} "
          f"max {pc.to_numpy().max():+.3f}  (per-seed values in the CSV)")
    print(f"\ndose-zero: {extra['dose_zero_mismatches']}/{extra['dose_zero_cells']} mismatches")
    print(f"max |perm letter dose-slope| = {extra['max_abs_perm_letter_slope']:.3f}")
    print(f"\nwrote -> {OUT}/apparatus_geometry_q05.csv|.json")


if __name__ == "__main__":
    main()
