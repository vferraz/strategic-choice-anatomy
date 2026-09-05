"""Step 0 — CPU geometry study for the steering permutation-null arm (matched control).

Free / CPU-only. Gates the within-family arm of the permutation-null design.

For each model and steer layer L in {65,79}:
  X   = 576 P1-baseline residual rows (144 fit games x 4 cb) from the corrected Akata substrate
        $SCA_DATA_ROOT/substrate/{model}/{game}/acts.npz, key p1_baseline_cb{cb}_l{L}
  y    = per-game canonical-signed incentive gap delta1_c (delta_tables_{model}.csv)
  d_inc_true       = analysis/.../directions_akata/{model}/directions.npz : d_inc_l{L}
  d_choice_perp    = same npz : d_choice_perp_l{L}
  ell_hat          = norm.weight (.)  (W_U[' J'] - W_U[' P']), unit  (perp letter readout axis)

Refit d_perm EXACTLY as fit_d_inc (imported, reused verbatim): centered covariance
Cov(X, delta1_c_perm), sign-align to the PERMUTED target, unit norm. Permutation is at the
GAME level over all 144 fit games (all 4 cb of a game share the permuted delta1_c):
  across-game        : permute delta1_c across all 144 games
  within-family      : permute delta1_c within each STRUCTURAL family block
                       (nagel_lk_type = DD/OD1/OD2/CO1/CO2/MP = game_slopes 'family';
                        deterministic from game structure, covers all 144)

Reports the distributions of corr(y_perm,y), cos(d_perm,d_inc_true), cos(d_perm,ell_hat),
cos(d_perm,d_choice_perp) and the WITHIN-FAMILY GATE verdict (median |cos(d_perm_withinfam,
d_inc_true)| > 0.6 at either layer -> within-family null unconstructible, drop it).

Outputs -> data/results/steering/perm_geometry/
  perm_geometry_permutations.csv  (full per-permutation record)
  perm_geometry_summary.csv       (quantiles per model/layer/perm-type/metric)
  PERM_GEOMETRY_NOTE.md           (gate verdict + headline numbers)
Run: uv run python analysis/steering/perm_geometry_study.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from steering.extract_directions import fit_d_inc  # exact reuse
from strategic_anatomy.config import repo_root, results_root, steering_root, substrate_root, taxonomy_dir

ROOT = repo_root()


MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
LAYERS = (65, 79)
N_PERM = 1000
GATE_THRESH = 0.60
BASE_SEED = 12345

SUB_ROOT = substrate_root()
DIRS = steering_root() / "directions" / "akata"
DELTA = results_root() / "steering" / "summary"
EQUIV = taxonomy_dir() / "equivalence_per_canonical.csv"
OUT = results_root() / "steering" / "perm_geometry"


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _games(model):
    base = SUB_ROOT / model
    return sorted(d.name for d in base.iterdir()
                  if d.is_dir() and d.name != "_tmp" and (d / "_DONE").exists())


def _load_X_meta(model, layers):
    """Return games list, {L: X}, meta DataFrame(game_code,counterbalance_id) aligned to X rows."""
    games = _games(model)
    Xs = {L: [] for L in layers}
    rows = []
    for g in games:
        z = np.load(SUB_ROOT / model / g / "acts.npz")
        try:
            for cb in range(4):
                present = all(f"p1_baseline_cb{cb}_l{L}" in z.files for L in layers)
                if not present:
                    continue
                for L in layers:
                    Xs[L].append(np.asarray(z[f"p1_baseline_cb{cb}_l{L}"], dtype=np.float32))
                rows.append({"game_code": g, "counterbalance_id": cb})
        finally:
            z.close()
    meta = pd.DataFrame(rows)
    X = {L: np.vstack(Xs[L]).astype(np.float32) for L in layers}
    return games, X, meta


def _perm_indices(games, fam_map, rng, within_family):
    """Return an index permutation over the game list."""
    n = len(games)
    if not within_family:
        return rng.permutation(n)
    perm = np.arange(n)
    fams = np.array([fam_map[g] for g in games])
    for f in np.unique(fams):
        idx = np.where(fams == f)[0]
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eq = pd.read_csv(EQUIV)
    fam_all = dict(zip(eq["bruns_name"], eq["nagel_lk_type"]))

    per_perm = []
    summary = []
    gate = {}
    for mi, model in enumerate(MODELS):
        games, X, meta = _load_X_meta(model, LAYERS)
        dt = pd.read_csv(DELTA / f"delta_tables_{model}.csv")
        d1c = dict(zip(dt["game_code"], dt["delta1_c"]))
        fam_map = {g: fam_all[g] for g in games}
        base_y = np.array([d1c[g] for g in games], dtype=np.float64)  # per-game, game-ordered
        z = np.load(DIRS / model / "directions.npz")
        ell = ELL_HAT[model]  # letter readout axis (checkpoint-reconstructed), layer-independent
        for L in LAYERS:
            d_inc_true = _unit(z[f"d_inc_l{L}"])
            d_choice_perp = _unit(z[f"d_choice_perp_l{L}"])
            fam_fp = {}
            for within in (False, True):
                ptype = "within_family" if within else "across_game"
                rng = np.random.default_rng((BASE_SEED, mi, L, int(within)))
                metrics = {"corr": [], "cos_dinc": [], "cos_ell": [], "cos_choiceperp": []}
                for it in range(N_PERM):
                    perm = _perm_indices(games, fam_map, rng, within)
                    yp_game = base_y[perm]
                    d1c_perm = {games[i]: float(yp_game[i]) for i in range(len(games))}
                    v, diag = fit_d_inc(X[L], meta, d1c_perm)
                    v = _unit(v)
                    corr = float(np.corrcoef(yp_game, base_y)[0, 1]) if np.std(yp_game) > 0 else 0.0
                    metrics["corr"].append(corr)
                    metrics["cos_dinc"].append(float(v @ d_inc_true))
                    metrics["cos_ell"].append(float(v @ ell))
                    metrics["cos_choiceperp"].append(float(v @ d_choice_perp))
                    per_perm.append({"model": model, "layer": L, "perm_type": ptype,
                                     "iter": it, "corr_yperm_y": corr,
                                     "cos_dinc": metrics["cos_dinc"][-1],
                                     "cos_ell": metrics["cos_ell"][-1],
                                     "cos_choiceperp": metrics["cos_choiceperp"][-1]})
                arr = {k: np.array(v) for k, v in metrics.items()}
                row = {"model": model, "layer": L, "perm_type": ptype, "n": N_PERM}
                for k, a in arr.items():
                    aa = np.abs(a)
                    row[f"{k}_median"] = float(np.median(a))
                    row[f"{k}_p05"] = float(np.quantile(a, 0.05))
                    row[f"{k}_p95"] = float(np.quantile(a, 0.95))
                    row[f"abs{k}_median"] = float(np.median(aa))
                    row[f"abs{k}_p95"] = float(np.quantile(aa, 0.95))
                summary.append(row)
                if within:
                    fam_fp["abscos_dinc_median"] = row["abscos_dinc_median"]
            gate[(model, L)] = fam_fp["abscos_dinc_median"]
        z.close()

    pp = pd.DataFrame(per_perm)
    pp.to_csv(OUT / "perm_geometry_permutations.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(OUT / "perm_geometry_summary.csv", index=False)

    # gate verdict
    worst = max(gate.values())
    drop = worst > GATE_THRESH
    lines = ["# Permutation-null geometry study (Step 0) — gate note", "",
             f"N_PERM = {N_PERM} per (model, layer, perm-type); base_seed = {BASE_SEED}.",
             "Structural family = nagel_lk_type (equivalence_per_canonical.csv) = game_slopes 'family'.",
             "",
             "## WITHIN-FAMILY GATE (median |cos(d_perm_withinfam, d_inc_true)| > 0.6 -> drop within-family arm)",
             ""]
    for (model, L), val in sorted(gate.items()):
        lines.append(f"- {model} L{L}: median |cos(within-fam d_perm, d_inc_true)| = {val:.4f}"
                     + ("  <-- EXCEEDS 0.6" if val > GATE_THRESH else ""))
    lines += ["", f"**Worst = {worst:.4f}.** "
              + ("Within-family null UNCONSTRUCTIBLE at >=1 layer -> DROP within-family arm; "
                 "across-game is the primary (and only) permuted control."
                 if drop else
                 "Within-family null is constructible (all medians <= 0.6); across-game remains primary."),
              "", "## Headline cos distributions (see perm_geometry_summary.csv for full quantiles)"]
    for model in MODELS:
        for L in LAYERS:
            for ptype in ("across_game", "within_family"):
                r = sm[(sm.model == model) & (sm.layer == L) & (sm.perm_type == ptype)].iloc[0]
                lines.append(
                    f"- {model} L{L} {ptype}: corr(y_perm,y) med={r.corr_median:+.3f} "
                    f"[p05 {r.corr_p05:+.3f}, p95 {r.corr_p95:+.3f}]; "
                    f"|cos d_inc| med={r.abscos_dinc_median:.3f} (p95 {r.abscos_dinc_p95:.3f}); "
                    f"|cos ell| med={r.abscos_ell_median:.3f}; "
                    f"|cos choice_perp| med={r.abscos_choiceperp_median:.3f}")
    lines += [
        "",
        "## Analysis pre-registration (for the downstream perm-arm analysis; honor these)",
        "",
        "- **Inference = MATCHED CONTROL, NOT a permutation p-value.** Pre-registered statistic: "
        "the paired game-level difference (real d_inc small-dose canonical effect on the DD family) "
        "minus (each of the 3 fixed permuted directions' effect), with a CI over games. A real "
        "permutation p-value would need K>=19 seeds; do NOT build or claim one from 3 seeds.",
        "- **PI FIX #2 (qwen power caveat).** qwen's true L65 canonical effect is small "
        "(mean +0.018) vs qwen_instruct's +0.057. qwen_instruct is the POWERED specificity test; "
        "qwen is SUPPORTING. The criterion stands for both; report qwen with this caveat.",
        "- **PI FIX #3 (report worst-of-3).** Report true - worst-of-3-seeds ALONGSIDE "
        "true - seed-mean, so a single favorable/unfavorable permuted draw cannot drive the verdict.",
        "- **Structural family = nagel_lk_type (= game_slopes 'family').** PROVENANCE FLAG: the PI "
        "spec said 'NOT nagel_lk_type', but game_slopes.csv's own `family` column is byte-identical "
        "to equivalence_per_canonical.csv `nagel_lk_type` (0/54 mismatches, full 144 coverage, "
        "deterministic from dominance_profile+num_pure_ne+level-k depth). This is the only partition "
        "consistent with game_slopes; confirm before finalizing.",
    ]
    (OUT / "PERM_GEOMETRY_NOTE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT/'perm_geometry_permutations.csv'} ({len(pp)} rows)")
    print(f"wrote {OUT/'perm_geometry_summary.csv'}")
    print(f"wrote {OUT/'PERM_GEOMETRY_NOTE.md'}")


# ell_hat (letter readout axis) reconstructed from the checkpoint; validated to reproduce the
# committed perp manifest cos_removed to ~1e-7 (see build note). Populated at import by _build_ell.
ELL_HAT = {}


def _build_ell():
    import glob, os
    from safetensors import safe_open
    from transformers import AutoTokenizer
    import torch
    repo = {"qwen": "Qwen/Qwen2.5-72B", "qwen_instruct": "Qwen/Qwen2.5-72B-Instruct",
            "llama31_instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct"}
    for m, r in repo.items():
        snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{r.replace('/','--')}/snapshots/*"))[0]
        tok = AutoTokenizer.from_pretrained(r, use_fast=True, local_files_only=True)
        jid = tok(" J", add_special_tokens=False)["input_ids"][-1]
        pid = tok(" P", add_special_tokens=False)["input_ids"][-1]
        wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
        with safe_open(os.path.join(snap, wm["lm_head.weight"]), framework="pt", device="cpu") as f:
            sl = f.get_slice("lm_head.weight")
            wj = sl[jid].to(torch.float32).numpy().astype(np.float64)
            wp = sl[pid].to(torch.float32).numpy().astype(np.float64)
        with safe_open(os.path.join(snap, wm["model.norm.weight"]), framework="pt", device="cpu") as f:
            nrm = f.get_tensor("model.norm.weight").to(torch.float32).numpy().astype(np.float64)
        ELL_HAT[m] = _unit(nrm * (wj - wp))


if __name__ == "__main__":
    _build_ell()
    main()
