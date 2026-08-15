"""Step 1 — build the 3 matched-control permuted steering directions (CPU, free).

For each model and steer layer L in {65,79}:
  * refit d_perm EXACTLY as fit_d_inc (imported) on an ACROSS-GAME game-level permutation of
    delta1_c (within-family dropped per the Step-0 gate), sign-aligned to the permuted target;
  * letter-orthogonalize with the EXACT ell_hat from the perp method:
        d = unit(d_perm - (d_perm . ell_hat) ell_hat)
        ell = norm.weight (.) (W_U[' J'] - W_U[' P']), unit  (checkpoint-reconstructed)
  * the ORTHOGONALIZED direction is stored (this is what gets injected).

Seeds: slots 0,1,2 start a priori from integer seeds 0,1,2; a candidate seed is ACCEPTED only if
|corr(y_perm, y)| <= 0.15 (game level, per model), else replaced by the next integer seed (recorded).
Acceptance order defines slots perm0/perm1/perm2.

Writes  $SCA_DATA_ROOT/steering/directions/akata_perm/{model}/
          directions.npz   keys d_inc_perm{slot}_l{L}  (the perp versions, injected)
          manifest.json    slot->actual seed, corr(y_perm,y), cos(d_perm,d_inc_true),
                           cos(perm_perp,ell_hat) [~0], cos(perm_perp,d_inc_true), norms, method,
                           git commit, jp_token_ids, ell validation vs committed perp manifest.

Run: uv run python steering/build_perm_directions.py
"""
from __future__ import annotations
import datetime as dt, glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from steering.extract_directions import fit_d_inc  # exact reuse
from strategic_anatomy.config import repo_root, results_root, steering_root, substrate_root

MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
REPO = {"qwen": "Qwen/Qwen2.5-72B", "qwen_instruct": "Qwen/Qwen2.5-72B-Instruct",
        "llama31_instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct"}
LAYERS = (65, 79)
N_SLOTS = 3
CORR_THRESH = 0.15
MAX_SEED = 200

ROOT = repo_root()
SUB_ROOT = substrate_root()
DIRS = steering_root() / "directions" / "akata"
DIRS_PERP = steering_root() / "directions" / "akata_perp"
DELTA = results_root() / "steering" / "summary"
OUT = steering_root() / "directions" / "akata_perm"


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        raise ValueError("zero/non-finite vector")
    return v / n


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=str(ROOT)).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _snap(repo):
    return glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/','--')}/snapshots/*"))[0]


def build_ell_hat(model):
    """ell_hat = unit(norm.weight (.) (W_U[' J'] - W_U[' P'])), bf16->fp32, from the checkpoint."""
    repo = REPO[model]; snap = _snap(repo)
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True, local_files_only=True)
    jid = tok(" J", add_special_tokens=False)["input_ids"][-1]
    pid = tok(" P", add_special_tokens=False)["input_ids"][-1]
    wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(snap, wm["lm_head.weight"]), framework="pt", device="cpu") as f:
        sl = f.get_slice("lm_head.weight")
        wj = sl[jid].to(torch.float32).numpy().astype(np.float64)
        wp = sl[pid].to(torch.float32).numpy().astype(np.float64)
    with safe_open(os.path.join(snap, wm["model.norm.weight"]), framework="pt", device="cpu") as f:
        nrm = f.get_tensor("model.norm.weight").to(torch.float32).numpy().astype(np.float64)
    return _unit(nrm * (wj - wp)), int(jid), int(pid)


def _games(model):
    base = SUB_ROOT / model
    return sorted(d.name for d in base.iterdir()
                  if d.is_dir() and d.name != "_tmp" and (d / "_DONE").exists())


def _load_X_meta(model, layers):
    games = _games(model)
    Xs = {L: [] for L in layers}
    rows = []
    for g in games:
        z = np.load(SUB_ROOT / model / g / "acts.npz")
        try:
            for cb in range(4):
                if not all(f"p1_baseline_cb{cb}_l{L}" in z.files for L in layers):
                    continue
                for L in layers:
                    Xs[L].append(np.asarray(z[f"p1_baseline_cb{cb}_l{L}"], dtype=np.float32))
                rows.append({"game_code": g, "counterbalance_id": cb})
        finally:
            z.close()
    return games, {L: np.vstack(Xs[L]).astype(np.float32) for L in layers}, pd.DataFrame(rows)


def validate_ell(model, ellhat):
    """Cross-check ell_hat against the committed perp manifest cos_removed (must match ~1e-6)."""
    man = json.load(open(DIRS_PERP / model / "manifest.json"))
    z = np.load(DIRS / model / "directions.npz")
    out = {}
    for L in LAYERS:
        d = _unit(z[f"d_inc_l{L}"])
        mine = float(d @ ellhat)
        ref = man["per_direction"][f"d_inc_l{L}"]["cos_removed"]
        out[f"d_inc_l{L}"] = {"my_cos": mine, "manifest_cos_removed": ref, "abs_diff": abs(mine - ref)}
        assert abs(mine - ref) < 1e-5, f"{model} L{L} ell mismatch {mine} vs {ref}"
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    for model in MODELS:
        games, X, meta = _load_X_meta(model, LAYERS)
        dt_df = pd.read_csv(DELTA / f"delta_tables_{model}.csv")
        d1c = dict(zip(dt_df["game_code"], dt_df["delta1_c"]))
        base_y = np.array([d1c[g] for g in games], dtype=np.float64)
        ellhat, jid, pid = build_ell_hat(model)
        ell_val = validate_ell(model, ellhat)
        z = np.load(DIRS / model / "directions.npz")
        d_inc_true = {L: _unit(z[f"d_inc_l{L}"]) for L in LAYERS}

        # ---- select the 3 seeds: start a priori at 0,1,2; replace failing seeds by next integer ----
        slot_seeds, slot_corr, tried = [], [], []
        seed = 0
        while len(slot_seeds) < N_SLOTS and seed < MAX_SEED:
            perm = np.random.default_rng(seed).permutation(len(games))
            yp = base_y[perm]
            corr = float(np.corrcoef(yp, base_y)[0, 1]) if np.std(yp) > 0 else 0.0
            accepted = abs(corr) <= CORR_THRESH
            tried.append({"seed": seed, "corr": corr, "accepted": bool(accepted)})
            if accepted:
                slot_seeds.append(seed); slot_corr.append(corr)
            seed += 1
        if len(slot_seeds) < N_SLOTS:
            raise RuntimeError(f"[{model}] could not find {N_SLOTS} seeds with |corr|<= {CORR_THRESH}")

        # ---- build perp directions for each accepted seed/slot ----
        directions, per_dir = {}, {}
        for slot, sd in enumerate(slot_seeds):
            perm = np.random.default_rng(sd).permutation(len(games))
            d1c_perm = {games[i]: float(base_y[perm][i]) for i in range(len(games))}
            for L in LAYERS:
                v, diag = fit_d_inc(X[L], meta, d1c_perm)
                if diag.get("status") != "OK":
                    raise RuntimeError(f"[{model}] slot{slot} L{L}: d_perm {diag}")
                d_perm = _unit(v)
                cos_dperm_dinc = float(d_perm @ d_inc_true[L])
                cos_removed = float(d_perm @ ellhat)
                perp = d_perm - cos_removed * ellhat
                perp_norm_before = float(np.linalg.norm(perp))
                perp = _unit(perp)
                key = f"d_inc_perm{slot}_l{L}"
                directions[key] = perp.astype(np.float32)
                per_dir[key] = {
                    "slot": slot, "seed": int(sd), "layer": L,
                    "corr_yperm_y": slot_corr[slot],
                    "cos_dperm_dinc_true": cos_dperm_dinc,
                    "cos_removed_ell": cos_removed,
                    "perp_norm_before_renorm": perp_norm_before,
                    "cos_perm_perp_ell": float(perp @ ellhat),        # MUST be ~0
                    "cos_perm_perp_dinc_true": float(perp @ d_inc_true[L]),
                    "norm": float(np.linalg.norm(directions[key])),
                }
        mdir = OUT / model
        mdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mdir / "directions.npz",
                            layers=np.array(LAYERS, dtype=np.int32),
                            **directions)
        manifest = {
            "method": "unit(d_perm - (d_perm.ell)ell); d_perm = fit_d_inc on ACROSS-GAME game-level "
                      "permuted delta1_c (sign-aligned to permuted target); ell = norm.weight * "
                      "(W_U[' J'] - W_U[' P']) unit-normalized (checkpoint bf16->fp32).",
            "model": model, "repo": REPO[model], "git_commit": commit,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "layers": list(LAYERS), "n_fit_games": len(games),
            "corr_threshold": CORR_THRESH,
            "within_family_dropped": "Step-0 gate: median|cos(within-fam d_perm,d_inc_true)|>0.6 all "
                                     "models/layers -> across-game is the only permuted control.",
            "jp_token_ids": {"J": jid, "P": pid},
            "ell_validation_vs_committed_perp_manifest": ell_val,
            "slots": {f"perm{slot}": {"seed": int(sd), "corr_yperm_y": slot_corr[slot]}
                      for slot, sd in enumerate(slot_seeds)},
            "seed_search_log": tried,
            "per_direction": per_dir,
            "direction_keys": sorted(directions),
            "vector_norms": {k: float(np.linalg.norm(v)) for k, v in directions.items()},
        }
        (mdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        z.close()
        print(f"[{model}] slots={[('perm%d' % i, s) for i, s in enumerate(slot_seeds)]} "
              f"corr={[round(c,4) for c in slot_corr]}")
        for k in sorted(per_dir):
            pd_ = per_dir[k]
            print(f"   {k}: seed={pd_['seed']} cos(d_perm,d_inc)={pd_['cos_dperm_dinc_true']:+.4f} "
                  f"cos_removed(ell)={pd_['cos_removed_ell']:+.4f} "
                  f"cos(perp,ell)={pd_['cos_perm_perp_ell']:+.2e} "
                  f"cos(perp,d_inc)={pd_['cos_perm_perp_dinc_true']:+.4f}")
        print(f"   wrote {mdir/'directions.npz'} + manifest.json")


if __name__ == "__main__":
    main()
