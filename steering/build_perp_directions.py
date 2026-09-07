"""Build the letter-orthogonalized (ℓ-⊥) perp directions — the ``main_perp`` steering variant.

For each model: ``ell = unit(norm.weight * (W_U[' J'] - W_U[' P']))`` (``build_ell_hat``,
checkpoint reconstruction), then for the steered keys (``d_inc``, ``d_choice_perp``) at every
layer: ``perp = unit(d - (d·ell)ell)``. This is what the small-dose runner loads from
``--dirs-perp``. Mirrors the method recorded in the empirical perp manifest, verifies
``cos(perp, ell) ~ 0``, and records the incentive belief in the manifest.

Reads:   {--dirs-in}/{model}/directions.npz
Writes:  {--dirs-out}/{model}/{directions.npz, manifest.json}

Usage (q = 0.5 arm, the one the paper reports)::

    python steering/build_perp_directions.py \
        --dirs-in  "$SCA_DATA_ROOT/steering/directions/akata_q05" \
        --dirs-out "$SCA_DATA_ROOT/steering/directions/akata_q05_perp"

Needs the [gpu] extra: ``build_ell_hat`` reconstructs the unembedding from the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from steering.build_perm_directions import build_ell_hat, _unit  # exact reuse

MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
PERP_KEYS = ("d_inc", "d_choice_perp")   # the directions the runner steers as main_perp


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs-in", required=True,
                   help="direction npz root to orthogonalize (e.g. .../directions/akata_q05).")
    p.add_argument("--dirs-out", required=True,
                   help="output root (e.g. .../directions/akata_q05_perp).")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--belief", choices=("empirical", "q05"), default="q05",
                   help="incentive basis of --dirs-in; recorded in the manifest, never assumed.")
    a = p.parse_args()
    din, dout = Path(a.dirs_in), Path(a.dirs_out)
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        z = np.load(din / m / "directions.npz")
        ell, jid, pid = build_ell_hat(m)
        ell = np.asarray(ell, np.float64)
        ell /= np.linalg.norm(ell)
        layers = sorted({int(re.search(r"_l(\d+)$", k).group(1)) for k in z.files
                         if re.search(r"^d_inc_l\d+$", k)})
        out, per = {}, {}
        for L in layers:
            for base in PERP_KEYS:
                k = f"{base}_l{L}"
                if k not in z.files:
                    continue
                d = z[k].astype(np.float64)
                if np.linalg.norm(d) == 0:
                    # Gated direction: pass the zero vector through rather than fabricate one.
                    out[k] = z[k].astype(np.float32)
                    per[k] = {"status": "NOT_ESTIMABLE_passthrough"}
                    continue
                d = d / np.linalg.norm(d)
                cos_removed = float(d @ ell)
                dp = _unit(d - cos_removed * ell)
                out[k] = dp.astype(np.float32)
                per[k] = {"cos_removed": abs(cos_removed), "cos_after": float(abs(dp @ ell)),
                          "cos_with_orig": float(abs(dp @ d))}
        (dout / m).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dout / m / "directions.npz", **out)
        (dout / m / "manifest.json").write_text(json.dumps({
            "method": "unit(d - (d.ell)ell), ell = norm.weight * (W_U[' J'] - W_U[' P']) unit",
            "incentive_belief": a.belief, "jp_token_ids": {"J": jid, "P": pid},
            "base_npz": str(din / m / "directions.npz"), "keys": sorted(out),
            "per_direction": per}, indent=2))
        mx = max((v.get("cos_after", 0) for v in per.values()), default=0)
        print(f"{m:18s} perp keys={len(out)} layers={len(layers)}  "
              f"max cos(perp,ell) after = {mx:.2e}  (must be ~0)")
        z.close()
    print("PERP_DONE")


if __name__ == "__main__":
    main()
