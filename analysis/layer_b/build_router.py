#!/usr/bin/env python3
"""S-router — descriptive GPT-OSS router decodability on the corrected Akata root.

Reads ``$SCA_DATA_ROOT/substrate/gptoss/{game}/router.npz`` and scores saved router
gate logits with game-grouped CV. This is descriptive evidence only: router state is read,
not edited, and no router do()-claim is made.

Outputs: tables/s_router_decodability.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402


TARGETS = [
    ("canonical_action", "Realized canonical choice"),
    ("sign_delta1c", "Incentive sign (Delta1c)"),
    ("sign_delta2c", "Opponent incentive sign (Delta2c)"),
]


def _router_panel(layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    dec = lib.decision_rows("gptoss", ok_only=True)
    dec = dec[(dec["player"] == 1) & (dec["condition"] == "baseline")].copy()
    rows, X = [], []
    for d in lib.game_dirs("gptoss"):
        g = d.name
        sub = dec[dec["game_code"] == g]
        if sub.empty or not (d / "router.npz").exists():
            continue
        with np.load(d / "router.npz") as z:
            for _, r in sub.iterrows():
                cb = int(r["cb"])
                key = f"p1_baseline_cb{cb}_l{layer}_gate"
                if key not in z.files:
                    continue
                X.append(np.asarray(z[key], dtype=np.float32))
                rows.append({
                    "game_code": g,
                    "counterbalance_id": cb,
                    "decoded_action": int(r["decoded_action"]),
                    "commit_type": r.get("commit_type", np.nan),
                })
    meta = pd.DataFrame(rows)
    arr = np.vstack(X).astype(np.float32) if X else np.zeros((0, 0), np.float32)
    return meta, arr


def build() -> pd.DataFrame:
    tt = lib.target_table()
    rows = []
    for L in lib.router_layers():
        meta, X = _router_panel(L)
        if meta.empty:
            continue
        meta = meta.merge(tt, on="game_code", how="left", suffixes=("", "_tt"))
        meta["canonical_action"] = (
            meta["decoded_action"].astype(int) == meta["canonical_action_p1"].astype(int)
        ).astype(int)
        games = meta["game_code"].to_numpy()
        for key, label in TARGETS:
            y = pd.to_numeric(meta[key], errors="coerce")
            keep = y.notna().to_numpy()
            if keep.sum() < 2 * lib.N_SPLITS or y[keep].nunique() < 2:
                rows.append({
                    "model": "gptoss", "layer": L, "target": key, "label": label,
                    "auc": np.nan, "lo": np.nan, "hi": np.nan, "shuffle_auc": np.nan,
                    "n": int(keep.sum()), "n_games": int(pd.unique(games[keep]).size),
                })
                continue
            yy = y[keep].astype(int).to_numpy()
            gg = games[keep]
            xx = X[keep]
            r = lib.probe_auc(xx, yy, gg, pca_k=min(32, xx.shape[1]),
                              n_splits=lib.N_SPLITS, seed=lib.CV_SEED, n_boot=lib.N_BOOT)
            rng = np.random.default_rng(lib.CV_SEED + 17 + L)
            ysh = rng.permutation(yy)
            rs = lib.probe_auc(xx, ysh, gg, pca_k=min(32, xx.shape[1]),
                               n_splits=lib.N_SPLITS, seed=lib.CV_SEED, n_boot=500)
            rows.append({
                "model": "gptoss", "layer": L, "target": key, "label": label,
                "auc": r["auc"], "lo": r["auc_lo"], "hi": r["auc_hi"],
                "shuffle_auc": rs["auc"], "n": r["n"], "n_games": r["n_games"],
            })
        best = [x for x in rows if x["layer"] == L and np.isfinite(x["auc"])]
        if best:
            msg = "  ".join(f"{r['target']}={r['auc']:.3f}" for r in best)
            print(f"[GPT-OSS router L{L}] {msg}")
    out = pd.DataFrame(rows)
    out.to_csv(lib.TAB_DIR / "s_router_decodability.csv", index=False)
    print(f"wrote {lib.TAB_DIR/'s_router_decodability.csv'}")
    return out


if __name__ == "__main__":
    build()
