#!/usr/bin/env python3
"""Flat cache of the GPT-OSS uniform-site recapture substrate, for the fusion geometry.

This is the consumer of the ``gptoss_recap`` deposit component. The main Layer-B residual
cache (``analysis/layer_b/build_residual_cache.py``) is built from ``substrate/``, where the
GPT-OSS read-out position is anchored to the model's own output — pure rows at the
commitment letter, mixed rows at the analysis->final transition. Cross-row geometry on that
substrate is capture-site confounded and is not a valid headline (``docs/METHODS.md``, the
uniform-site amendment). The recapture reads every row at the *same* site, so it needs its
own cache with its own target column.

Population and target (locked; see the amendment):

* rows          all 576 P1-baseline cells (144 games x 4 counterbalance forms)
* analysis mask ``use_in_neural_target == 1`` -> 541 rows = 525 pure + 16 stated-mixed;
                the 35 harness-imputed ``default_uniform`` rows are flagged out, never imputed
* target        ``p_canonical`` = the model's STATED policy mapped to P(canonical action);
                pure -> 0/1, mixed -> the stated probability, canonical-mapped

All 576 rows are cached and the mask is applied at run time, so the cache itself stays
scope-neutral and a sensitivity that wants the excluded rows can still reach them.

Row order is ``sort_values(["game_code", "counterbalance_id"])`` — sorted game-code strings,
counterbalance 0..3 within game. This order is part of the locked estimator: the permutation
null draws game blocks in it, so changing it changes every published p-value.

Outputs, into ``$SCA_DATA_ROOT/layer_b_cache/recap_baseline/``:
  meta_gptoss.parquet     576 rows x 7 columns
  X_gptoss_l{0..36}.npy   576 x 2880 float32, one file per residual layer

Router gate logits are NOT cached here — ``oss_router_fusion.py`` reads ``router.npz``
straight from the recap root, as it always has for the main substrate.

Run:  uv run python analysis/layer_b/fusion_associative/recap_cache.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategic_anatomy.config import gptoss_recap_root, layer_b_cache_root, require

MODEL = "gptoss"
N_LAYERS = 37
HIDDEN = 2880
N_ROWS = 576
N_GAMES = 144
N_TARGET = 541

META_COLS = [
    "game_code",
    "counterbalance_id",
    "commit_type",
    "prob_source",
    "p_canonical",
    "use_in_neural_target",
    "canonical_action_p1",
]

RECAP_ROOT = gptoss_recap_root()
OUT = layer_b_cache_root() / "recap_baseline"
DEFAULT_RECAP_ROOT, DEFAULT_OUT = RECAP_ROOT, OUT


def load_meta(recap_root: Path) -> pd.DataFrame:
    """P1-baseline metadata for every recap game, in the locked row order."""
    files = sorted(recap_root.glob("*/results.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No results.parquet under {recap_root}. Fetch the gptoss_recap component with\n"
            f"    python scripts/download_data.py --component gptoss_recap"
        )
    frames = [pd.read_parquet(f, columns=META_COLS + ["player", "condition"]) for f in files]
    meta = pd.concat(frames, ignore_index=True)
    meta = meta[(meta["player"] == 1) & (meta["condition"] == "baseline")][META_COLS]
    return meta.sort_values(["game_code", "counterbalance_id"]).reset_index(drop=True)


def check_invariants(meta: pd.DataFrame) -> None:
    """The population gates from the uniform-site amendment. Loud, not silent."""
    assert len(meta) == N_ROWS, f"expected {N_ROWS} P1-baseline rows, got {len(meta)}"
    assert meta["game_code"].nunique() == N_GAMES, "expected 144 games"
    assert sorted(meta["counterbalance_id"].unique().tolist()) == [0, 1, 2, 3], "expected cb 0..3"
    keep = meta[meta["use_in_neural_target"] == 1]
    assert len(keep) == N_TARGET, f"expected {N_TARGET} in-target rows, got {len(keep)}"
    counts = keep["prob_source"].value_counts().to_dict()
    assert counts.get("pure") == 525 and counts.get("stated") == 16, f"prob_source split {counts}"
    excluded = meta[meta["use_in_neural_target"] != 1]["prob_source"].value_counts().to_dict()
    assert excluded == {"default_uniform": 35}, f"excluded rows are not the 35 imputed ones: {excluded}"
    print(f"[recap] invariants OK: {N_ROWS} rows / {N_GAMES} games / cb0-3 | "
          f"mask {N_TARGET} = 525 pure + 16 stated | 35 default_uniform excluded")


def build_residuals(meta: pd.DataFrame, recap_root: Path) -> np.ndarray:
    """(576, 37, 2880) float32, rows aligned to ``meta``. One npz open per game."""
    X = np.empty((len(meta), N_LAYERS, HIDDEN), dtype=np.float32)
    row_of = {(r.game_code, int(r.counterbalance_id)): i for i, r in meta.iterrows()}
    l0_ref, l0_dev = None, 0.0
    games = sorted(meta["game_code"].unique())
    for gi, g in enumerate(games):
        with np.load(recap_root / g / "acts.npz") as z:
            for cb in range(4):
                i = row_of.get((g, cb))
                if i is None:
                    continue
                for L in range(N_LAYERS):
                    X[i, L] = z[f"p1_baseline_cb{cb}_l{L}"]
                v = np.asarray(z[f"p1_baseline_cb{cb}_l0"], dtype=np.float64)
                if l0_ref is None:
                    l0_ref = v
                else:
                    l0_dev = max(l0_dev, float(np.max(np.abs(v - l0_ref))))
        if gi % 36 == 0:
            print(f"[recap] loaded {gi + 1}/{len(games)} games", flush=True)
    # The uniform capture site is one shared token, so its layer-0 embedding must be
    # identical across every row; a non-zero deviation means the rows are not co-sited.
    print(f"[recap] layer-0 anchor max abs deviation across all {len(meta)} rows: {l0_dev:.3e}")
    if l0_dev != 0.0:
        raise AssertionError(f"layer-0 anchor is not invariant (max dev {l0_dev:.3e})")
    return X


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="build the GPT-OSS uniform-site recap cache")
    p.add_argument("--recap-root", default=str(DEFAULT_RECAP_ROOT),
                   help="recapture substrate root (default: $SCA_DATA_ROOT/gptoss_recap)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="cache output directory (default: $SCA_DATA_ROOT/layer_b_cache/recap_baseline)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = _parse_args(argv)
    recap_root, out = Path(a.recap_root), Path(a.out)
    # Construct-identity guard, same shape as the q05 steering analyzers: a non-default
    # input substrate must never be written under the name the released analyses read.
    if recap_root != DEFAULT_RECAP_ROOT and out == DEFAULT_OUT:
        sys.exit(f"[recap] REFUSING to write a non-default substrate (recap_root={recap_root}) "
                 f"into the canonical cache {DEFAULT_OUT}. Pass an explicit --out.")
    require(recap_root, "GPT-OSS recapture substrate")
    print(f"[recap] recap_root={recap_root}\n[recap] out={out}")

    meta = load_meta(recap_root)
    check_invariants(meta)
    X = build_residuals(meta, recap_root)

    out.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out / f"meta_{MODEL}.parquet")
    for L in range(N_LAYERS):
        np.save(out / f"X_{MODEL}_l{L}.npy", X[:, L])
    print(f"[recap] wrote meta_{MODEL}.parquet + {N_LAYERS} layer files -> {out}")


if __name__ == "__main__":
    main()
