#!/usr/bin/env python3
"""Fig S-distributed (N5) — distributed vs localized code.

Pre-empts the "is it a few memorized neurons?" objection. For the decision axis and the incentive
axis, per model: scale-fair participation ratio (PR/d) on standardized logistic weights, and the
greedy-ablation curve (fraction of dims to reach 90% of full AUC). The cross-model contrast is the
point: a distributed code that ISN'T recruited (Llama) vs distributed AND recruited (Qwen-I).

Reuses analysis/block_b/levelk_geometry_oneshot.run_n5 verbatim (N5 machinery), fed the cached
P1-baseline residuals (steer layers only -- N5 is the expensive test). Supplementary, supporting.

  .venv/bin/python analysis/layer_b/build_distributed.py
Outputs: tables/s_distributed.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


from analysis.layer_b import lib  # noqa: E402
from analysis.layer_b.levelk_geometry_oneshot import attach_manifest, run_n5  # noqa: E402


def build() -> None:
    frames = []
    for m in lib.MODELS:
        layers = sorted(lib.steer_layers(m))
        meta, X = lib.load_baseline(m, layers)
        em = attach_manifest(meta, m)
        frames.append(run_n5(m, layers, pre=(em, X)))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(lib.TAB_DIR / "s_distributed.csv", index=False)
    print(f"wrote {lib.TAB_DIR/'s_distributed.csv'}")


if __name__ == "__main__":
    build()
