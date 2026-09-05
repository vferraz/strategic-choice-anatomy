#!/usr/bin/env python3
"""Layer-A foundation validation gate on corrected-root decisions.

This gate intentionally does *not* read the old slot-argmax behaviour helpers.
It validates the caches produced by ``build_data_layer.py``:

* ``layerA_game_level.parquet``: per-game generated/committed behaviour;
* ``layerA_cells_p1baseline.parquet``: usable P1 baseline generated/committed cells.

Dense parse failures and GPT-OSS no-commit cells are dropped. GPT-OSS mixed-strategy
cells are included after their resolved 0/1 ``realized_action``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from strategic_anatomy.config import results_root, taxonomy_dir

ROOT = Path(__file__).resolve().parents[2]
from analysis.layer_a.layer1_lambda_delta1 import fit_lambda, boot_cluster_lambda  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402

DATA = results_root() / "layer_a" / "_data"
GL = DATA / "layerA_game_level.parquet"
CELLS = DATA / "layerA_cells_p1baseline.parquet"
MASTER = taxonomy_dir() / "human_game_master_per_canonical.csv"

ORDER = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]

PCANON_WINDOWS = {
    "qwen": (0.68, 0.75),
    "qwen_instruct": (0.69, 0.76),
    "llama31_instruct": (0.70, 0.78),
    "gptoss": (0.85, 0.93),
}

LAMBDA_Q05_WINDOWS = {
    "qwen": (1.70, 2.35),
    "qwen_instruct": (1.60, 2.20),
    "llama31_instruct": (1.60, 2.20),
    "gptoss": (0.65, 1.10),
}


def _in_window(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def main() -> int:
    cfg = CS.assert_complete_pm1()
    gl = pd.read_parquet(GL)
    cells = pd.read_parquet(CELLS)
    master = pd.read_csv(MASTER)

    print("=== corrected substrate root ===")
    print(f"  root: {CS.CORRECTED_ROOT}")
    print(f"  configs: {len(cfg)}; rows/game by model:")
    print(cfg.groupby("model")["n_rows"].first().reindex(ORDER).to_string())

    print("=== corrected cache counts ===")
    print(f"  game_level: models={gl['model'].nunique()} games={gl['game_code'].nunique()} rows={len(gl)}")
    print(f"  cells:      models={cells['model'].nunique()} games={cells['game_code'].nunique()} rows={len(cells)}")
    assert gl["model"].nunique() == 4 and gl["game_code"].nunique() == 144
    assert cells["model"].nunique() == 4 and cells["game_code"].nunique() == 144
    assert set(ORDER) <= set(gl["model"]) and set(ORDER) <= set(cells["model"])

    print("\n=== cell coverage (usable generated/commit cells; no argmax fallback) ===")
    coverage = cells.groupby("model").agg(n_cells=("y_canon", "size"), n_games=("game_code", "nunique"))
    print(coverage.reindex(ORDER).to_string())
    assert (coverage["n_games"] == 144).all()
    assert (coverage["n_cells"] == 576).all(), "expected 4 cells x 144 games per model"

    print("\n=== GPT-OSS mixed-strategy provenance ===")
    gpt = CS.normalized_results("gptoss")
    print(gpt["commit_type"].value_counts(dropna=False).to_string())
    assert int(gpt["commit_type"].eq("mixed").sum()) > 0
    assert int(gpt["commit_type"].eq("none").sum()) == 3

    print("\n=== Nagel axis-flip sanity ===")
    print(f"  mean nagel_frac_choose_act0_canonical = {master['nagel_frac_choose_act0_canonical'].mean():.3f}")
    print(f"  mean nagel_pcanon                    = {master['nagel_pcanon'].mean():.3f}")
    assert 0.70 <= master["nagel_pcanon"].mean() <= 0.80, "Nagel flip sanity failed"

    print("\n=== p(canonical) cache canaries ===")
    means = gl.groupby("model")["p_canon"].mean().reindex(ORDER)
    for model, value in means.items():
        lo, hi = PCANON_WINDOWS[model]
        ok = _in_window(float(value), lo, hi)
        print(f"  {model:18s} {value:.3f}  expected [{lo:.2f}, {hi:.2f}]  {'OK' if ok else 'FAIL'}")
        assert ok, f"{model} p_canon={value:.3f} outside corrected-source window"

    print("\n=== q=0.5 action-space lambda canaries ===")
    for model in ORDER:
        d = cells[cells["model"].eq(model)]
        lam, alpha = fit_lambda(d["delta1c_q05"].to_numpy(float), d["y_canon"].to_numpy(int), "hard")
        lo, hi = boot_cluster_lambda(
            d["delta1c_q05"].to_numpy(float),
            d["y_canon"].to_numpy(int),
            d["game_code"].to_numpy(),
            "hard",
            n_boot=2000,
        )
        wlo, whi = LAMBDA_Q05_WINDOWS[model]
        ok = _in_window(float(lam), wlo, whi)
        print(f"  {model:18s} λ={lam:+.3f} [{lo:+.2f}, {hi:+.2f}]  "
              f"expected [{wlo:+.2f}, {whi:+.2f}]  {'OK' if ok else 'FAIL'}")
        assert ok, f"{model} lambda={lam:.3f} outside corrected-source window"

    print("\n=== verdict ===")
    print("  corrected integrated-root cache is paper-ready.")
    print("  GPT-OSS mixed cells are included as resolved actions; no-commit cells are dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
