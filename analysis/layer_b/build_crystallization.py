#!/usr/bin/env python3
"""B1b — the decision crystallizes across depth.

Per model, per captured layer: out-of-fold AUC of the canonical-action decision readout
(aligned = decoded_action == canonical_action_p1), folds by game, bootstrap-by-game CI.
We report AUC *and* the GAIN OVER THE EMBEDDING (layer-0) floor -- the embedding already
encodes the stimulus, so the depth-emergent, attributable signal is auc(L) - auc(L0).

Storyline beat (surfaced in the summary + caption): Qwen-base CRYSTALLIZES the decision
representation across depth then COLLAPSES before commit -- represent-without-recruit within a
single model. peak_auc - final_auc quantifies the collapse.

GPT-OSS residual is plotted but its nulls are router-mediated (see Fig S-router); the dense
residual under-reads an MoE model (docs/GPTOSS_LAYER_B_HANDLING.md).

  .venv/bin/python analysis/layer_b/build_crystallization.py
Outputs: tables/b1_crystallization.csv (per layer), tables/b1_crystallization_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

ONSET_AUC = 0.65  # mirrors c1_crystallization.py:64 / levelk_geometry_oneshot.ONSET_AUC


def build(models=None) -> None:
    models = list(models or lib.MODELS)
    curve_rows, summ_rows = [], []
    for m in models:
        meta, X = lib.load_baseline(m)                     # all cached layers
        meta = lib.attach_baseline_commit_type(meta, m)
        layers = sorted(X.keys())
        Lmax = max(layers)
        keep = (meta["commit_type"].eq("pure").to_numpy()
                if m == "gptoss" else np.ones(len(meta), dtype=bool))
        games = meta.loc[keep, "game_code"].to_numpy()
        y = (meta.loc[keep, "decoded_action"] == meta.loc[keep, "canonical_action_p1"]).astype(int).to_numpy()
        row_scope = "pure_commitment" if m == "gptoss" else "parsed_generated_choice"
        per = {}
        for L in layers:
            r = lib.probe_auc(np.asarray(X[L], np.float32)[keep], y, games,
                              pca_k=lib.PCA_K, n_splits=lib.N_SPLITS, seed=lib.CV_SEED, n_boot=1000)
            per[L] = r
        a0 = per[0]["auc"] if 0 in per else np.nan
        for L in layers:
            r = per[L]
            curve_rows.append(dict(model=m, layer=L, depth_frac=L / Lmax, auc=r["auc"],
                                   lo=r["auc_lo"], hi=r["auc_hi"],
                                   auc_gain_over_embedding=r["auc"] - a0,
                                   row_scope=row_scope, n_obs=r["n"], n_games=r["n_games"]))
        aucs = np.array([per[L]["auc"] for L in layers], float)
        peak_i = int(np.nanargmax(aucs))
        peak_L, peak = layers[peak_i], float(aucs[peak_i])
        final = float(aucs[-1])
        onset = next((L / Lmax for L in layers if np.isfinite(per[L]["auc"]) and per[L]["auc"] >= ONSET_AUC),
                     np.nan)
        summ_rows.append(dict(model=m, embedding_auc=a0, onset_depth_frac=onset,
                              peak_auc=peak, peak_depth_frac=peak_L / Lmax, final_auc=final,
                              collapse=peak - final, row_scope=row_scope,
                              n_obs=int(keep.sum()), n_games=int(pd.unique(games).size)))
        print(f"[{lib.SHORT[m]}] embed={a0:.3f} onset={onset if not np.isnan(onset) else float('nan'):.3f} "
              f"peak={peak:.3f}@{peak_L/Lmax:.2f} final={final:.3f} collapse(peak-final)={peak-final:+.3f}")

    curve = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summ_rows)
    curve_path = lib.TAB_DIR / "b1_crystallization.csv"
    summary_path = lib.TAB_DIR / "b1_crystallization_summary.csv"
    if curve_path.exists() and set(models) != set(lib.MODELS):
        old = pd.read_csv(curve_path)
        curve = pd.concat([old[~old.model.isin(models)], curve], ignore_index=True)
    if summary_path.exists() and set(models) != set(lib.MODELS):
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old.model.isin(models)], summary], ignore_index=True)
    curve["row_scope"] = curve.get("row_scope", pd.Series(index=curve.index, dtype=object)).fillna(
        "parsed_generated_choice"
    )
    curve["n_obs"] = curve.get("n_obs", pd.Series(index=curve.index, dtype=float)).fillna(576).astype(int)
    curve["n_games"] = curve.get("n_games", pd.Series(index=curve.index, dtype=float)).fillna(144).astype(int)
    summary["row_scope"] = summary.get(
        "row_scope", pd.Series(index=summary.index, dtype=object)
    ).fillna("parsed_generated_choice")
    summary["n_obs"] = summary.get("n_obs", pd.Series(index=summary.index, dtype=float)).fillna(576).astype(int)
    summary["n_games"] = summary.get(
        "n_games", pd.Series(index=summary.index, dtype=float)
    ).fillna(144).astype(int)
    curve.sort_values(["model", "layer"]).to_csv(curve_path, index=False)
    summary.sort_values("model").to_csv(summary_path, index=False)
    print(f"wrote {lib.TAB_DIR/'b1_crystallization.csv'} (+summary)")


if __name__ == "__main__":
    build(sys.argv[1:] or None)
