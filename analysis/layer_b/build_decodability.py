#!/usr/bin/env python3
"""B1a (deepest-layer decodability + controls) and B1d (equilibrium-structure 3-class).

Organizing principle: decodability proves the game is ENCODED, never competence. So this panel
leads with the controls visible -- a perfectly-balanced stimulus floor (cell(0,0) rank, 3-class)
and the Llama label-axis control -- and the README forbids ranking models by absolute AUC (the
cb density inflates all magnitudes; report per-model PATTERN + cross-model RANK).

All probes: deepest layer, P1 baseline decision slot, game-grouped CV (folds by GAME),
bootstrap-by-game CI. Binary -> probe_auc; 3-class -> probe_auc_multiclass (macro-OvR + per-class).

Incentive signs are the UNIFORM-belief (q=0.5) stimulus version (lib.target_table): a pure
function of the payoff matrix, model-independent (the geometry panels use the empirical-belief
Delta1c instead -- see build_geometry.py).

  .venv/bin/python analysis/layer_b/build_decodability.py
Outputs: tables/b1_decodability.csv, tables/b1_equilibrium_structure.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402

# binary probes: (probe key, label, source). source 'row' = from baseline meta; 'game' = target_table col
BINARY_PROBES = [
    ("canonical_action", "Canonical action", "row"),     # decision readout: model chose canonical
    ("sign_delta1c", "Incentive sign (Δ1c)", "game"),     # stimulus: own incentive favours canonical
    ("dominant_action_p1", "Dominant action", "game"),    # stimulus, defined on 72 games
    ("sign_delta2c", "Opponent incentive (Δ2c)", "game"), # one-shot H3 analog (decoded from P1 residual)
    ("label_axis_J", "Emitted label = J", "row"),         # J/P label-axis control
]


def build() -> None:
    tt = lib.target_table()
    rows, eq_rows = [], []
    for m in lib.MODELS:
        L = lib.deepest_layer(m)
        meta, Xd = lib.load_baseline(m, [L])
        X = np.asarray(Xd[L], dtype=np.float32)
        meta = lib.attach_baseline_commit_type(meta, m)
        meta = meta.merge(tt, on="game_code", how="left", suffixes=("", "_tt"))
        games = meta["game_code"].to_numpy()
        # row-level targets
        meta["canonical_action"] = (meta["decoded_action"] == meta["canonical_action_p1"]).astype(int)
        if m == "gptoss":
            # Mixed commitments are captured at a different site and their binary action is a
            # seeded resolution, not a literal commitment.  They remain available to the
            # game-target probes below but are not a valid coming-choice target.
            meta.loc[meta["commit_type"] != "pure", "canonical_action"] = np.nan
        lab = meta["decoded_label"].fillna("").astype(str).str.upper()
        meta["label_axis_J"] = np.where(lab.isin(["J", "P"]), (lab == "J").astype(int), np.nan)

        for key, label, src in BINARY_PROBES:
            y = pd.to_numeric(meta[key], errors="coerce")
            keep = y.notna().to_numpy()
            if keep.sum() < 2 * lib.N_SPLITS or y[keep].nunique() < 2:
                rows.append(dict(model=m, probe=key, label=label, auc=np.nan, lo=np.nan, hi=np.nan,
                                 n=int(keep.sum()), n_games=int(pd.unique(games[keep]).size),
                                 base_rate=float(y[keep].mean()) if keep.any() else np.nan))
                continue
            r = lib.probe_auc(X[keep], y[keep].astype(int).to_numpy(), games[keep],
                              pca_k=lib.PCA_K, n_splits=lib.N_SPLITS, seed=lib.CV_SEED, n_boot=lib.N_BOOT)
            if m == "gptoss" and key in {"canonical_action", "label_axis_J"}:
                row_scope = "pure_commitment"
            elif src == "game":
                row_scope = "all_p1_baseline_captures"
            else:
                row_scope = "parsed_generated_choice"
            rows.append(dict(model=m, probe=key, label=label, auc=r["auc"], lo=r["auc_lo"],
                             hi=r["auc_hi"], n=r["n"], n_games=r["n_games"],
                             base_rate=r["base_rate"], row_scope=row_scope))

        # stimulus-control floor: cell(0,0) payoff rank, 3-class {1,2,3} (balanced 48/48/48)
        sc = lib.probe_auc_multiclass(X, meta["stim_cell00"].to_numpy(), games,
                                      classes=[1, 2, 3], pca_k=lib.PCA_K, n_boot=lib.N_BOOT)
        rows.append(dict(model=m, probe="stim_control_cell00", label="Stimulus control (cell(0,0) rank)",
                         auc=sc["macro_auc"], lo=sc["macro_lo"], hi=sc["macro_hi"],
                         n=sc["n"], n_games=sc["n_games"], base_rate=np.nan,
                         row_scope="all_p1_baseline_captures"))

        # B1d equilibrium structure: num_pure_ne 3-class {0=MP, 1=OD, 2=CO}
        eq = lib.probe_auc_multiclass(X, meta["num_pure_ne"].to_numpy(), games,
                                      classes=[0, 1, 2], pca_k=lib.PCA_K, n_boot=lib.N_BOOT)
        eq_rows.append(dict(model=m, macro_auc=eq["macro_auc"], macro_lo=eq["macro_lo"],
                            macro_hi=eq["macro_hi"], bal_acc=eq["bal_acc"],
                            auc_MP=eq["per_class"].get("0", np.nan),
                            auc_OD=eq["per_class"].get("1", np.nan),
                            auc_CO=eq["per_class"].get("2", np.nan),
                            n=eq["n"], n_games=eq["n_games"],
                            stim_floor_macro_auc=sc["macro_auc"]))
        print(f"[{lib.SHORT[m]}] decode: "
              + "  ".join(f"{r['probe']}={r['auc']:.3f}" for r in rows if r["model"] == m
                          and np.isfinite(r["auc"]))
              + f"  | eq3class macro={eq['macro_auc']:.3f} (floor {sc['macro_auc']:.3f})")

    pd.DataFrame(rows).to_csv(lib.TAB_DIR / "b1_decodability.csv", index=False)
    pd.DataFrame(eq_rows).to_csv(lib.TAB_DIR / "b1_equilibrium_structure.csv", index=False)
    print(f"wrote {lib.TAB_DIR/'b1_decodability.csv'} and {lib.TAB_DIR/'b1_equilibrium_structure.csv'}")


if __name__ == "__main__":
    build()
