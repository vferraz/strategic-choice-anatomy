#!/usr/bin/env python3
"""ONE-TIME heavy pass: extract decision/commit-slot residuals from the corrected Akata substrate
into analysis/layer_b/_data/ so every downstream build is CPU-cheap and never re-reads it.

Per (model, game) we open acts.npz ONCE and pull:
  - baseline cache : P1 baseline decision slot at ALL capture layers (key p1_baseline_cb{cb}_l{L})
  - cue cache      : P1 baseline + 5 trait cues + placebo at the DEEPEST layer only
                     (key p1_{condition}_cb{cb}_l{Ldeep}; condition in results.parquet, e.g.
                      "cue_risk_aversion" -> key prefix "p1_cue_risk_aversion")

Decision/commit-slot residuals only -- never _seq keys. P2 residuals are NOT needed
for the descriptive Layer B figures (the opponent-incentive probe decodes sign(Delta2c) from P1's
own residual), so we cache P1 only.

The corrected Akata recollection has 4 counterbalance cells per game. Baseline meta matches
the analysis/block_b/levelk_geometry_oneshot.load_p1_baseline contract
(reset index, X rows aligned 1:1) so it is directly usable as a run_n5/run_n6 ``pre`` after
attach_manifest. Idempotent: skip a model whose _DONE marker exists unless --force.

Run (smoke first, then full):
  .venv/bin/python analysis/layer_b/build_residual_cache.py --models gptoss --smoke
  .venv/bin/python analysis/layer_b/build_residual_cache.py            # all 4 models, full
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_b import lib  # noqa: E402


def _incentive_targets():
    """Per-game ``delta1c`` / ``delta2c`` / ``stim_control_cell00`` for the baseline meta.

    These three columns were present in the 2026-07-10 cache generation and dropped from a
    later one; three downstream scripts still read them off the cached meta
    (``rebuild/rebuild_bridge_variants.py``, ``rebuild/rebuild_missing_tables.py``,
    ``rebuild/rebuild_round2.py``), so a fresh cache broke all three.

    The definitions are IMPORTED, not re-derived, from ``rebuild_missing_tables._targets()``
    -- the single authority for all three (playbook rule 5). Verified against the 2026-07-10
    cache as an oracle before this was written: all three columns reproduce EXACTLY
    (max |delta| = 0.0) on all four models, 2304 rows. That check settled all three open
    questions about them: ``delta1c``/``delta2c`` are the CONTINUOUS canonical-signed gaps
    rather than their signs; ``delta1c`` comes from
    ``data/results/layer_c/incentive_delta1c.csv``; and ``stim_control_cell00`` is the P1
    payoff rank at canonical cell (0,0), balanced 48/48/48 -- the same quantity
    ``lib.target_table()`` calls ``stim_cell00``.

    The import points at the ``rebuild/`` tree rather than the other way round because that
    is where the definition lives; duplicating it here is exactly the drift this restores.
    """
    tgt = __import__("analysis.layer_b.rebuild.rebuild_missing_tables",
                    fromlist=["_targets"])._targets()
    return (tgt["delta1c"].to_dict(), tgt["delta2c"].to_dict(),
            tgt["stim_control_cell00"].to_dict())


def _canon_maps():
    feats = pd.read_csv(lib.FEATURES)
    c1 = {r.game_code: int(r.canonical_action_p1) for r in feats.itertuples()
          if pd.notna(r.canonical_action_p1)}
    c2 = {r.game_code: int(r.canonical_action_p2) for r in feats.itertuples()
          if pd.notna(r.canonical_action_p2)}
    return c1, c2


def _game_dirs(model: str):
    yield from lib.game_dirs(model)


def _clear_model_cache(model: str) -> None:
    for root in (lib.BASE_CACHE, lib.CUE_CACHE):
        pats = [f"meta_{model}.parquet", f"X_{model}_l*.npy", f"_DONE_{model}"]
        for pat in pats:
            for p in root.glob(pat):
                if p.is_file():
                    p.unlink()


def build_model(model: str, *, smoke: bool = False, force: bool = False) -> None:
    if not force and lib.baseline_done(model) and lib.cues_done(model):
        print(f"[{model}] cache present (_DONE); skip. Use --force to rebuild.")
        return
    if force:
        _clear_model_cache(model)
    layers = lib.capture_layers(model)
    if smoke:
        layers = lib.steer_layers(model)             # few layers for a fast structural check
    Ldeep = lib.deepest_layer(model)
    if Ldeep not in layers:
        layers = sorted(set(layers) | {Ldeep})
    c1map, c2map = _canon_maps()
    d1map, d2map, scmap = _incentive_targets()
    conds = lib.ALL_CONDS                            # baseline + 5 cues + placebo
    decisions = lib.decision_rows(model, ok_only=True)

    base_meta: list[dict] = []
    base_X: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    cue_meta: list[dict] = []
    cue_X: list[np.ndarray] = []

    t0 = time.time()
    games = list(_game_dirs(model))
    if smoke:
        games = games[:8]
    for gi, d in enumerate(games):
        g = d.name
        p1 = decisions[(decisions["game_code"] == g) & (decisions["player"] == 1)]
        c1 = c1map.get(g, -1)
        c2 = c2map.get(g, -1)
        z = np.load(d / "acts.npz")
        try:
            files = set(z.files)
            # ---- baseline at ALL layers ----
            bl = p1[(p1["condition"] == "baseline") & (p1["decoded_action"].isin([0, 1]))]
            for _, r in bl.iterrows():
                cb = int(r["cb"])
                keys = {L: f"p1_baseline_cb{cb}_l{L}" for L in layers}
                if not all(k in files for k in keys.values()):
                    continue
                for L in layers:
                    base_X[L].append(np.asarray(z[keys[L]], dtype=np.float32))
                base_meta.append(dict(game_code=g, counterbalance_id=cb,
                                      decoded_action=int(r["decoded_action"]),
                                      decoded_label=str(r["decoded_label"]),
                                      emitted_label=str(r["decoded_label"]),
                                      action_source=str(r["action_source"]),
                                      commit_type=r.get("commit_type", np.nan),
                                      stated_p_act0=r.get("stated_p_act0", np.nan),
                                      prob_source=r.get("prob_source", np.nan),
                                      canonical_action_p1=int(c1), canonical_action_p2=int(c2),
                                      delta1c=float(d1map[g]), delta2c=float(d2map[g]),
                                      stim_control_cell00=int(scmap[g])))
            # ---- baseline + cues at DEEPEST only ----
            for cond in conds:
                cc = p1[(p1["condition"] == cond) & (p1["decoded_action"].isin([0, 1]))]
                prefix = f"p1_{cond}"                  # baseline -> p1_baseline ; cue_X -> p1_cue_X
                for _, r in cc.iterrows():
                    cb = int(r["cb"])
                    key = f"{prefix}_cb{cb}_l{Ldeep}"
                    if key not in files:
                        continue
                    cue_X.append(np.asarray(z[key], dtype=np.float32))
                    cue_meta.append(dict(game_code=g, counterbalance_id=cb, condition=cond,
                                         decoded_action=int(r["decoded_action"]),
                                         decoded_label=str(r["decoded_label"]),
                                         emitted_label=str(r["decoded_label"]),
                                         action_source=str(r["action_source"]),
                                         commit_type=r.get("commit_type", np.nan),
                                         stated_p_act0=r.get("stated_p_act0", np.nan),
                                         prob_source=r.get("prob_source", np.nan),
                                         canonical_action_p1=int(c1)))
        finally:
            z.close()
        if (gi + 1) % 24 == 0:
            print(f"  [{model}] {gi + 1}/{len(games)} games  ({time.time() - t0:.0f}s)")

    # ---- persist baseline ----
    bmeta = pd.DataFrame(base_meta).reset_index(drop=True)
    bmeta.to_parquet(lib.BASE_CACHE / f"meta_{model}.parquet")
    for L in layers:
        X = np.vstack(base_X[L]).astype(np.float32) if base_X[L] else np.zeros((0, 0), np.float32)
        np.save(lib.BASE_CACHE / f"X_{model}_l{L}.npy", X)
    (lib.BASE_CACHE / f"_DONE_{model}").write_text(
        f"rows={len(bmeta)} layers={len(layers)} smoke={smoke}\n")

    # ---- persist cues (deepest) ----
    cmeta = pd.DataFrame(cue_meta).reset_index(drop=True)
    cmeta.to_parquet(lib.CUE_CACHE / f"meta_{model}.parquet")
    Xc = np.vstack(cue_X).astype(np.float32) if cue_X else np.zeros((0, 0), np.float32)
    np.save(lib.CUE_CACHE / f"X_{model}_l{Ldeep}.npy", Xc)
    (lib.CUE_CACHE / f"_DONE_{model}").write_text(
        f"rows={len(cmeta)} layer={Ldeep} conds={len(conds)} smoke={smoke}\n")

    d0 = base_X[layers[0]][0].shape[0] if base_X[layers[0]] else 0
    print(f"[{model}] DONE in {time.time() - t0:.0f}s | baseline {len(bmeta)} rows x {d0} dims "
          f"x {len(layers)} layers | cues {len(cmeta)} rows @ l{Ldeep} "
          f"({cmeta['condition'].nunique() if len(cmeta) else 0} conds)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract one-shot decision-slot residual cache.")
    ap.add_argument("--models", nargs="*", default=list(lib.MODELS))
    ap.add_argument("--smoke", action="store_true", help="few layers, 8 games (structural check)")
    ap.add_argument("--force", action="store_true", help="rebuild even if _DONE present")
    args = ap.parse_args()
    models = [m for m in args.models if (lib.SUBSTRATE / m).exists()]
    print(f"PRIMARY_ROOT_USED = {lib.SUBSTRATE}")
    print(f"models = {models}  smoke = {args.smoke}")
    for m in models:
        build_model(m, smoke=args.smoke, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
