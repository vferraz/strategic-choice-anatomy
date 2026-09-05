"""Rebuild the accepted within-model recruitment bridge and scoped sensitivities.

The bridge uses game-held-out incentive directions and game-level slopes.  Dense
models use their parsed generated choices.  GPT-OSS uses *pure commitments only*
for the accepted headline because mixed answers are captured at a different
post-reasoning site and their binary actions are seeded resolutions.  The former
heterogeneous 576-row estimate and a 108-game complete-pure-block estimate are
retained in ``within_model_bridge_gptoss_sensitivity.csv``.

Variants:
  V1_gamez    signed OOF incentive projection, game mean, z-scored across games;
  V2_strength absolute OOF projection;
  V3_perp     incentive direction orthogonalised to the decision direction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from analysis.layer_b.rebuild.rebuild_lib import MODELS, ROOT, games, meta, load_game_rows, realized_aligned, boot_ci
from strategic_anatomy.config import data_root, results_root

CAND = {
    "qwen": [30, 63, 65, 75, 78, 79, 80],
    "qwen_instruct": [30, 62, 65, 75, 78, 79, 80],
    "llama31_instruct": [30, 60, 65, 75, 78, 79, 80],
    "gptoss": [1, 6, 9, 22, 24, 35, 36],
}

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "out"
TABLES = results_root() / "layer_b" / "recruitment"
DATA = results_root() / "layer_b" / "recruitment" / "_data"
MAIN_TABLE = TABLES / "within_model_bridge.csv"
POINTS_TABLE = TABLES / "within_model_bridge_points.csv"
SENS_TABLE = TABLES / "within_model_bridge_gptoss_sensitivity.csv"
OUT.mkdir(parents=True, exist_ok=True)


def slope(y, x):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def residualize(v, on):
    on = np.asarray(on, float)
    A = np.c_[np.ones_like(on), on]
    beta, *_ = np.linalg.lstsq(A, np.asarray(v, float), rcond=None)
    return np.asarray(v, float) - A @ beta


def _estimate_scope(X, y, ok, grp, dlt, mask, *, model, layer, row_scope):
    X = np.asarray(X, np.float64)[mask]
    y, ok, grp, dlt = y[mask], ok[mask], grp[mask], dlt[mask]
    proj_s = np.full(len(y), np.nan)
    proj_p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, groups=grp):
        mu = X[tr].mean(0)
        Xt = X[tr] - mu
        d = Xt.T @ (dlt[tr] - dlt[tr].mean())
        n = np.linalg.norm(d)
        d = d / n if n > 0 else d
        proj_s[te] = (X[te] - mu) @ d
        valid = ok[tr]
        if (y[tr][valid] == 1).any() and (y[tr][valid] == 0).any():
            ddec = (
                X[tr][valid & (y[tr] == 1)].mean(0)
                - X[tr][valid & (y[tr] == 0)].mean(0)
            )
            nd = np.linalg.norm(ddec)
            if nd > 0:
                ddec /= nd
                dp = d - (d @ ddec) * ddec
                np_ = np.linalg.norm(dp)
                dp = dp / np_ if np_ > 0 else dp
                proj_p[te] = (X[te] - mu) @ dp

    gids = pd.unique(grp)
    yg = np.array([
        y[(grp == game) & ok].mean() if ((grp == game) & ok).any() else np.nan
        for game in gids
    ])
    dg = np.array([dlt[grp == game][0] for game in gids])
    rows, points = [], None
    for name, pr in (
        ("V1_gamez", proj_s),
        ("V2_strength", np.abs(proj_s)),
        ("V3_perp", proj_p),
    ):
        xraw = np.array([np.nanmean(pr[grp == game]) for game in gids])
        xg = (xraw - np.nanmean(xraw)) / np.nanstd(xraw)
        keep = np.isfinite(yg) & np.isfinite(xg)
        xk, yk, dk = xg[keep], yg[keep], dg[keep]
        raw = slope(yk, xk)
        par = slope(residualize(yk, dk), residualize(xk, dk))

        def st_raw(take):
            return slope(yk[take], xk[take])

        def st_par(take):
            return slope(
                residualize(yk[take], dk[take]),
                residualize(xk[take], dk[take]),
            )

        rlo, rhi = boot_ci(xk, st_raw)
        plo, phi = boot_ci(xk, st_par)
        rows.append(
            dict(
                model=model,
                layer=layer,
                variant=name,
                row_scope=row_scope,
                raw=raw,
                raw_lo=rlo,
                raw_hi=rhi,
                partial=par,
                partial_lo=plo,
                partial_hi=phi,
                n_rows=int(mask.sum()),
                n_games=int(keep.sum()),
            )
        )
        if name == "V1_gamez":
            points = pd.DataFrame(
                {
                    "model": model,
                    "game_code": gids,
                    "p_canonical": yg,
                    "inc_projection": xraw,
                    "delta1c_q05": dg,
                    "n_cells": [int((grp == game).sum()) for game in gids],
                    "inc_projection_z": xg,
                    "delta1c_q05_z": (dg - np.nanmean(dg)) / np.nanstd(dg),
                    "row_scope": row_scope,
                    "layer": layer,
                }
            )
    return rows, points


def run_model(model):
    M = meta()
    gl = games(model)
    layers = CAND[model]
    X_all = {L: [] for L in layers}
    y_rows, ok_rows, g_rows, d_rows, commit_rows = [], [], [], [], []
    for gi, game in enumerate(gl):
        X, df = load_game_rows(model, game, layers)
        y, ok = realized_aligned(df, model, M.loc[game, "canonical_action_p1"])
        for L in layers:
            X_all[L].append(X[L].astype(np.float32))
        y_rows.append(y)
        ok_rows.append(ok)
        g_rows.append(np.full(len(df), game, dtype=object))
        d_rows.append(np.full(len(df), M.loc[game, "delta1c"], dtype=float))
        commit_rows.append(
            df["commit_type"].astype(str).to_numpy()
            if model == "gptoss"
            else np.full(len(df), "parsed_generated_choice", dtype=object)
        )
        if gi % 48 == 0:
            print(f"[{model}] loaded {gi + 1}/{len(gl)}", flush=True)

    y = np.concatenate(y_rows)
    ok = np.concatenate(ok_rows)
    grp = np.concatenate(g_rows)
    dlt = np.concatenate(d_rows)
    commit = np.concatenate(commit_rows)
    if model == "gptoss":
        pure = commit == "pure"
        complete_games = [game for game in pd.unique(grp) if pure[grp == game].all()]
        scopes = (
            ("pure_commitment", pure),
            ("heterogeneous_all", np.ones(len(y), dtype=bool)),
            ("pure_complete_games", np.isin(grp, complete_games)),
        )
    else:
        scopes = (("parsed_generated_choice", np.ones(len(y), dtype=bool)),)

    all_rows, all_points = [], []
    for L in layers:
        X = np.concatenate(X_all[L]).astype(np.float64)
        for row_scope, mask in scopes:
            rows, points = _estimate_scope(
                X, y, ok, grp, dlt, mask, model=model, layer=L, row_scope=row_scope
            )
            all_rows.extend(rows)
            if L == max(layers):
                all_points.append(points)
            v1 = rows[0]
            print(
                f"[{model}] L{L} {row_scope}: raw {v1['raw']:+.3f} "
                f"[{v1['raw_lo']:+.3f},{v1['raw_hi']:+.3f}] | partial "
                f"{v1['partial']:+.3f} [{v1['partial_lo']:+.3f},{v1['partial_hi']:+.3f}]",
                flush=True,
            )
    out = pd.DataFrame(all_rows)
    pts = pd.concat(all_points, ignore_index=True)
    out.to_csv(OUT / f"bridge_variants_{model}.csv", index=False)
    pts.to_csv(OUT / f"bridge_points_{model}.csv", index=False)
    return out, pts


def _accepted_row(r):
    return {
        "model": r.model,
        "layer": int(r.layer),
        "slope_pcanonical_per_sd_neural_inc": r.raw,
        "ci_lo": r.raw_lo,
        "ci_hi": r.raw_hi,
        "partial_slope_control_delta1c": r.partial,
        "partial_ci_lo": r.partial_lo,
        "partial_ci_hi": r.partial_hi,
        "row_scope": r.row_scope,
        "n_rows": int(r.n_rows),
        "n_games": int(r.n_games),
    }


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)


def _update_gptoss_final_geometry_bridge(bridge, points):
    """Remove the heterogeneous-site GPT row from the legacy compact bridge table."""
    path = TABLES / "final_geometry_bridge.csv"
    if not path.exists():
        return
    md = pd.read_parquet(
        data_root() / "layer_b_cache" / "baseline" / "meta_gptoss.parquet"
    ).reset_index(drop=True)
    commits = []
    for game in games("gptoss"):
        d = pd.read_parquet(
            Path(ROOT) / "gptoss" / game / "results.parquet",
            columns=["player", "condition", "counterbalance_id", "commit_type"],
        )
        d = d[(d.player == 1) & (d.condition == "baseline")].copy()
        d["game_code"] = game
        commits.append(d[["game_code", "counterbalance_id", "commit_type"]])
    # Third site of the commit_type writer/reader desync (PATCH 1/2 class): a cache built by
    # the current build_residual_cache.py already carries commit_type, so this merge would
    # collide into commit_type_x/_y and the attribute access below would fail. Drop the
    # cache-borne copy and keep results.parquet as the authority.
    md = md.drop(columns=["commit_type"], errors="ignore")
    md = md.merge(
        pd.concat(commits, ignore_index=True),
        on=["game_code", "counterbalance_id"],
        validate="one_to_one",
    )
    pure = md.commit_type.eq("pure").to_numpy()
    X = np.load(
        data_root() / "layer_b_cache" / "baseline" / "X_gptoss_l36.npy"
    ).astype(float)[pure]
    y = md.loc[pure, "decoded_action"].eq(md.loc[pure, "canonical_action_p1"]).astype(int).to_numpy()
    d = md.loc[pure, "delta1c"].to_numpy(float)
    Xc = X - X.mean(0)
    d_dec = _unit(X[y == 1].mean(0) - X[y == 0].mean(0))
    dc = d - d.mean()
    d_inc = _unit((Xc.T @ dc) / float(dc @ dc))
    angle = float(np.degrees(np.arccos(np.clip(d_dec @ d_inc, -1, 1))))
    proj_dec, proj_inc = Xc @ d_dec, Xc @ d_inc

    gpt_bridge = bridge[bridge.model.eq("gptoss")].iloc[0]
    gp = points[points.model.eq("gptoss")].copy()
    rx = residualize(gp.inc_projection_z, gp.delta1c_q05)
    ry = residualize(gp.p_canonical, gp.delta1c_q05)
    updates = {
        "final_layer": 36,
        "angle_decision_incentive_deg": angle,
        "decision_coordinate_delta1c_r": float(np.corrcoef(proj_dec, d)[0, 1]),
        "incentive_coordinate_delta1c_r": float(np.corrcoef(proj_inc, d)[0, 1]),
        "decision_coordinate_choice_r": float(np.corrcoef(proj_dec, y)[0, 1]),
        "layer": 36,
        "slope_pcanonical_per_sd_neural_inc": gpt_bridge.slope_pcanonical_per_sd_neural_inc,
        "ci_lo": gpt_bridge.ci_lo,
        "ci_hi": gpt_bridge.ci_hi,
        "r": float(np.corrcoef(gp.inc_projection_z, gp.p_canonical)[0, 1]),
        "partial_slope_control_delta1c": gpt_bridge.partial_slope_control_delta1c,
        "partial_ci_lo": gpt_bridge.partial_ci_lo,
        "partial_ci_hi": gpt_bridge.partial_ci_hi,
        "partial_r": float(np.corrcoef(rx, ry)[0, 1]),
        "n_games": 144,
        "n": 144,
        "row_scope": "pure_commitment",
        "n_rows": 525,
    }
    out = pd.read_csv(path)
    if "row_scope" not in out:
        out["row_scope"] = "parsed_generated_choice"
    if "n_rows" not in out:
        out["n_rows"] = 576
    mask = out.model.eq("gptoss")
    for key, value in updates.items():
        out.loc[mask, key] = value
    rho = out["decision_coordinate_delta1c_r"].corr(
        out["slope_pcanonical_per_sd_neural_inc"], method="spearman"
    )
    out["spearman_geometry_behavior_rho"] = rho
    out.to_csv(path, index=False)
    DATA.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA / "final_geometry_bridge.parquet", index=False)


def ship_headline(models_run):
    """Replace only rebuilt model rows in the accepted bridge tables."""
    TABLES.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(MAIN_TABLE) if MAIN_TABLE.exists() else pd.DataFrame()
    replacements = []
    point_replacements = []
    for model in models_run:
        d = pd.read_csv(OUT / f"bridge_variants_{model}.csv")
        scope = "pure_commitment" if model == "gptoss" else "parsed_generated_choice"
        r = d[
            (d.layer == max(CAND[model]))
            & (d.variant == "V1_gamez")
            & (d.row_scope == scope)
        ].iloc[0]
        replacements.append(_accepted_row(r))
        p = pd.read_csv(OUT / f"bridge_points_{model}.csv")
        point_replacements.append(p[(p.layer == max(CAND[model])) & (p.row_scope == scope)])

    repl = pd.DataFrame(replacements)
    if not existing.empty:
        existing = existing[~existing.model.isin(models_run)].copy()
        if "row_scope" not in existing:
            existing["row_scope"] = "parsed_generated_choice"
        if "n_rows" not in existing:
            existing["n_rows"] = 576
        if "n_games" not in existing:
            existing["n_games"] = 144
        repl = pd.concat([existing, repl], ignore_index=True)
    repl.sort_values("model").to_csv(MAIN_TABLE, index=False)
    DATA.mkdir(parents=True, exist_ok=True)
    repl.sort_values("model").to_parquet(DATA / "within_model_bridge.parquet", index=False)

    points = pd.read_csv(POINTS_TABLE) if POINTS_TABLE.exists() else pd.DataFrame()
    if not points.empty:
        points = points[~points.model.isin(models_run)].copy()
        if "row_scope" not in points:
            points["row_scope"] = "parsed_generated_choice"
        if "layer" not in points:
            points["layer"] = points.model.map({m: max(CAND[m]) for m in CAND})
    points = pd.concat([points, *point_replacements], ignore_index=True)
    points.sort_values(["model", "game_code"]).to_csv(POINTS_TABLE, index=False)
    points.sort_values(["model", "game_code"]).to_parquet(
        DATA / "within_model_bridge_points.parquet", index=False
    )

    if "gptoss" in models_run:
        d = pd.read_csv(OUT / "bridge_variants_gptoss.csv")
        sens = d[(d.layer == max(CAND["gptoss"])) & (d.variant == "V1_gamez")].copy()
        sens.to_csv(SENS_TABLE, index=False)
        _update_gptoss_final_geometry_bridge(repl, points)
    print(f"shipped accepted bridge -> {MAIN_TABLE}", flush=True)


if __name__ == "__main__":
    selected = sys.argv[1:] or MODELS
    for m in selected:
        run_model(m)
    ship_headline(selected)
    print("DONE", flush=True)
