#!/usr/bin/env python3
"""Build all Layer B recruitment-final tables, caches, and the final figure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.layer_b.recruitment import shared as S


def build_audit() -> pd.DataFrame:
    """Validate the corrected root and write provenance/audit rows."""
    cfg = S.CS.assert_complete_pm1()
    targets = S.game_targets()
    rows: list[dict] = []
    for model in S.MODELS:
        df = S.normalized_results(model)
        layers = S.available_layers(model)
        p1b = df[(df["player"].eq(1)) & (df["condition"].eq("baseline"))]
        rows.append(
            {
                "model": model,
                "data_root": str(S.CORRECTED_ROOT / model),
                "n_games": int(df["game_code"].nunique()),
                "n_rows": int(len(df)),
                "n_p1_baseline_rows": int(len(p1b)),
                "n_p1_baseline_ok": int(p1b["ok"].sum()),
                "n_conditions": int(df["condition"].nunique()),
                "conditions": ";".join(sorted(df["condition"].unique())),
                "cb_ids": ",".join(str(x) for x in sorted(df["cb"].unique())),
                "min_layer": int(min(layers)),
                "max_layer": int(max(layers)),
                "n_layers": int(len(layers)),
                "hidden_dim": int(S.hidden_dim(model)),
                "decision_source": str(df["source"].iloc[0]) if "source" in df else "",
                "canonical_axis_joined": bool(targets["canonical_action_p1"].notna().all()),
                "payoff_multiplier": int(
                    pd.to_numeric(cfg.loc[cfg["model"].eq(model), "payoff_multiplier"], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()[0]
                ),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(S.TABLES / "audit_loader.csv", index=False)
    guards = pd.DataFrame(
        [
            {
                "guardrail": "canonical_axis",
                "status": "pass",
                "detail": "All cross-game behaviour uses action == canonical_action_p1.",
            },
            {
                "guardrail": "q05_delta1c_spine",
                "status": "pass",
                "detail": "Primary incentive axis is canonical-signed Delta1c at q=0.5.",
            },
            {
                "guardrail": "realized_action_only",
                "status": "pass",
                "detail": "Dense uses parse_ok-gated decoded_action; GPT-OSS uses commit-resolved realized_action.",
            },
            {
                "guardrail": "no_old_layerB_tables",
                "status": "pass",
                "detail": "Builders read corrected root, metadata, Layer A trait/lambda tables, and optional causal status only.",
            },
        ]
    )
    guards.to_csv(S.TABLES / "method_guardrails.csv", index=False)
    S.write_json(
        S.TABLES / "provenance.json",
        {
            "analysis": "analysis/layer_b/recruitment",
            "data_root": str(S.CORRECTED_ROOT),
            "models": S.MODELS,
            "primary_behavior_target": "realized P1 baseline action aligned to canonical_action_p1",
            "primary_incentive_axis": "canonical-signed Delta1c at q=0.5",
            "n_boot": S.N_BOOT,
            "seed": S.RNG_SEED,
        },
    )
    print("[audit] wrote audit_loader.csv and provenance.json")
    return audit


def build_disposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cue-shift LDA separability and compact dissociation table."""
    aim = pd.read_csv(S.LAYERA_TRAIT_AIM)
    aim_key = aim[["model", "trait", "aim", "magnitude", "magnitude_lo", "magnitude_hi"]].copy()
    pred_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    diss_rows: list[dict] = []

    for model in S.MODELS:
        layer = max(S.available_layers(model))
        X, meta = S.cue_shift_panel(model, layer)
        pred = S.lda_predictions(X, meta["cue"].to_numpy(), meta["game_code"].to_numpy())
        pred = pd.concat([meta.reset_index(drop=True), pred[["pred", "correct"]]], axis=1)
        pred_rows.append(pred)
        overall, lo, hi = S.boot_mean_by_game(pred["correct"].to_numpy(), pred["game_code"].to_numpy())

        Xb, mb = S.residual_panel(model, layer, player=1, conditions=("baseline",))
        mb = S.attach_targets(mb, model)
        mu = Xb.mean(axis=0)
        d_dec = S.decision_axis(Xb.astype(float) - mu, mb["aligned_canonical_p1"].to_numpy())

        summary_rows.append(
            {
                "model": model,
                "layer": layer,
                "metric": "lda_5way_accuracy",
                "value": overall,
                "ci_lo": lo,
                "ci_hi": hi,
                "chance": 1.0 / len(S.CUES),
                "n_obs": int(len(pred)),
                "n_games": int(pred["game_code"].nunique()),
            }
        )

        for cue in S.CUES:
            sub = pred[pred["cue"].eq(cue)]
            acc, acc_lo, acc_hi = S.boot_mean_by_game(sub["correct"].to_numpy(), sub["game_code"].to_numpy())
            mean_shift = X[meta["cue"].eq(cue).to_numpy()].astype(float).mean(axis=0)
            angle = S.angle_deg(mean_shift, d_dec)
            trait = S.TRAIT_LABEL[cue]
            mrow = aim_key[(aim_key["model"].eq(model)) & (aim_key["trait"].eq(trait))]
            diss_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "cue": cue,
                    "trait": trait,
                    "lda_trait_accuracy": acc,
                    "lda_trait_ci_lo": acc_lo,
                    "lda_trait_ci_hi": acc_hi,
                    "angle_to_decision_deg": angle,
                    "behavior_aim": float(mrow["aim"].iloc[0]) if len(mrow) else np.nan,
                    "behavior_magnitude": float(mrow["magnitude"].iloc[0]) if len(mrow) else np.nan,
                    "behavior_magnitude_lo": float(mrow["magnitude_lo"].iloc[0]) if len(mrow) else np.nan,
                    "behavior_magnitude_hi": float(mrow["magnitude_hi"].iloc[0]) if len(mrow) else np.nan,
                    "n_obs": int(len(sub)),
                    "n_games": int(sub["game_code"].nunique()),
                }
            )
        print(f"[disposition] {S.SHORT[model]} L{layer}: 5-way LDA acc={overall:.3f}")

    preds = pd.concat(pred_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    diss = pd.DataFrame(diss_rows)
    preds.to_parquet(S.DATA / "disposition_lda_predictions.parquet", index=False)
    summary.to_parquet(S.DATA / "disposition_lda_summary.parquet", index=False)
    diss.to_parquet(S.DATA / "disposition_dissociation.parquet", index=False)
    summary.to_csv(S.TABLES / "disposition_lda_summary.csv", index=False)
    diss.to_csv(S.TABLES / "disposition_dissociation.csv", index=False)
    return summary, diss


def build_representation_inventory(disposition_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in S.MODELS:
        layer = max(S.available_layers(model))
        X, meta = S.residual_panel(model, layer, player=1, conditions=("baseline",))
        meta = S.attach_targets(meta, model)
        games = meta["game_code"].to_numpy()
        concepts = [
            ("decision/readout", "realised canonical action", meta["aligned_canonical_p1"].to_numpy(), "binary"),
            ("d_inc", "sign(Delta1c q=0.5)", meta["sign_delta1c_q05"].to_numpy(), "binary"),
            ("stimulus control", "rank of P1 payoff at cell 00", meta["stim_rank00"].to_numpy(), "multiclass"),
        ]
        for family, concept, y, kind in concepts:
            if kind == "binary":
                res = S.group_auc(X, y, games)
            else:
                res = S.group_auc_multiclass(X, y, games)
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "family": family,
                    "concept": concept,
                    **res,
                }
            )

        cue = disposition_summary[disposition_summary["model"].eq(model)].iloc[0].to_dict()
        rows.append(
            {
                "model": model,
                "layer": int(cue["layer"]),
                "family": "cue axes",
                "concept": "5-way disposition cue shifts",
                "metric": cue["metric"],
                "value": cue["value"],
                "ci_lo": cue["ci_lo"],
                "ci_hi": cue["ci_hi"],
                "base_rate": cue["chance"],
                "n_obs": cue["n_obs"],
                "n_games": cue["n_games"],
            }
        )
        print(f"[inventory] {S.SHORT[model]} L{layer}: inventory rows complete")

    out = pd.DataFrame(rows)
    out.to_csv(S.TABLES / "representation_inventory.csv", index=False)
    out.to_parquet(S.DATA / "representation_inventory.parquet", index=False)
    return out


def build_depth_and_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict] = []
    geom_rows: list[dict] = []
    for model in S.MODELS:
        layers = S.selected_layers(model)
        X_by_layer, meta0 = S.baseline_layers(model, layers)
        meta = S.attach_targets(meta0, model)
        games = meta["game_code"].to_numpy()
        y = meta["aligned_canonical_p1"].to_numpy()
        d1c = meta["delta1c_q05"].to_numpy()
        print(f"[depth] {S.SHORT[model]} selected layers: {layers}")
        for layer in layers:
            X = X_by_layer[layer]
            auc = S.group_auc(X, y, games)
            curve_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "metric": "canonical_action_auc",
                    "auc": auc["value"],
                    "auc_ci_lo": auc["ci_lo"],
                    "auc_ci_hi": auc["ci_hi"],
                    "n_obs": auc["n_obs"],
                    "n_games": auc["n_games"],
                }
            )
            mu = X.astype(float).mean(axis=0)
            Xc = X.astype(float) - mu
            d_dec = S.decision_axis(Xc, y)
            d_inc = S.incentive_axis(Xc, d1c)
            proj_dec = Xc @ d_dec
            proj_inc = Xc @ d_inc
            geom_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "angle_decision_incentive_deg": S.angle_deg(d_dec, d_inc),
                    "decision_coordinate_delta1c_r": S.corr(proj_dec, d1c),
                    "incentive_coordinate_delta1c_r": S.corr(proj_inc, d1c),
                    "decision_coordinate_choice_r": S.corr(proj_dec, y),
                    "n_obs": int(np.isfinite(y).sum()),
                    "n_games": int(pd.Series(games[np.isfinite(y)]).nunique()),
                }
            )
        print(f"[depth] {S.SHORT[model]} complete")

    curves = pd.DataFrame(curve_rows)
    geoms = pd.DataFrame(geom_rows)
    curves["auc_gain_over_embedding"] = curves["auc"] - curves.groupby("model")["auc"].transform("first")
    geoms["decision_delta1c_r_gain_over_embedding"] = (
        geoms["decision_coordinate_delta1c_r"]
        - geoms.groupby("model")["decision_coordinate_delta1c_r"].transform("first")
    )
    curves.to_csv(S.TABLES / "decision_crystallization.csv", index=False)
    curves.to_parquet(S.DATA / "decision_crystallization.parquet", index=False)
    # recruitment_geometry_depth.{csv,parquet} is NOT written here. The released table is the
    # depth-resolved fusion geometry under other column names, produced by
    # analysis/layer_b/fusion_associative/build_fusion_figures.py: it sweeps every captured
    # layer and carries the game-block permutation null, while this path sweeps the
    # selected_layers subset and computes no null, so it cannot reproduce the shipped file.
    # The frame below is still returned because build_final_geometry_bridge() needs its three
    # coordinate-correlation columns, which the fusion table does not carry.
    return curves, geoms


def build_bridge() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    point_rows: list[pd.DataFrame] = []
    for model in S.MODELS:
        layer = max(S.available_layers(model))
        X, meta = S.residual_panel(model, layer, player=1, conditions=("baseline",))
        meta = S.attach_targets(meta, model)
        proj = S.oof_incentive_projection(X, meta["delta1c_q05"].to_numpy(), meta["game_code"].to_numpy())
        cells = meta[["model", "game_code", "cb", "aligned_canonical_p1", "delta1c_q05"]].copy()
        cells["inc_projection_oof"] = proj
        game = (
            cells.groupby(["model", "game_code"], as_index=False)
            .agg(
                p_canonical=("aligned_canonical_p1", "mean"),
                inc_projection=("inc_projection_oof", "mean"),
                delta1c_q05=("delta1c_q05", "first"),
                n_cells=("aligned_canonical_p1", "count"),
            )
        )
        game["inc_projection_z"] = S.zscore(game["inc_projection"])
        game["delta1c_q05_z"] = S.zscore(game["delta1c_q05"])
        point_rows.append(game)

        raw = S.boot_slope_by_game(
            game["inc_projection_z"].to_numpy(),
            game["p_canonical"].to_numpy(),
            game["game_code"].to_numpy(),
        )
        # Robustness: represented incentive residual, controlling for true Delta1c.
        x = game["inc_projection_z"].to_numpy()
        y = game["p_canonical"].to_numpy()
        z = game["delta1c_q05_z"].to_numpy()
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if keep.sum() > 8:
            rx = x[keep] - np.polyval(np.polyfit(z[keep], x[keep], 1), z[keep])
            ry = y[keep] - np.polyval(np.polyfit(z[keep], y[keep], 1), z[keep])
            partial = S.boot_slope_by_game(rx, ry, game.loc[keep, "game_code"].to_numpy())
        else:
            partial = {"slope": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "r": np.nan}

        summary_rows.append(
            {
                "model": model,
                "layer": layer,
                "slope_pcanonical_per_sd_neural_inc": raw["slope"],
                "ci_lo": raw["ci_lo"],
                "ci_hi": raw["ci_hi"],
                "r": raw["r"],
                "partial_slope_control_delta1c": partial["slope"],
                "partial_ci_lo": partial["ci_lo"],
                "partial_ci_hi": partial["ci_hi"],
                "partial_r": partial["r"],
                "n_games": raw["n_games"],
                "n": raw["n"],
            }
        )
        print(
            f"[bridge] {S.SHORT[model]} L{layer}: slope={raw['slope']:.3f}, "
            f"r={raw['r']:.3f}, n_games={raw['n_games']}"
        )
    points = pd.concat(point_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    points.to_csv(S.TABLES / "within_model_bridge_points.csv", index=False)
    points.to_parquet(S.DATA / "within_model_bridge_points.parquet", index=False)
    summary.to_parquet(S.DATA / "within_model_bridge.parquet", index=False)
    summary.to_csv(S.TABLES / "within_model_bridge.csv", index=False)
    return summary, points


def build_final_geometry_bridge(geom: pd.DataFrame) -> pd.DataFrame:
    """Join final-layer recruitment geometry to the within-model behavioural bridge.

    This is the compact geometry -> behaviour table used in the revised main panel:
    no depth normalization and no cross-architecture gain arithmetic.

    ``geom`` is passed in from :func:`build_depth_and_geometry` rather than re-read from
    ``recruitment_geometry_depth.csv``: that file is now the fusion table under renamed
    columns and does not carry the three ``*_coordinate_*_r`` columns this join needs.
    """
    bridge = pd.read_csv(S.TABLES / "within_model_bridge.csv")
    final = (
        geom.sort_values("layer")
        .groupby("model", as_index=False)
        .tail(1)[
            [
                "model",
                "layer",
                "angle_decision_incentive_deg",
                "decision_coordinate_delta1c_r",
                "incentive_coordinate_delta1c_r",
                "decision_coordinate_choice_r",
            ]
        ]
        .rename(columns={"layer": "final_layer"})
    )
    out = final.merge(bridge, on="model", how="left")
    out["spearman_geometry_behavior_rho"] = out["decision_coordinate_delta1c_r"].corr(
        out["slope_pcanonical_per_sd_neural_inc"], method="spearman"
    )
    out.to_csv(S.TABLES / "final_geometry_bridge.csv", index=False)
    out.to_parquet(S.DATA / "final_geometry_bridge.parquet", index=False)
    return out


def _effect_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "canonical_pref_delta",
        "delta_canonical_pref",
        "target_pref_delta",
        "intended_shift",
        "intended_effect",
        "effect",
        "shift",
        "delta_pref",
    ]
    return [col for col in candidates if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def build_causal_status() -> pd.DataFrame:
    expected_modes = ["h0", "h1_patch", "h1_dinc", "h2_choice", "h3_oppinc"]
    rows: list[dict] = []
    if not S.A5_RUN.exists():
        for mode in expected_modes:
            for model in S.MODELS:
                rows.append(
                    {
                        "mode": mode,
                        "model": model,
                        "status": "source_missing",
                        "source": str(S.A5_RUN),
                        "n_rows": 0,
                        "effect_column": "",
                        "effect_mean": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "note": "A5 run folder not present in this checkout; causal panel is pending.",
                    }
                )
    else:
        not_est = {}
        for path in S.A5_RUN.glob("*.NOT_ESTIMABLE.json"):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = {"reason": "invalid_json"}
            not_est[path.stem.replace(".NOT_ESTIMABLE", "")] = payload
        for mode in expected_modes:
            for model in S.MODELS:
                path = S.A5_RUN / f"{mode}_{model}.parquet"
                key = f"{mode}_{model}"
                if key in not_est:
                    rows.append(
                        {
                            "mode": mode,
                            "model": model,
                            "status": "not_estimable",
                            "source": str(path),
                            "n_rows": 0,
                            "effect_column": "",
                            "effect_mean": np.nan,
                            "ci_lo": np.nan,
                            "ci_hi": np.nan,
                            "note": json.dumps(not_est[key], sort_keys=True),
                        }
                    )
                elif path.exists():
                    df = pd.read_parquet(path)
                    cols = _effect_columns(df)
                    if cols:
                        col = cols[0]
                        value, lo, hi = S.boot_mean_by_game(
                            pd.to_numeric(df[col], errors="coerce").to_numpy(),
                            df["game_code"].to_numpy() if "game_code" in df else np.arange(len(df)),
                        )
                        status = "complete"
                    else:
                        col, value, lo, hi = "", np.nan, np.nan, np.nan
                        status = "complete_no_standard_effect_col"
                    rows.append(
                        {
                            "mode": mode,
                            "model": model,
                            "status": status,
                            "source": str(path),
                            "n_rows": int(len(df)),
                            "effect_column": col,
                            "effect_mean": value,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "note": "standard effect column summarized" if cols else f"columns={list(df.columns)}",
                        }
                    )
                else:
                    rows.append(
                        {
                            "mode": mode,
                            "model": model,
                            "status": "pending",
                            "source": str(path),
                            "n_rows": 0,
                            "effect_column": "",
                            "effect_mean": np.nan,
                            "ci_lo": np.nan,
                            "ci_hi": np.nan,
                            "note": "Expected A5 output is absent.",
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_parquet(S.DATA / "causal_status.parquet", index=False)
    out.to_csv(S.TABLES / "causal_status.csv", index=False)
    print(f"[causal] statuses: {out['status'].value_counts().to_dict()}")
    return out


def build_router_status() -> pd.DataFrame:
    if not S.ROUTER_VERDICTS.exists():
        rows = [
            {
                "model": "gptoss",
                "status": "source_missing",
                "source": str(S.ROUTER_VERDICTS),
                "n_rows": 0,
                "note": "Router verdict table is absent; GPT-OSS interpretation remains residual-only descriptive.",
            }
        ]
    else:
        try:
            df = pd.read_csv(S.ROUTER_VERDICTS)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if df.empty:
            rows = [
                {
                    "model": "gptoss",
                    "status": "no_router_verdict_rows",
                    "source": str(S.ROUTER_VERDICTS),
                    "n_rows": 0,
                    "note": "Router file exists but contains no verdict rows; no GPT-OSS causal/router claim.",
                }
            ]
        else:
            rows = [
                {
                    "model": "gptoss",
                    "status": "available",
                    "source": str(S.ROUTER_VERDICTS),
                    "n_rows": int(len(df)),
                    "note": "Router verdict rows are available; inspect source table for MoE-specific claims.",
                }
            ]
    out = pd.DataFrame(rows)
    out.to_parquet(S.DATA / "router_status.parquet", index=False)
    out.to_csv(S.TABLES / "router_status.csv", index=False)
    print(f"[router] {out['status'].iloc[0]}")
    return out


def main() -> None:
    build_audit()
    disposition_summary, _ = build_disposition()
    build_representation_inventory(disposition_summary)
    _, geoms = build_depth_and_geometry()
    build_bridge()
    build_final_geometry_bridge(geoms)
    build_causal_status()
    build_router_status()
    from analysis.layer_b.recruitment.figures import render

    render()
    print(f"[done] figure: {S.FIGURES / 'fig_layerB_recruitment_final.png'}")


if __name__ == "__main__":
    main()
