"""Analyze the corrected Akata small-dose steering result package.

This is a local reporting wrapper around the locked small-dose outputs in
``$SCA_DATA_ROOT/steering/smalldose``. It keeps the central decomposition
from ``analysis.steering.analyze_smalldose`` but writes cross-model tables and
figures for the Layer-B steering report folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from strategic_anatomy.config import game_features_csv, manifests_root, results_root, steering_root

ROOT = Path(__file__).resolve().parents[3]

from collection.akata_common import akata_cb_grid  # noqa: E402

GRID = {c["counterbalance_id"]: c for c in akata_cb_grid()}
MODELS = ("qwen", "qwen_instruct", "llama31_instruct")
MODES = ("h1_dinc", "h2_choice")
LAYERS = (65, 79)
VARIANTS = ("main", "main_perp", "random")
DOSES = (-0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25)
BOOT_RNG = np.random.default_rng(20260705)


def boot_mean_ci(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    draws = BOOT_RNG.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_package(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        fps = sorted(fp for mode in MODES for fp in root.glob(f"steer_{model}_{mode}_*.parquet"))
        if len(fps) != 2:
            raise SystemExit(f"expected 2 parquet files for {model}, found {len(fps)} under {root}")
        frames.extend(pd.read_parquet(fp) for fp in fps)
    df = pd.concat(frames, ignore_index=True)
    df["j_is_act0"] = df["counterbalance_id"].map(lambda c: GRID[int(c)]["letter_to_action"]["J"] == 0)
    df["pref_act0"] = np.where(df["j_is_act0"], df["slot_pref_J"], 1.0 - df["slot_pref_J"])
    df["pref_canon"] = np.where(
        df["canonical_action_p1"].astype(int) == 0,
        df["pref_act0"],
        1.0 - df["pref_act0"],
    )
    df["beh_letter_J"] = np.where(
        df["realized_letter"] == "",
        np.nan,
        (df["realized_letter"] == "J").astype(float),
    )
    df["beh_act0"] = np.where(
        df["realized_action"] < 0,
        np.nan,
        (df["realized_action"] == 0).astype(float),
    )
    df["beh_canon"] = df["realized_canonical"].astype(float)
    return df


def attach_game_metadata(df: pd.DataFrame) -> pd.DataFrame:
    sample = pd.read_csv(manifests_root() / "steer_sample_54.csv")
    features = pd.read_csv(game_features_csv())
    keep = [
        "game_code",
        "dominance_profile",
        "iesds_depth",
        "num_pure_ne",
        "payoff_conflict",
        "complexity_score",
        "complexity_score_n_components",
    ]
    meta = sample.merge(features[keep], on="game_code", how="left", suffixes=("", "_features"))
    return df.merge(meta, on=["game_code", "wave"], how="left", suffixes=("", "_sample"))


def add_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "mode", "game_code", "counterbalance_id", "steer_layer", "variant"]
    base_cols = keys + ["slot_pref_J", "pref_act0", "pref_canon", "beh_letter_J", "beh_act0", "beh_canon"]
    base = df[df["dose"] == 0][base_cols].rename(
        columns={
            "slot_pref_J": "base_letter",
            "pref_act0": "base_act0",
            "pref_canon": "base_canon",
            "beh_letter_J": "base_beh_letter",
            "beh_act0": "base_beh_act0",
            "beh_canon": "base_beh_canon",
        }
    )
    out = df.merge(base, on=keys, how="left", validate="many_to_one")
    out["d_letter"] = out["slot_pref_J"] - out["base_letter"]
    out["d_act0"] = out["pref_act0"] - out["base_act0"]
    out["d_canon"] = out["pref_canon"] - out["base_canon"]
    out["d_beh_letter"] = out["beh_letter_J"] - out["base_beh_letter"]
    out["d_beh_act0"] = out["beh_act0"] - out["base_beh_act0"]
    out["d_beh_canon"] = out["beh_canon"] - out["base_beh_canon"]
    return out


def per_game_components(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "mode", "steer_layer", "variant", "game_code", "dose"]
    value_cols = ["d_letter", "d_act0", "d_canon", "d_beh_letter", "d_beh_act0", "d_beh_canon"]
    meta_cols = ["wave", "family", "num_pure_ne", "complexity_score", "payoff_conflict"]
    pg = df.groupby(group_cols, as_index=False)[value_cols].mean()
    meta = df[["model", "mode", "steer_layer", "variant", "game_code"] + meta_cols].drop_duplicates()
    return pg.merge(meta, on=["model", "mode", "steer_layer", "variant", "game_code"], how="left")


def slope_table(pg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_cols = ["d_letter", "d_act0", "d_canon", "d_beh_letter", "d_beh_act0", "d_beh_canon"]
    keys = ["model", "mode", "steer_layer", "variant", "game_code"]
    for key, sub in pg.groupby(keys):
        row = dict(zip(keys, key))
        for col in value_cols:
            clean = sub[["dose", col]].dropna()
            if clean["dose"].nunique() >= 3:
                row[f"{col}_slope"] = float(np.polyfit(clean["dose"], clean[col], 1)[0])
                row[f"{col}_range"] = float(clean[col].max() - clean[col].min())
                pos = clean.loc[np.isclose(clean["dose"], 0.25), col]
                neg = clean.loc[np.isclose(clean["dose"], -0.25), col]
                row[f"{col}_at_pos025"] = float(pos.iloc[0]) if len(pos) else np.nan
                row[f"{col}_at_neg025"] = float(neg.iloc[0]) if len(neg) else np.nan
            else:
                row[f"{col}_slope"] = np.nan
                row[f"{col}_range"] = np.nan
                row[f"{col}_at_pos025"] = np.nan
                row[f"{col}_at_neg025"] = np.nan
        meta = sub.iloc[0]
        row["wave"] = meta.get("wave")
        row["family"] = meta.get("family")
        row["num_pure_ne"] = meta.get("num_pure_ne")
        row["complexity_score"] = meta.get("complexity_score")
        row["payoff_conflict"] = meta.get("payoff_conflict")
        rows.append(row)
    return pd.DataFrame(rows)


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model", "mode", "steer_layer", "variant", "game_code", "counterbalance_id"]
    for key, sub in df.groupby(keys):
        row = dict(zip(keys, key))
        row["range_letter"] = float(sub["slot_pref_J"].max() - sub["slot_pref_J"].min())
        row["range_act0"] = float(sub["pref_act0"].max() - sub["pref_act0"].min())
        row["range_canon"] = float(sub["pref_canon"].max() - sub["pref_canon"].min())
        row["flip_letter"] = int(sub.loc[sub["realized_letter"] != "", "realized_letter"].nunique() > 1)
        row["flip_action"] = int(sub.loc[sub["realized_action"] >= 0, "realized_action"].nunique() > 1)
        row["flip_canon"] = int(sub["realized_canonical"].dropna().nunique() > 1)
        row["family"] = sub["family"].iloc[0]
        row["wave"] = int(sub["wave"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def coherence_table(df: pd.DataFrame) -> pd.DataFrame:
    nonzero = df[~np.isclose(df["dose"], 0.0)].copy()
    rows: list[dict[str, object]] = []
    axes = {"letter": "d_letter", "act0": "d_act0", "canon": "d_canon"}
    keys = ["model", "mode", "steer_layer", "variant", "game_code"]
    for key, sub in nonzero.groupby(keys):
        row = dict(zip(keys, key))
        for axis, col in axes.items():
            delta = sub[col].to_numpy(float)
            signed = np.sign(sub["dose"].to_numpy(float)) * delta
            nz = np.abs(delta) > 1e-12
            row[f"{axis}_nonzero_rate"] = float(nz.mean())
            row[f"{axis}_dose_aligned_rate_all"] = float((signed > 0).mean())
            row[f"{axis}_dose_aligned_rate_nonzero"] = float((signed[nz] > 0).mean()) if nz.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(slopes: pd.DataFrame, cells: pd.DataFrame, coh: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model", "mode", "steer_layer", "variant"]
    for key, s in slopes.groupby(keys):
        row = dict(zip(keys, key))
        row["n_games"] = int(s["game_code"].nunique())
        for col in ["d_letter", "d_act0", "d_canon", "d_beh_letter", "d_beh_act0", "d_beh_canon"]:
            vals = s[f"{col}_slope"].to_numpy(float)
            lo, hi = boot_mean_ci(vals)
            row[f"{col}_slope_mean"] = float(np.nanmean(vals))
            row[f"{col}_slope_ci_low"] = lo
            row[f"{col}_slope_ci_high"] = hi
            row[f"{col}_range_mean"] = float(np.nanmean(s[f"{col}_range"]))
            row[f"{col}_abs_pos025_mean"] = float(np.nanmean(np.abs(s[f"{col}_at_pos025"])))
            row[f"{col}_abs_neg025_mean"] = float(np.nanmean(np.abs(s[f"{col}_at_neg025"])))
        c = cells
        for k_name, k_val in zip(keys, key):
            c = c[c[k_name] == k_val]
        row["n_cells"] = int(len(c))
        for col in ["letter", "act0", "canon"]:
            row[f"cell_{col}_range_mean"] = float(c[f"range_{col}"].mean())
            row[f"cell_{col}_range_p95"] = float(c[f"range_{col}"].quantile(0.95))
        for col in ["letter", "action", "canon"]:
            game_rates = c.groupby("game_code")[f"flip_{col}"].mean().to_numpy(float)
            lo, hi = boot_mean_ci(game_rates)
            row[f"flip_{col}_rate"] = float(np.nanmean(game_rates))
            row[f"flip_{col}_rate_ci_low"] = lo
            row[f"flip_{col}_rate_ci_high"] = hi
        h = coh
        for k_name, k_val in zip(keys, key):
            h = h[h[k_name] == k_val]
        for axis in ["letter", "act0", "canon"]:
            for metric in ["nonzero_rate", "dose_aligned_rate_all", "dose_aligned_rate_nonzero"]:
                game_vals = h.groupby("game_code")[f"{axis}_{metric}"].mean().to_numpy(float)
                row[f"{axis}_{metric}"] = float(np.nanmean(game_vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def summarize_family(slopes: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model", "mode", "steer_layer", "variant", "family"]
    for key, s in slopes.groupby(keys):
        row = dict(zip(keys, key))
        row["n_games"] = int(s["game_code"].nunique())
        for col in ["d_letter", "d_canon", "d_beh_canon"]:
            row[f"{col}_slope_mean"] = float(s[f"{col}_slope"].mean())
            row[f"{col}_range_mean"] = float(s[f"{col}_range"].mean())
        c = cells
        for k_name, k_val in zip(keys, key):
            c = c[c[k_name] == k_val]
        row["flip_canon_rate"] = float(c.groupby("game_code")["flip_canon"].mean().mean())
        row["cell_canon_range_mean"] = float(c["range_canon"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def summarize_contrasts(slopes: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "d_letter_slope",
        "d_canon_slope",
        "d_beh_canon_slope",
        "d_letter_range",
        "d_canon_range",
        "d_beh_canon_range",
    ]
    cell_game = (
        cells.groupby(["model", "mode", "steer_layer", "variant", "game_code"], as_index=False)
        [["flip_canon", "range_canon"]]
        .mean()
        .rename(columns={"flip_canon": "cell_flip_canon", "range_canon": "cell_range_canon"})
    )
    merged = slopes.merge(
        cell_game,
        on=["model", "mode", "steer_layer", "variant", "game_code"],
        how="left",
    )
    for (model, mode, layer), sub in merged.groupby(["model", "mode", "steer_layer"]):
        for variant in ("main", "main_perp"):
            row: dict[str, object] = {
                "model": model,
                "mode": mode,
                "steer_layer": layer,
                "contrast": f"{variant}-random",
            }
            left = sub[sub["variant"] == variant].set_index("game_code")
            right = sub[sub["variant"] == "random"].set_index("game_code")
            games = sorted(set(left.index).intersection(right.index))
            row["n_games"] = len(games)
            for metric in metrics + ["cell_flip_canon", "cell_range_canon"]:
                diff = left.loc[games, metric].to_numpy(float) - right.loc[games, metric].to_numpy(float)
                lo, hi = boot_mean_ci(diff)
                row[f"{metric}_diff_mean"] = float(np.nanmean(diff))
                row[f"{metric}_diff_ci_low"] = lo
                row[f"{metric}_diff_ci_high"] = hi
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "mode", "steer_layer", "contrast"]).reset_index(drop=True)


def build_verdict_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, mode, layer), sub in summary.groupby(["model", "mode", "steer_layer"]):
        by_variant = sub.set_index("variant")
        row = {"model": model, "mode": mode, "steer_layer": layer}
        for variant in VARIANTS:
            if variant not in by_variant.index:
                continue
            v = by_variant.loc[variant]
            row[f"{variant}_canon_slope"] = float(v["d_canon_slope_mean"])
            row[f"{variant}_letter_slope"] = float(v["d_letter_slope_mean"])
            row[f"{variant}_canon_range"] = float(v["cell_canon_range_mean"])
            row[f"{variant}_canon_flip_rate"] = float(v["flip_canon_rate"])
            row[f"{variant}_canon_coherence"] = float(v["canon_dose_aligned_rate_nonzero"])
        if {"main", "random"}.issubset(by_variant.index):
            row["main_minus_random_canon_slope"] = float(
                by_variant.loc["main", "d_canon_slope_mean"]
                - by_variant.loc["random", "d_canon_slope_mean"]
            )
            row["main_over_random_canon_range"] = float(
                by_variant.loc["main", "cell_canon_range_mean"]
                / max(by_variant.loc["random", "cell_canon_range_mean"], 1e-12)
            )
        if {"main_perp", "random"}.issubset(by_variant.index):
            row["main_perp_minus_random_canon_slope"] = float(
                by_variant.loc["main_perp", "d_canon_slope_mean"]
                - by_variant.loc["random", "d_canon_slope_mean"]
            )
            row["main_perp_over_random_canon_range"] = float(
                by_variant.loc["main_perp", "cell_canon_range_mean"]
                / max(by_variant.loc["random", "cell_canon_range_mean"], 1e-12)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "mode", "steer_layer"]).reset_index(drop=True)


def audit_table(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model"):
        rows.append(
            {
                "model": model,
                "rows": int(len(sub)),
                "files": len([fp for mode in MODES for fp in root.glob(f"steer_{model}_{mode}_*.parquet")]),
                "games": int(sub["game_code"].nunique()),
                "modes": ",".join(sorted(sub["mode"].unique())),
                "layers": ",".join(map(str, sorted(sub["steer_layer"].unique()))),
                "variants": ",".join(sorted(sub["variant"].unique())),
                "doses": ",".join(map(str, sorted(sub["dose"].unique()))),
                "parse_failure_rows": int((sub["realized_letter"] == "").sum()),
                "status_values": ",".join(sorted(sub["status"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def write_figures(summary: pd.DataFrame, pg: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = {"main": "#3B6FB6", "main_perp": "#2F9C67", "random": "#9A9A9A"}
    markers = {"h1_dinc": "o", "h2_choice": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = summary[summary["model"] == model]
        for _, row in sub.iterrows():
            ax.scatter(
                row["d_letter_slope_mean"],
                row["d_canon_slope_mean"],
                s=72 if row["steer_layer"] == 79 else 48,
                marker=markers[row["mode"]],
                color=colors[row["variant"]],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
            )
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.axvline(0, color="#333333", linewidth=0.7)
        ax.set_title(model)
        ax.set_xlabel("Letter slope: d pref(J) / dose")
    axes[0].set_ylabel("Canonical strategic slope: d pref(canon) / dose")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="main", markerfacecolor=colors["main"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="main_perp", markerfacecolor=colors["main_perp"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="random", markerfacecolor=colors["random"], markersize=8),
        plt.Line2D([0], [0], marker="o", color="#333333", label="h1_dinc", markerfacecolor="#333333", linestyle=""),
        plt.Line2D([0], [0], marker="s", color="#333333", label="h2_choice", markerfacecolor="#333333", linestyle=""),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(fig_dir / "slope_letter_vs_canonical.png", dpi=220)
    plt.close(fig)

    plot = summary.copy()
    plot["group"] = (
        plot["model"].str.replace("_instruct", "-inst", regex=False)
        + "\n"
        + plot["mode"].str.replace("_", " ")
        + " L"
        + plot["steer_layer"].astype(str)
    )
    groups = plot[["model", "mode", "steer_layer", "group"]].drop_duplicates().sort_values(
        ["model", "mode", "steer_layer"]
    )
    x = np.arange(len(groups))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True)
    for i, variant in enumerate(VARIANTS):
        vals = []
        flips = []
        for _, g in groups.iterrows():
            row = plot[
                (plot["model"] == g["model"])
                & (plot["mode"] == g["mode"])
                & (plot["steer_layer"] == g["steer_layer"])
                & (plot["variant"] == variant)
            ]
            vals.append(float(row["d_canon_range_mean"].iloc[0]) if len(row) else np.nan)
            flips.append(float(row["flip_canon_rate"].iloc[0]) if len(row) else np.nan)
        axes[0].bar(x + (i - 1) * width, vals, width=width, color=colors[variant], label=variant)
        axes[1].bar(x + (i - 1) * width, flips, width=width, color=colors[variant], label=variant)
    axes[0].set_ylabel("CB-mean canonical\nprobability range")
    axes[1].set_ylabel("Canonical decision\nflip rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups["group"], rotation=45, ha="right")
    axes[0].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "canonical_range_and_flips.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=False)
    for r, model in enumerate(MODELS):
        for c, mode in enumerate(MODES):
            ax = axes[r, c]
            sub = pg[(pg["model"] == model) & (pg["mode"] == mode) & (pg["steer_layer"] == 79)]
            for variant in VARIANTS:
                sv = sub[sub["variant"] == variant]
                if sv.empty:
                    continue
                curve = sv.groupby("dose", as_index=False)[["d_letter", "d_canon"]].mean()
                ax.plot(curve["dose"], curve["d_letter"], color=colors[variant], linestyle="--", alpha=0.8)
                ax.plot(curve["dose"], curve["d_canon"], color=colors[variant], linestyle="-", label=variant)
            ax.axhline(0, color="#333333", linewidth=0.7)
            ax.set_title(f"{model} / {mode} / L79")
            if c == 0:
                ax.set_ylabel("Mean delta vs dose 0")
            if r == 2:
                ax.set_xlabel("Dose")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 1].legend(frameon=False)
    fig.text(0.5, 0.01, "Solid = canonical strategic component; dashed = letter component", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(fig_dir / "dose_curves_l79.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=steering_root() / "smalldose")
    parser.add_argument("--out", type=Path, default=results_root() / "steering")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)

    df = attach_game_metadata(load_package(args.root))
    df = add_baseline_deltas(df)
    pg = per_game_components(df)
    slopes = slope_table(pg)
    cells = cell_table(df)
    coh = coherence_table(df)
    summary = summarize_groups(slopes, cells, coh)
    family = summarize_family(slopes, cells)
    contrasts = summarize_contrasts(slopes, cells)
    verdict = build_verdict_table(summary)
    audit = audit_table(df, args.root)

    tables = {
        "audit_summary.csv": audit,
        "group_summary.csv": summary,
        "per_game_slopes.csv": slopes,
        "per_cell_ranges_flips.csv": cells,
        "coherence_by_game.csv": coh,
        "family_summary.csv": family,
        "contrast_summary.csv": contrasts,
        "verdict_by_model_mode_layer.csv": verdict,
    }
    for name, table in tables.items():
        table.to_csv(args.out / "tables" / name, index=False)

    extremes = slopes.sort_values("d_canon_range", ascending=False).head(30)
    extremes.to_csv(args.out / "tables" / "top_canonical_range_games.csv", index=False)
    write_figures(summary, pg, args.out)

    meta = {
        "source_root": str(args.root.relative_to(ROOT)),
        "output_root": str(args.out.relative_to(ROOT)),
        "rows": int(len(df)),
        "models": list(MODELS),
        "modes": list(MODES),
        "layers": list(LAYERS),
        "variants": list(VARIANTS),
        "doses": list(DOSES),
        "games": int(df["game_code"].nunique()),
        "parse_failure_rows": int((df["realized_letter"] == "").sum()),
        "status_values": sorted(df["status"].astype(str).unique().tolist()),
    }
    (args.out / "tables" / "analysis_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
