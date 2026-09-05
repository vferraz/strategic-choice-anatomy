"""Candidate final Layer C figures for the corrected Akata one-shot substrate.

This script treats the saved logit-lens values as descriptive policy-evidence
readouts, not causal attributions. It reads:

- $SCA_DATA_ROOT/layerc/{model}/{game}/tokens.parquet
- $SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet

and writes candidate figures/tables under analysis/layer_c/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_layerc_final")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from analysis.layer_a.trait_steering_proper import trait_targets
from strategic_anatomy.config import game_features_csv, layerc_root, repo_root, results_root, substrate_root

ROOT = repo_root()



LAYERC_ROOT = layerc_root()
BEHAV_ROOT = substrate_root()
OUT_DIR = ROOT / "analysis" / "layer_c"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

MODELS = ["qwen", "qwen_instruct", "llama31_instruct"]
MODEL_LABELS = {
    "qwen": "Qwen",
    "qwen_instruct": "Qwen-Instruct",
    "llama31_instruct": "Llama-3.1-Instruct",
}
MODEL_COLORS = {
    "qwen": "#4169a8",
    "qwen_instruct": "#198754",
    "llama31_instruct": "#c45a32",
}
REGION_ORDER = [
    "cue_prefix",
    "intro",
    "rule",
    "own_payoff",
    "opponent_payoff",
    "label_token",
    "question",
    "answer_prefix",
]
REGION_LABELS = {
    "cue_prefix": "cue/header",
    "intro": "intro",
    "rule": "rules",
    "own_payoff": "own payoff",
    "opponent_payoff": "other payoff",
    "label_token": "option labels",
    "question": "question",
    "answer_prefix": "answer prefix",
}
TRAITS = {
    "risk_aversion": "risk",
    "loss_aversion": "loss",
    "inequity_aversion": "inequity",
    "maximin": "maximin",
    "selfish_maximizer": "selfish",
}
TRAIT_LABELS = {
    "risk_aversion": "risk",
    "loss_aversion": "loss",
    "inequity_aversion": "inequity",
    "maximin": "maximin",
    "selfish_maximizer": "selfish",
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _read_tokens(columns: list[str], *, condition_filter: set[str] | None = None) -> pd.DataFrame:
    parts = []
    for model in MODELS:
        for path in sorted((LAYERC_ROOT / model).glob("*/tokens.parquet")):
            df = pd.read_parquet(path, columns=columns)
            if condition_filter is not None:
                df = df[df["condition"].isin(condition_filter)]
            parts.append(df)
    if not parts:
        raise RuntimeError(f"no token files found under {LAYERC_ROOT}")
    return pd.concat(parts, ignore_index=True)


def _local_delta(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    keys = ["model", "game_code", "cb_id", "condition", "layer"]
    df = df.sort_values(keys + ["token_index"]).copy()
    # Local policy-evidence change. The first token is set to zero so the map
    # emphasises prompt-local changes rather than model prior at BOS/header.
    df[out_col] = df.groupby(keys)[value_col].diff().fillna(0.0)
    return df


def _final_token_scores(tokens: pd.DataFrame, score_col: str) -> pd.DataFrame:
    idx = tokens.groupby(["model", "game_code", "cb_id", "condition", "layer"])["token_index"].idxmax()
    return tokens.loc[idx, ["model", "game_code", "cb_id", "condition", "layer", score_col]].copy()


def load_behavior_game_rates() -> pd.DataFrame:
    gf = pd.read_csv(game_features_csv(),
                     usecols=["game_code", "canonical_action_p1"])
    parts = []
    for model in MODELS:
        for path in sorted((BEHAV_ROOT / model).glob("*/results.parquet")):
            df = pd.read_parquet(path)
            df = df[(df["player"].eq(1)) & (df["condition"].eq("baseline")) & (df["parse_ok"].eq(1))]
            parts.append(df[["model", "game_code", "counterbalance_id", "decoded_action"]])
    beh = pd.concat(parts, ignore_index=True).merge(gf, on="game_code", how="left")
    beh["chose_canonical"] = (beh["decoded_action"].astype(int) == beh["canonical_action_p1"].astype(int)).astype(float)
    return beh.groupby(["model", "game_code"], as_index=False)["chose_canonical"].mean()


def figure_policy_bridge() -> pd.DataFrame:
    tokens = _read_tokens(
        ["model", "game_code", "cb_id", "condition", "layer", "token_index", "score_canonical"],
        condition_filter={"baseline"},
    )
    final = _final_token_scores(tokens[tokens["layer"].eq(79)], "score_canonical")
    final_game = (
        final.groupby(["model", "game_code"], as_index=False)["score_canonical"]
        .mean()
        .rename(columns={"score_canonical": "final_l79_canonical_score"})
    )
    beh = load_behavior_game_rates()
    dat = final_game.merge(beh, on=["model", "game_code"], how="inner")
    dat.to_csv(TABLE_DIR / "c1_policy_bridge_game_level.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = dat[dat["model"].eq(model)].copy()
        x = sub["final_l79_canonical_score"].to_numpy()
        y = sub["chose_canonical"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        ax.scatter(x, y, s=18, alpha=0.7, color=MODEL_COLORS[model], edgecolor="none")
        if len(sub) > 2:
            coef = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#202020", lw=1.5)
        ax.axhline(0.5, color="#9a9a9a", lw=0.8, ls=":")
        ax.axvline(0.0, color="#9a9a9a", lw=0.8, ls=":")
        ax.set_title(f"{MODEL_LABELS[model]}\nr = {r:.2f}", fontsize=10)
        ax.set_xlabel("final L79 canonical logit margin")
        ax.set_ylim(-0.03, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("generated P(canonical), by game")
    fig.suptitle("Layer C bridge: final policy evidence predicts generated choice", y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figC1_policy_bridge.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC1_policy_bridge.pdf", bbox_inches="tight")
    plt.close(fig)
    return dat


def figure_semantic_evidence() -> pd.DataFrame:
    tokens = _read_tokens(
        ["model", "game_code", "cb_id", "condition", "layer", "token_index", "region", "score_canonical"],
        condition_filter={"baseline"},
    )
    tokens = tokens[tokens["layer"].eq(79)]
    tokens = _local_delta(tokens, "score_canonical", "d_score")
    tokens["abs_d_score"] = tokens["d_score"].abs()
    reg = (
        tokens.groupby(["model", "game_code", "cb_id", "layer", "region"], as_index=False)
        .agg(abs_mass=("abs_d_score", "sum"), n_tokens=("abs_d_score", "size"))
    )
    totals = reg.groupby(["model", "game_code", "cb_id", "layer"], as_index=False)["abs_mass"].sum()
    totals = totals.rename(columns={"abs_mass": "total_abs_mass"})
    reg = reg.merge(totals, on=["model", "game_code", "cb_id", "layer"], how="left")
    reg["abs_share"] = reg["abs_mass"] / reg["total_abs_mass"].replace(0, np.nan)
    reg["abs_density"] = reg["abs_mass"] / reg["n_tokens"].replace(0, np.nan)

    summary = (
        reg.groupby(["model", "region"], as_index=False)
        .agg(mean_abs_share=("abs_share", "mean"), mean_abs_density=("abs_density", "mean"))
    )
    summary = summary[summary["region"].isin(REGION_ORDER)]
    summary.to_csv(TABLE_DIR / "c2_semantic_evidence_l79_baseline.csv", index=False)

    # Baseline raw Qwen prompts have no cue_prefix, while Llama chat headers are
    # mapped there. Drop it from the baseline semantic comparison so the figure
    # compares task text rather than wrapper/header tokens.
    regions = [r for r in REGION_ORDER if r != "cue_prefix" and r in set(summary["region"])]
    share = summary.pivot(index="region", columns="model", values="mean_abs_share").reindex(regions)
    density = summary.pivot(index="region", columns="model", values="mean_abs_density").reindex(regions)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.6), constrained_layout=True)
    for ax, mat, title, cmap, fmt in [
        (axes[0], share, "total local evidence share", "Blues", ".2f"),
        (axes[1], density, "local evidence per token", "Oranges", ".1f"),
    ]:
        vals = mat[MODELS].to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m].replace("-Instruct", "\nInstr.") for m in MODELS], fontsize=8)
        ax.set_yticks(range(len(regions)), [REGION_LABELS[r] for r in regions], fontsize=9)
        ax.set_title(title, fontsize=10)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, format(vals[i, j], fmt), ha="center", va="center", fontsize=7, color="#202020")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Where canonical policy evidence changes at L79", fontsize=12)
    fig.savefig(FIG_DIR / "figC2_semantic_evidence_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC2_semantic_evidence_map.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def figure_trait_redirection() -> pd.DataFrame:
    cols = [
        "model", "game_code", "cb_id", "condition", "layer", "token_index",
        "region", "score_canonical", "score_trait_target",
    ]
    condition_filter = {"cue_length_match_null"} | {f"cue_{t}" for t in TRAITS}
    tokens = _read_tokens(cols, condition_filter=condition_filter)
    tokens = tokens[tokens["layer"].eq(79)].copy()

    gf = pd.read_csv(game_features_csv(),
                     usecols=["game_code", "canonical_action_p1"])
    targets = trait_targets().merge(gf, on="game_code", how="left")

    records = []
    for trait, col_suffix in TRAITS.items():
        cue = f"cue_{trait}"
        tmp = tokens[tokens["condition"].isin([cue, "cue_length_match_null"])].merge(
            targets[["game_code", "canonical_action_p1", f"a_{col_suffix}"]],
            on="game_code",
            how="left",
        )
        tmp = tmp[tmp[f"a_{col_suffix}"].isin([0, 1])].copy()
        # Final-analysis trait panel: focus on games where the cue target is
        # identifiable against the canonical/rational action.
        tmp = tmp[tmp[f"a_{col_suffix}"].astype(int) != tmp["canonical_action_p1"].astype(int)]
        if tmp.empty:
            continue
        same_as_canon = tmp[f"a_{col_suffix}"].astype(int) == tmp["canonical_action_p1"].astype(int)
        tmp["target_score"] = np.where(
            tmp["condition"].eq(cue),
            tmp["score_trait_target"],
            np.where(same_as_canon, tmp["score_canonical"], -tmp["score_canonical"]),
        )
        tmp = _local_delta(tmp, "target_score", "d_target")
        reg = (
            tmp.groupby(["model", "game_code", "cb_id", "condition", "region"], as_index=False)
            .agg(net_target_change=("d_target", "sum"))
        )
        wide = reg.pivot_table(
            index=["model", "game_code", "cb_id", "region"],
            columns="condition",
            values="net_target_change",
            aggfunc="first",
        ).reset_index()
        if cue not in wide or "cue_length_match_null" not in wide:
            continue
        wide["trait"] = trait
        wide["cue_minus_placebo"] = wide[cue] - wide["cue_length_match_null"]
        records.append(wide[["model", "game_code", "cb_id", "trait", "region", "cue_minus_placebo"]])
    dat = pd.concat(records, ignore_index=True)
    dat = dat[dat["region"].isin(REGION_ORDER)]
    summary = (
        dat.groupby(["model", "trait", "region"], as_index=False)
        .agg(mean_cue_minus_placebo=("cue_minus_placebo", "mean"),
             n_cells=("cue_minus_placebo", "size"))
    )
    summary.to_csv(TABLE_DIR / "c3_trait_target_redirection_l79_conflict_regions.csv", index=False)

    regions = ["cue_prefix", "rule", "own_payoff", "opponent_payoff", "label_token", "question", "answer_prefix"]
    traits = [t for t in TRAITS if t in set(summary["trait"])]
    vals_all = summary["mean_cue_minus_placebo"].to_numpy()
    vmax = float(np.nanquantile(np.abs(vals_all), 0.95))
    vmax = max(vmax, 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        mat = (
            summary[summary["model"].eq(model)]
            .pivot(index="trait", columns="region", values="mean_cue_minus_placebo")
            .reindex(index=traits, columns=regions)
        )
        vals = mat.to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xticks(range(len(regions)), [REGION_LABELS[r] for r in regions], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(traits)), [TRAIT_LABELS[t] for t in traits], fontsize=9)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="cue - placebo target-axis change")
    fig.suptitle("Trait-target policy evidence redirection at L79 (conflict games)", fontsize=12)
    fig.savefig(FIG_DIR / "figC3_trait_target_redirection.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC3_trait_target_redirection.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def figure_incentive_curves() -> pd.DataFrame:
    """Cleaner bridge: Layer-C policy evidence as a function of incentive gap."""
    tokens = _read_tokens(
        ["model", "game_code", "cb_id", "condition", "layer", "token_index", "score_canonical"],
        condition_filter={"baseline"},
    )
    final = _final_token_scores(tokens[tokens["layer"].eq(79)], "score_canonical")
    final_game = (
        final.groupby(["model", "game_code"], as_index=False)["score_canonical"]
        .mean()
        .rename(columns={"score_canonical": "final_l79_canonical_score"})
    )
    layera = pd.read_parquet(
        results_root() / "layer_a" / "_data" / "layerA_game_level.parquet",
        columns=["model", "game_code", "delta1c_q05", "p_canon"],
    )
    layera = layera[layera["model"].isin(MODELS)]
    dat = final_game.merge(layera, on=["model", "game_code"], how="inner")
    dat["lens_z"] = dat.groupby("model")["final_l79_canonical_score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    # Common incentive bins keep the panel visually stable and close to the
    # Layer-A quantal-response story.
    dat["gap_bin"] = pd.qcut(dat["delta1c_q05"], q=7, duplicates="drop")
    summary = (
        dat.groupby(["model", "gap_bin"], observed=True)
        .agg(delta1c=("delta1c_q05", "mean"),
             lens_z=("lens_z", "mean"),
             p_canon=("p_canon", "mean"),
             n_games=("game_code", "nunique"))
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "c1b_incentive_curve_l79.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
    for model in MODELS:
        sub = summary[summary["model"].eq(model)].sort_values("delta1c")
        label = MODEL_LABELS[model]
        axes[0].plot(sub["delta1c"], sub["lens_z"], marker="o", lw=2, ms=4,
                     color=MODEL_COLORS[model], label=label)
        axes[1].plot(sub["delta1c"], sub["p_canon"], marker="o", lw=2, ms=4,
                     color=MODEL_COLORS[model], label=label)
    axes[0].axhline(0, color="#9a9a9a", lw=0.8, ls=":")
    axes[1].axhline(0.5, color="#9a9a9a", lw=0.8, ls=":")
    axes[0].set_ylabel("L79 canonical policy evidence (z)")
    axes[1].set_ylabel("generated P(canonical)")
    for ax in axes:
        ax.set_xlabel("canonical-signed incentive gap, $\\Delta_1^c$")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("vocabulary-level readout")
    axes[1].set_title("generated behaviour")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Layer C follows the same incentive gradient as behaviour", y=1.03, fontsize=12)
    fig.subplots_adjust(bottom=0.28, top=0.78, left=0.08, right=0.98, wspace=0.32)
    fig.savefig(FIG_DIR / "figC1b_incentive_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC1b_incentive_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def figure_task_region_bars() -> pd.DataFrame:
    """Cleaner region figure: task-bearing evidence share plus payoff split."""
    tokens = _read_tokens(
        ["model", "game_code", "cb_id", "condition", "layer", "token_index", "region", "score_canonical"],
        condition_filter={"baseline"},
    )
    tokens = tokens[tokens["layer"].eq(79)]
    tokens = _local_delta(tokens, "score_canonical", "d_score")
    tokens["abs_d_score"] = tokens["d_score"].abs()
    tokens["region_group"] = tokens["region"].replace({
        "own_payoff": "payoff_numbers",
        "opponent_payoff": "payoff_numbers",
        "answer_prefix": "question_answer",
        "question": "question_answer",
    })
    keep = ["rule", "payoff_numbers", "label_token", "question_answer"]
    grp = (
        tokens[tokens["region_group"].isin(keep)]
        .groupby(["model", "game_code", "cb_id", "region_group"], as_index=False)
        .agg(abs_mass=("abs_d_score", "sum"))
    )
    totals = grp.groupby(["model", "game_code", "cb_id"], as_index=False)["abs_mass"].sum()
    totals = totals.rename(columns={"abs_mass": "task_abs_total"})
    grp = grp.merge(totals, on=["model", "game_code", "cb_id"], how="left")
    grp["share_of_task_evidence"] = grp["abs_mass"] / grp["task_abs_total"].replace(0, np.nan)
    summary = (
        grp.groupby(["model", "region_group"], as_index=False)
        .agg(mean_share=("share_of_task_evidence", "mean"))
    )

    payoff = (
        tokens[tokens["region"].isin(["own_payoff", "opponent_payoff"])]
        .groupby(["model", "game_code", "cb_id", "region"], as_index=False)
        .agg(abs_mass=("abs_d_score", "sum"))
        .pivot_table(index=["model", "game_code", "cb_id"], columns="region", values="abs_mass", fill_value=0)
        .reset_index()
    )
    payoff["opponent_share_payoff"] = payoff["opponent_payoff"] / (
        payoff["own_payoff"] + payoff["opponent_payoff"]
    ).replace(0, np.nan)
    payoff_summary = (
        payoff.groupby("model", as_index=False)
        .agg(opponent_share_payoff=("opponent_share_payoff", "mean"))
    )
    out = summary.merge(payoff_summary, on="model", how="left")
    out.to_csv(TABLE_DIR / "c2b_task_region_evidence_l79.csv", index=False)

    order = ["rule", "payoff_numbers", "label_token", "question_answer"]
    labels = {
        "rule": "rules",
        "payoff_numbers": "payoff\nnumbers",
        "label_token": "option\nlabels",
        "question_answer": "question /\nanswer",
    }
    colors = {
        "rule": "#4c78a8",
        "payoff_numbers": "#72b7b2",
        "label_token": "#f58518",
        "question_answer": "#b279a2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))
    for region in order:
        vals = [
            float(summary[(summary["model"].eq(m)) & (summary["region_group"].eq(region))]["mean_share"].iloc[0])
            for m in MODELS
        ]
        ax.barh(y, vals, left=left, height=0.58, label=labels[region], color=colors[region])
        left += np.array(vals)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of task-text local policy evidence")
    ax.set_title("")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.22))
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    vals = [
        float(payoff_summary[payoff_summary["model"].eq(m)]["opponent_share_payoff"].iloc[0])
        for m in MODELS
    ]
    ax.bar(np.arange(len(MODELS)), vals, color=[MODEL_COLORS[m] for m in MODELS], width=0.62)
    ax.axhline(0.5, color="#333333", lw=1, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(MODELS)), [MODEL_LABELS[m].replace("-Instruct", "\nInstr.") for m in MODELS], fontsize=8)
    ax.set_ylabel("share of payoff-number evidence\non opponent payoff")
    ax.set_title("own vs other payoff")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Semantic sources of L79 canonical policy evidence", y=1.08, fontsize=12)
    fig.subplots_adjust(top=0.74, bottom=0.18, left=0.16, right=0.98, wspace=0.42)
    fig.savefig(FIG_DIR / "figC2b_task_region_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC2b_task_region_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def figure_trait_final_shift() -> pd.DataFrame:
    """Cleaner trait figure: final target-evidence shift, not token-region map."""
    cols = [
        "model", "game_code", "cb_id", "condition", "layer", "token_index",
        "score_canonical", "score_trait_target",
    ]
    condition_filter = {"cue_length_match_null"} | {f"cue_{t}" for t in TRAITS}
    tokens = _read_tokens(cols, condition_filter=condition_filter)
    tokens = tokens[tokens["layer"].eq(79)].copy()
    final = _final_token_scores(tokens, "score_canonical")
    final_tt = _final_token_scores(tokens, "score_trait_target")
    final["score_trait_target"] = final_tt["score_trait_target"].to_numpy()

    gf = pd.read_csv(game_features_csv(),
                     usecols=["game_code", "canonical_action_p1"])
    targets = trait_targets().merge(gf, on="game_code", how="left")
    records = []
    for trait, col_suffix in TRAITS.items():
        cue = f"cue_{trait}"
        tmp = final[final["condition"].isin([cue, "cue_length_match_null"])].merge(
            targets[["game_code", "canonical_action_p1", f"a_{col_suffix}"]],
            on="game_code",
            how="left",
        )
        tmp = tmp[tmp[f"a_{col_suffix}"].isin([0, 1])].copy()
        same_as_canon = tmp[f"a_{col_suffix}"].astype(int) == tmp["canonical_action_p1"].astype(int)
        tmp["target_score"] = np.where(
            tmp["condition"].eq(cue),
            tmp["score_trait_target"],
            np.where(same_as_canon, tmp["score_canonical"], -tmp["score_canonical"]),
        )
        wide = tmp.pivot_table(
            index=["model", "game_code", "cb_id"], columns="condition", values="target_score", aggfunc="first"
        ).reset_index()
        wide["trait"] = trait
        wide["cue_minus_placebo"] = wide[cue] - wide["cue_length_match_null"]
        records.append(wide[["model", "game_code", "cb_id", "trait", "cue_minus_placebo"]])
    dat = pd.concat(records, ignore_index=True)
    summary = (
        dat.groupby(["model", "trait"], as_index=False)
        .agg(mean_shift=("cue_minus_placebo", "mean"),
             sem=("cue_minus_placebo", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
             n=("cue_minus_placebo", "size"))
    )
    summary["lo"] = summary["mean_shift"] - 1.96 * summary["sem"]
    summary["hi"] = summary["mean_shift"] + 1.96 * summary["sem"]
    summary.to_csv(TABLE_DIR / "c3b_trait_final_target_shift_l79.csv", index=False)

    trait_order = list(TRAITS)
    y_base = np.arange(len(trait_order))
    offsets = {"qwen": -0.22, "qwen_instruct": 0.0, "llama31_instruct": 0.22}
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.axvline(0, color="#333333", lw=1, ls=":")
    for model in MODELS:
        sub = summary[summary["model"].eq(model)].set_index("trait").reindex(trait_order)
        y = y_base + offsets[model]
        x = sub["mean_shift"].to_numpy()
        xerr = np.vstack([x - sub["lo"].to_numpy(), sub["hi"].to_numpy() - x])
        ax.errorbar(x, y, xerr=xerr, fmt="o", ms=5, lw=1.3, capsize=2,
                    color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_yticks(y_base, [TRAIT_LABELS[t] for t in trait_order])
    ax.invert_yaxis()
    ax.set_xlabel("final L79 target-policy evidence shift (cue - placebo)")
    ax.set_title("Trait cues redirect final policy evidence unevenly")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figC3b_trait_final_shift.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figC3b_trait_final_shift.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    _ensure_dirs()
    bridge = figure_policy_bridge()
    semantic = figure_semantic_evidence()
    trait = figure_trait_redirection()
    incentive = figure_incentive_curves()
    task_regions = figure_task_region_bars()
    trait_final = figure_trait_final_shift()
    print("Wrote candidate Layer C figures:")
    for path in sorted(FIG_DIR.glob("figC*.png")):
        print(f"  {path.relative_to(ROOT)}")
    print("Key bridge correlations:")
    for model in MODELS:
        sub = bridge[bridge["model"].eq(model)]
        r = np.corrcoef(sub["final_l79_canonical_score"], sub["chose_canonical"])[0, 1]
        print(f"  {model}: r={r:.3f}, n={len(sub)}")
    print(f"Semantic summary rows: {len(semantic)}")
    print(f"Trait summary rows: {len(trait)}")
    print(f"Clean incentive-curve rows: {len(incentive)}")
    print(f"Clean task-region rows: {len(task_regions)}")
    print(f"Clean trait-final rows: {len(trait_final)}")


if __name__ == "__main__":
    main()
