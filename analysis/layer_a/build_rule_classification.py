#!/usr/bin/env python3
"""Build MGN rule-classification tables for Layer A Figure 3 panel B.

This intentionally reads corrected generated/commit choices directly from
``$SCA_DATA_ROOT/substrate/{model}/{game}/results.parquet``.

It does not depend on the potentially stale layerA_game_level cache.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


from strategic_anatomy.games import bruns_games  # noqa: E402
from analysis.layer_a import mgn_rules as MGN  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402
from strategic_anatomy.config import game_features_csv, human_refs_root, repo_root, results_root

ROOT = repo_root()


TAB_DIR = results_root() / "layer_a"
NAGEL = human_refs_root() / "raw" / "nagel"
GAME_FEATURES = game_features_csv()
SEED = 20260520
N_BOOT = 2000

LLMS = ("qwen", "qwen_instruct", "llama31_instruct", "gptoss")
AGENTS = LLMS + ("nagel",)
DISPLAY = {
    "qwen": "Qwen2.5-72B",
    "qwen_instruct": "Qwen2.5-72B-Instruct",
    "llama31_instruct": "Llama-3.1-70B-Instruct",
    "gptoss": "GPT-OSS-120B",
    "nagel": "Human (MGN)",
}


def _vec_tuple(x) -> tuple[int, ...]:
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return tuple(int(v) for v in x)


def project_game_table() -> pd.DataFrame:
    mat = loadmat(human_refs_root() / "raw" / "nagel" / "tuple_comparison" / "bruns_games.mat")["unnamed1"]
    mgn_vecs = {_vec_tuple(row) for row in mat}
    rows = []
    for game_code, payload in bruns_games.items():
        vec = _vec_tuple(payload[0])
        rows.append({"game_code": game_code, "vec8": vec})
    games = pd.DataFrame(rows).sort_values("game_code").reset_index(drop=True)
    if len(games) != 144 or games["vec8"].nunique() != 144:
        raise AssertionError("project Bruns game table is not 144 unique 8-vectors")
    missing = set(games["vec8"]) - mgn_vecs
    extra = mgn_vecs - set(games["vec8"])
    if missing or extra:
        raise AssertionError(f"MGN/project 8-vector mismatch: missing={len(missing)} extra={len(extra)}")
    return games


def mgn_perspective_map() -> pd.DataFrame:
    ros = pd.read_csv(NAGEL / "df_ros.csv")
    ros["vec8"] = ros["game_vector"].map(_vec_tuple)
    if ros["vec8"].nunique() != 144:
        raise AssertionError("df_ros does not contain 144 unique perspective vectors")
    return ros[["code", "vec8"]].copy()


def dominant_row_action(vec8: tuple[int, ...]) -> int | None:
    R, _ = MGN.reshape8(vec8)
    if np.all(R[0, :] > R[1, :]):
        return 0
    if np.all(R[1, :] > R[0, :]):
        return 1
    return None


def rule_prediction_table(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in games.itertuples(index=False):
        for rule in MGN.PANEL_RULES:
            rec = MGN.row_action_set_for_rule(rule, r.vec8)
            rows.append({
                "game_code": r.game_code,
                "rule": rule,
                "family": MGN.RULE_FAMILY.get(rule, "Other"),
                "rec_set": "".join(str(x) for x in sorted(rec)),
                "unique_action": next(iter(rec)) if len(rec) == 1 else np.nan,
                "unique": len(rec) == 1,
            })
    pred = pd.DataFrame(rows)
    wide = pred.pivot(index="game_code", columns="rule", values="unique_action")
    diagnostic = []
    for game_code, row in wide.iterrows():
        vals = {int(v) for v in row.dropna().astype(int).tolist()}
        diagnostic.append({"game_code": game_code, "diagnostic": len(vals) >= 2})
    return pred.merge(pd.DataFrame(diagnostic), on="game_code", how="left")


def validate_orientation(games: pd.DataFrame) -> dict[str, int]:
    feats = pd.read_csv(GAME_FEATURES)[["game_code", "dominant_action_p1"]]
    d = games.merge(feats, on="game_code", how="left")
    d["dominant_port"] = d["vec8"].map(dominant_row_action)
    chk = d.dropna(subset=["dominant_action_p1", "dominant_port"]).copy()
    chk["dominant_action_p1"] = chk["dominant_action_p1"].astype(int)
    chk["dominant_port"] = chk["dominant_port"].astype(int)
    n = len(chk)
    ok = int((chk["dominant_action_p1"] == chk["dominant_port"]).sum())
    if n != 72 or ok != 72:
        bad = chk[chk["dominant_action_p1"] != chk["dominant_port"]]
        raise AssertionError(f"orientation gate failed: {ok}/{n}; examples={bad.head().to_dict('records')}")
    return {"dominant_defined": n, "dominant_matches": ok}


def validate_mgn_rules() -> pd.DataFrame:
    ros = pd.read_csv(NAGEL / "df_ros.csv")
    choices = pd.read_csv(NAGEL / "normalized_data_20241612.csv")
    subjects = pd.read_csv(NAGEL / "subjects_merged_manually.csv")
    vecs = {r.code: _vec_tuple(r.game_vector) for r in ros.itertuples(index=False)}
    rows = []
    for rule in MGN.VALIDATION_RULES:
        if rule not in subjects.columns:
            continue
        per_game = []
        for code in [f"P{i}" for i in range(1, 145)]:
            y = choices[code].map({"A": 0, "B": 1}).to_numpy()
            pred = MGN.unique_action_for_rule(rule, vecs[code])
            hit = np.zeros(len(y), dtype=float) if pred is None else (y == pred).astype(float)
            per_game.append(hit)
        calc = np.vstack(per_game).T.mean(axis=1).mean()
        target = float(subjects[rule].mean())
        diff = float(calc - target)
        rows.append({
            "rule": rule,
            "computed_mean_subject_fit": calc,
            "mgn_column_mean": target,
            "abs_diff": abs(diff),
            "pass": abs(diff) <= 0.005,
        })
    out = pd.DataFrame(rows)
    if not bool(out["pass"].all()):
        raise AssertionError("MGN validation gate failed:\n" + out[~out["pass"]].to_string(index=False))
    return out


def load_llm_cells() -> pd.DataFrame:
    CS.assert_complete_pm1()
    frames = []
    for model in LLMS:
        d = CS.decisions(model)
        b = d[(d["player"].eq(1)) & (d["condition"].eq("baseline")) & (d["ok"])].copy()
        b = b.rename(columns={"cb": "cell_id"})
        b["agent"] = model
        frames.append(b[["agent", "game_code", "cell_id", "action"]])
    cells = pd.concat(frames, ignore_index=True)
    cells["action"] = cells["action"].astype(int)
    return cells


def modal_choices(cells: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent, game), d in cells.groupby(["agent", "game_code"], sort=True):
        counts = d["action"].value_counts()
        n0 = int(counts.get(0, 0))
        n1 = int(counts.get(1, 0))
        modal = np.nan if n0 == n1 else (0 if n0 > n1 else 1)
        rows.append({
            "agent": agent,
            "display": DISPLAY[agent],
            "game_code": game,
            "n_cells": int(len(d)),
            "n_act0": n0,
            "n_act1": n1,
            "modal_action": modal,
            "modal_unique": not np.isnan(modal),
        })
    out = pd.DataFrame(rows)
    full = pd.MultiIndex.from_product([LLMS, games["game_code"]], names=["agent", "game_code"]).to_frame(index=False)
    out = full.merge(out, on=["agent", "game_code"], how="left")
    out["display"] = out["agent"].map(DISPLAY)
    out["n_cells"] = out["n_cells"].fillna(0).astype(int)
    out["n_act0"] = out["n_act0"].fillna(0).astype(int)
    out["n_act1"] = out["n_act1"].fillna(0).astype(int)
    out["modal_unique"] = out["modal_unique"].fillna(False)
    return out


def human_game_hits() -> pd.DataFrame:
    # The human reference lives on MGN's P1..P144 perspective set. This is not the
    # same exact 144-vector set as the project's Bruns canonical game list, so we
    # score humans on the validated MGN perspective frame rather than forcing a
    # lossy project-game join.
    pmap = mgn_perspective_map()
    choices = pd.read_csv(NAGEL / "normalized_data_20241612.csv")
    rows = []
    for r in pmap.itertuples(index=False):
        y = choices[r.code].map({"A": 0, "B": 1}).to_numpy()
        unique = {MGN.unique_action_for_rule(rule, r.vec8) for rule in MGN.PANEL_RULES}
        unique = {int(v) for v in unique if v is not None}
        diagnostic = len(unique) >= 2
        for rule in MGN.PANEL_RULES:
            p = MGN.unique_action_for_rule(rule, r.vec8)
            hit = np.zeros(len(y), dtype=float) if p is None else (y == int(p)).astype(float)
            rows.append({
                "agent": "nagel",
                "display": DISPLAY["nagel"],
                "game_code": r.code,
                "rule": rule,
                "hit": float(hit.mean()),
                "diagnostic": bool(diagnostic),
            })
    return pd.DataFrame(rows)


def llm_game_hits(modal: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    d = modal.merge(pred[["game_code", "rule", "family", "unique_action", "diagnostic"]],
                    on="game_code", how="inner")
    d["hit"] = [
        MGN.rule_hit(set() if pd.isna(p) else {int(p)}, None if pd.isna(a) else int(a))
        for p, a in zip(d["unique_action"], d["modal_action"])
    ]
    return d[["agent", "display", "game_code", "rule", "hit", "diagnostic"]]


def _boot_ci_by_game(values: pd.DataFrame, game_col="game_code", value_col="hit") -> tuple[float, float]:
    games = values[game_col].to_numpy()
    vals = values[value_col].to_numpy(float)
    uniq = np.array(sorted(pd.unique(games)))
    idx = {g: np.where(games == g)[0] for g in uniq}
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        means.append(float(vals[ii].mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_rule_fit(hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent, rule), d in hits.groupby(["agent", "rule"], sort=False):
        rec = {"agent": agent, "display": DISPLAY[agent], "rule": rule,
               "family": MGN.RULE_FAMILY.get(rule, "Other")}
        for name, sub in [("diagnostic", d[d["diagnostic"]]), ("full", d)]:
            rec[f"n_games_{name}"] = int(sub["game_code"].nunique())
            rec[f"fit_{name}"] = float(sub["hit"].mean())
            lo, hi = _boot_ci_by_game(sub)
            rec[f"ci_lo_{name}"] = lo
            rec[f"ci_hi_{name}"] = hi
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_winners(rule_fit: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for agent in AGENTS:
        d = rule_fit[rule_fit["agent"].eq(agent)].sort_values(
            ["fit_diagnostic", "rule"], ascending=[False, True]
        )
        primary = d.iloc[0]
        secondary = d.iloc[1]
        hd = hits[(hits["agent"].eq(agent)) & (hits["diagnostic"])].copy()
        games = np.array(sorted(hd["game_code"].unique()))
        wins = 0
        for _ in range(N_BOOT):
            pick = rng.choice(games, size=len(games), replace=True)
            sub = pd.concat([hd[hd["game_code"].eq(g)] for g in pick], ignore_index=True)
            means = sub.groupby("rule")["hit"].mean()
            winner = means.sort_values(ascending=False).index[0]
            wins += int(winner == primary["rule"])
        rows.append({
            "agent": agent,
            "display": DISPLAY[agent],
            "primary_rule": primary["rule"],
            "secondary_rule": secondary["rule"],
            "primary_fit": primary["fit_diagnostic"],
            "primary_ci_lo": primary["ci_lo_diagnostic"],
            "primary_ci_hi": primary["ci_hi_diagnostic"],
            "secondary_fit": secondary["fit_diagnostic"],
            "stability": wins / N_BOOT,
            "classification": "pure" if primary["fit_diagnostic"] >= 0.90 else "hybrid",
            "n_games_diagnostic": int(primary["n_games_diagnostic"]),
        })
    return pd.DataFrame(rows)


def human_reference_table(human_hits: pd.DataFrame) -> pd.DataFrame:
    subjects = pd.read_csv(NAGEL / "subjects_merged_manually.csv")
    shares = subjects["winning_rule"].value_counts(normalize=True)
    rows = []
    for rule in MGN.VALIDATION_RULES:
        rec = {"rule": rule, "winning_rule_share": float(shares.get(rule, 0.0))}
        if rule in subjects.columns:
            rec["mean_subject_fit_full_mgn"] = float(subjects[rule].mean())
        else:
            rec["mean_subject_fit_full_mgn"] = np.nan
        if rule in set(MGN.PANEL_RULES):
            sub = human_hits[human_hits["rule"].eq(rule)]
            rec["mean_subject_fit_diagnostic"] = float(sub[sub["diagnostic"]]["hit"].mean())
            rec["mean_subject_fit_full_recomputed"] = float(sub["hit"].mean())
        else:
            rec["mean_subject_fit_diagnostic"] = np.nan
            rec["mean_subject_fit_full_recomputed"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def build_all() -> dict[str, pd.DataFrame]:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    games = project_game_table()
    orient = validate_orientation(games)
    validation = validate_mgn_rules()
    pred = rule_prediction_table(games)
    cells = load_llm_cells()
    modal = modal_choices(cells, games)
    llm_hits = llm_game_hits(modal, pred)
    human_hits = human_game_hits()
    hits = pd.concat([llm_hits, human_hits], ignore_index=True)
    rule_fit = summarize_rule_fit(hits)
    winners = summarize_winners(rule_fit, hits)
    human_ref = human_reference_table(human_hits)

    validation.to_csv(TAB_DIR / "b_rule_validation.csv", index=False)
    rule_fit.to_csv(TAB_DIR / "b_rule_fit.csv", index=False)
    winners.to_csv(TAB_DIR / "b_winning_rule.csv", index=False)
    human_ref.to_csv(TAB_DIR / "b_human_rule_reference.csv", index=False)

    print("MGN validation max abs diff:", round(float(validation["abs_diff"].max()), 6))
    print(f"orientation gate: {orient['dominant_matches']}/{orient['dominant_defined']}")
    print("diagnostic games:", int(pred[["game_code", "diagnostic"]].drop_duplicates()["diagnostic"].sum()))
    print("\n--- b_winning_rule ---")
    print(winners.round(3).to_string(index=False))
    return {
        "validation": validation,
        "rule_fit": rule_fit,
        "winners": winners,
        "human_ref": human_ref,
    }


def main() -> int:
    build_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
