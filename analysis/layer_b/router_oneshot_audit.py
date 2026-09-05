#!/usr/bin/env python3
"""One-shot GPT-OSS router audit for Layer B.

This is deliberately descriptive. It reads the corrected integrated one-shot root
(`$SCA_DATA_ROOT/substrate/gptoss`) and does three things:

1. prove provenance/coverage for commit-token router tensors;
2. compare router gate logits, top-k expert-load vectors, and commit residuals on
   the same rows and targets;
3. test whether router state predicts pure final-channel choices beyond objective
   incentive/counterbalance covariates.

No causal claims are made here. Mixed GPT-OSS commitments are excluded from the
primary choice target because their resolved `realized_action` is a deterministic
seeded draw, not a pure action commitment.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "4")


from collection.oneshot_common import canonical_sign, delta1, u_mats  # noqa: E402
from strategic_anatomy.config import game_features_csv, repo_root, results_root, substrate_root, taxonomy_dir

ROOT = repo_root()


DATA_ROOT = substrate_root() / "gptoss"
FEATURES = game_features_csv()
MASTER = taxonomy_dir() / "equivalence_per_canonical.csv"
OUT = results_root() / "layer_b"
OUT.mkdir(parents=True, exist_ok=True)

CV_SEED = 20260628
BOOT_SEED = 20260627
N_SPLITS = 5
N_BOOT = 1000


def game_dirs() -> list[Path]:
    dirs = [p for p in sorted(DATA_ROOT.iterdir()) if (p / "results.parquet").exists()]
    if len(dirs) != 144:
        raise AssertionError(f"expected 144 GPT-OSS games in {DATA_ROOT}, found {len(dirs)}")
    return dirs


def router_layers() -> list[int]:
    cfg = json.loads((game_dirs()[0] / "config.json").read_text())
    layers = [int(x) for x in cfg["router_layers"]]
    if not layers:
        raise AssertionError("no router layers in GPT-OSS config")
    return layers


def targets() -> pd.DataFrame:
    master = pd.read_csv(MASTER)[["bruns_name", "canonical_8vec"]].rename(
        columns={"bruns_name": "game_code"}
    )
    features = pd.read_csv(FEATURES)[
        [
            "game_code",
            "canonical_action_p1",
            "num_pure_ne",
            "dominance_profile",
            "complexity_score",
        ]
    ]
    meta = master.merge(features, on="game_code", how="left")
    rows = []
    for _, row in meta.iterrows():
        vec = tuple(float(x) for x in str(row["canonical_8vec"]).split(","))
        if len(vec) != 8:
            raise ValueError(f"expected canonical 8-vector, got {row['canonical_8vec']!r}")
        c1 = int(row["canonical_action_p1"])
        d1 = canonical_sign(delta1(vec, 0.5), c1)
        u1, _ = u_mats(vec)
        rows.append(
            {
                "game_code": row["game_code"],
                "canonical_action_p1": c1,
                "delta1c_q05": float(d1),
                "sign_delta1c_q05": int(d1 > 0),
                "abs_delta1c_q05": abs(float(d1)),
                "stim_cell01_ismax": int(round(float(u1[0, 1])) == 4),
                "stim_cell00_rank": int(round(float(u1[0, 0]))),
                "num_pure_ne": int(row["num_pure_ne"]),
                "dominance_profile": int(row["dominance_profile"]),
                "complexity_score": float(row["complexity_score"]),
            }
        )
    out = pd.DataFrame(rows)
    if out["game_code"].nunique() != 144:
        raise AssertionError("target table is not 144 games")
    return out


def all_results() -> pd.DataFrame:
    frames = []
    for gdir in game_dirs():
        df = pd.read_parquet(gdir / "results.parquet").rename(
            columns={"counterbalance_id": "cb"}
        )
        df["game_code"] = gdir.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(targets(), on="game_code", how="left")
    df["captured"] = pd.to_numeric(df["captured"], errors="coerce").fillna(0).astype(int)
    df["decoded_action_num"] = pd.to_numeric(df["decoded_action"], errors="coerce")
    df["realized_action_num"] = pd.to_numeric(df["realized_action"], errors="coerce")
    df["pure_commit"] = df["commit_type"].eq("pure")
    df["mixed_commit"] = df["commit_type"].eq("mixed")
    df["label_map_j_is_act0"] = df["label_map_p1"].fillna(df["label_map"]).eq("J=act0").astype(int)
    df["q_order_jp"] = df["q_order"].eq("JP").astype(int)
    df["literal_j_choice"] = df["move_letter"].eq("J").astype(float)
    df.loc[~df["pure_commit"], "literal_j_choice"] = np.nan
    df["aligned_pure"] = np.where(
        df["pure_commit"] & df["decoded_action_num"].isin([0, 1]),
        (df["decoded_action_num"] == df["canonical_action_p1"]).astype(int),
        np.nan,
    )
    df["aligned_realized"] = np.where(
        df["realized_action_num"].isin([0, 1]),
        (df["realized_action_num"] == df["canonical_action_p1"]).astype(int),
        np.nan,
    )
    return df


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cfg0 = json.loads((game_dirs()[0] / "config.json").read_text())
    rows.append(
        {
            "scope": "root",
            "n_games": df["game_code"].nunique(),
            "n_rows": len(df),
            "n_router_files": sum((g / "router.npz").exists() for g in game_dirs()),
            "substrate": cfg0.get("substrate"),
            "decoder": cfg0.get("decoder"),
            "feeding": cfg0.get("feeding"),
            "capture": cfg0.get("capture"),
            "cb": cfg0.get("cb"),
            "router_layers": ",".join(map(str, cfg0.get("router_layers", []))),
            "max_new_tokens": cfg0.get("max_new_tokens"),
        }
    )
    by = (
        df.groupby(["player", "condition", "commit_type"], dropna=False)
        .agg(n=("game_code", "size"), captured=("captured", "sum"))
        .reset_index()
    )
    for _, r in by.iterrows():
        rows.append(
            {
                "scope": f"player{int(r['player'])}:{r['condition']}:{r['commit_type']}",
                "n_games": df.loc[
                    (df["player"].eq(r["player"]))
                    & (df["condition"].eq(r["condition"]))
                    & (df["commit_type"].eq(r["commit_type"])),
                    "game_code",
                ].nunique(),
                "n_rows": int(r["n"]),
                "n_captured": int(r["captured"]),
                "capture_rate": float(r["captured"] / r["n"]) if r["n"] else np.nan,
                "substrate": "",
                "decoder": "",
                "feeding": "",
                "capture": "",
                "cb": "",
                "router_layers": "",
                "max_new_tokens": "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "router_oneshot_coverage.csv", index=False)
    return out


def decision_process_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize GPT-OSS token length before final-channel commitment.

    This is a process measure, not a reasoning-round count. The stored
    `final_text` is truncated, so level-k-style iteration counting would require
    regeneration from full traces.
    """
    rows: list[dict] = []
    if "n_new_tokens" not in df.columns:
        out = pd.DataFrame(
            [{"scope": "missing", "metric": "n_new_tokens", "note": "column unavailable"}]
        )
        out.to_csv(OUT / "router_oneshot_decision_process.csv", index=False)
        return out

    scopes = {
        "all_cells": df,
        "p1_baseline": _analysis_rows(df),
    }
    for scope, sub in scopes.items():
        for commit_type, g in sub.groupby("commit_type", dropna=False):
            token = pd.to_numeric(g["n_new_tokens"], errors="coerce").dropna()
            gen_s = pd.to_numeric(g.get("gen_s", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(
                {
                    "scope": scope,
                    "commit_type": commit_type,
                    "n": int(len(g)),
                    "n_games": int(g["game_code"].nunique()),
                    "n_new_tokens_median": float(token.median()) if len(token) else np.nan,
                    "n_new_tokens_mean": float(token.mean()) if len(token) else np.nan,
                    "n_new_tokens_q25": float(token.quantile(0.25)) if len(token) else np.nan,
                    "n_new_tokens_q75": float(token.quantile(0.75)) if len(token) else np.nan,
                    "n_new_tokens_max": float(token.max()) if len(token) else np.nan,
                    "gen_s_median": float(gen_s.median()) if len(gen_s) else np.nan,
                }
            )

    base = _analysis_rows(df)
    per_game = base.groupby("game_code")["n_new_tokens"].median()
    rows.append(
        {
            "scope": "p1_baseline_per_game_median",
            "commit_type": "all",
            "n": int(len(base)),
            "n_games": int(per_game.size),
            "n_new_tokens_median": float(per_game.median()),
            "n_new_tokens_mean": float(per_game.mean()),
            "n_new_tokens_q25": float(per_game.quantile(0.25)),
            "n_new_tokens_q75": float(per_game.quantile(0.75)),
            "n_new_tokens_min": float(per_game.min()),
            "n_new_tokens_max": float(per_game.max()),
            "gen_s_median": np.nan,
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "router_oneshot_decision_process.csv", index=False)
    return out


def _read_features(layer: int, rows: pd.DataFrame, feature_set: str) -> np.ndarray:
    X = []
    for _, r in rows.iterrows():
        gdir = DATA_ROOT / str(r["game_code"])
        cb = int(r["cb"])
        prefix = f"p{int(r['player'])}_{r['condition']}_cb{cb}_l{layer}"
        if feature_set == "residual":
            with np.load(gdir / "acts.npz") as z:
                X.append(np.asarray(z[prefix], dtype=np.float32))
        else:
            with np.load(gdir / "router.npz") as z:
                if feature_set == "router_gate":
                    X.append(np.asarray(z[f"{prefix}_gate"], dtype=np.float32))
                elif feature_set == "router_topk_weight":
                    idx = np.asarray(z[f"{prefix}_idx"], dtype=int)
                    weight = np.asarray(z[f"{prefix}_w"], dtype=np.float32)
                    vec = np.zeros(128, dtype=np.float32)
                    vec[idx] = weight
                    X.append(vec)
                elif feature_set == "router_topk_binary":
                    idx = np.asarray(z[f"{prefix}_idx"], dtype=int)
                    vec = np.zeros(128, dtype=np.float32)
                    vec[idx] = 1.0
                    X.append(vec)
                else:
                    raise ValueError(f"unknown feature_set: {feature_set}")
    return np.vstack(X).astype(np.float32)


def _layer_feature_bundle(layer: int, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Read all feature sets for a layer in one pass over game files.

    The row order matches ``rows.reset_index(drop=True)``. This avoids opening
    compressed NPZ files once per row per feature set.
    """
    rows = rows.reset_index(drop=True)
    n = len(rows)
    out: dict[str, list[np.ndarray | None]] = {
        "router_gate": [None] * n,
        "router_topk_weight": [None] * n,
        "router_topk_binary": [None] * n,
        "residual": [None] * n,
    }
    for game, idxs in rows.groupby("game_code").groups.items():
        gdir = DATA_ROOT / str(game)
        with np.load(gdir / "router.npz") as rz, np.load(gdir / "acts.npz") as az:
            for i in idxs:
                r = rows.iloc[int(i)]
                cb = int(r["cb"])
                prefix = f"p{int(r['player'])}_{r['condition']}_cb{cb}_l{layer}"
                gate = np.asarray(rz[f"{prefix}_gate"], dtype=np.float32)
                expert_idx = np.asarray(rz[f"{prefix}_idx"], dtype=int)
                expert_w = np.asarray(rz[f"{prefix}_w"], dtype=np.float32)
                topk_w = np.zeros(128, dtype=np.float32)
                topk_w[expert_idx] = expert_w
                topk_bin = np.zeros(128, dtype=np.float32)
                topk_bin[expert_idx] = 1.0
                out["router_gate"][int(i)] = gate
                out["router_topk_weight"][int(i)] = topk_w
                out["router_topk_binary"][int(i)] = topk_bin
                out["residual"][int(i)] = np.asarray(az[prefix], dtype=np.float32)
    bundled: dict[str, np.ndarray] = {}
    for key, vals in out.items():
        if any(v is None for v in vals):
            missing = sum(v is None for v in vals)
            raise RuntimeError(f"missing {missing} rows for feature set {key} at layer {layer}")
        bundled[key] = np.vstack(vals).astype(np.float32)  # type: ignore[arg-type]
    return bundled


def _transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    pca_dim: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    if pca_dim is None:
        return Xtr, Xte
    k = min(int(pca_dim), Xtr.shape[0] - 1, Xtr.shape[1])
    if k < 1:
        return Xtr, Xte
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(Xtr), pca.transform(Xte)


def cv_binary_scores(
    X: np.ndarray,
    y: np.ndarray,
    games: np.ndarray,
    *,
    pca_dim: int | None,
    seed: int = CV_SEED,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    X = np.asarray(X, dtype=float)[keep]
    yy = y[keep].astype(int)
    gg = np.asarray(games)[keep]
    score = np.full(len(yy), np.nan)
    n_splits = min(N_SPLITS, len(np.unique(gg)))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, yy, groups=gg)):
        if len(np.unique(yy[train])) < 2:
            continue
        Xtr, Xte = _transform_train_test(
            X[train], X[test], pca_dim=pca_dim, seed=seed + fold
        )
        clf = LogisticRegression(C=0.5, max_iter=2000, random_state=seed + fold)
        clf.fit(Xtr, yy[train])
        classes = list(clf.classes_)
        score[test] = clf.predict_proba(Xte)[:, classes.index(1)]
    out = np.full(len(y), np.nan)
    out[np.where(keep)[0]] = score
    return out


def _auc_ci_by_game(y: np.ndarray, score: np.ndarray, games: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    games = np.asarray(games)
    keep = np.isfinite(y) & np.isfinite(score)
    if keep.sum() < 8 or len(np.unique(y[keep])) < 2:
        return np.nan, np.nan, np.nan
    auc = float(roc_auc_score(y[keep].astype(int), score[keep]))
    unique = np.unique(games[keep])
    idx = {g: np.where((games == g) & keep)[0] for g in unique}
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        boots.append(roc_auc_score(y[rows].astype(int), score[rows]))
    return auc, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _logloss_ci_improvement(
    y: np.ndarray,
    base_score: np.ndarray,
    aug_score: np.ndarray,
    games: np.ndarray,
) -> tuple[float, float, float, float, float]:
    y = np.asarray(y, dtype=float)
    base_score = np.asarray(base_score, dtype=float)
    aug_score = np.asarray(aug_score, dtype=float)
    games = np.asarray(games)
    keep = np.isfinite(y) & np.isfinite(base_score) & np.isfinite(aug_score)
    yy = y[keep].astype(int)
    bs = np.clip(base_score[keep], 1e-6, 1 - 1e-6)
    ag = np.clip(aug_score[keep], 1e-6, 1 - 1e-6)
    gg = games[keep]
    if keep.sum() < 8 or len(np.unique(yy)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    base_ll = float(log_loss(yy, bs, labels=[0, 1]))
    aug_ll = float(log_loss(yy, ag, labels=[0, 1]))
    imp = base_ll - aug_ll
    unique = np.unique(gg)
    idx = {g: np.where(gg == g)[0] for g in unique}
    rng = np.random.default_rng(BOOT_SEED + 31)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        if len(np.unique(yy[rows])) < 2:
            continue
        boots.append(
            log_loss(yy[rows], bs[rows], labels=[0, 1])
            - log_loss(yy[rows], ag[rows], labels=[0, 1])
        )
    return imp, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), base_ll, aug_ll


def baseline_design(rows: pd.DataFrame) -> np.ndarray:
    return rows[
        [
            "delta1c_q05",
            "abs_delta1c_q05",
            "label_map_j_is_act0",
            "q_order_jp",
            "complexity_score",
        ]
    ].to_numpy(dtype=float)


def _analysis_rows(df: pd.DataFrame, *, condition: str = "baseline") -> pd.DataFrame:
    rows = df[
        (df["player"].eq(1))
        & (df["condition"].eq(condition))
        & (df["captured"].eq(1))
    ].copy()
    rows = rows.sort_values(["game_code", "cb"]).reset_index(drop=True)
    return rows


def decodability(df: pd.DataFrame) -> pd.DataFrame:
    base = _analysis_rows(df)
    target_specs = [
        ("incentive_sign_q05", "sign_delta1c_q05", "all", "objective incentive sign"),
        ("pure_canonical_choice", "aligned_pure", "pure", "pure final-channel canonical choice"),
        ("realized_canonical_choice", "aligned_realized", "all", "realized canonical choice incl mixed"),
        ("literal_J_choice", "literal_j_choice", "pure", "literal J/P choice control"),
        ("stimulus_control", "stim_cell01_ismax", "all", "strategically inert payoff control"),
    ]
    feature_sets = [
        ("router_gate", None),
        ("router_topk_weight", None),
        ("router_topk_binary", None),
        ("residual", 64),
    ]
    rows_out = []
    for layer in router_layers():
        bundle = _layer_feature_bundle(layer, base)
        for fs, pca_dim in feature_sets:
            X_all = bundle[fs]
            for target, col, scope, label in target_specs:
                sub = base.copy()
                X = X_all
                if scope == "pure":
                    mask = sub["pure_commit"].to_numpy()
                    sub = sub.loc[mask].reset_index(drop=True)
                    X = X_all[mask]
                y = pd.to_numeric(sub[col], errors="coerce").to_numpy()
                games = sub["game_code"].to_numpy()
                if np.isfinite(y).sum() < 8 or len(np.unique(y[np.isfinite(y)])) < 2:
                    auc = lo = hi = np.nan
                    n = int(np.isfinite(y).sum())
                    n_games = int(pd.Series(games[np.isfinite(y)]).nunique())
                else:
                    score = cv_binary_scores(X, y, games, pca_dim=pca_dim)
                    auc, lo, hi = _auc_ci_by_game(y, score, games)
                    n = int(np.isfinite(score).sum())
                    n_games = int(pd.Series(games[np.isfinite(score)]).nunique())
                rows_out.append(
                    {
                        "layer": layer,
                        "feature_set": fs,
                        "target": target,
                        "target_label": label,
                        "scope": scope,
                        "auc": auc,
                        "auc_lo": lo,
                        "auc_hi": hi,
                        "n": n,
                        "n_games": n_games,
                    }
                )
            print(f"[decodability] L{layer:02d} {fs}", flush=True)
    out = pd.DataFrame(rows_out)
    out.to_csv(OUT / "router_oneshot_decodability.csv", index=False)
    return out


def behavior_link(df: pd.DataFrame) -> pd.DataFrame:
    base = _analysis_rows(df)
    feature_sets = [
        ("router_gate", None),
        ("router_topk_weight", None),
        ("router_topk_binary", None),
        ("residual", 64),
    ]
    target_specs = [
        ("pure_canonical_choice", "aligned_pure", "pure"),
        ("realized_canonical_choice", "aligned_realized", "all"),
    ]
    rows_out = []
    for target, col, scope in target_specs:
        target_rows = base.copy()
        if scope == "pure":
            target_rows = target_rows[target_rows["pure_commit"]].reset_index(drop=True)
        y = pd.to_numeric(target_rows[col], errors="coerce").to_numpy()
        games = target_rows["game_code"].to_numpy()
        B = baseline_design(target_rows)
        base_score = cv_binary_scores(B, y, games, pca_dim=None, seed=CV_SEED + 101)
        base_auc, base_auc_lo, base_auc_hi = _auc_ci_by_game(y, base_score, games)
        for layer in router_layers():
            bundle = _layer_feature_bundle(layer, base)
            for fs, pca_dim in feature_sets:
                X_all = bundle[fs]
                if scope == "pure":
                    X_feat = X_all[base["pure_commit"].to_numpy()]
                else:
                    X_feat = X_all
                X_aug = np.hstack([B, X_feat])
                # Residual dimensionality needs compression; router features do not.
                aug_pca = 64 if fs == "residual" else None
                aug_score = cv_binary_scores(
                    X_aug, y, games, pca_dim=aug_pca, seed=CV_SEED + 201
                )
                aug_auc, aug_auc_lo, aug_auc_hi = _auc_ci_by_game(y, aug_score, games)
                imp, imp_lo, imp_hi, base_ll, aug_ll = _logloss_ci_improvement(
                    y, base_score, aug_score, games
                )
                rows_out.append(
                    {
                        "layer": layer,
                        "feature_set": fs,
                        "target": target,
                        "scope": scope,
                        "objective_baseline_auc": base_auc,
                        "objective_baseline_auc_lo": base_auc_lo,
                        "objective_baseline_auc_hi": base_auc_hi,
                        "augmented_auc": aug_auc,
                        "augmented_auc_lo": aug_auc_lo,
                        "augmented_auc_hi": aug_auc_hi,
                        "logloss_improvement": imp,
                        "logloss_improvement_lo": imp_lo,
                        "logloss_improvement_hi": imp_hi,
                        "objective_baseline_logloss": base_ll,
                        "augmented_logloss": aug_ll,
                        "n": int(np.isfinite(aug_score).sum()),
                        "n_games": int(pd.Series(games[np.isfinite(aug_score)]).nunique()),
                    }
                )
            print(f"[behavior-link] {target} L{layer:02d}", flush=True)
    out = pd.DataFrame(rows_out)
    out.to_csv(OUT / "router_oneshot_behavior_link.csv", index=False)
    return out


def best_summary(dec: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in dec["target"].unique():
        for fs in dec["feature_set"].unique():
            sub = dec[(dec["target"].eq(target)) & (dec["feature_set"].eq(fs))]
            sub = sub.dropna(subset=["auc"]).sort_values("auc", ascending=False)
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    {
                        "analysis": "decodability",
                        "target": target,
                        "feature_set": fs,
                        "best_layer": int(r["layer"]),
                        "value": float(r["auc"]),
                        "lo": float(r["auc_lo"]),
                        "hi": float(r["auc_hi"]),
                        "n": int(r["n"]),
                        "n_games": int(r["n_games"]),
                    }
                )
    for target in link["target"].unique():
        for fs in link["feature_set"].unique():
            sub = link[(link["target"].eq(target)) & (link["feature_set"].eq(fs))]
            sub = sub.dropna(subset=["logloss_improvement"]).sort_values(
                "logloss_improvement", ascending=False
            )
            if not sub.empty:
                r = sub.iloc[0]
                rows.append(
                    {
                        "analysis": "behavior_link_logloss_improvement",
                        "target": target,
                        "feature_set": fs,
                        "best_layer": int(r["layer"]),
                        "value": float(r["logloss_improvement"]),
                        "lo": float(r["logloss_improvement_lo"]),
                        "hi": float(r["logloss_improvement_hi"]),
                        "n": int(r["n"]),
                        "n_games": int(r["n_games"]),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "router_oneshot_best_summary.csv", index=False)
    return out


def write_report(
    cov: pd.DataFrame,
    dec: pd.DataFrame,
    link: pd.DataFrame,
    summary: pd.DataFrame,
    process: pd.DataFrame,
) -> None:
    root_row = cov[cov["scope"].eq("root")].iloc[0]
    pure = cov[cov["scope"].str.contains("player1:baseline:pure", regex=False)]
    mixed = cov[cov["scope"].str.contains("player1:baseline:mixed", regex=False)]
    none = cov[cov["scope"].str.contains("player1:baseline:none", regex=False)]
    pure_proc = process[
        process["scope"].eq("all_cells") & process["commit_type"].eq("pure")
    ]
    mixed_proc = process[
        process["scope"].eq("all_cells") & process["commit_type"].eq("mixed")
    ]
    per_game_proc = process[process["scope"].eq("p1_baseline_per_game_median")]
    lines = [
        "# GPT-OSS One-Shot Router Audit",
        "",
        "Descriptive only. No causal steering or router-edit claim is made.",
        "",
        "## Provenance",
        "",
        f"- Root: `{DATA_ROOT}`",
        f"- Substrate: `{root_row['substrate']}`",
        f"- Decoder: `{root_row['decoder']}`",
        f"- Feeding: `{root_row['feeding']}`",
        f"- Capture: `{root_row['capture']}`",
        f"- Counterbalance: `{root_row['cb']}`",
        f"- Router layers: `{root_row['router_layers']}`",
        f"- Games: {int(root_row['n_games'])}; rows: {int(root_row['n_rows'])}; router files: {int(root_row['n_router_files'])}",
        "",
        "## Baseline Choice Rows",
        "",
        f"- Pure P1 baseline commits: {int(pure['n_rows'].sum()) if not pure.empty else 0}",
        f"- Mixed P1 baseline commits: {int(mixed['n_rows'].sum()) if not mixed.empty else 0}",
        f"- None P1 baseline commits: {int(none['n_rows'].sum()) if not none.empty else 0}",
        "",
        "Primary router-to-choice analyses use pure final-channel commitments only. "
        "All-row realized-action analyses are robustness/comparability only because mixed rows are seeded resolutions.",
        "",
        "## Decision Process Length",
        "",
        "This is a token-to-final-commit measure, not a discrete reasoning-round count.",
    ]
    if not pure_proc.empty and not mixed_proc.empty:
        lines.extend(
            [
                f"- All cells, pure commits: median {pure_proc.iloc[0]['n_new_tokens_median']:.0f} new tokens "
                f"({pure_proc.iloc[0]['gen_s_median']:.2f} s median generation).",
                f"- All cells, mixed commits: median {mixed_proc.iloc[0]['n_new_tokens_median']:.0f} new tokens "
                f"({mixed_proc.iloc[0]['gen_s_median']:.2f} s median generation).",
            ]
        )
    if not per_game_proc.empty:
        r = per_game_proc.iloc[0]
        lines.append(
            f"- P1 baseline per-game median range: {r['n_new_tokens_min']:.0f} to "
            f"{r['n_new_tokens_max']:.1f} new tokens."
        )
    lines.extend(
        [
            "",
            "## Headline Summary",
            "",
        ]
    )
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['analysis']} · {r['target']} · {r['feature_set']} · L{int(r['best_layer'])}: "
            f"{r['value']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] (n={int(r['n'])}, games={int(r['n_games'])})"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Router gate logits/top-k vectors are read at the GPT-OSS commit token, not at the prompt slot.",
            "- Decodability of objective incentive is representation, not use.",
            "- Decodability of pure canonical choice is a readout of final commitment, not causality.",
            "- Log-loss improvement over objective incentive/counterbalance covariates is the descriptive behavior-link statistic.",
            "- Direct router editing is not claimed here; GPT-OSS MXFP4 routing is treated as read-only.",
        ]
    )
    (OUT / "router_oneshot_audit_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    print("[router-audit] loading GPT-OSS integrated one-shot root")
    df = all_results()
    cov = coverage_table(df)
    print("[router-audit] coverage written")
    process = decision_process_table(df)
    print("[router-audit] decision process written")
    dec = decodability(df)
    print("[router-audit] decodability written")
    link = behavior_link(df)
    print("[router-audit] behavior link written")
    summary = best_summary(dec, link)
    write_report(cov, dec, link, summary, process)
    print(f"[router-audit] wrote tables under {OUT}")


if __name__ == "__main__":
    main()
