#!/usr/bin/env python3
"""Shared loaders and estimator primitives for Layer B recruitment final.

This module is deliberately tied to the corrected integrated root:
``$SCA_DATA_ROOT/substrate``. It reuses current project metadata and pure
math helpers, but it does not read old Layer B panel tables.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "4")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


from collection.oneshot_common import canonical_sign, delta1, delta2  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402
from analysis.layer_a.src.shared_data import parse_8vec  # noqa: E402
from strategic_anatomy.config import game_features_csv, repo_root, results_root, substrate_root, taxonomy_dir

ROOT = repo_root()


HERE = Path(__file__).resolve().parents[1]
DATA = results_root() / "layer_b" / "recruitment" / "_data"
TABLES = results_root() / "layer_b" / "recruitment"
FIGURES = HERE / "figures"
for _path in (DATA, TABLES, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)

CORRECTED_ROOT = substrate_root()
GAME_FEATURES = game_features_csv()
MASTER_CANON = taxonomy_dir() / "human_game_master_per_canonical.csv"
LAYERA_TRAIT_AIM = results_root() / "layer_a" / "f2_trait_aim.csv"
LAYERA_LAMBDA = results_root() / "layer_a" / "f3_lambda.csv"
A5_RUN = (
    ROOT
    / "analysis"
    / "block_b"
    / "tables"
    / "causal_oneshot"
    / "runs"
    / "a5full_20260620_200509"
)
ROUTER_VERDICTS = (
    ROOT
    / "analysis"
    / "block_b"
    / "tables"
    / "moe_router_causal"
    / "summary"
    / "router_causal_verdicts.csv"
)

MODELS = ["qwen_instruct", "qwen", "llama31_instruct", "gptoss"]
SHORT = {
    "qwen_instruct": "Qwen-I",
    "qwen": "Qwen-B",
    "llama31_instruct": "Llama",
    "gptoss": "GPT-OSS",
}
NICE = {
    "qwen_instruct": "Qwen2.5-72B-Instruct",
    "qwen": "Qwen2.5-72B",
    "llama31_instruct": "Llama-3.1-70B-Instruct",
    "gptoss": "GPT-OSS-120B",
}
COLORS = {
    "qwen_instruct": "#2f6fbb",
    "qwen": "#2a9d55",
    "llama31_instruct": "#c84a3a",
    "gptoss": "#7d5cc6",
}

CUES = ["risk_aversion", "loss_aversion", "inequity_aversion", "maximin", "selfish_maximizer"]
PLACEBO = "length_match_null"
TRAIT_LABEL = {
    "risk_aversion": "risk",
    "loss_aversion": "loss",
    "inequity_aversion": "inequity",
    "maximin": "maximin",
    "selfish_maximizer": "selfish",
}

RNG_SEED = 20260627
N_BOOT = 1000
N_SPLITS = 5
PCA_DIM = 96
PCA_DIM_LDA = 80
LOGREG_MAX_ITER = 2000


def model_root(model: str) -> Path:
    root = CORRECTED_ROOT / model
    if not root.exists():
        raise FileNotFoundError(f"corrected model root missing: {root}")
    return root


def game_dirs(model: str) -> list[Path]:
    dirs = [p for p in sorted(model_root(model).iterdir()) if (p / "results.parquet").exists()]
    if len(dirs) != 144:
        raise AssertionError(f"{model}: expected 144 corrected games, found {len(dirs)}")
    return dirs


def read_results(model: str, columns: list[str] | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(g / "results.parquet", columns=columns) for g in game_dirs(model)]
    out = pd.concat(frames, ignore_index=True).rename(columns={"counterbalance_id": "cb"})
    out["model"] = model
    return out


def normalized_results(model: str) -> pd.DataFrame:
    return CS.normalized_results(model)


def decisions(model: str) -> pd.DataFrame:
    return CS.decisions(model)


def game_targets() -> pd.DataFrame:
    """One row per game with canonical-axis targets and q=0.5 incentive gaps."""
    master = pd.read_csv(MASTER_CANON)
    features = pd.read_csv(GAME_FEATURES)
    cols = [
        "game_code",
        "canonical_action_p1",
        "canonical_action_p2",
        "dominant_action_p1",
        "dominant_action_p2",
        "num_pure_ne",
        "complexity_score",
    ]
    meta = master[["game_code", "canonical_8vec", "nagel_lk_type"]].merge(
        features[cols], on="game_code", how="left"
    )
    rows: list[dict] = []
    for _, row in meta.iterrows():
        vec = parse_8vec(row["canonical_8vec"])
        c1 = int(row["canonical_action_p1"])
        c2 = int(row["canonical_action_p2"])
        p1 = np.asarray(vec[:4], dtype=float)
        d1c_q05 = canonical_sign(delta1(vec, 0.5), c1)
        d2c_q05 = canonical_sign(delta2(vec, 0.5), c2)
        rows.append(
            {
                "game_code": row["game_code"],
                "canonical_8vec": row["canonical_8vec"],
                "canonical_action_p1": c1,
                "canonical_action_p2": c2,
                "dominant_action_p1": row["dominant_action_p1"],
                "dominant_action_p2": row["dominant_action_p2"],
                "num_pure_ne": int(row["num_pure_ne"]),
                "complexity_score": float(row["complexity_score"]),
                "nagel_lk_type": row["nagel_lk_type"],
                "stim_rank00": int(np.argsort(np.argsort(p1))[0] + 1),
                "delta1c_q05": float(d1c_q05),
                "sign_delta1c_q05": int(d1c_q05 > 0),
                "delta2c_q05": float(d2c_q05),
                "sign_delta2c_q05": int(d2c_q05 > 0),
            }
        )
    out = pd.DataFrame(rows)
    if out["game_code"].nunique() != 144:
        raise AssertionError("game target join did not produce 144 games")
    return out


def q2_map(model: str) -> pd.Series:
    d = decisions(model)
    p2 = d[(d["player"].eq(2)) & (d["condition"].eq("baseline")) & (d["ok"])]
    return p2.groupby("game_code")["action"].apply(lambda x: float((x.astype(int) == 0).mean()))


def attach_targets(meta: pd.DataFrame, model: str) -> pd.DataFrame:
    d = decisions(model)[["game_code", "player", "condition", "cb", "action", "ok"]]
    out = meta.merge(d, on=["game_code", "player", "condition", "cb"], how="left")
    out = out.merge(game_targets(), on="game_code", how="left")
    out["ok"] = out["ok"].fillna(False).astype(bool)
    is_p1 = out["player"].eq(1)
    out["aligned_canonical_p1"] = np.where(
        is_p1 & out["ok"],
        (out["action"].astype(float) == out["canonical_action_p1"].astype(float)).astype(int),
        np.nan,
    )
    q2 = q2_map(model)
    vecs = dict(zip(out["game_code"], out["canonical_8vec"]))
    emp = []
    for game, canon in zip(out["game_code"], out["canonical_action_p1"]):
        q = q2.get(game, np.nan)
        if pd.isna(q):
            emp.append(np.nan)
        else:
            emp.append(canonical_sign(delta1(parse_8vec(vecs[game]), float(q)), int(canon)))
    out["delta1c_emp"] = emp
    return out


def available_layers(model: str) -> list[int]:
    sample = game_dirs(model)[0] / "acts.npz"
    with np.load(sample) as z:
        layers = sorted(
            {
                int(key.rsplit("_l", 1)[1])
                for key in z.files
                if "_seq" not in key and "_l" in key and key.rsplit("_l", 1)[1].isdigit()
            }
        )
    return layers


def selected_layers(model: str) -> list[int]:
    layers = available_layers(model)
    max_layer = max(layers)
    if max_layer >= 80:
        wanted = [0, 1] + list(range(5, max_layer + 1, 5)) + [max_layer]
    else:
        wanted = [0, 1] + list(range(3, max_layer + 1, 3)) + [max_layer]
    return sorted(set(layer for layer in wanted if layer in layers))


def hidden_dim(model: str) -> int:
    sample = game_dirs(model)[0] / "acts.npz"
    with np.load(sample) as z:
        return int(z[z.files[0]].shape[0])


def residual_panel(
    model: str,
    layer: int,
    *,
    player: int = 1,
    conditions: Iterable[str] = ("baseline",),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load residual vectors for a set of conditions at one layer.

    The row order is deterministic and is keyed by game, condition, and cb. Only
    residual keys are read; sequence activations are ignored.
    """
    X: list[np.ndarray] = []
    rows: list[dict] = []
    wanted = list(conditions)
    for game_dir in game_dirs(model):
        game = game_dir.name
        with np.load(game_dir / "acts.npz") as z:
            keys = set(z.files)
            for condition in wanted:
                for cb in range(4):
                    key = f"p{player}_{condition}_cb{cb}_l{layer}"
                    if key in keys:
                        X.append(np.asarray(z[key], dtype=np.float32))
                        rows.append(
                            {
                                "model": model,
                                "game_code": game,
                                "player": player,
                                "condition": condition,
                                "cb": cb,
                                "layer": layer,
                            }
                        )
    if not X:
        return np.zeros((0, hidden_dim(model)), dtype=np.float32), pd.DataFrame(rows)
    return np.stack(X), pd.DataFrame(rows)


def baseline_layers(model: str, layers: Iterable[int]) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Load P1 baseline residuals for several layers while opening each npz once."""
    layers = list(layers)
    X_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    rows: list[dict] = []
    for game_dir in game_dirs(model):
        game = game_dir.name
        with np.load(game_dir / "acts.npz") as z:
            keys = set(z.files)
            for cb in range(4):
                if all(f"p1_baseline_cb{cb}_l{layer}" in keys for layer in layers):
                    for layer in layers:
                        X_by_layer[layer].append(np.asarray(z[f"p1_baseline_cb{cb}_l{layer}"], dtype=np.float32))
                    rows.append(
                        {
                            "model": model,
                            "game_code": game,
                            "player": 1,
                            "condition": "baseline",
                            "cb": cb,
                        }
                    )
    out = {layer: np.stack(vals) for layer, vals in X_by_layer.items()}
    return out, pd.DataFrame(rows)


def cue_shift_panel(model: str, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Cue minus baseline residual shifts for the five disposition cues."""
    X: list[np.ndarray] = []
    rows: list[dict] = []
    for game_dir in game_dirs(model):
        game = game_dir.name
        with np.load(game_dir / "acts.npz") as z:
            keys = set(z.files)
            for cb in range(4):
                bkey = f"p1_baseline_cb{cb}_l{layer}"
                if bkey not in keys:
                    continue
                base = np.asarray(z[bkey], dtype=np.float32)
                for cue in CUES:
                    cond = f"cue_{cue}"
                    ckey = f"p1_{cond}_cb{cb}_l{layer}"
                    if ckey not in keys:
                        continue
                    X.append(np.asarray(z[ckey], dtype=np.float32) - base)
                    rows.append(
                        {
                            "model": model,
                            "game_code": game,
                            "player": 1,
                            "condition": cond,
                            "cue": cue,
                            "cb": cb,
                            "layer": layer,
                        }
                    )
    return np.stack(X), pd.DataFrame(rows)


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0 or not np.isfinite(n):
        return np.zeros_like(v)
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au, bu = unit(a), unit(b)
    if not au.any() or not bu.any():
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(au @ bu), -1.0, 1.0))))


def decision_axis(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
        return np.zeros(X.shape[1])
    Xk = np.asarray(X[keep], dtype=float)
    yk = y[keep].astype(int)
    return unit(Xk[yk == 1].mean(axis=0) - Xk[yk == 0].mean(axis=0))


def incentive_axis(X: np.ndarray, d1c: np.ndarray) -> np.ndarray:
    d1c = np.asarray(d1c, dtype=float)
    keep = np.isfinite(d1c)
    if keep.sum() < 5 or np.nanstd(d1c[keep]) == 0:
        return np.zeros(X.shape[1])
    Xk = np.asarray(X[keep], dtype=float)
    Xc = Xk - Xk.mean(axis=0)
    z = d1c[keep] - d1c[keep].mean()
    return unit((Xc.T @ z) / float(z @ z))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 5 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def _fold_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    pca_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    k = int(min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1]))
    if k < 1:
        return Xtr, Xte
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(Xtr), pca.transform(Xte)


def _boot_groups(games: np.ndarray) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    unique = np.unique(games)
    return unique, {game: np.where(games == game)[0] for game in unique}


def boot_mean_by_game(
    values: np.ndarray,
    games: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    games = np.asarray(games)
    keep = np.isfinite(values)
    values = values[keep]
    games = games[keep]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    unique, idx = _boot_groups(games)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([idx[g] for g in sampled])
        boots[i] = values[rows].mean()
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _boot_auc(y: np.ndarray, score: np.ndarray, games: np.ndarray, *, multiclass: bool = False) -> tuple[float, float]:
    unique, idx = _boot_groups(games)
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([idx[g] for g in sampled])
        try:
            if multiclass:
                yy = y[rows]
                ss = score[rows]
                present = yy.sum(axis=0) > 0
                if present.sum() >= 2:
                    vals.append(float(roc_auc_score(yy[:, present], ss[:, present], average="macro")))
            elif len(np.unique(y[rows])) >= 2:
                vals.append(float(roc_auc_score(y[rows], score[rows])))
        except ValueError:
            continue
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def group_auc(X: np.ndarray, y: np.ndarray, games: np.ndarray, *, pca_dim: int = PCA_DIM) -> dict:
    y = np.asarray(y)
    games = np.asarray(games)
    keep = pd.notna(pd.Series(y)).to_numpy()
    X = X[keep]
    y = y[keep].astype(int)
    games = games[keep]
    out = {
        "metric": "auc",
        "value": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "base_rate": np.nan,
        "n_obs": int(len(y)),
        "n_games": int(np.unique(games).size) if len(games) else 0,
    }
    if len(y) < 8 or len(np.unique(y)) < 2:
        return out
    out["base_rate"] = float(y.mean())
    score = np.full(len(y), np.nan)
    n_splits = min(N_SPLITS, len(np.unique(games)))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=games)):
        if len(np.unique(y[train])) < 2:
            continue
        Xtr, Xte = _fold_pipeline(X[train], X[test], pca_dim=pca_dim, seed=RNG_SEED + fold)
        clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RNG_SEED + fold)
        clf.fit(Xtr, y[train])
        classes = list(clf.classes_)
        score[test] = clf.predict_proba(Xte)[:, classes.index(1)]
    used = np.isfinite(score)
    if used.sum() and len(np.unique(y[used])) >= 2:
        out["value"] = float(roc_auc_score(y[used], score[used]))
        out["ci_lo"], out["ci_hi"] = _boot_auc(y[used], score[used], games[used])
        out["n_obs"] = int(used.sum())
        out["n_games"] = int(np.unique(games[used]).size)
    return out


def group_auc_multiclass(X: np.ndarray, y: np.ndarray, games: np.ndarray, *, pca_dim: int = PCA_DIM) -> dict:
    y = np.asarray(y)
    games = np.asarray(games)
    keep = pd.notna(pd.Series(y)).to_numpy()
    X = X[keep]
    y = y[keep]
    games = games[keep]
    classes = sorted(pd.unique(pd.Series(y)).tolist())
    out = {
        "metric": "macro_ovr_auc",
        "value": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "base_rate": np.nan,
        "n_obs": int(len(y)),
        "n_games": int(np.unique(games).size) if len(games) else 0,
    }
    if len(classes) < 2 or len(y) < 8:
        return out
    probs = np.full((len(y), len(classes)), np.nan)
    col = {cls: i for i, cls in enumerate(classes)}
    n_splits = min(N_SPLITS, len(np.unique(games)))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=games)):
        if len(np.unique(y[train])) < 2:
            continue
        Xtr, Xte = _fold_pipeline(X[train], X[test], pca_dim=pca_dim, seed=RNG_SEED + fold)
        clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=RNG_SEED + fold)
        clf.fit(Xtr, y[train])
        pred = clf.predict_proba(Xte)
        block = np.zeros((len(test), len(classes)))
        for j, cls in enumerate(clf.classes_):
            block[:, col[cls]] = pred[:, j]
        probs[test] = block
    used = np.isfinite(probs).all(axis=1)
    if used.sum():
        yy = pd.get_dummies(pd.Categorical(y[used], categories=classes)).to_numpy()
        present = yy.sum(axis=0) > 0
        if present.sum() >= 2:
            out["value"] = float(roc_auc_score(yy[:, present], probs[used][:, present], average="macro"))
            out["ci_lo"], out["ci_hi"] = _boot_auc(yy, probs[used], games[used], multiclass=True)
            out["n_obs"] = int(used.sum())
            out["n_games"] = int(np.unique(games[used]).size)
    return out


def lda_predictions(X: np.ndarray, labels: np.ndarray, games: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(labels)
    games = np.asarray(games)
    pred = np.full(len(labels), None, dtype=object)
    n_splits = min(N_SPLITS, len(np.unique(games)))
    for fold, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, labels, groups=games)):
        Xtr, Xte = _fold_pipeline(X[train], X[test], pca_dim=PCA_DIM_LDA, seed=RNG_SEED + fold)
        clf = LinearDiscriminantAnalysis()
        clf.fit(Xtr, labels[train])
        pred[test] = clf.predict(Xte)
    correct = pred == labels
    return pd.DataFrame({"label": labels, "pred": pred, "correct": correct.astype(float), "game_code": games})


def oof_incentive_projection(X: np.ndarray, d1c: np.ndarray, games: np.ndarray) -> np.ndarray:
    d1c = np.asarray(d1c, dtype=float)
    games = np.asarray(games)
    out = np.full(len(games), np.nan)
    n_splits = min(N_SPLITS, len(np.unique(games)))
    for train, test in GroupKFold(n_splits=n_splits).split(X, d1c, groups=games):
        mu = X[train].astype(float).mean(axis=0)
        axis = incentive_axis(X[train].astype(float) - mu, d1c[train])
        proj_train = (X[train].astype(float) - mu) @ axis
        if corr(proj_train, d1c[train]) < 0:
            axis = -axis
        out[test] = (X[test].astype(float) - mu) @ axis
    return out


def boot_slope_by_game(x: np.ndarray, y: np.ndarray, games: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    games = np.asarray(games)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    games = games[keep]
    out = {
        "slope": np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "r": np.nan,
        "n": int(len(x)),
        "n_games": int(np.unique(games).size) if len(games) else 0,
    }
    if len(x) < 8 or np.std(x) == 0:
        return out
    out["slope"] = float(np.polyfit(x, y, 1)[0])
    out["r"] = corr(x, y)
    unique, idx = _boot_groups(games)
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([idx[g] for g in sampled])
        if np.std(x[rows]) > 0:
            vals.append(float(np.polyfit(x[rows], y[rows], 1)[0]))
    if vals:
        out["ci_lo"] = float(np.percentile(vals, 2.5))
        out["ci_hi"] = float(np.percentile(vals, 97.5))
    return out


def zscore(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return arr * np.nan
    return (arr - mu) / sd


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
