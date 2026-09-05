#!/usr/bin/env python3
"""PI-authorized Round-2 fixes for the Layer-B evidence freeze (§10).

The runner is fixed to ``$SCA_DATA_ROOT/substrate`` and never reads an
archive/legacy path.  Commands:

  geometry     reshape the passing objective fusion implementation as Table C
  disposition run explicit GPT-OSS none-exclusion/inclusion and ship Table D
  grid-a       execute the staged, bounded Table-A protocol grid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from strategic_anatomy.config import results_root

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "4")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

from analysis.layer_b.rebuild.rebuild_lib import MODELS, N_LAYERS, ROOT, games  # noqa: E402
from analysis.layer_b.rebuild.rebuild_missing_tables import (  # noqa: E402
    CACHE,
    CUES,
    EQUIV,
    FINAL_LAYER,
    FINAL_TABLES,
    LAYERA_UPTAKE,
    PCA_K,
    PLACEBO,
    RECRUIT_TABLES,
    SEED,
    _assert_root,
    _load_meta,
    _load_x,
    _targets,
)

ROOT = Path(ROOT)
OUT = HERE / "out"
GRID_CACHE = OUT / "cache_round2"
FUSION = results_root() / "layer_b" / "fusion" / "fusion_depth_table.csv"

TOL = 0.005
TARGET = {
    "qwen_instruct": {"sign_delta1c": .869, "canonical_action": .866, "control": .834},
    "qwen": {"sign_delta1c": .847, "canonical_action": .827, "control": .802},
    "llama31_instruct": {"sign_delta1c": .767, "canonical_action": .787, "control": .829},
    "gptoss": {"sign_delta1c": .894, "canonical_action": .802, "control": .567},
}

ROWSETS = ("p1_baseline", "p1_baseline_cues", "p1_p2_baseline")
CONDITIONS = ("baseline", *CUES.values(), PLACEBO)


# ---------------------------------------------------------------------------
# 10.1: Table C from the passing fusion implementation


def _independent_null_final(model: str, n_perm: int = 200) -> float:
    """Independent row shuffles of choice and Delta1c, fusion-identical math."""
    meta = _load_meta(model)
    scope = np.ones(len(meta), dtype=bool)
    if model == "gptoss":
        parts = []
        for game in games(model):
            d = pd.read_parquet(
                ROOT / model / game / "results.parquet",
                columns=["player", "condition", "counterbalance_id", "commit_type"],
            )
            d = d[(d.player == 1) & (d.condition == "baseline")].copy()
            d["game_code"] = game
            parts.append(d[["game_code", "counterbalance_id", "commit_type"]])
        commit = pd.concat(parts, ignore_index=True)
        meta = meta.merge(
            commit,
            on=["game_code", "counterbalance_id"],
            how="left",
            validate="one_to_one",
        )
        pure = meta.commit_type.eq("pure")
        scope = pure.groupby(meta.game_code).transform("all").to_numpy()
    y = meta.decoded_action.eq(meta.canonical_action_p1).astype(int).to_numpy()
    d = meta.delta1c.astype(np.float32).to_numpy()
    y, d = y[scope], d[scope]
    X = np.asarray(_load_x(model, FINAL_LAYER[model]), dtype=np.float32)[scope]
    n = len(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    rng = np.random.default_rng(20260702)
    pa = np.asarray([rng.permutation(n) for _ in range(n_perm)])
    pd_ = np.asarray([rng.permutation(n) for _ in range(n_perm)])
    al = y[pa]
    W = np.where(al == 1, 1.0 / n1, -1.0 / n0).T.astype(np.float32)
    dv = d[pd_]
    D = (dv - dv.mean(1, keepdims=True)).T.astype(np.float32)
    Xc = X - X.mean(0)
    A, B = X.T @ W, Xc.T @ D
    A /= np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-12)
    cos = np.clip(np.einsum("dp,dp->p", A, B), -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(cos))))


def geometry_from_fusion() -> None:
    _assert_root()
    if not FUSION.exists():
        raise FileNotFoundError(f"missing passing fusion table: {FUSION}")
    src = pd.read_csv(FUSION)
    if set(src.belief.unique()) != {"uniform"}:
        raise AssertionError("Table C must use the objective/uniform fusion table")
    out = src.rename(
        columns={
            "angle_deg": "angle_decision_incentive_deg",
            "null_lo": "null_lo_deg",
            "null_med": "null_median_deg",
            "null_hi": "null_hi_deg",
            "sig": "below_null",
        }
    ).copy()
    keep = [
        "model",
        "layer",
        "angle_decision_incentive_deg",
        "null_median_deg",
        "null_lo_deg",
        "null_hi_deg",
        "perm_p",
        "below_null",
    ]
    for optional in ("row_scope", "n_obs", "n_games"):
        if optional in out.columns:
            keep.append(optional)
    out = out[keep]
    if "n_obs" not in out:
        out["n_obs"] = 576
    if "n_games" not in out:
        out["n_games"] = 144
    if "row_scope" not in out:
        out["row_scope"] = "unspecified"
    out["n_perm"] = 200
    out["independent_shuffle_median_deg"] = np.nan
    independent = {}
    for model in MODELS:
        val = _independent_null_final(model)
        independent[model] = val
        mask = out.model.eq(model) & out.layer.eq(FINAL_LAYER[model])
        out.loc[mask, "independent_shuffle_median_deg"] = val
    RECRUIT_TABLES.mkdir(parents=True, exist_ok=True)
    out.sort_values(["model", "layer"]).to_csv(
        RECRUIT_TABLES / "recruitment_geometry_depth.csv", index=False
    )

    def onset(g: pd.DataFrame) -> float | None:
        g = g[g.layer > 0].sort_values("layer").reset_index(drop=True)
        sig = g.below_null.astype(bool).to_numpy()
        for i in range(len(sig) - 2):
            if sig[i : i + 3].all():
                return round(float(g.loc[i, "layer"] / g.layer.max()), 2)
        return None

    summary = []
    for model, g in out.groupby("model"):
        summary.append(
            {
                "model": model,
                "onset_depth": onset(g),
                "final_angle": float(g.sort_values("layer").iloc[-1].angle_decision_incentive_deg),
                "median_null": float(g.null_median_deg.median()),
                "independent_shuffle_final": independent[model],
                "n_below_null": int(g.below_null.sum()),
            }
        )
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "round2_geometry_summary.csv", index=False)
    print(s.to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
# 10.2: explicit disposition inclusion variant


def _cue_shifts(
    model: str, *, include_none: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    layer = FINAL_LAYER[model]
    shifts: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    placebo: list[np.ndarray] = []
    placebo_groups: list[str] = []
    stats = {
        "none_rows_seen": 0,
        "none_rows_with_capture": 0,
        "none_rows_excluded": 0,
        "missing_capture_rows": 0,
    }
    for gi, game in enumerate(games(model)):
        gd = ROOT / model / game
        results = pd.read_parquet(gd / "results.parquet")
        results = results[results.player.eq(1)].copy()
        row = {
            (str(r.condition), int(r.counterbalance_id)): r
            for r in results.itertuples(index=False)
        }
        with np.load(gd / "acts.npz") as z:
            for cb in range(4):
                bk = f"p1_baseline_cb{cb}_l{layer}"
                if bk not in z.files:
                    stats["missing_capture_rows"] += 1
                    continue
                base = np.asarray(z[bk], dtype=np.float32)
                for j, cond in enumerate(CUES.values()):
                    rr = row[(cond, cb)]
                    is_none = model == "gptoss" and str(getattr(rr, "commit_type", "")) == "none"
                    ck = f"p1_{cond}_cb{cb}_l{layer}"
                    if is_none:
                        stats["none_rows_seen"] += 1
                        if ck in z.files:
                            stats["none_rows_with_capture"] += 1
                        if not include_none:
                            stats["none_rows_excluded"] += 1
                            continue
                    if ck not in z.files:
                        stats["missing_capture_rows"] += 1
                        continue
                    shifts.append(np.asarray(z[ck], dtype=np.float32) - base)
                    labels.append(j)
                    groups.append(game)
                rr = row[(PLACEBO, cb)]
                is_none = model == "gptoss" and str(getattr(rr, "commit_type", "")) == "none"
                pk = f"p1_{PLACEBO}_cb{cb}_l{layer}"
                if not (is_none and not include_none) and pk in z.files:
                    placebo.append(np.asarray(z[pk], dtype=np.float32) - base)
                    placebo_groups.append(game)
        if gi % 36 == 0 or gi + 1 == 144:
            print(f"[cue-load] {model} include_none={include_none}: {gi+1}/144", flush=True)
    return (
        np.vstack(shifts).astype(np.float32),
        np.asarray(labels, dtype=int),
        np.asarray(groups),
        np.vstack(placebo).astype(np.float32),
        np.asarray(placebo_groups),
        stats,
    )


def _fit_disposition(model: str, *, include_none: bool) -> dict[str, object]:
    X, y, groups, P, pgroups, stats = _cue_shifts(model, include_none=include_none)
    pred = np.full(len(y), -1, dtype=int)
    placebo_pred: list[int] = []
    fold_acc = []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=groups)):
        scaler = StandardScaler().fit(X[tr])
        xtr = scaler.transform(X[tr])
        xte = scaler.transform(X[te])
        k = min(PCA_K, xtr.shape[0] - 1, xtr.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=SEED + fold)
        ztr, zte = pca.fit_transform(xtr), pca.transform(xte)
        lda = LinearDiscriminantAnalysis(solver="svd").fit(ztr, y[tr])
        pred[te] = lda.predict(zte)
        fold_acc.append(float(accuracy_score(y[te], pred[te])))
        test_games = np.unique(groups[te])
        pmask = np.isin(pgroups, test_games)
        if pmask.any():
            placebo_pred.extend(lda.predict(pca.transform(scaler.transform(P[pmask]))).tolist())
    if (pred < 0).any():
        raise AssertionError("missing OOF disposition predictions")
    per = {
        trait: float(accuracy_score(y[y == j], pred[y == j]))
        for j, trait in enumerate(CUES)
    }
    return {
        "model": model,
        "include_none": include_none,
        "accuracy": float(accuracy_score(y, pred)),
        "per_class": per,
        "fold_min": min(fold_acc),
        "fold_max": max(fold_acc),
        "placebo_n": len(placebo_pred),
        "n_rows": len(y),
        **stats,
    }


def disposition_round2() -> None:
    _assert_root()
    if not LAYERA_UPTAKE.exists():
        raise FileNotFoundError(f"Layer-A uptake table absent: {LAYERA_UPTAKE}")
    uptake = pd.read_csv(LAYERA_UPTAKE)
    fits: dict[str, dict[str, object]] = {}
    for model in MODELS:
        fits[model] = _fit_disposition(model, include_none=False)
        print(f"[disposition] {model} exclusion={fits[model]['accuracy']:.6f}", flush=True)
    inclusion = _fit_disposition("gptoss", include_none=True)
    print(f"[disposition] gptoss inclusion={inclusion['accuracy']:.6f}", flush=True)
    use_inclusion = float(inclusion["accuracy"]) >= .969
    chosen = dict(fits)
    if use_inclusion:
        chosen["gptoss"] = inclusion
    protocol = "include_none" if use_inclusion else "exclude_none_adopt_0.965"

    rows = []
    for model in MODELS:
        fit = chosen[model]
        sub = uptake[uptake.model.eq(model)].set_index("trait")
        for trait in CUES:
            r = sub.loc[trait]
            rows.append(
                {
                    "model": model,
                    "trait": trait,
                    "lda_trait_accuracy": float(fit["accuracy"]),
                    "lda_trait_class_accuracy": float(fit["per_class"][trait]),
                    "behavior_magnitude": float(r.magnitude),
                    "behavior_magnitude_lo": float(r.magnitude_lo),
                    "behavior_magnitude_hi": float(r.magnitude_hi),
                    "behavior_n_games": int(r.n_games),
                    "layer": FINAL_LAYER[model],
                    "pca_components": PCA_K,
                    "gptoss_none_protocol": protocol if model == "gptoss" else "not_applicable",
                }
            )
    out = pd.DataFrame(rows)
    RECRUIT_TABLES.mkdir(parents=True, exist_ok=True)
    out.sort_values(["model", "trait"]).to_csv(
        RECRUIT_TABLES / "disposition_dissociation.csv", index=False
    )
    audit_rows = []
    for model, fit in fits.items():
        audit_rows.append({k: v for k, v in fit.items() if k != "per_class"})
    audit_rows.append({k: v for k, v in inclusion.items() if k != "per_class"})
    audit = pd.DataFrame(audit_rows)
    audit["chosen_protocol"] = protocol
    audit.to_csv(RECRUIT_TABLES / "disposition_lda_summary.csv", index=False)
    print("[disposition] chosen", protocol, flush=True)
    print(out.groupby("trait").behavior_magnitude.mean().to_string(), flush=True)


# ---------------------------------------------------------------------------
# 10.3: staged, bounded protocol grid for Table A


@dataclass(frozen=True)
class Protocol:
    stage: int
    protocol_id: str
    rowset: str = "p1_baseline"
    layer_mode: str = "final"
    folds: str = "gkf5"
    standardization: str = "per_fold"
    C: float = 1.0
    class_weight: str = "none"
    auc_mode: str = "pooled_oof"
    control_variant: str = "rank00_macro_ovr"


def _rowset_filter(df: pd.DataFrame, rowset: str) -> pd.DataFrame:
    if rowset == "p1_baseline":
        return df[df.player.eq(1) & df.condition.eq("baseline")]
    if rowset == "p1_baseline_cues":
        return df[df.player.eq(1) & df.condition.isin(CONDITIONS)]
    if rowset == "p1_p2_baseline":
        return df[df.condition.eq("baseline") & df.player.isin([1, 2])]
    raise ValueError(rowset)


def _grid_meta(model: str, rowset: str) -> pd.DataFrame:
    tm = _targets()
    parts = []
    for game in games(model):
        df = pd.read_parquet(ROOT / model / game / "results.parquet")
        df = _rowset_filter(df, rowset).copy()
        if model == "gptoss":
            ok = df.commit_type.astype(str).ne("none")
            action = pd.to_numeric(df.realized_action, errors="coerce")
        else:
            ok = df.parse_ok.astype(bool)
            action = pd.to_numeric(df.decoded_action, errors="coerce")
        df = df.loc[ok & action.isin([0, 1])].copy()
        df["action"] = action.loc[df.index].astype(int)
        df["game_code"] = game
        df["canonical_action_p1"] = int(tm.loc[game, "canonical_action_p1"])
        df["delta1c"] = float(tm.loc[game, "delta1c"])
        df["control_rank"] = int(tm.loc[game, "stim_control_cell00"])
        u1 = np.asarray([float(x) for x in str(tm.loc[game, "canonical_p1"]).split(",")])
        df["control_binary"] = int(u1[1] == 4)
        df["aligned"] = df.action.eq(df.canonical_action_p1).astype(int)
        parts.append(
            df[
                [
                    "game_code",
                    "player",
                    "condition",
                    "counterbalance_id",
                    "action",
                    "aligned",
                    "delta1c",
                    "control_rank",
                    "control_binary",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True).sort_values(
        ["game_code", "player", "condition", "counterbalance_id"]
    ).reset_index(drop=True)


def _ensure_final_grid_cache(model: str, rowset: str) -> tuple[pd.DataFrame, np.ndarray]:
    GRID_CACHE.mkdir(parents=True, exist_ok=True)
    stem = GRID_CACHE / f"{model}_{rowset}"
    mp, xp = stem.with_suffix(".parquet"), stem.with_suffix(".npy")
    if mp.exists() and xp.exists():
        return pd.read_parquet(mp), np.load(xp, mmap_mode="r")
    meta = _grid_meta(model, rowset)
    first = games(model)[0]
    with np.load(ROOT / model / first / "acts.npz") as z:
        dim = len(z[f"p1_baseline_cb0_l{FINAL_LAYER[model]}"])
    X = np.lib.format.open_memmap(
        xp, mode="w+", dtype=np.float32, shape=(len(meta), dim)
    )
    for game, idxs in meta.groupby("game_code").groups.items():
        with np.load(ROOT / model / game / "acts.npz") as z:
            for i in idxs:
                r = meta.iloc[int(i)]
                key = (
                    f"p{int(r.player)}_{r.condition}_cb{int(r.counterbalance_id)}_"
                    f"l{FINAL_LAYER[model]}"
                )
                if key.endswith("_seq") or key not in z.files:
                    raise KeyError(f"missing fixed read-position key {model}/{game}/{key}")
                X[int(i)] = np.asarray(z[key], dtype=np.float32)
    X.flush()
    del X
    meta.to_parquet(mp, index=False)
    return meta, np.load(xp, mmap_mode="r")


def _layer_grid_matrix(model: str, meta: pd.DataFrame, layer: int) -> np.ndarray:
    if (
        len(meta) == 576
        and meta.player.eq(1).all()
        and meta.condition.eq("baseline").all()
    ):
        return np.asarray(_load_x(model, layer), dtype=np.float32)
    first = games(model)[0]
    with np.load(ROOT / model / first / "acts.npz") as z:
        dim = len(z[f"p1_baseline_cb0_l{layer}"])
    X = np.empty((len(meta), dim), dtype=np.float32)
    for game, idxs in meta.groupby("game_code").groups.items():
        with np.load(ROOT / model / game / "acts.npz") as z:
            for i in idxs:
                r = meta.iloc[int(i)]
                key = f"p{int(r.player)}_{r.condition}_cb{int(r.counterbalance_id)}_l{layer}"
                X[int(i)] = np.asarray(z[key], dtype=np.float32)
    return X


def _splits(groups: np.ndarray, folds: str):
    dummy = np.zeros(len(groups), dtype=int)
    if folds == "gkf5":
        return list(GroupKFold(5).split(dummy, groups=groups))
    if folds == "gkf10":
        return list(GroupKFold(10).split(dummy, groups=groups))
    if folds == "logo":
        return list(LeaveOneGroupOut().split(dummy, groups=groups))
    raise ValueError(folds)


def _auc_value(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(
        roc_auc_score(y, proba, labels=classes, multi_class="ovr", average="macro")
    )


def _grid_probe(
    X: np.ndarray, meta: pd.DataFrame, target: str, protocol: Protocol
) -> float:
    if target == "sign_delta1c":
        mask = meta.delta1c.ne(0).to_numpy()
        y = meta.loc[mask, "delta1c"].gt(0).astype(int).to_numpy()
    elif target == "canonical_action":
        mask = np.ones(len(meta), dtype=bool)
        y = meta.aligned.astype(int).to_numpy()
    elif target == "control":
        mask = np.ones(len(meta), dtype=bool)
        col = "control_rank" if protocol.control_variant == "rank00_macro_ovr" else "control_binary"
        y = meta[col].astype(int).to_numpy()
    else:
        raise ValueError(target)
    Xk = np.asarray(X[mask], dtype=np.float32)
    groups = meta.loc[mask, "game_code"].to_numpy()
    classes = np.unique(y)
    split = _splits(groups, protocol.folds)
    pred = np.full((len(y), len(classes)), np.nan, dtype=float)
    fold_auc = []
    Xglobal = None
    if protocol.standardization == "global":
        Xglobal = StandardScaler().fit_transform(Xk)
    for tr, te in split:
        if protocol.standardization == "per_fold":
            scaler = StandardScaler().fit(Xk[tr])
            xtr, xte = scaler.transform(Xk[tr]), scaler.transform(Xk[te])
        elif protocol.standardization == "global":
            xtr, xte = Xglobal[tr], Xglobal[te]
        elif protocol.standardization == "none":
            xtr, xte = Xk[tr], Xk[te]
        else:
            raise ValueError(protocol.standardization)
        cw = None if protocol.class_weight == "none" else "balanced"
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=protocol.C, class_weight=cw
        ).fit(xtr, y[tr])
        pp = clf.predict_proba(xte)
        for j, c in enumerate(clf.classes_):
            pred[te, int(np.flatnonzero(classes == c)[0])] = pp[:, j]
        if protocol.auc_mode == "per_fold_mean" and len(np.unique(y[te])) == len(classes):
            fold_auc.append(_auc_value(y[te], pred[te], classes))
    if not np.isfinite(pred).all():
        raise AssertionError(f"missing OOF predictions for {target}/{protocol.protocol_id}")
    if protocol.auc_mode == "per_fold_mean":
        return float(np.mean(fold_auc))
    return _auc_value(y, pred, classes)


def _eval_final(protocol: Protocol) -> dict[str, dict[str, float]]:
    out = {}
    for model in MODELS:
        meta, X = _ensure_final_grid_cache(model, protocol.rowset)
        vals = {
            target: _grid_probe(X, meta, target, protocol)
            for target in ("sign_delta1c", "canonical_action", "control")
        }
        out[model] = vals
        print(
            f"[grid] {protocol.protocol_id} {model}: "
            + " ".join(f"{k}={v:.3f}" for k, v in vals.items()),
            flush=True,
        )
    return out


def _eval_peak(protocol: Protocol) -> dict[str, dict[str, float]]:
    out = {}
    for model in MODELS:
        meta, _ = _ensure_final_grid_cache(model, protocol.rowset)
        best = {t: (-np.inf, -1) for t in ("sign_delta1c", "canonical_action", "control")}
        for layer in range(N_LAYERS[model]):
            X = _layer_grid_matrix(model, meta, layer)
            for target in best:
                val = _grid_probe(X, meta, target, protocol)
                if val > best[target][0]:
                    best[target] = (val, layer)
            if layer % 10 == 0 or layer + 1 == N_LAYERS[model]:
                print(f"[grid-peak] {protocol.rowset} {model} L{layer}", flush=True)
        out[model] = {t: float(v[0]) for t, v in best.items()}
        out[model].update({f"{t}_layer": int(v[1]) for t, v in best.items()})
    return out


def _score(values: dict[str, dict[str, float]]) -> int:
    return sum(
        abs(values[m][t] - TARGET[m][t]) <= TOL
        for m in MODELS
        for t in ("sign_delta1c", "canonical_action", "control")
    )


def _log_protocol(
    logs: dict[str, list[dict]], protocol: Protocol, values, *, status: str, matches: int
) -> None:
    for model in MODELS:
        row = asdict(protocol)
        row.update(
            {
                "status": status,
                "global_matches_of_12": matches,
                "identified": bool(matches >= 11),
            }
        )
        if values is not None:
            for target in ("sign_delta1c", "canonical_action", "control"):
                got = float(values[model][target])
                row[f"auc_{target}"] = got
                row[f"target_{target}"] = TARGET[model][target]
                row[f"match_{target}"] = abs(got - TARGET[model][target]) <= TOL
                if f"{target}_layer" in values[model]:
                    row[f"layer_{target}"] = int(values[model][f"{target}_layer"])
        logs[model].append(row)


def _write_logs(logs: dict[str, list[dict]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model, rows in logs.items():
        pd.DataFrame(rows).to_csv(OUT / f"gridA_{model}.csv", index=False)


def grid_a() -> None:
    _assert_root()
    logs = {m: [] for m in MODELS}
    identified: Protocol | None = None
    identified_values = None

    # Stage 1: rows x layer, all other fields at §0 defaults.  Final is run
    # first; a peak cell is exactly prunable if >=2 values already exceed the
    # target+tolerance because a per-target maximum cannot lower them and the
    # qualification threshold permits only one miss.
    final_by_rowset = {}
    for rowset in ROWSETS:
        p = Protocol(1, f"s1_{rowset}_final", rowset=rowset)
        vals = _eval_final(p)
        final_by_rowset[rowset] = vals
        n = _score(vals)
        _log_protocol(logs, p, vals, status="COMPLETE", matches=n)
        _write_logs(logs)
        if n >= 11:
            identified, identified_values = p, vals
            break
    if identified is None:
        for rowset in ROWSETS:
            p = Protocol(1, f"s1_{rowset}_per_target_peak", rowset=rowset, layer_mode="per_target_peak")
            high = sum(
                final_by_rowset[rowset][m][t] > TARGET[m][t] + TOL
                for m in MODELS
                for t in ("sign_delta1c", "canonical_action", "control")
            )
            if high >= 2:
                _log_protocol(logs, p, None, status=f"PRUNED_MONOTONIC_{high}_ABOVE", matches=0)
                _write_logs(logs)
                continue
            vals = _eval_peak(p)
            n = _score(vals)
            _log_protocol(logs, p, vals, status="COMPLETE", matches=n)
            _write_logs(logs)
            if n >= 11:
                identified, identified_values = p, vals
                break

    # Stage 2: folds x standardization, defaults elsewhere.
    if identified is None:
        for folds in ("gkf5", "gkf10", "logo"):
            for std in ("per_fold", "global", "none"):
                p = Protocol(2, f"s2_{folds}_{std}", folds=folds, standardization=std)
                vals = _eval_final(p)
                n = _score(vals)
                _log_protocol(logs, p, vals, status="COMPLETE", matches=n)
                _write_logs(logs)
                if n >= 11:
                    identified, identified_values = p, vals
                    break
            if identified is not None:
                break

    # Stage 3: C x class weight x AUC x control, defaults elsewhere.
    if identified is None:
        for C in (.01, .1, 1.0):
            for cw in ("none", "balanced"):
                for auc in ("pooled_oof", "per_fold_mean"):
                    for control in ("rank00_macro_ovr", "binary_cell01_is4"):
                        p = Protocol(
                            3,
                            f"s3_C{C}_{cw}_{auc}_{control}",
                            C=C,
                            class_weight=cw,
                            auc_mode=auc,
                            control_variant=control,
                        )
                        vals = _eval_final(p)
                        n = _score(vals)
                        _log_protocol(logs, p, vals, status="COMPLETE", matches=n)
                        _write_logs(logs)
                        if n >= 11:
                            identified, identified_values = p, vals
                            break
                    if identified is not None:
                        break
                if identified is not None:
                    break
            if identified is not None:
                break

    result = {
        "identified": identified is not None,
        "threshold": "at least 11 of 12 values within +/-0.005",
        "protocol": asdict(identified) if identified else None,
        "values": identified_values,
        "n_protocols_logged": {m: len(v) for m, v in logs.items()},
    }
    (OUT / "gridA_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["geometry", "disposition", "grid-a"])
    args = ap.parse_args()
    {"geometry": geometry_from_fusion, "disposition": disposition_round2, "grid-a": grid_a}[
        args.command
    ]()


if __name__ == "__main__":
    main()
