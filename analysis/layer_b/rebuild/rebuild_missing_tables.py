#!/usr/bin/env python3
"""Regenerate the missing Layer-B evidence-freeze tables.

This runner is deliberately fixed to ``$SCA_DATA_ROOT/substrate`` and the
protocol locked 2026-07-09 in ``REBUILD_ALL_SPEC.md``.  It never inspects
archived or legacy paths.

Subcommands are ordered to match the spec:

  cache          Build float32 P1-baseline, all-layer matrices.
  decodability   Table A, final-layer strategic/control probes.
  crystallize    Table B, canonical-choice probe at every layer.
  geometry       Table C, decision-incentive angle and joint game null.
  disposition    Table D, held-out cue-shift PCA(30) -> Fisher LDA plus
                 committed Layer-A conflict-set uptake.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from strategic_anatomy.config import data_root, game_features_csv, results_root, substrate_root, taxonomy_dir

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
from analysis.layer_b.rebuild.rebuild_lib import (  # noqa: E402
    DELTA1C,
    MODELS,
    N_LAYERS,
    ROOT,
    games,
    load_game_rows,
    meta as base_meta,
    realized_aligned,
)

ROOT = Path(ROOT)
FEATURES = game_features_csv()
EQUIV = taxonomy_dir() / "equivalence_per_canonical.csv"
LAYERA_UPTAKE = results_root() / "layer_a" / "f2_trait_aim.csv"
CACHE = data_root() / "layer_b_cache" / "baseline"
FINAL_TABLES = results_root() / "layer_b"
RECRUIT_TABLES = results_root() / "layer_b" / "recruitment"

SEED = 20260627
N_BOOT = 1000
N_PERM = 200
PCA_K = 30  # established active disposition implementation

FINAL_LAYER = {"qwen": 80, "qwen_instruct": 80, "llama31_instruct": 80, "gptoss": 36}
TEX_DEC = {
    "qwen_instruct": {"sign_delta1c": .869, "canonical_action": .866, "stim_control_cell00": .834},
    "qwen": {"sign_delta1c": .847, "canonical_action": .827, "stim_control_cell00": .802},
    "llama31_instruct": {"sign_delta1c": .767, "canonical_action": .787, "stim_control_cell00": .829},
    "gptoss": {"sign_delta1c": .894, "canonical_action": .832, "stim_control_cell00": .567},
}
TEX_CRY = {
    "qwen_instruct": {"l0": .441, "peak": .917, "peak_layer": 62, "final": .866},
    "qwen": {"l0": .429, "peak": .898, "peak_layer": 63, "final": .827},
    "llama31_instruct": {"l0": .418, "peak": .841, "peak_layer": 60, "final": .787},
    "gptoss": {"l0": .583, "peak": .838, "peak_layer": 24, "final": .832},
}

CUES = {
    "risk": "cue_risk_aversion",
    "loss": "cue_loss_aversion",
    "inequity": "cue_inequity_aversion",
    "maximin": "cue_maximin",
    "selfish": "cue_selfish_maximizer",
}
PLACEBO = "cue_length_match_null"


def _targets() -> pd.DataFrame:
    feat = pd.read_csv(FEATURES)[
        ["game_code", "canonical_action_p1", "canonical_action_p2"]
    ]
    d1 = pd.read_csv(DELTA1C)
    eq = pd.read_csv(EQUIV)[["bruns_name", "canonical_p1", "canonical_p2"]].rename(
        columns={"bruns_name": "game_code"}
    )
    out = feat.merge(d1, on="game_code", how="inner").merge(eq, on="game_code", how="inner")
    if len(out) != 144:
        raise AssertionError(f"target join expected 144 games, got {len(out)}")

    def vals(s: str) -> np.ndarray:
        a = np.asarray([float(x) for x in str(s).split(",")], dtype=float)
        if a.shape != (4,):
            raise ValueError(f"expected four canonical payoffs, got {s!r}")
        return a

    d2, ctrl = [], []
    for r in out.itertuples(index=False):
        u1, u2 = vals(r.canonical_p1), vals(r.canonical_p2)
        raw2 = .5 * (u2[0] + u2[2]) - .5 * (u2[1] + u2[3])
        d2.append(raw2 if int(r.canonical_action_p2) == 0 else -raw2)
        ctrl.append(int(u1[0]))
    out["delta2c"] = d2
    out["stim_control_cell00"] = ctrl
    counts = out["stim_control_cell00"].value_counts().sort_index().to_dict()
    if counts != {1: 48, 2: 48, 3: 48}:
        raise AssertionError(f"stimulus control is not balanced 48/48/48: {counts}")
    return out.set_index("game_code")


def _assert_root() -> None:
    expected = (substrate_root()).resolve()
    if ROOT.resolve() != expected:
        raise AssertionError(f"fixed root violation: {ROOT.resolve()} != {expected}")
    for model in MODELS:
        gl = games(model)
        if len(gl) != 144:
            raise AssertionError(f"{model}: expected 144 games, got {len(gl)}")
        for g in (gl[0], gl[-1]):
            cfg = json.loads((ROOT / model / g / "config.json").read_text())
            if cfg.get("substrate") != "akata_oneshot":
                raise AssertionError(f"{model}/{g}: substrate={cfg.get('substrate')!r}")


def _valid_rows(model: str, game: str, tm: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / model / game / "results.parquet")
    df = df[(df.player == 1) & (df.condition == "baseline")].sort_values(
        "counterbalance_id"
    ).copy()
    _, ok = realized_aligned(df, model, tm.loc[game, "canonical_action_p1"])
    if model == "gptoss":
        act = pd.to_numeric(df["realized_action"], errors="coerce")
    else:
        act = pd.to_numeric(df["decoded_action"], errors="coerce")
    df = df.loc[ok].copy()
    df["decoded_action"] = act.loc[ok].astype(int).to_numpy()
    df["game_code"] = game
    for c in ["canonical_action_p1", "delta1c", "delta2c", "stim_control_cell00"]:
        df[c] = tm.loc[game, c]
    return df[
        [
            "game_code",
            "counterbalance_id",
            "decoded_action",
            "canonical_action_p1",
            "delta1c",
            "delta2c",
            "stim_control_cell00",
        ]
    ]


def build_cache(models: list[str]) -> None:
    _assert_root()
    CACHE.mkdir(parents=True, exist_ok=True)
    tm = _targets()
    manifest: dict[str, object] = {
        "data_root": str(ROOT.relative_to(REPO)),
        "dtype": "float32",
        "rows": "valid realized P1-baseline actions only, sorted game/counterbalance",
        "models": {},
    }
    for model in models:
        gl = games(model)
        parts = [_valid_rows(model, g, tm) for g in gl]
        md = pd.concat(parts, ignore_index=True).sort_values(
            ["game_code", "counterbalance_id"]
        ).reset_index(drop=True)
        if len(md) != 576:
            raise AssertionError(f"{model}: expected 576 valid P1-baseline rows, got {len(md)}")
        md.to_parquet(CACHE / f"meta_{model}.parquet", index=False)
        row_index = {
            (r.game_code, int(r.counterbalance_id)): i for i, r in md.iterrows()
        }
        layers = list(range(N_LAYERS[model]))
        first_game = gl[0]
        with np.load(ROOT / model / first_game / "acts.npz") as z:
            dim = int(z[f"p1_baseline_cb0_l0"].shape[0])
        mats = {
            L: np.lib.format.open_memmap(
                CACHE / f"X_{model}_l{L}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(md), dim),
            )
            for L in layers
        }
        for gi, game in enumerate(gl):
            with np.load(ROOT / model / game / "acts.npz") as z:
                for cb in range(4):
                    i = row_index[(game, cb)]
                    for L in layers:
                        key = f"p1_baseline_cb{cb}_l{L}"
                        if key not in z.files or key.endswith("_seq"):
                            raise KeyError(f"missing answer/commit capture {model}/{game}/{key}")
                        mats[L][i] = np.asarray(z[key], dtype=np.float32)
            if gi % 12 == 0 or gi + 1 == len(gl):
                print(f"[cache] {model}: {gi + 1}/{len(gl)} games", flush=True)
        for m in mats.values():
            m.flush()
        del mats
        manifest["models"][model] = {
            "n_games": 144,
            "n_rows": len(md),
            "n_layers": len(layers),
            "hidden_dim": dim,
        }
    (CACHE / "cache_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _load_meta(model: str) -> pd.DataFrame:
    p = CACHE / f"meta_{model}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing cache metadata: {p}; run cache first")
    return pd.read_parquet(p).reset_index(drop=True)


def _load_x(model: str, layer: int) -> np.ndarray:
    p = CACHE / f"X_{model}_l{layer}.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing layer cache: {p}; run cache first")
    return np.load(p, mmap_mode="r")


def _oof_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    classes = np.unique(y)
    pred = np.full((len(y), len(classes)), np.nan, dtype=float)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=groups)):
        scaler = StandardScaler().fit(X[tr])
        xtr = scaler.transform(X[tr])
        xte = scaler.transform(X[te])
        clf = LogisticRegression(solver="lbfgs", max_iter=2000).fit(xtr, y[tr])
        pp = clf.predict_proba(xte)
        for j, c in enumerate(clf.classes_):
            pred[te, int(np.where(classes == c)[0][0])] = pp[:, j]
        print(f"    fold {fold + 1}/5", flush=True)
    if not np.isfinite(pred).all():
        raise AssertionError("OOF predictions contain missing values")
    return pred, classes


def _auc(y: np.ndarray, pred: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) == 2:
        return float(roc_auc_score(y, pred[:, 1]))
    return float(roc_auc_score(y, pred, labels=classes, multi_class="ovr", average="macro"))


def _auc_ci(
    y: np.ndarray, pred: np.ndarray, classes: np.ndarray, groups: np.ndarray
) -> tuple[float, float, float]:
    point = _auc(y, pred, classes)
    ug = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        if len(np.unique(y[idx])) != len(classes):
            continue
        vals.append(_auc(y[idx], pred[idx], classes))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def _probe(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float, float, int, int]:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    yk, gk = np.asarray(y)[mask], np.asarray(groups)[mask]
    pred, classes = _oof_proba(X[mask], yk, gk)
    auc, lo, hi = _auc_ci(yk, pred, classes, gk)
    return auc, lo, hi, int(len(yk)), int(len(np.unique(gk)))


def decodability(models: list[str]) -> None:
    _assert_root()
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        md = _load_meta(model)
        X = _load_x(model, FINAL_LAYER[model])
        groups = md["game_code"].to_numpy()
        targets = [
            ("sign_delta1c", (md.delta1c > 0).astype(int).to_numpy(), md.delta1c.ne(0).to_numpy()),
            (
                "canonical_action",
                md.decoded_action.eq(md.canonical_action_p1).astype(int).to_numpy(),
                np.ones(len(md), dtype=bool),
            ),
            ("sign_delta2c", (md.delta2c > 0).astype(int).to_numpy(), md.delta2c.ne(0).to_numpy()),
            (
                "stim_control_cell00",
                md.stim_control_cell00.astype(int).to_numpy(),
                np.ones(len(md), dtype=bool),
            ),
        ]
        for name, y, mask in targets:
            print(f"[decodability] {model} L{FINAL_LAYER[model]} {name}", flush=True)
            auc, lo, hi, n, ng = _probe(X, y, groups, mask)
            rows.append(
                dict(
                    model=model,
                    probe=name,
                    layer=FINAL_LAYER[model],
                    auc=auc,
                    lo=lo,
                    hi=hi,
                    n_obs=n,
                    n_games=ng,
                    cv="GroupKFold(5)",
                    behavior_label="realized canonical action" if name == "canonical_action" else "game target",
                )
            )
            target = TEX_DEC.get(model, {}).get(name)
            suffix = f" target={target:.3f}" if target is not None else ""
            print(f"  AUC={auc:.3f} [{lo:.3f},{hi:.3f}]{suffix}", flush=True)
    out = pd.DataFrame(rows)
    dest = FINAL_TABLES / "b1_decodability.csv"
    if dest.exists() and set(out.model) != set(MODELS):
        old = pd.read_csv(dest)
        out = pd.concat([old[~old.model.isin(models)], out], ignore_index=True)
    out.sort_values(["model", "probe"]).to_csv(dest, index=False)


def crystallize(models: list[str]) -> None:
    _assert_root()
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        md = _load_meta(model)
        y = md.decoded_action.eq(md.canonical_action_p1).astype(int).to_numpy()
        groups = md.game_code.to_numpy()
        model_rows = []
        for L in range(N_LAYERS[model]):
            print(f"[crystallize] {model} L{L}/{N_LAYERS[model]-1}", flush=True)
            auc, lo, hi, n, ng = _probe(_load_x(model, L), y, groups)
            model_rows.append(
                dict(
                    model=model,
                    layer=L,
                    depth_frac=L / (N_LAYERS[model] - 1),
                    auc=auc,
                    lo=lo,
                    hi=hi,
                    n_obs=n,
                    n_games=ng,
                )
            )
            print(f"  AUC={auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
        floor = model_rows[0]["auc"]
        for r in model_rows:
            r["gain_over_embedding"] = r["auc"] - floor
        rows.extend(model_rows)
        peak = max(model_rows, key=lambda r: r["auc"])
        t = TEX_CRY[model]
        print(
            f"[accept] {model}: L0 {floor:.3f} (target {t['l0']:.3f}); "
            f"peak {peak['auc']:.3f}@L{peak['layer']} "
            f"(target {t['peak']:.3f}@L{t['peak_layer']}); "
            f"final {model_rows[-1]['auc']:.3f} (target {t['final']:.3f})",
            flush=True,
        )
    out = pd.DataFrame(rows)
    dest = FINAL_TABLES / "b1_crystallization.csv"
    if dest.exists() and set(out.model) != set(MODELS):
        old = pd.read_csv(dest)
        out = pd.concat([old[~old.model.isin(models)], out], ignore_index=True)
    out.sort_values(["model", "layer"]).to_csv(dest, index=False)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.zeros_like(v)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = _unit(a), _unit(b)
    return float(np.degrees(np.arccos(np.clip(float(aa @ bb), -1.0, 1.0))))


def _axes(X: np.ndarray, y: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf = np.asarray(X, dtype=np.float64)
    d_dec = xf[y == 1].mean(0) - xf[y == 0].mean(0)
    xc = xf - xf.mean(0)
    dc = d - d.mean()
    d_inc = xc.T @ dc
    return _unit(d_dec), _unit(d_inc)


def _game_row_permutations(groups: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    ug = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in ug]
    sizes = {len(x) for x in idx}
    if sizes != {4}:
        raise AssertionError(f"joint game null requires four cb rows/game, got sizes {sizes}")
    out = np.empty((n, len(groups)), dtype=int)
    for b in range(n):
        perm = rng.permutation(len(ug))
        for target, source in enumerate(perm):
            out[b, idx[target]] = idx[source]
    return out


def _joint_null(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
    *,
    independent: bool = False,
) -> np.ndarray:
    xf = np.asarray(X, dtype=np.float32)
    yc = np.asarray(y, dtype=int)
    dc0 = np.asarray(d, dtype=np.float32)
    py = _game_row_permutations(groups, rng, N_PERM)
    pd_ = _game_row_permutations(groups, rng, N_PERM) if independent else py
    yp = yc[py]
    dp = dc0[pd_]
    n1 = yp.sum(1).astype(np.float32)
    n0 = len(yc) - n1
    W = np.where(yp == 1, 1.0 / n1[:, None], -1.0 / n0[:, None]).T.astype(np.float32)
    D = (dp - dp.mean(1, keepdims=True)).T.astype(np.float32)
    Xc = xf - xf.mean(0)
    A = xf.T @ W
    B = Xc.T @ D
    A /= np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-12)
    cos = np.clip(np.einsum("dp,dp->p", A, B), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def geometry(models: list[str]) -> None:
    _assert_root()
    RECRUIT_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        md = _load_meta(model)
        y = md.decoded_action.eq(md.canonical_action_p1).astype(int).to_numpy()
        d = md.delta1c.astype(float).to_numpy()
        groups = md.game_code.to_numpy()
        rng = np.random.default_rng(SEED)
        for L in range(N_LAYERS[model]):
            print(f"[geometry] {model} L{L}/{N_LAYERS[model]-1}", flush=True)
            X = _load_x(model, L)
            d_dec, d_inc = _axes(X, y, d)
            angle = _angle(d_dec, d_inc)
            null = _joint_null(X, y, d, groups, rng)
            lo, med, hi = np.percentile(null, [2.5, 50, 97.5])
            p = (float(np.sum(null <= angle)) + 1.0) / (N_PERM + 1.0)
            independent_med = np.nan
            if L == FINAL_LAYER[model]:
                independent_med = float(np.median(_joint_null(X, y, d, groups, rng, independent=True)))
            rows.append(
                dict(
                    model=model,
                    layer=L,
                    angle_decision_incentive_deg=angle,
                    null_median_deg=float(med),
                    null_lo_deg=float(lo),
                    null_hi_deg=float(hi),
                    perm_p=p,
                    below_null=bool(p < .05),
                    independent_shuffle_median_deg=independent_med,
                    n_obs=len(md),
                    n_games=md.game_code.nunique(),
                    n_perm=N_PERM,
                )
            )
            print(
                f"  angle={angle:.1f} null={med:.1f} [{lo:.1f},{hi:.1f}] p={p:.3f}",
                flush=True,
            )
    out = pd.DataFrame(rows)
    dest = RECRUIT_TABLES / "recruitment_geometry_depth.csv"
    if dest.exists() and set(out.model) != set(MODELS):
        old = pd.read_csv(dest)
        out = pd.concat([old[~old.model.isin(models)], out], ignore_index=True)
    out.sort_values(["model", "layer"]).to_csv(dest, index=False)


def _cue_shifts(
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layer = FINAL_LAYER[model]
    shifts, labels, groups, placebo, placebo_groups = [], [], [], [], []
    gl = games(model)
    for gi, game in enumerate(gl):
        gd = ROOT / model / game
        with np.load(gd / "acts.npz") as z:
            for cb in range(4):
                base_key = f"p1_baseline_cb{cb}_l{layer}"
                if base_key not in z.files:
                    continue
                b = np.asarray(z[base_key], dtype=np.float32)
                for j, (_, cond) in enumerate(CUES.items()):
                    cue_key = f"p1_{cond}_cb{cb}_l{layer}"
                    if cue_key not in z.files:
                        continue
                    shifts.append(np.asarray(z[cue_key], dtype=np.float32) - b)
                    labels.append(j)
                    groups.append(game)
                placebo_key = f"p1_{PLACEBO}_cb{cb}_l{layer}"
                if placebo_key in z.files:
                    placebo.append(np.asarray(z[placebo_key], dtype=np.float32) - b)
                    placebo_groups.append(game)
        if gi % 24 == 0 or gi + 1 == len(gl):
            print(f"[disposition-load] {model}: {gi + 1}/{len(gl)}", flush=True)
    return (
        np.vstack(shifts).astype(np.float32),
        np.asarray(labels, dtype=int),
        np.asarray(groups),
        np.vstack(placebo).astype(np.float32),
        np.asarray(placebo_groups),
    )


def disposition(models: list[str]) -> None:
    _assert_root()
    if not LAYERA_UPTAKE.exists():
        raise FileNotFoundError(
            f"Layer-A per-cue conflict-set uptake is absent: {LAYERA_UPTAKE}; STOP"
        )
    uptake = pd.read_csv(LAYERA_UPTAKE)
    needed = {"model", "trait", "magnitude", "magnitude_lo", "magnitude_hi"}
    if not needed.issubset(uptake.columns):
        raise AssertionError(f"Layer-A uptake schema missing {sorted(needed - set(uptake.columns))}")
    rows, summary = [], []
    for model in models:
        X, y, groups, P, placebo_groups = _cue_shifts(model)
        pred = np.full(len(y), -1, dtype=int)
        fold_acc = []
        placebo_pred = []
        for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, groups=groups)):
            scaler = StandardScaler().fit(X[tr])
            xtr = scaler.transform(X[tr])
            xte = scaler.transform(X[te])
            k = min(PCA_K, xtr.shape[0] - 1, xtr.shape[1])
            pca = PCA(n_components=k, svd_solver="randomized", random_state=SEED + fold)
            ztr = pca.fit_transform(xtr)
            zte = pca.transform(xte)
            lda = LinearDiscriminantAnalysis(solver="svd").fit(ztr, y[tr])
            pred[te] = lda.predict(zte)
            fold_acc.append(accuracy_score(y[te], pred[te]))
            test_games = np.unique(groups[te])
            pmask = np.isin(placebo_groups, test_games)
            pz = pca.transform(scaler.transform(P[pmask]))
            placebo_pred.extend(lda.predict(pz).tolist())
            print(f"[disposition] {model}: fold {fold + 1}/5 acc={fold_acc[-1]:.3f}", flush=True)
        if (pred < 0).any():
            raise AssertionError("missing disposition OOF predictions")
        overall = float(accuracy_score(y, pred))
        per = {name: float(accuracy_score(y[y == j], pred[y == j])) for j, name in enumerate(CUES)}
        sub = uptake[uptake.model.eq(model)].set_index("trait")
        for trait in CUES:
            if trait not in sub.index:
                raise AssertionError(f"Layer-A uptake missing {model}/{trait}")
            r = sub.loc[trait]
            rows.append(
                dict(
                    model=model,
                    trait=trait,
                    lda_trait_accuracy=overall,
                    lda_trait_class_accuracy=per[trait],
                    behavior_magnitude=float(r.magnitude),
                    behavior_magnitude_lo=float(r.magnitude_lo),
                    behavior_magnitude_hi=float(r.magnitude_hi),
                    behavior_n_games=int(r.n_games),
                    layer=FINAL_LAYER[model],
                    pca_components=PCA_K,
                )
            )
        pc = pd.Series(placebo_pred).value_counts(normalize=True).sort_index()
        summary.append(
            dict(
                model=model,
                lda_trait_accuracy=overall,
                fold_accuracy_min=float(np.min(fold_acc)),
                fold_accuracy_max=float(np.max(fold_acc)),
                placebo_projected_n=len(placebo_pred),
                placebo_nearest_risk=float(pc.get(0, 0.0)),
                placebo_nearest_loss=float(pc.get(1, 0.0)),
                placebo_nearest_inequity=float(pc.get(2, 0.0)),
                placebo_nearest_maximin=float(pc.get(3, 0.0)),
                placebo_nearest_selfish=float(pc.get(4, 0.0)),
            )
        )
        print(f"[accept] {model}: held-out 5-way LDA={overall:.3f}", flush=True)
    out = pd.DataFrame(rows)
    dest = RECRUIT_TABLES / "disposition_dissociation.csv"
    if dest.exists() and set(out.model) != set(MODELS):
        old = pd.read_csv(dest)
        out = pd.concat([old[~old.model.isin(models)], out], ignore_index=True)
    out.sort_values(["model", "trait"]).to_csv(dest, index=False)
    pd.DataFrame(summary).sort_values("model").to_csv(
        RECRUIT_TABLES / "disposition_lda_summary.csv", index=False
    )
    means = out.groupby("trait").behavior_magnitude.mean().sort_index()
    print("[accept] behavior uptake means:\n" + means.to_string(), flush=True)


def _models(value: str | None) -> list[str]:
    if not value:
        return list(MODELS)
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(out) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["cache", "decodability", "crystallize", "geometry", "disposition"])
    ap.add_argument("--models", help="comma-separated subset; default all four")
    args = ap.parse_args()
    selected = _models(args.models)
    {
        "cache": build_cache,
        "decodability": decodability,
        "crystallize": crystallize,
        "geometry": geometry,
        "disposition": disposition,
    }[args.command](selected)


if __name__ == "__main__":
    main()
