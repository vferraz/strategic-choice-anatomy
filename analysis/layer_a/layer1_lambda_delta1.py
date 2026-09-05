#!/usr/bin/env python3
"""Canonical Layer-1 behavioral lambda and Delta1 artifacts.

This module is the single source of truth for the behavior -> mechanism coupling
used by Layer 1 and Layer 2. The canonical estimator is the current
``analysis/block_a/layer1_behavior.ipynb`` Pillar-4a estimator:

    logit P(P1 act0) = alpha + lambda * Delta1

where ``Delta1 = EU1(act0) - EU1(act1)``. The canonical/headline row uses each
model's empirical baseline P2 act0 rate as the opponent belief qhat, because
that is the Layer-1 ``lambda_hard`` reported in the notebook. The uniform q=0.5
variant is persisted beside it for incentive-clarity/gating analyses.

The legacy Layer-2 fits are still written to
``analysis/outputs/layer2a/lambda_per_model.parquet`` as robustness-only
diagnostics. They are not the headline behavioral lambda.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from strategic_anatomy.config import data_root, human_refs_root

LOG = logging.getLogger("layer1_lambda_delta1")

ROOT = Path(".")
UNIFIED = human_refs_root() / "unified_pairs.parquet"
# NOTE (phase-4): MASTER_LONG and RAW_ROOT name the superseded DESIGN_V2 round-based
# substrate and its derived gameplay tables. Neither is part of this release (plan §1/§5)
# nor of the data deposit, so both resolve under $SCA_DATA_ROOT to paths that do not
# exist. They are read ONLY by get_rounds_baseline() and compute_layer2_robustness(),
# which in turn are reached only from this module's own main(). main() now checks for
# them up front and exits with an actionable message rather than silently writing tables
# with the gpt-oss rounds missing. Everything the release actually uses from this module
# — fit_lambda, boot_cluster_lambda, delta1, EPS, imported by fig3_qre_to_levelk,
# validate_foundation, levelk_precision_oneshot and probe_common — is pure estimator
# code and touches neither constant.
MASTER_LONG = data_root() / "analysis_outputs" / "01_gameplay/master_behavior_long.parquet"
RAW_ROOT = data_root() / "design_v2_main"
OUT_ROOT = data_root() / "analysis_outputs"
OUT_LAYER2A = OUT_ROOT / "layer2a"
BEHAVIORAL_LAMBDA = OUT_ROOT / "behavioral_lambda.parquet"
DELTA1_OUT = OUT_LAYER2A / "delta1_per_game.parquet"
ROBUSTNESS_OUT = OUT_LAYER2A / "lambda_per_model.parquet"

MODELS = ("qwen", "qwen_instruct", "llama31_instruct", "gptoss")
MODEL_LABEL = {
    "qwen": "Qwen2.5-72B (base)",
    "qwen_instruct": "Qwen2.5-72B-Instruct",
    "llama31_instruct": "Llama-3.1-70B-Instruct",
    "gptoss": "GPT-OSS-120B (MoE)",
    "nagel": "Nagel (human)",
}
BOOT_N = int(os.environ.get("LAYER1_BOOT_N", "4000"))
RNG_SEED = 20260520
EPS = 1e-6
RUN_RE = re.compile(r"^(.+?)_s(\d+)_(.+)$")


def parse_vec(x: str | list[float]) -> list[float]:
    return json.loads(x) if isinstance(x, str) else list(x)


def u_mats(vec8: str | list[float]) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(parse_vec(vec8), float)
    return v[:4].reshape(2, 2), v[4:].reshape(2, 2)


def delta1(vec8: str | list[float], q: float) -> float:
    """EU1(act0) - EU1(act1) when P2 plays act0 with probability q."""
    u1, _ = u_mats(vec8)
    return float((q * u1[0, 0] + (1 - q) * u1[0, 1])
                 - (q * u1[1, 0] + (1 - q) * u1[1, 1]))


def fit_lambda(x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    """Binomial GLM slope/intercept. ``kind='soft'`` clips fractional pref0."""
    X = sm.add_constant(np.asarray(x, float))
    yy = np.asarray(y, float)
    try:
        yy = np.clip(yy, EPS, 1 - EPS) if kind == "soft" else yy
        m = sm.GLM(yy, X, family=sm.families.Binomial()).fit()
        return float(m.params[1]), float(m.params[0])
    except Exception:
        return np.nan, np.nan


def boot_cluster_lambda(
    x: np.ndarray,
    y: np.ndarray,
    gids: np.ndarray,
    kind: str,
    *,
    n_boot: int = BOOT_N,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    gids = np.asarray(gids)
    games = np.unique(gids)
    rows = [np.where(gids == g)[0] for g in games]
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for pick in rng.integers(0, len(games), size=(n_boot, len(games))):
        idx = np.concatenate([rows[k] for k in pick])
        lam, _ = fit_lambda(x[idx], y[idx], kind)
        if np.isfinite(lam):
            out.append(lam)
    if not out:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _cell_type(tp1: str, tp2: str) -> str:
    if tp1 == "observed" or tp2 == "observed":
        return "observed"
    if tp1 == "baseline" and tp2 == "baseline":
        return "baseline"
    if tp2 == "baseline" and tp1 != "baseline":
        return "typeA"
    if tp1 == tp2 and tp1 != "baseline":
        return "typeB"
    return "other"


def _ensure_unified() -> None:
    if UNIFIED.exists():
        return
    raise FileNotFoundError(
        f"missing {UNIFIED}; run `uv run python features/build_unified_pairs.py` first"
    )


def load_unified() -> pd.DataFrame:
    _ensure_unified()
    up = pd.read_parquet(UNIFIED)
    up = up.loc[:, ~up.columns.duplicated()].copy()
    if "cell" not in up.columns:
        up["cell"] = [_cell_type(a, b) for a, b in zip(up["trait_p1"], up["trait_p2"])]
    return up[up["in_model_set"]].copy()


def baseline_frame(up: pd.DataFrame, agent: str) -> pd.DataFrame:
    d = up[(up["sample"] == agent) & (up["cell"] == "baseline")].copy()
    cols = ["game_code", "matrix_8vec", "realized_p1", "realized_p2", "pref0_p1"]
    return d[cols].sort_values("game_code").reset_index(drop=True)


def _master_model_col(df: pd.DataFrame) -> str:
    if "model" in df.columns:
        return "model"
    if "model_kind" in df.columns:
        return "model_kind"
    raise KeyError("master behavior table has neither model nor model_kind")


def get_rounds_baseline() -> pd.DataFrame:
    """Baseline round-level rows for all four models, including raw gpt-oss."""
    keep = ["agent", "game_code", "seed", "round", "cell", "move1", "move2", "pref0_p1", "pref0_p2"]
    m = pd.read_parquet(MASTER_LONG)
    model_col = _master_model_col(m)
    llm = m[m["cell_name"].eq("baseline")].rename(
        columns={
            model_col: "agent",
            "cell_name": "cell",
            "pref_opt0_p1_round": "pref0_p1",
            "pref_opt0_p2_round": "pref0_p2",
        }
    )
    llm = llm[[c for c in keep if c in llm.columns]].copy()

    rows = []
    for f in glob.glob(str(RAW_ROOT / "gptoss/*_baseline/results.parquet")):
        base = os.path.basename(os.path.dirname(f))
        mt = RUN_RE.match(base)
        if not mt:
            continue
        d = pd.read_parquet(
            f,
            columns=[
                "game_code", "round", "move1", "move2",
                "pref_opt0_p1_round", "pref_opt0_p2_round",
            ],
        )
        d["seed"] = int(mt.group(2))
        d["cell"] = "baseline"
        d["agent"] = "gptoss"
        d = d.rename(columns={"pref_opt0_p1_round": "pref0_p1", "pref_opt0_p2_round": "pref0_p2"})
        rows.append(d[keep])
    return pd.concat([llm, *rows], ignore_index=True)


def delta1_per_game(up: pd.DataFrame) -> pd.DataFrame:
    bf = baseline_frame(up, "qwen")
    out = bf[["game_code", "matrix_8vec"]].copy()
    out["q"] = 0.5
    out["q_kind"] = "uniform"
    out["delta1"] = [delta1(v, 0.5) for v in out["matrix_8vec"]]
    out["abs_delta1"] = out["delta1"].abs()
    return out[["game_code", "matrix_8vec", "q_kind", "q", "delta1", "abs_delta1"]]


def compute_behavioral_lambda(up: pd.DataFrame) -> pd.DataFrame:
    rounds = get_rounds_baseline()
    recs: list[dict[str, object]] = []
    for model in MODELS:
        bf = baseline_frame(up, model)
        if bf.empty:
            continue
        bf["delta1_empirical"] = [delta1(v, q) for v, q in zip(bf["matrix_8vec"], bf["realized_p2"])]
        bf["delta1_uniform"] = [delta1(v, 0.5) for v in bf["matrix_8vec"]]
        r1 = rounds[(rounds["agent"] == model) & (rounds["round"] == 1)].copy()
        r1 = r1.merge(bf[["game_code", "delta1_empirical", "delta1_uniform"]], on="game_code", how="inner")
        r1 = r1.dropna(subset=["move1"]).copy()
        r1["y_hard"] = (r1["move1"].astype(int) == 0).astype(int)

        for q_kind, xcol, canonical in [
            ("empirical_baseline_p2_rate", "delta1_empirical", True),
            ("uniform_q50", "delta1_uniform", False),
        ]:
            qf = bf.dropna(subset=[xcol, "pref0_p1"]).copy()
            rr = r1.dropna(subset=[xcol, "y_hard"]).copy()
            lam_soft, alpha_soft = fit_lambda(qf[xcol].to_numpy(), qf["pref0_p1"].to_numpy(), "soft")
            lam_hard, alpha_hard = fit_lambda(rr[xcol].to_numpy(), rr["y_hard"].to_numpy(), "hard")
            soft_lo, soft_hi = boot_cluster_lambda(
                qf[xcol].to_numpy(), qf["pref0_p1"].to_numpy(), qf["game_code"].to_numpy(), "soft"
            )
            hard_lo, hard_hi = boot_cluster_lambda(
                rr[xcol].to_numpy(), rr["y_hard"].to_numpy(), rr["game_code"].to_numpy(), "hard"
            )
            recs.append(
                {
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "q_kind": q_kind,
                    "q": "per-game baseline P2 act0 rate" if canonical else "0.5",
                    "is_canonical": canonical,
                    "lam_hard": lam_hard,
                    "lam_hard_ci_low": hard_lo,
                    "lam_hard_ci_high": hard_hi,
                    "lam_soft": lam_soft,
                    "lam_soft_ci_low": soft_lo,
                    "lam_soft_ci_high": soft_hi,
                    "alpha": alpha_hard,
                    "alpha_hard": alpha_hard,
                    "alpha_soft": alpha_soft,
                    "n_matches": int(rr[["game_code", "seed"]].drop_duplicates().shape[0]),
                    "n_obs": int(len(rr)),
                    "n_games": int(rr["game_code"].nunique()),
                    "outcome_axis": "act0",
                    "delta1_definition": "EU1(act0)-EU1(act1)",
                    "estimator": "Layer1 Pillar-4a binomial GLM, round-1 baseline",
                    "source": "layer1_behavior.ipynb Pillar 4a (private repo provenance)",
                }
            )

    if "nagel" in set(up["sample"]):
        bf = baseline_frame(up, "nagel")
        if not bf.empty:
            bf["delta1_empirical"] = [delta1(v, q) for v, q in zip(bf["matrix_8vec"], bf["realized_p2"])]
            lam_h, alpha_h = fit_lambda(bf["delta1_empirical"].to_numpy(), bf["realized_p1"].to_numpy(), "soft")
            lo, hi = boot_cluster_lambda(
                bf["delta1_empirical"].to_numpy(), bf["realized_p1"].to_numpy(), bf["game_code"].to_numpy(), "soft"
            )
            recs.append(
                {
                    "model": "nagel",
                    "model_label": MODEL_LABEL["nagel"],
                    "q_kind": "empirical_baseline_p2_rate",
                    "q": "per-game baseline P2 act0 rate",
                    "is_canonical": True,
                    "lam_hard": lam_h,
                    "lam_hard_ci_low": lo,
                    "lam_hard_ci_high": hi,
                    "lam_soft": np.nan,
                    "lam_soft_ci_low": np.nan,
                    "lam_soft_ci_high": np.nan,
                    "alpha": alpha_h,
                    "alpha_hard": alpha_h,
                    "alpha_soft": np.nan,
                    "n_matches": int(len(bf)),
                    "n_obs": int(len(bf)),
                    "n_games": int(bf["game_code"].nunique()),
                    "outcome_axis": "act0",
                    "delta1_definition": "EU1(act0)-EU1(act1)",
                    "estimator": "Layer1 Pillar-4a binomial GLM, realized human rate",
                    "source": "layer1_behavior.ipynb Pillar 4a (private repo provenance)",
                }
            )
    return pd.DataFrame(recs)


def _robust_fit(sub: pd.DataFrame, formula: str) -> dict[str, float | int]:
    res = smf.logit(formula, data=sub).fit(
        disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": sub["match_id"]}
    )
    ci = res.conf_int().loc["delta1"]
    return {
        "lambda": float(res.params["delta1"]),
        "lambda_ci_low": float(ci[0]),
        "lambda_ci_high": float(ci[1]),
        "intercept": float(res.params["Intercept"]),
        "n_obs": int(res.nobs),
        "n_matches": int(sub["match_id"].nunique()),
        "mean_aligned": float(sub["aligned_canonical_p1"].mean()),
    }


def compute_layer2_robustness(up: pd.DataFrame) -> pd.DataFrame:
    """Old all-round/cell-FE fits retained only as robustness diagnostics."""
    rounds = []
    m = pd.read_parquet(MASTER_LONG)
    model_col = _master_model_col(m)
    base_cols = ["game_code", "round", "move1", "cell_name", "seed", model_col]
    m = m[m["move1"].isin([0, 1])][base_cols + ["canonical_action_p1"]].rename(columns={model_col: "model"})
    rounds.append(m)
    feat = up[["game_code", "matrix_8vec", "canonical_action_p1"]].drop_duplicates("game_code").copy()
    act0_gap = np.array([delta1(v, 0.5) for v in feat["matrix_8vec"]], dtype=float)
    c1 = feat["canonical_action_p1"].astype(int).to_numpy()
    feat["delta1_layer2_robustness"] = np.where(c1 == 0, act0_gap, -act0_gap)
    for f in glob.glob(str(RAW_ROOT / "gptoss/*/results.parquet")):
        mt = RUN_RE.match(os.path.basename(os.path.dirname(f)))
        if not mt:
            continue
        d = pd.read_parquet(f, columns=["game_code", "round", "move1", "cell_name"])
        d = d[d["move1"].isin([0, 1])].copy()
        d["model"] = "gptoss"
        d["seed"] = int(mt.group(2))
        d = d.merge(feat[["game_code", "canonical_action_p1"]], on="game_code", how="left")
        rounds.append(d[["game_code", "round", "move1", "cell_name", "seed", "model", "canonical_action_p1"]])
    b = pd.concat(rounds, ignore_index=True)
    d1 = feat[["game_code", "delta1_layer2_robustness"]].rename(
        columns={"delta1_layer2_robustness": "delta1"}
    )
    b = b.merge(d1, on="game_code", how="left")
    b["aligned_canonical_p1"] = (b["move1"].astype(int) == b["canonical_action_p1"].astype(int)).astype(int)
    b["match_id"] = b["model"] + "::" + b["game_code"] + "_s" + b["seed"].astype(str) + "_" + b["cell_name"]

    recs = []
    for model in MODELS:
        sub_all = b[b["model"] == model].dropna(subset=["delta1", "aligned_canonical_p1"])
        base = sub_all[sub_all["cell_name"] == "baseline"]
        for fit, data, formula in [
            ("baseline_primary", base, "aligned_canonical_p1 ~ delta1"),
            ("allcells_cellFE", sub_all, "aligned_canonical_p1 ~ delta1 + C(cell_name)"),
        ]:
            try:
                r = _robust_fit(data, formula)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("robustness fit failed for %s/%s: %s", model, fit, exc)
                r = {"lambda": np.nan, "n_obs": int(len(data)), "n_matches": int(data["match_id"].nunique())}
            r.update(
                model=model,
                fit=fit,
                is_headline=False,
                delta1_definition="EU1(canonical_action)-EU1(other_action), q=0.5",
                outcome_axis="aligned_canonical_p1",
                note="Layer-2 robustness only; do not use as canonical lambda",
            )
            recs.append(r)
    return pd.DataFrame(recs)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    missing = [p for p in (MASTER_LONG, RAW_ROOT) if not p.exists()]
    if missing:
        LOG.error(
            "This CLI rebuilds the DESIGN_V2 round-based Layer-2 inputs, which are NOT part "
            "of this release or the data deposit. Missing: %s. The estimator functions this "
            "module exports (fit_lambda, boot_cluster_lambda, delta1) are what the released "
            "analysis imports and they need none of this — see docs/DATA.md.",
            ", ".join(str(p) for p in missing),
        )
        return 2
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_LAYER2A.mkdir(parents=True, exist_ok=True)
    up = load_unified()
    d1 = delta1_per_game(up)
    lam = compute_behavioral_lambda(up)
    robustness = compute_layer2_robustness(up)
    d1.to_parquet(DELTA1_OUT, index=False)
    lam.to_parquet(BEHAVIORAL_LAMBDA, index=False)
    robustness.to_parquet(ROBUSTNESS_OUT, index=False)
    print("\n=== canonical behavioral lambda ===")
    show = lam[lam["is_canonical"]][
        ["model", "lam_hard", "lam_hard_ci_low", "lam_hard_ci_high", "lam_soft", "alpha", "q_kind", "n_matches"]
    ]
    print(show.round(3).to_string(index=False))
    print(f"\nwrote {BEHAVIORAL_LAMBDA}")
    print(f"wrote {DELTA1_OUT}")
    print(f"wrote robustness-only {ROBUSTNESS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
