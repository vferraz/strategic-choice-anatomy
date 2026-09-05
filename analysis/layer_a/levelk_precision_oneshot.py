#!/usr/bin/env python3
"""Level-k x precision behavioral estimator on the one-shot substrate (SPEC §1).

Replaces single-lambda QRE with a two-axis bounded-rationality fingerprint:
reasoning depth tau (level-k belief: L1 best-responds to uniform/L0, L2 to L1,
L3 to L2) x response precision lambda (logit best-response sharpness). Per model
we fit, for each belief b,

    Logit(aligned_canonical_p1 ~ const + gap_b_canonical)

over the one-shot P1 baseline cells (16 counterbalance cells x 144 games). The
best-fitting belief is the model's revealed reasoning level; the slope is the
precision lambda at that belief.

CANONICAL AXIS (docs/METHODS.md HC-2): the outcome is ``aligned_canonical_p1`` and the
regressor is the canonical-signed gap (``canonical_sign``). We never pool raw
``decoded_action == 0`` across games. ``gap_b`` is positional (EU1(act0)-EU1(act1)
at belief q_b) and is then oriented to the canonical action.

Beliefs (opponent P2 act0 probability q_b), per game:
  L1 -> q = 0.5                       (best-response to uniform / L0)
  L2 -> P2 plays l1_action_p2         (q = 1 if act0, 0 if act1, 0.5 if undefined)
  L3 -> P2 plays l2_action_p2         (same mapping)
  EQ -> empirical q-hat = per-game P2-baseline act0 rate (the natural "what the
        opponent actually does" belief; reuses the one-shot baselines only).

Note on q-hat granularity: the SPEC text says "per model"; the repo's
``build_delta_tables`` machinery uses the per-game P2-baseline act0 rate. We use
the per-game rate (the natural empirical belief) and document the choice here.

Per-game revealed level (the manifest N1/N2 consume): using the model-level
fitted (alpha_b, lambda_b), assign each game ``argmax_{b in {L1,L2,L3}}`` of the
per-game log-likelihood of its 16 cells. Default to L1 on a tie or when the
level beliefs coincide for that game. THIS RULE IS A FAITHFUL RECONSTRUCTION OF
THE SPEC (the scratch proof-of-concept scripts no longer exist); the population
R^2 / lambda results do not depend on it, only N1's target labels do.

Outputs (analysis/block_a/{tables,figures}/):
  tables/levelk_fit.csv             per (model, belief): McFadden R^2 (full + 48
                                    diagnostic games), llf, dLL vs EQ, lambda+CI.
  tables/levelk_revealed_level.csv  per (model, game): revealed_level, p_canon,
                                    q_*, gap_*_canonical, complexity_score.
  figures/levelk_precision.{png,pdf}

CPU only; reads behavior (results.parquet) from $SCA_DATA_ROOT/substrate via
steering.extract_directions.load_behavior.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.probe_common import (  # noqa: E402
    game_meta, parse_8vec, MODELS, SHORT, COL, style,
)
from collection.oneshot_common import delta1, canonical_sign  # noqa: E402
from steering.extract_directions import load_behavior  # noqa: E402
from analysis.layer_a.layer1_lambda_delta1 import (  # noqa: E402
    fit_lambda, boot_cluster_lambda,
)

import statsmodels.api as sm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from strategic_anatomy.config import repo_root, results_root, substrate_root

ROOT = repo_root()


TAB_DIR = results_root() / "layer_a"
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
BELIEFS = ["L1", "L2", "L3", "EQ"]
LEVELS = ["L1", "L2", "L3"]  # candidate revealed levels (EQ is a belief, not a level)


# ---------------------------------------------------------------------------
def _q_from_action(a: int) -> float:
    """Opponent (P2) plays deterministic action a -> P2 act0 probability.

    a == 0 -> q = 1.0; a == 1 -> q = 0.0; a == -1 (tie / undefined) -> 0.5.
    """
    a = int(a)
    if a == 0:
        return 1.0
    if a == 1:
        return 0.0
    return 0.5  # undefined level action -> indifferent opponent randomizes


def _qhat_p2(model: str) -> pd.Series:
    """Per-game empirical P2-baseline act0 rate (the EQ belief)."""
    beh = load_behavior(model)
    p2b = beh[(beh["player"] == 2) & (beh["condition"] == "baseline")]
    p2b = p2b[p2b["decoded_action"].isin([0, 1])]
    return p2b.groupby("game_code")["decoded_action"].apply(lambda s: float((s == 0).mean()))


def build_game_beliefs(meta: pd.DataFrame, qhat_p2: pd.Series) -> pd.DataFrame:
    """Per game: belief q_b and canonical-signed gap_b for each belief, plus the
    level-implied P1 actions used to flag the diagnostic games."""
    rows = []
    for _, r in meta.iterrows():
        g = r["game_code"]
        vec = parse_8vec(r["canonical_8vec"])
        canon = int(r["canonical_action_p1"])
        q = {
            "L1": 0.5,
            "L2": _q_from_action(r["l1_action_p2"]),
            "L3": _q_from_action(r["l2_action_p2"]),
            "EQ": float(qhat_p2.get(g, 0.5)),
        }
        rec = {"game_code": g, "canonical_action_p1": canon,
               "complexity_score": float(r["complexity_score"]),
               "iesds_depth": int(r["iesds_depth"]),
               "nagel_lk_type": r["nagel_lk_type"]}
        for b in BELIEFS:
            rec[f"q_{b}"] = q[b]
            rec[f"gap_{b}"] = canonical_sign(delta1(vec, q[b]), canon)
        # theoretical level-ladder P1 actions (model-independent; for the
        # diagnostic mask). Undefined/tie actions are stored as -1.
        rec["act_L0"] = int(r["l0_action_p1"]) if pd.notna(r["l0_action_p1"]) else -1
        rec["act_L1"] = int(r["l1_action_p1"]) if pd.notna(r["l1_action_p1"]) else -1
        rec["act_L2"] = int(r["l2_action_p1"]) if pd.notna(r["l2_action_p1"]) else -1
        rec["dominant_action_p1"] = int(r["dominant_action_p1"]) if pd.notna(r["dominant_action_p1"]) else -1
        rows.append(rec)
    gb = pd.DataFrame(rows)
    # diagnostic games: the theoretical level ladder predicts different P1 acts
    # (model-independent; "kills dominance collinearity" because dominant games
    # have all levels agree on the dominant action). Ignore -1 undefined entries.
    def _disagree(row) -> bool:
        acts = {int(row[c]) for c in ("act_L0", "act_L1", "act_L2") if int(row[c]) in (0, 1)}
        return len(acts) > 1
    gb["diagnostic"] = gb.apply(_disagree, axis=1)
    return gb


def _p1_baseline_choices(model: str) -> pd.DataFrame:
    """Per-cell P1 baseline choices: game_code, counterbalance_id, decoded_action."""
    beh = load_behavior(model)
    p1b = beh[(beh["player"] == 1) & (beh["condition"] == "baseline")].copy()
    p1b = p1b[p1b["decoded_action"].isin([0, 1])]
    return p1b[["game_code", "counterbalance_id", "decoded_action"]].reset_index(drop=True)


def _fit_belief(df: pd.DataFrame, xcol: str) -> dict:
    """Logit(y ~ const + xcol). Returns McFadden R^2, llf, lambda(+CI), n."""
    d = df.dropna(subset=[xcol, "y"])
    out = {"r2": np.nan, "llf": np.nan, "lam": np.nan,
           "lam_lo": np.nan, "lam_hi": np.nan,
           "n_obs": int(len(d)), "n_games": int(d["game_code"].nunique())}
    if len(d) < 8 or d["y"].nunique() < 2:
        return out
    x = d[xcol].to_numpy(float)
    y = d["y"].to_numpy(int)
    try:
        res = sm.Logit(y, sm.add_constant(x)).fit(disp=0, maxiter=200)
        out["r2"] = float(res.prsquared)
        out["llf"] = float(res.llf)
        out["lam"] = float(res.params[1])
    except Exception:
        # fall back to GLM slope (no R^2) if Logit fails to converge
        lam, _ = fit_lambda(x, y, "hard")
        out["lam"] = lam
    lo, hi = boot_cluster_lambda(x, y, d["game_code"].to_numpy(), "hard", n_boot=2000)
    out["lam_lo"], out["lam_hi"] = lo, hi
    return out


def _assign_levels(cells: pd.DataFrame, fits: dict[str, dict]) -> pd.DataFrame:
    """Per game, revealed_level = argmax_{L1,L2,L3} per-game log-likelihood under
    the model-level fitted (alpha_b, lambda_b). Default L1 on tie / coincident."""
    # recover (alpha, lambda) per level from a fresh fit so we have the intercept
    ab = {}
    for b in LEVELS:
        d = cells.dropna(subset=[f"gap_{b}", "y"])
        try:
            res = sm.Logit(d["y"].to_numpy(int),
                           sm.add_constant(d[f"gap_{b}"].to_numpy(float))).fit(disp=0, maxiter=200)
            ab[b] = (float(res.params[0]), float(res.params[1]))
        except Exception:
            lam, alpha = fit_lambda(d[f"gap_{b}"].to_numpy(float), d["y"].to_numpy(int), "hard")
            ab[b] = (alpha if np.isfinite(alpha) else 0.0, lam if np.isfinite(lam) else 0.0)

    def _ll_game(sub: pd.DataFrame, b: str) -> float:
        alpha, lam = ab[b]
        z = alpha + lam * sub[f"gap_{b}"].to_numpy(float)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-9, 1 - 1e-9)
        y = sub["y"].to_numpy(int)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    rows = []
    for g, sub in cells.groupby("game_code"):
        lls = {b: _ll_game(sub, b) for b in LEVELS}
        # if every level's gap is identical for this game, levels are indistinguishable
        gaps = {round(float(sub[f"gap_{b}"].iloc[0]), 9) for b in LEVELS}
        if len(gaps) == 1:
            best = "L1"
        else:
            best = max(LEVELS, key=lambda b: (lls[b], -LEVELS.index(b)))  # tie -> lower level
        rows.append({"game_code": g, "revealed_level": best,
                     **{f"ll_{b}": lls[b] for b in LEVELS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def run_model(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = game_meta()
    qhat = _qhat_p2(model)
    gb = build_game_beliefs(meta, qhat)
    choices = _p1_baseline_choices(model)
    cells = choices.merge(gb, on="game_code", how="inner")
    cells["y"] = (cells["decoded_action"].astype(int)
                  == cells["canonical_action_p1"].astype(int)).astype(int)

    diag = gb.loc[gb["diagnostic"], "game_code"]
    cells_diag = cells[cells["game_code"].isin(diag)]

    fits = {b: _fit_belief(cells, f"gap_{b}") for b in BELIEFS}
    fits_diag = {b: _fit_belief(cells_diag, f"gap_{b}") for b in BELIEFS}
    ll_eq = fits["EQ"]["llf"]

    fit_rows = []
    for b in BELIEFS:
        f = fits[b]
        fit_rows.append({
            "model": model, "belief": b,
            "mcfadden_r2": f["r2"],
            "mcfadden_r2_diagnostic": fits_diag[b]["r2"],
            "llf": f["llf"],
            "dLL_vs_EQ": (f["llf"] - ll_eq) if (np.isfinite(f["llf"]) and np.isfinite(ll_eq)) else np.nan,
            "lambda": f["lam"], "lambda_ci_low": f["lam_lo"], "lambda_ci_high": f["lam_hi"],
            "n_obs": f["n_obs"], "n_games": f["n_games"],
            "n_diagnostic_games": int(diag.nunique()),
        })
    fit_df = pd.DataFrame(fit_rows)

    # per-game revealed level manifest
    levels = _assign_levels(cells, fits)
    p_canon = cells.groupby("game_code")["y"].mean().rename("p_canon")
    manifest = (gb.merge(levels, on="game_code", how="left")
                  .merge(p_canon, on="game_code", how="left"))
    manifest.insert(0, "model", model)
    keep = (["model", "game_code", "revealed_level", "p_canon", "complexity_score",
             "iesds_depth", "nagel_lk_type", "diagnostic"]
            + [f"q_{b}" for b in BELIEFS] + [f"gap_{b}" for b in BELIEFS])
    manifest = manifest[keep]

    # depth <-> competence and level counts (printed)
    sub = manifest.dropna(subset=["p_canon", "complexity_score"])
    rho, pval = spearmanr(sub["p_canon"], sub["complexity_score"]) if len(sub) > 3 else (np.nan, np.nan)
    counts = manifest["revealed_level"].value_counts().reindex(LEVELS, fill_value=0)
    fit_df["rho_pcanon_complexity"] = float(rho)
    fit_df["rho_pval"] = float(pval)
    for lv in LEVELS:
        fit_df[f"n_{lv}"] = int(counts[lv])

    print(f"\n[{SHORT.get(model, model)}] revealed levels "
          f"L1/L2/L3 = {counts['L1']}/{counts['L2']}/{counts['L3']}  "
          f"| diagnostic games = {int(diag.nunique())}  "
          f"| rho(p_canon, complexity) = {rho:.3f}")
    print(fit_df[["belief", "mcfadden_r2", "mcfadden_r2_diagnostic", "dLL_vs_EQ",
                  "lambda", "lambda_ci_low", "lambda_ci_high"]].round(3).to_string(index=False))
    return fit_df, manifest


# ---------------------------------------------------------------------------
def _make_figure(fit_all: pd.DataFrame, man_all: pd.DataFrame, models: list[str]) -> None:
    PS = style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: McFadden R^2 by belief, grouped by model
    ax = axes[0]
    width = 0.8 / max(1, len(models))
    xpos = np.arange(len(BELIEFS))
    for i, m in enumerate(models):
        sub = fit_all[fit_all["model"] == m].set_index("belief").reindex(BELIEFS)
        ax.bar(xpos + i * width, sub["mcfadden_r2"].to_numpy(), width,
               label=SHORT.get(m, m), color=COL.get(m, None))
    ax.set_xticks(xpos + width * (len(models) - 1) / 2)
    ax.set_xticklabels(BELIEFS)
    ax.set_ylabel("McFadden $R^2$")
    ax.set_title("Belief fit by reasoning level")
    ax.legend(fontsize=7, frameon=False)

    # Panel 2: precision lambda at the L1 belief, by model
    ax = axes[1]
    sub = fit_all[fit_all["belief"] == "L1"].set_index("model").reindex(models)
    yerr = np.vstack([
        (sub["lambda"] - sub["lambda_ci_low"]).to_numpy(),
        (sub["lambda_ci_high"] - sub["lambda"]).to_numpy(),
    ])
    ax.bar(range(len(models)), sub["lambda"].to_numpy(),
           color=[COL.get(m) for m in models])
    ax.errorbar(range(len(models)), sub["lambda"].to_numpy(), yerr=np.abs(yerr),
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([SHORT.get(m, m) for m in models], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"precision $\lambda$ (L1 belief)")
    ax.set_title("Response precision")

    # Panel 3: depth <-> competence (p_canon vs complexity)
    ax = axes[2]
    for m in models:
        sub = man_all[man_all["model"] == m].dropna(subset=["p_canon", "complexity_score"])
        ax.scatter(sub["complexity_score"], sub["p_canon"], s=10, alpha=0.5,
                   color=COL.get(m), label=SHORT.get(m, m))
    ax.set_xlabel("complexity_score")
    ax.set_ylabel(r"$p_\mathrm{canon}$")
    ax.set_title("Depth vs competence")
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"levelk_precision.{ext}", dpi=300)
    plt.close(fig)
    print(f"\nwrote {FIG_DIR / 'levelk_precision.png'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Level-k x precision behavioral estimator (one-shot, SPEC §1).")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    fit_frames, man_frames, done = [], [], []
    for m in args.models:
        # phase-4: guard repointed from the superseded A/B root (output/oneshot_main/substrate,
        # which cannot exist in a clone, so every model was skipped) to the released root that
        # load_behavior() actually reads.
        if not (substrate_root() / m).exists():
            print(f"[{m}] no substrate at {substrate_root() / m} — skipping")
            continue
        fit_df, man = run_model(m)
        fit_frames.append(fit_df)
        man_frames.append(man)
        done.append(m)

    if not fit_frames:
        print("no models processed")
        return 1
    fit_all = pd.concat(fit_frames, ignore_index=True)
    man_all = pd.concat(man_frames, ignore_index=True)
    fit_all.to_csv(TAB_DIR / "levelk_fit.csv", index=False)
    man_all.to_csv(TAB_DIR / "levelk_revealed_level.csv", index=False)
    print(f"\nwrote {TAB_DIR / 'levelk_fit.csv'}")
    print(f"wrote {TAB_DIR / 'levelk_revealed_level.csv'}  ({len(man_all)} rows)")
    if not args.no_fig:
        _make_figure(fit_all, man_all, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
