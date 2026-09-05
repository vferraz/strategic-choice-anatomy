#!/usr/bin/env python3
"""Retired Fig 3 compatibility module.

The canonical paper Fig 3 writer is now
``analysis/layer_a/figscripts/fig3_qre_to_levelk.py``.  This source-module
implementation is retained only for historical debugging and must not own
``tables/f3_lambda.csv``.

Legacy implementation below: QRE -> level-k (one-shot).

The new Layer-A result. Three panels, styled from ``_paper_style`` to match Figs 1-2:
  (a) Quantal response curve: P(canonical) vs the canonical-signed incentive gap at the
      uniform belief q=0.5, logistic fit per model + the Nagel human (Moore, Germano &
      Nagel 2026) curve overlaid. Annotates the precision lambda. PRIMARY lambda = hard /
      the realized generated/committed move (binary, apples-to-apples with the human's
      binary choice); soft (pref0) lambda is reported as robustness in f3_lambda_legacy.csv.
  (b) Model selection: leave-games-out (GroupKFold-by-game) predictive log-loss for
      {L0,NE,QRE,Lk,QLk,CH,QCH} per agent + human, bootstrap-by-game CIs. NE catastrophic,
      QLk/QCH win, capable LLMs + human in the same family, Llama closest to chance.
  (c) Fingerprint: each agent on the bounded-rationality plane (revealed reasoning depth
      x precision lambda). Nash = the deep + infinitely-precise limiting corner.

Tables: f3_lambda_legacy.csv, f3_model_selection_cv.csv (built by model_selection_cv),
        f3_fingerprint.csv, f3_human_lambda.csv (built by human_lambda_mgn).

Payoff scale (Methods + caption): LLMs and MGN are both on ranks {1,2,3,4}
(payoff_multiplier=1) — lambdas are per-rank and directly comparable; Griffiths is
cardinal and lives on a separate axis (not shown here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from analysis.layer_a.src import human_lambda_mgn as HL  # noqa: E402
from analysis.layer_a.src import model_selection_cv as MSC  # noqa: E402
from analysis.layer_a.layer1_lambda_delta1 import fit_lambda, boot_cluster_lambda  # noqa: E402
import analysis.layer_a.levelk_precision_oneshot as LK  # noqa: E402
from strategic_anatomy.config import repo_root, results_root

ROOT = repo_root()


DATA = SD.DATA
OUTDIR = ROOT / "analysis" / "layer_a" / "figures"
TABLES = results_root() / "layer_a"
AGENTS = SD.MODELS + ["nagel_human"]
AGENT_COL = {**SD.COL, "nagel_human": SD.HUMAN_COL["nagel"]}
AGENT_LBL = {**SD.SHORT, "nagel_human": "Human"}


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---- lambda table (model hard q05/emp + soft; human binary q05) -------------
def build_lambda_table(cells, human_row):
    rows = []
    for mod in SD.MODELS:
        d = cells[cells.model == mod]
        y = d["y_canon"].to_numpy(int)
        canon = d["canonical_action_p1"].to_numpy(int)
        psoft = np.where(canon == 0, d["pref0"].to_numpy(float), 1 - d["pref0"].to_numpy(float))
        g = d["game_code"].to_numpy()
        for belief, x in (("q05", d["delta1c_q05"].to_numpy(float)),
                          ("emp", d["delta1c_emp"].to_numpy(float))):
            lam, a = fit_lambda(x, y, "hard")
            lo, hi = boot_cluster_lambda(x, y, g, "hard", n_boot=SD.BOOT_N)
            rows.append({"agent": mod, "readout": "hard_generate", "belief": belief,
                         "lambda": lam, "alpha": a, "lo": lo, "hi": hi, "n_games": d["game_code"].nunique()})
        lam_s, a_s = fit_lambda(d["delta1c_q05"].to_numpy(float), psoft, "soft")
        lo_s, hi_s = boot_cluster_lambda(d["delta1c_q05"].to_numpy(float), psoft, g, "soft", n_boot=SD.BOOT_N)
        rows.append({"agent": mod, "readout": "soft_pref0", "belief": "q05",
                     "lambda": lam_s, "alpha": a_s, "lo": lo_s, "hi": hi_s, "n_games": d["game_code"].nunique()})
    rows.append({"agent": "nagel_human", "readout": "binary_choice", "belief": "q05",
                 "lambda": human_row["lambda"], "alpha": human_row["alpha"],
                 "lo": human_row["lambda_boot_game_lo"], "hi": human_row["lambda_boot_game_hi"],
                 "n_games": human_row["n_games"]})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "f3_lambda_legacy.csv", index=False)
    return df


# ---- fingerprint: revealed depth x precision --------------------------------
def build_fingerprint(human_row, lam_tab, cells):
    """Depth x precision fingerprint computed on the CORRECTED decision data.

    Reuses levelk_precision_oneshot's per-game level-assignment logic (LK._assign_levels)
    but feeds it the generate()/commit decisions — NOT levelk's own output, which is built
    on the substrate slot argmax (load_behavior) and is therefore on the wrong decoder.
    Precision λ is the q=0.5 hard/generate slope from the corrected λ table."""
    gp = MSC.game_predictors().reset_index()[["game_code", "gc_L1", "gc_L2", "gc_L3"]]
    lvl_num = {"L1": 1, "L2": 2, "L3": 3}

    def _levels(yc_df):  # yc_df: game_code, y(0/1) decision; -> revealed-level dist
        c = yc_df.merge(gp, on="game_code", how="left").rename(
            columns={"gc_L1": "gap_L1", "gc_L2": "gap_L2", "gc_L3": "gap_L3"})
        lev = LK._assign_levels(c, {})
        dist = lev["revealed_level"].value_counts(normalize=True)
        mean_lvl = float(np.mean([lvl_num[x] for x in lev["revealed_level"]]))
        return dist, mean_lvl

    rows = []
    for mod in SD.MODELS:
        yc = cells[cells.model == mod][["game_code", "y_canon"]].rename(columns={"y_canon": "y"})
        dist, mean_lvl = _levels(yc)
        lam = float(lam_tab[(lam_tab.agent == mod) & (lam_tab.readout == "hard_generate")
                            & (lam_tab.belief == "q05")]["lambda"].iloc[0])
        rows.append({"agent": mod, "mean_level": mean_lvl, "lambda": lam,
                     "frac_L1": float(dist.get("L1", 0)), "frac_L2": float(dist.get("L2", 0)),
                     "frac_L3": float(dist.get("L3", 0))})
    # human: same machinery on MGN individual decisions
    yc = HL.individual_choices()[["game_code", "y_canon"]].rename(columns={"y_canon": "y"})
    dist, mean_lvl = _levels(yc)
    rows.append({"agent": "nagel_human", "mean_level": mean_lvl, "lambda": human_row["lambda"],
                 "frac_L1": float(dist.get("L1", 0)), "frac_L2": float(dist.get("L2", 0)),
                 "frac_L3": float(dist.get("L3", 0))})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "f3_fingerprint.csv", index=False)
    return df


# ---- figure -----------------------------------------------------------------
def render(lam_tab, cv, fp, gl, human_curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from strategic_anatomy import paper_style as S
    S.apply()

    fig = plt.figure(figsize=(7.2, 2.7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.42,
                           left=0.075, right=0.985, top=0.84, bottom=0.20)

    # ---- (a) quantal response curve ----
    axa = fig.add_subplot(gs[0])
    xs = np.linspace(-3, 3, 100)
    for mod in SD.MODELS:
        d = gl[gl.model == mod]
        axa.scatter(d["delta1c_q05"], d["p_canon"], s=5, alpha=0.18, color=SD.COL[mod], edgecolor="none")
        r = lam_tab[(lam_tab.agent == mod) & (lam_tab.readout == "hard_generate") & (lam_tab.belief == "q05")].iloc[0]
        axa.plot(xs, _sig(r["alpha"] + r["lambda"] * xs), color=SD.COL[mod], lw=1.5,
                 label=f"{SD.SHORT[mod]} λ={r['lambda']:.2f}")
    hr = lam_tab[lam_tab.agent == "nagel_human"].iloc[0]
    axa.plot(xs, _sig(hr["alpha"] + hr["lambda"] * xs), color=SD.HUMAN_COL["nagel"], lw=1.8, ls="--",
             label=f"Human λ={hr['lambda']:.2f}")
    axa.axhline(0.5, color=S.GREY, lw=0.6, ls=":"); axa.axvline(0, color=S.GREY, lw=0.6, ls=":")
    axa.set_xlim(-3, 3); axa.set_ylim(0, 1)
    axa.set_xlabel("canonical incentive gap Δ₁ᶜ  (q=0.5)", fontsize=S.FS_AXIS)
    axa.set_ylabel("P(canonical action)", fontsize=S.FS_AXIS)
    axa.tick_params(labelsize=S.FS_TICK)
    axa.legend(fontsize=5.6, frameon=False, loc="lower right", handlelength=1.2)
    S.panel_title(axa, "a", "Quantal response", fontsize=S.FS_PANEL)

    # ---- (b) model selection log-loss ----
    axb = fig.add_subplot(gs[1])
    fams = MSC.MODELS_GRID
    xpos = np.arange(len(fams))
    w = 0.15
    for j, ag in enumerate(AGENTS):
        sub = cv[cv.agent == ag].set_index("model").reindex(fams)
        yerr = np.vstack([(sub["cv_logloss"] - sub["cv_logloss_lo"]).to_numpy(),
                          (sub["cv_logloss_hi"] - sub["cv_logloss"]).to_numpy()])
        axb.bar(xpos + (j - 2) * w, sub["cv_logloss"].to_numpy(), w, color=AGENT_COL[ag],
                label=AGENT_LBL[ag], yerr=np.abs(yerr), error_kw=dict(elinewidth=0.5, capsize=0))
    axb.axhline(0.693, color=S.INK, lw=0.8, ls="--")
    axb.text(6.4, 0.693, "chance", fontsize=S.FS_FOOT, va="bottom", ha="right", color=S.INK)
    axb.set_xticks(xpos); axb.set_xticklabels(fams, fontsize=S.FS_TICK, rotation=0)
    axb.set_ylabel("CV predictive log-loss\n(lower = better)", fontsize=S.FS_AXIS)
    axb.set_ylim(0.35, float(cv["cv_logloss_hi"].max()) * 1.04)  # fit all data; never clip Lk/NE
    axb.tick_params(labelsize=S.FS_TICK)
    axb.legend(fontsize=5.6, frameon=False, ncol=5, loc="upper center", columnspacing=0.8,
               handlelength=1.0, handletextpad=0.3, bbox_to_anchor=(0.5, 1.02))
    S.panel_title(axb, "b", "Model selection (leave-games-out)", fontsize=S.FS_PANEL)

    # ---- (c) fingerprint ----
    axc = fig.add_subplot(gs[2])
    for _, r in fp.iterrows():
        ag = r["agent"]
        mk = "D" if ag == "nagel_human" else "o"
        axc.scatter(r["mean_level"], r["lambda"], s=70, color=AGENT_COL[ag], marker=mk,
                    edgecolor="white", lw=0.8, zorder=4)
        axc.annotate(AGENT_LBL[ag], (r["mean_level"], r["lambda"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=S.FS_FOOT, color=AGENT_COL[ag])
    axc.annotate("Nash\n(λ→∞, deep)", (3.0, axc.get_ylim()[1] if False else 1.75),
                 fontsize=S.FS_FOOT, color=S.GREY, ha="center", va="top")
    axc.set_xlabel("revealed reasoning depth\n(mean level)", fontsize=S.FS_AXIS)
    axc.set_ylabel("precision λ (q=0.5)", fontsize=S.FS_AXIS)
    axc.set_xlim(0.95, 1.6); axc.set_ylim(0, 1.85)
    axc.tick_params(labelsize=S.FS_TICK)
    S.panel_title(axc, "c", "Bounded-rationality fingerprint", fontsize=S.FS_PANEL)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig3_behavioral_model.{ext}", dpi=600 if ext == "png" else 300,
                    bbox_inches="tight")
    plt.close(fig)
    print("[fig3] wrote fig3_behavioral_model.{png,pdf}")


def _human_row_from_cache():
    p = TABLES / "f3_human_lambda.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p).iloc[0].to_dict()
    return r


def _human_curve():
    """Per-game human P(canonical) vs delta1c (fast; no bootstrap)."""
    df = HL.individual_choices()
    return (df.groupby("game_code")
              .agg(p_canon=("y_canon", "mean"), delta1c=("delta1c", "first")).reset_index())


def _legacy_main(force: bool = False) -> int:
    gl = pd.read_parquet(DATA / "layerA_game_level.parquet")
    cells = pd.read_parquet(DATA / "layerA_cells_p1baseline.parquet")

    cached_human = None if force else _human_row_from_cache()
    if cached_human is not None:
        human_row = cached_human
        human_curve = _human_curve()
        print(f"[fig3] human λ (cached) = {human_row['lambda']:.3f}")
    else:
        human_row, human_curve = HL.fit_human_lambda()

    lam_cache = TABLES / "f3_lambda_legacy.csv"
    if lam_cache.exists() and not force:
        lam_tab = pd.read_csv(lam_cache)
        print("[fig3] λ table loaded from cache")
    else:
        lam_tab = build_lambda_table(cells, human_row)
    print("\n[fig3 a] precision λ table (hard/Generate primary):")
    print(lam_tab[lam_tab.readout != "soft_pref0"][["agent", "belief", "lambda", "lo", "hi"]].round(3).to_string(index=False))

    cv_cache = TABLES / "f3_model_selection_cv.csv"
    if not cv_cache.exists() or force:
        MSC.run()
    cv = pd.read_csv(cv_cache)
    fp = build_fingerprint(human_row, lam_tab, cells)
    print("\n[fig3 c] fingerprint (depth x precision):")
    print(fp[["agent", "mean_level", "lambda", "frac_L1", "frac_L2", "frac_L3"]].round(3).to_string(index=False))

    render(lam_tab, cv, fp, gl, human_curve)
    return 0


def main(force: bool = False) -> int:
    """Compatibility wrapper; delegates to the canonical paper Fig 3 script."""
    from analysis.layer_a.figscripts import fig3_qre_to_levelk

    print("[fig3] src.fig3_behavioral_model is retired; delegating to figscripts/fig3_qre_to_levelk.py")
    return fig3_qre_to_levelk.main()


if __name__ == "__main__":
    raise SystemExit(main())
