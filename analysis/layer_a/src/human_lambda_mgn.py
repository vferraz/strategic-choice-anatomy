#!/usr/bin/env python3
"""Layer A final — human precision lambda on MGN *individual* decisions (SPEC §honesty).

The human comparison is apples-to-apples with the model's GENERATED move: a binary
human choice vs the binary model GENERATED/committed move. So lambda is fit on 451
complete sessions from 450 unique participants, each covering 144 perspectives in
the Moore-Germano-Nagel (2026) individual decisions
(``datasets/nagel/normalized_data_20241612.csv``), NOT on aggregate frequencies; the
``nagel_frac_choose_act0_canonical`` column is only the collapsed summary.

Perspective matching (REQUIRED): each canonical game's P1 view maps to one MGN
perspective via ``nagel_p_id`` (-> column ``P{id}``). The raw choice is "A"/"B" in the
perspective's own row frame; the canonical orientation flips the row in 72/144
perspectives (verified), so we orient each subject's choice to the canonical act0 axis
by the master mapping (cross-checked against ``nagel_swap_sr``). Then:
    y_canon  = 1 if the subject played the CANONICAL action
    delta1c  = canonical-signed incentive gap at the uniform belief q=0.5
               (same belief basis as the model human-comparison fit)
and lambda is the slope of  Logit(y_canon ~ const + delta1c).

CIs: session-clustered robust SE AND bootstrap-by-game (cluster=game, the
project default, comparable to the model lambdas). Stochasticity-source asymmetry
(human = 451 sessions from 450 people; model = 4 corrected-root counterbalance
frames) is stated, not hidden.

Output: tables/f3_human_lambda.csv ; returns the per-game P(canonical) curve too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.layer_a.src import shared_data as SD  # noqa: E402
from collection.oneshot_common import delta1, canonical_sign  # noqa: E402
from analysis.layer_a.layer1_lambda_delta1 import fit_lambda, boot_cluster_lambda  # noqa: E402
from strategic_anatomy.config import human_refs_root, repo_root, results_root

ROOT = repo_root()


NORM = human_refs_root() / "raw" / "nagel" / "normalized_data_20241612.csv"
TABLES = results_root() / "layer_a"


def individual_choices() -> pd.DataFrame:
    """Long frame: one row per (session, canonical game) with oriented y_canon + delta1c."""
    m = SD.master()
    nd = pd.read_csv(NORM)
    # `Name`/`QT.Username` have one duplicate in this export; ResponseId is the
    # unique session identifier used for clustered uncertainty.
    if "QT.ResponseId" in nd.columns:
        subj_col = "QT.ResponseId"
    elif "Unnamed: 0" in nd.columns:
        subj_col = "Unnamed: 0"
    else:
        subj_col = "Name" if "Name" in nd.columns else nd.columns[0]

    rows = []
    flips = []
    for _, r in m.iterrows():
        g = r["game_code"]
        k = int(r["nagel_p_id"])
        col = f"P{k}"
        if col not in nd.columns:
            continue
        canon = int(r["canonical_action_p1"])
        vec = SD.parse_8vec(r["canonical_8vec"])
        d1c = canonical_sign(delta1(vec, 0.5), canon)         # q=0.5 belief basis

        ch = nd[[subj_col, col]].dropna(subset=[col])
        raw_A = (ch[col] == "A").astype(int).to_numpy()       # 1 if chose row "A" (perspective act0)
        # orient to canonical act0: pick the mapping matching the master aggregate
        targ = float(r["nagel_frac_choose_act0_canonical"])
        no_flip_err = abs(raw_A.mean() - targ)
        flip_err = abs((1 - raw_A).mean() - targ)
        flip = flip_err < no_flip_err
        flips.append({"game_code": g, "flip": bool(flip),
                      "swap_sr": bool(r.get("nagel_swap_sr", False))})
        act0_canon = (1 - raw_A) if flip else raw_A           # 1 if played canonical act0
        y_canon = act0_canon if canon == 0 else (1 - act0_canon)  # 1 if played canonical action
        for sid, y in zip(ch[subj_col].to_numpy(), y_canon):
            rows.append({"session_id": sid, "game_code": g, "y_canon": int(y), "delta1c": d1c})

    df = pd.DataFrame(rows)
    fl = pd.DataFrame(flips)
    # cross-check: orientation flip should equal the row-swap flag
    agree = float((fl["flip"] == fl["swap_sr"]).mean())
    df.attrs["flip_vs_swapsr_agreement"] = agree
    df.attrs["n_flipped"] = int(fl["flip"].sum())
    df.attrs["n_sessions"] = int(nd[subj_col].nunique())
    person_col = "Name" if "Name" in nd.columns else "QT.Username"
    df.attrs["n_unique_participants"] = int(nd[person_col].nunique())
    return df


def _clustered_lambda(df: pd.DataFrame):
    """Logit slope with session-clustered robust SE -> (lambda, lo, hi)."""
    import statsmodels.api as sm
    X = sm.add_constant(df["delta1c"].to_numpy(float))
    y = df["y_canon"].to_numpy(int)
    res = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                             cov_kwds={"groups": pd.factorize(df["session_id"])[0]})
    lam = float(res.params[1])
    ci = res.conf_int()
    return lam, float(ci[1][0]), float(ci[1][1])


def fit_human_lambda():
    df = individual_choices()
    n_sessions = int(df.attrs["n_sessions"])
    n_unique_participants = int(df.attrs["n_unique_participants"])
    n_games = df["game_code"].nunique()
    print(f"[human] {len(df)} individual decisions = {n_sessions} sessions "
          f"from {n_unique_participants} participants x {n_games} games")
    print(f"[human] orientation flips: {df.attrs['n_flipped']}/144 perspectives; "
          f"flip == nagel_swap_sr agreement = {df.attrs['flip_vs_swapsr_agreement']:.3f}")

    lam_c, lo_c, hi_c = _clustered_lambda(df)
    x = df["delta1c"].to_numpy(float); y = df["y_canon"].to_numpy(int)
    g = df["game_code"].to_numpy()
    lam_h, alpha = fit_lambda(x, y, "hard")
    lo_g, hi_g = boot_cluster_lambda(x, y, g, "hard", n_boot=SD.BOOT_N)

    # per-game P(canonical) curve (human overlay for Fig 3a)
    curve = (df.groupby("game_code")
               .agg(p_canon=("y_canon", "mean"), delta1c=("delta1c", "first"),
                    n_sessions=("session_id", "nunique")).reset_index())

    row = {"agent": "nagel_human", "readout": "binary_choice", "belief": "q05",
           "lambda": lam_h, "alpha": float(alpha),
           "lambda_boot_game_lo": lo_g, "lambda_boot_game_hi": hi_g,
           "lambda_session_clust_lo": lo_c, "lambda_session_clust_hi": hi_c,
           "n_obs": int(len(df)), "n_sessions": n_sessions,
           "n_unique_participants": n_unique_participants, "n_games": int(n_games)}
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(TABLES / "f3_human_lambda.csv", index=False)
    print(f"[human] lambda (q=0.5, binary individual choices) = {lam_h:.3f}")
    print(f"        bootstrap-by-game 95% CI [{lo_g:.3f}, {hi_g:.3f}]")
    print(f"        session-clustered 95% CI [{lo_c:.3f}, {hi_c:.3f}]")
    print(f"wrote {TABLES/'f3_human_lambda.csv'}")
    return row, curve


def main() -> int:
    fit_human_lambda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
