#!/usr/bin/env python3
"""Do prompt traits steer strategic play toward the payoffs the trait favors?

This replaces the `make_trait_direction_*` figures. It answers the question in
the order the analyst asked for it:

  (1) MAGNITUDE   - how much does a unilateral cue move P1's action frequency,
                    in absolute terms (the "~0.2" question)?  Benchmarked
                    against a content-free prompt of matched length and against
                    the uncued opponent.
  (2) DIRECTION   - of that movement, how much is aimed at the action the trait
                    actually prefers (placebo-corrected signed shift +
                    a directionality index)?
  (3) DOSE-RESPONSE - does the steering scale with how strongly the game's
                    payoffs differentiate the two actions on the trait's own
                    dimension (the "right reason" test, robust to ceilings)?

Design facts that make this clean (verified in-repo):
  * Unilateral cue (typeA): P1 gets the trait system-prefix, P2 stays baseline.
  * Conditions are PAIRED BY SEED: seed s renders the identical prompt
    (same A/B counterbalance) across baseline and every cue, so the only thing
    that changes is the trait prefix -> a true within-rendering manipulation.
  * Round 1 decoding is deterministic: the uncued P2's round-1 move is
    byte-identical between baseline and cue, so the P2 round-1 shift is an exact
    null (an apparatus check, printed below).

Trait operationalizations are read literally from src/traits.json and turned
into (a) a preferred action a*(game) and (b) a SIGNED payoff dose d(game),
where d>0 means canonical act0 is the trait-preferred action and |d| is how
strongly the payoffs favor it.  Derived preferred actions are asserted to match
the repo's precomputed predictors (game_features.csv) before anything is
aggregated, satisfying CLAUDE.md hard-constraint #2 (align per game to the
trait-predicted action; never pool raw move==0 across games).

Outputs (analysis/_shared/datasets/, analysis/block_a/figures/):
  trait_steering_proper_per_game.csv   - one row per (model, round, trait, game)
  trait_steering_proper_summary.csv    - one row per (model, round, trait)
  fig_trait_steering_magnitude_direction.{png,pdf}
  fig_trait_steering_dose_response.{png,pdf}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from strategic_anatomy.config import game_features_csv, human_refs_root, repo_root

ROOT = repo_root()
# NOTE (phase-2 port): MASTER and RAW_ROOT name the superseded DESIGN_V2 substrate,
# which is excluded from the release (plan §1/§5). They are read only by functions
# outside the released trait-target path; kept verbatim and flagged rather than
# repointed, since no released root corresponds to them.
MASTER = ROOT / "analysis/notebooks/outputs/01_gameplay/master_behavior_long.parquet"
RAW_ROOT = ROOT / "output/design_v2_main"
UNIFIED = human_refs_root() / "unified_pairs.parquet"
FEATURES = game_features_csv()
DATA_DIR = human_refs_root()
FIG_DIR = ROOT / "analysis" / "layer_a" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260521)
N_BOOT = 4000

MODELS = ["qwen", "qwen_instruct", "llama31_instruct", "gptoss"]
MODEL_LABEL = {
    "qwen": "Qwen",
    "qwen_instruct": "Qwen-Instruct",
    "llama31_instruct": "Llama-3.1-Instruct",
    "gptoss": "GPT-OSS",
}
# Traits with a payoff-grounded target.  level0_naive is excluded (its prompt is
# an own-max "first glance" heuristic, not a canonical rule with a clean target).
PLACEBO = "length_match_null"
TRAITS = ["risk", "loss", "inequity", "selfish", "maximin", "level1", "level2"]
TRAIT_LABEL = {
    "risk": "Risk-averse",
    "loss": "Loss-averse",
    "inequity": "Inequity-averse",
    "selfish": "Selfish-max",
    "maximin": "Maximin (rule)",
    "level1": "Level-1 (rule)",
    "level2": "Level-2 (rule)",
}
TRAIT_FAMILY = {
    "risk": "trait", "loss": "trait", "inequity": "trait", "selfish": "trait",
    "maximin": "rule", "level1": "rule", "level2": "rule",
}
CUE_TO_CELL = {
    "risk": "risk_aversion_typeA",
    "loss": "loss_aversion_typeA",
    "inequity": "inequity_aversion_typeA",
    "selfish": "selfish_maximizer_typeA",
    "maximin": "maximin_typeA",
    "level1": "level1_typeA",
    "level2": "level2_typeA",
}
PLACEBO_CELL = "length_match_null_typeA"
RUN_RE = re.compile(r"(?P<game>.+)_s(?P<seed>\d+)_(?P<condition>.+)$")


# --------------------------------------------------------------------------- #
# 1. Behavior: unified long table (move1/move2 per model/game/seed/round/cell)
# --------------------------------------------------------------------------- #
def _cells_needed() -> set[str]:
    return {"baseline", PLACEBO_CELL, *CUE_TO_CELL.values()}


def load_behavior() -> pd.DataFrame:
    """Round 1 + 10 realized moves for all four models, only the cells we need.

    Models present in the merged master parquet are read from it; GPT-OSS
    (absent from the parquet) is read from raw design_v2_main results.
    """
    cells = _cells_needed()
    frames: list[pd.DataFrame] = []

    cols = ["model", "game_code", "seed", "cell_name", "round", "move1", "move2"]
    master = pd.read_parquet(MASTER, columns=cols)
    master = master[master.cell_name.isin(cells) & master["round"].isin([1, 10])]
    frames.append(master)

    parquet_models = set(master.model.unique())
    if "gptoss" not in parquet_models:
        rows: list[pd.DataFrame] = []
        for path in sorted(RAW_ROOT.glob("gptoss/*/results.parquet")):
            mobj = RUN_RE.match(path.parent.name)
            if not mobj or mobj.group("condition") not in cells:
                continue
            d = pq.read_table(path, columns=["game_code", "round", "move1", "move2"]).to_pandas()
            d = d[d["round"].isin([1, 10])]
            d["model"] = "gptoss"
            d["seed"] = int(mobj.group("seed"))
            d["cell_name"] = mobj.group("condition")
            rows.append(d)
        if rows:
            frames.append(pd.concat(rows, ignore_index=True)[cols])

    beh = pd.concat(frames, ignore_index=True)
    beh = beh[beh.move1.isin([0, 1]) & beh.move2.isin([0, 1])].copy()  # drop parse fails
    beh["move1"] = beh["move1"].astype(int)
    beh["move2"] = beh["move2"].astype(int)
    return beh


# --------------------------------------------------------------------------- #
# 2. Targets + signed payoff doses from the literal prompt semantics
# --------------------------------------------------------------------------- #
def _br(vals: list[float], *, high: bool = True) -> float:
    if abs(float(vals[0]) - float(vals[1])) < 1e-12:
        return np.nan
    return float(np.argmax(vals) if high else np.argmin(vals))


def trait_targets() -> pd.DataFrame:
    """Per game: preferred action a* and signed dose d (d>0 -> act0 preferred)."""
    u = pd.read_parquet(UNIFIED, columns=["game_code", "matrix_8vec", "agent_kind", "condition"])
    g = (
        u[(u.agent_kind == "llm") & (u.condition == "baseline")]
        .drop_duplicates("game_code")[["game_code", "matrix_8vec"]]
        .sort_values("game_code")
        .reset_index(drop=True)
    )
    out = []
    for _, r in g.iterrows():
        v = np.array(json.loads(r.matrix_8vec), dtype=float)
        p1, p2 = v[:4].reshape(2, 2), v[4:].reshape(2, 2)
        worst = [float(p1[i, :].min()) for i in range(2)]               # risk/loss/maximin
        expo = [float(p1[i, :].mean()) for i in range(2)]               # selfish/level1 (BR uniform)
        gap = [float(np.mean(np.abs(p1[i, :] - p2[i, :]))) for i in range(2)]  # inequity
        l1_p2 = _br([0.5 * p2[0, j] + 0.5 * p2[1, j] for j in range(2)])
        l2_d = np.nan if pd.isna(l1_p2) else (p1[0, int(l1_p2)] - p1[1, int(l1_p2)])
        a_worst, a_expo = _br(worst), _br(expo)
        a_gap = _br(gap, high=False)
        a_l2 = np.nan if pd.isna(l1_p2) else _br([p1[i, int(l1_p2)] for i in range(2)])
        rec = {
            "game_code": r.game_code,
            "d_risk": worst[0] - worst[1], "a_risk": a_worst,
            "d_loss": worst[0] - worst[1], "a_loss": a_worst,
            "d_maximin": worst[0] - worst[1], "a_maximin": a_worst,
            "d_selfish": expo[0] - expo[1], "a_selfish": a_expo,
            "d_level1": expo[0] - expo[1], "a_level1": a_expo,
            "d_inequity": gap[1] - gap[0], "a_inequity": a_gap,
            "d_level2": l2_d, "a_level2": a_l2,
        }
        out.append(rec)
    targets = pd.DataFrame(out)
    _validate_targets(targets)
    return targets


def _validate_targets(t: pd.DataFrame) -> None:
    """Assert derived preferred actions match the repo's precomputed predictors."""
    f = pd.read_csv(FEATURES)[["game_code", "p1_maximin_action", "l1_action_p1", "l2_action_p1"]]
    m = t.merge(f, on="game_code")
    checks = [("a_risk", "p1_maximin_action"), ("a_selfish", "l1_action_p1"), ("a_level2", "l2_action_p1")]
    for derived, ref in checks:
        ok = (~m[derived].isna()) & (~m[ref].isna())
        n_ok = int((m.loc[ok, derived] == m.loc[ok, ref]).sum())
        if n_ok != int(ok.sum()):
            raise AssertionError(f"target mismatch {derived} vs {ref}: {n_ok}/{int(ok.sum())}")


# --------------------------------------------------------------------------- #
# 3. Per-(model, round, trait, game) effect table
# --------------------------------------------------------------------------- #
def per_game_freq(beh: pd.DataFrame, model: str, round_id: int, cell: str) -> pd.Series:
    sub = beh[(beh.model == model) & (beh["round"] == round_id) & (beh.cell_name == cell)]
    return sub.groupby("game_code").move1.apply(lambda s: float((s == 0).mean()))


def per_game_seed_move(beh: pd.DataFrame, model: str, round_id: int, cell: str) -> pd.DataFrame:
    sub = beh[(beh.model == model) & (beh["round"] == round_id) & (beh.cell_name == cell)]
    return sub.groupby(["game_code", "seed"]).move1.first()


def build_per_game(beh: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODELS:
        if model not in set(beh.model.unique()):
            continue
        for round_id in (1, 10):
            base0 = per_game_freq(beh, model, round_id, "baseline")
            null0 = per_game_freq(beh, model, round_id, PLACEBO_CELL)
            base_seed = per_game_seed_move(beh, model, round_id, "baseline")
            null_seed = per_game_seed_move(beh, model, round_id, PLACEBO_CELL)
            for trait in TRAITS:
                cell = CUE_TO_CELL[trait]
                cue0 = per_game_freq(beh, model, round_id, cell)
                cue_seed = per_game_seed_move(beh, model, round_id, cell)
                a = targets.set_index("game_code")[f"a_{trait}"]
                d = targets.set_index("game_code")[f"d_{trait}"]
                for game in base0.index:
                    if game not in cue0.index or game not in null0.index:
                        continue
                    astar = a.get(game, np.nan)
                    if pd.isna(astar):       # trait does not differentiate this game
                        continue
                    astar = int(astar)
                    b, c, n = float(base0[game]), float(cue0[game]), float(null0[game])
                    # signed canonical-0 shifts
                    shift0 = c - b
                    nullshift0 = n - b
                    # toward preferred action
                    sgn = 1.0 if astar == 0 else -1.0
                    # per-rendering flip stats (paired seeds)
                    flips = aligned = n_pairs = 0
                    try:
                        bs, cs = base_seed.loc[game], cue_seed.loc[game]
                        for s in bs.index.intersection(cs.index):
                            n_pairs += 1
                            if int(bs[s]) != int(cs[s]):
                                flips += 1
                                if int(cs[s]) == astar:
                                    aligned += 1
                    except KeyError:
                        pass
                    rows.append({
                        "model": model, "round": round_id, "trait": trait,
                        "family": TRAIT_FAMILY[trait], "game": game,
                        "a_star": astar, "dose": float(d[game]),
                        "base0": b, "cue0": c, "null0": n,
                        "shift0": shift0, "nullshift0": nullshift0,
                        "abs_shift": abs(shift0), "abs_nullshift": abs(nullshift0),
                        "shift_pref": sgn * shift0,            # signed toward a*
                        "nullshift_pref": sgn * nullshift0,
                        "net_pref": sgn * (shift0 - nullshift0),  # placebo-corrected
                        "base_pref": b if astar == 0 else 1 - b,  # baseline P(a*) -> ceiling
                        "n_pairs": n_pairs, "flips": flips, "flips_aligned": aligned,
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Summaries with cluster-bootstrap (resample games)
# --------------------------------------------------------------------------- #
def _ci(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    draws = vals[RNG.integers(0, len(vals), size=(N_BOOT, len(vals)))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _slope_ci(d: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """OLS slope of y~d with cluster bootstrap; returns slope, lo, hi, pearson r."""
    m = (~np.isnan(d)) & (~np.isnan(y))
    d, y = d[m], y[m]
    if len(d) < 5 or np.std(d) < 1e-9:
        return np.nan, np.nan, np.nan, np.nan
    slope = float(np.polyfit(d, y, 1)[0])
    r = float(np.corrcoef(d, y)[0, 1])
    boots = []
    n = len(d)
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        dd, yy = d[idx], y[idx]
        if np.std(dd) < 1e-9:
            continue
        boots.append(np.polyfit(dd, yy, 1)[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return slope, float(lo), float(hi), r


def summarize(pg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, round_id, trait), grp in pg.groupby(["model", "round", "trait"]):
        mag, mag_lo, mag_hi = _ci(grp.abs_shift.to_numpy())
        magn, _, _ = _ci(grp.abs_nullshift.to_numpy())
        s, s_lo, s_hi = _ci(grp.shift_pref.to_numpy())
        net, net_lo, net_hi = _ci(grp.net_pref.to_numpy())
        slope, sl_lo, sl_hi, r = _slope_ci(grp.dose.to_numpy(), grp.shift0.to_numpy())
        slope_n, _, _, _ = _slope_ci(grp.dose.to_numpy(), grp.nullshift0.to_numpy())
        flips = int(grp.flips.sum()); pairs = int(grp.n_pairs.sum()); al = int(grp.flips_aligned.sum())
        # Ceiling robustness: restrict to "movable" games (baseline P(a*) in [0.2, 0.8]),
        # where the apparatus actually has headroom to register a positive shift.
        mv = grp[(grp.base_pref >= 0.2) & (grp.base_pref <= 0.8)]
        net_mv, net_mv_lo, net_mv_hi = _ci(mv.net_pref.to_numpy()) if len(mv) else (np.nan, np.nan, np.nan)
        slope_mv, slmv_lo, slmv_hi, _ = _slope_ci(mv.dose.to_numpy(), mv.shift0.to_numpy()) if len(mv) >= 5 else (np.nan, np.nan, np.nan, np.nan)
        rows.append({
            "model": model, "round": int(round_id), "trait": trait,
            "family": TRAIT_FAMILY[trait], "n_games": int(grp.game.nunique()),
            # magnitude
            "magnitude": mag, "magnitude_lo": mag_lo, "magnitude_hi": mag_hi,
            "magnitude_null": magn,
            "flip_rate": flips / pairs if pairs else np.nan,
            "flip_rate_aligned": al / flips if flips else np.nan,
            # direction
            "dir_shift": s, "dir_lo": s_lo, "dir_hi": s_hi,
            "dir_net": net, "dir_net_lo": net_lo, "dir_net_hi": net_hi,
            "directionality_index": (s / mag) if mag and not np.isnan(mag) and mag > 1e-9 else np.nan,
            "base_pref_mean": float(grp.base_pref.mean()),
            # dose response
            "dose_slope": slope, "dose_slope_lo": sl_lo, "dose_slope_hi": sl_hi,
            "dose_r": r, "dose_slope_null": slope_n,
            # ceiling-robust (movable games only)
            "n_movable": int(mv.game.nunique()),
            "dir_net_movable": net_mv, "dir_net_movable_lo": net_mv_lo, "dir_net_movable_hi": net_mv_hi,
            "dose_slope_movable": slope_mv, "dose_slope_movable_lo": slmv_lo, "dose_slope_movable_hi": slmv_hi,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Apparatus check: uncued-P2 round-1 null
# --------------------------------------------------------------------------- #
def apparatus_check(beh: pd.DataFrame) -> str:
    lines = ["apparatus check - uncued P2 round-1 shift (should be ~0):"]
    for model in [m for m in MODELS if m in set(beh.model.unique())]:
        diffs = []
        for cell in CUE_TO_CELL.values():
            b = beh[(beh.model == model) & (beh["round"] == 1) & (beh.cell_name == "baseline")]
            c = beh[(beh.model == model) & (beh["round"] == 1) & (beh.cell_name == cell)]
            bp = b.groupby("game_code").move2.apply(lambda s: (s == 0).mean())
            cp = c.groupby("game_code").move2.apply(lambda s: (s == 0).mean())
            j = pd.concat([bp, cp], axis=1).dropna()
            diffs.append((j.iloc[:, 1] - j.iloc[:, 0]).abs().mean())
        lines.append(f"  {MODEL_LABEL[model]:20s} mean|dP2| across cues = {np.mean(diffs):.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. Figures
# --------------------------------------------------------------------------- #
GREEN, RED, GREY, NULLC = "#0b7a45", "#c83b31", "#5f5f5f", "#9a9a9a"


def fig_magnitude_direction(summary: pd.DataFrame) -> None:
    d1 = summary[summary["round"] == 1]
    models = [m for m in MODELS if m in set(d1.model.unique())]
    # Layout: dispositional TRAITS on top, explicit RULES below, separated by a gap.
    layout = ["risk", "loss", "inequity", "selfish", "_gap", "maximin", "level1", "level2"]
    order = [t for t in layout if t != "_gap"]
    ypos: dict[str, float] = {}
    y = len(layout) - 1
    for key in layout:
        if key != "_gap":
            ypos[key] = float(y)
        y -= 1
    xmax = max(0.34, float(d1.magnitude.max()) * 1.12)

    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 7.6), squeeze=False)
    for col, model in enumerate(models):
        dm = d1[d1.model == model].set_index("trait")
        # --- Row 0: MAGNITUDE (absolute frequency shift), trait vs null placebo
        ax = axes[0][col]
        for t in order:
            if t not in dm.index:
                continue
            r = dm.loc[t]; yy = ypos[t]
            ax.plot([0, r.magnitude], [yy, yy], color=GREY, lw=6.0, solid_capstyle="butt", alpha=0.85, zorder=3)
            ax.plot([r.magnitude_lo, r.magnitude_hi], [yy, yy], color="#111", lw=1.0, zorder=4)
            ax.scatter(r.magnitude_null, yy, marker="|", s=150, color=RED, zorder=5, linewidths=1.8)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(layout) - 0.4)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([TRAIT_LABEL[t] for t in order] if col == 0 else [], fontsize=8)
        ax.set_title(MODEL_LABEL[model], fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7.5)
        if col == 0:
            ax.set_ylabel("MAGNITUDE\nmean |Δ freq of act0|", fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # --- Row 1: DIRECTION (placebo-corrected signed shift toward a*), + ceiling
        ax = axes[1][col]
        ax.axvspan(0, 0.55, color="#e6f4e6", alpha=0.5, zorder=0)
        ax.axvspan(-0.55, 0, color="#f8e9e5", alpha=0.5, zorder=0)
        ax.axvline(0, color="#222", lw=0.9, zorder=2)
        for t in order:
            if t not in dm.index:
                continue
            r = dm.loc[t]; yy = ypos[t]
            col_pt = GREEN if r.dir_net >= 0 else RED
            ceil = r.base_pref_mean >= 0.8  # near-ceiling: interpret with caution
            ax.plot([r.dir_net_lo, r.dir_net_hi], [yy, yy], color=col_pt, lw=2.0, solid_capstyle="round",
                    alpha=0.45 if ceil else 1.0, zorder=4)
            ax.scatter(r.dir_net, yy, s=46, color=col_pt, edgecolor="white", linewidth=0.7,
                       alpha=0.45 if ceil else 1.0, zorder=5)
            ax.text(0.475, yy, f"{r.base_pref_mean:.2f}", transform=ax.get_yaxis_transform(),
                    fontsize=6.0, color=("#b06a00" if ceil else "#777"), va="center", ha="left")
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.6, len(layout) - 0.4)
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([TRAIT_LABEL[t] for t in order] if col == 0 else [], fontsize=8)
        ax.set_xticks([-0.3, 0, 0.3])
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel("Δ P(trait-preferred action) vs placebo", fontsize=8)
        if col == 0:
            ax.set_ylabel("DIRECTION\n(placebo-corrected)", fontsize=8.5)
        ax.text(0.475, len(layout) - 0.45, "base\nP(a*)", transform=ax.get_yaxis_transform(),
                fontsize=5.6, color="#777", va="center", ha="left", linespacing=0.9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Group dividers / labels on the left column
    for row in (0, 1):
        ax = axes[row][0]
        gy = ypos["selfish"] - 0.5
        ax.axhline(gy, color="#ddd", lw=0.8, ls="--")
    fig.text(0.012, 0.70, "TRAITS", rotation=90, fontsize=8, color="#888", va="center", fontweight="bold")
    fig.text(0.012, 0.34, "RULES", rotation=90, fontsize=8, color="#888", va="center", fontweight="bold")

    fig.suptitle("Do prompt traits steer play toward the payoffs they favor?",
                 fontsize=13.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "Unilateral cue on P1, round 1.  TOP: how much the cue moves choice frequency "
             "(grey bar) vs a content-free prompt of matched length (red tick).  BOTTOM: net shift toward the "
             "trait-preferred action after subtracting that placebo; faded = baseline already ≥0.80 (ceiling).",
             ha="center", fontsize=7.8, color="#444")
    handles = [
        Line2D([0], [0], color=GREY, lw=5, label="trait magnitude |Δ freq|"),
        Line2D([0], [0], marker="|", color=RED, lw=0, ms=11, mew=1.8, label="placebo (length-null) magnitude"),
        Line2D([0], [0], marker="o", color=GREEN, lw=2, ms=6, label="net steering toward trait action (95% CI)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_trait_steering_magnitude_direction.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_dose_response(pg: pd.DataFrame, summary: pd.DataFrame) -> None:
    d1pg = pg[pg["round"] == 1]
    d1s = summary[summary["round"] == 1].set_index(["model", "trait"])
    models = [m for m in MODELS if m in set(d1pg.model.unique())]
    show = ["risk", "selfish", "inequity"]  # one per distinct trait dimension
    fig, axes = plt.subplots(len(show), len(models), figsize=(3.0 * len(models), 2.7 * len(show)), squeeze=False)
    for ri, trait in enumerate(show):
        for ci, model in enumerate(models):
            ax = axes[ri][ci]
            g = d1pg[(d1pg.model == model) & (d1pg.trait == trait)]
            ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
            ax.scatter(g.dose, g.shift0, s=16, alpha=0.55, color="#1f6f8b", edgecolor="none")
            try:
                r = d1s.loc[(model, trait)]
                xs = np.array([g.dose.min(), g.dose.max()])
                b = np.polyfit(g.dose, g.shift0, 1)
                ax.plot(xs, b[0] * xs + b[1], color=RED, lw=1.6)
                ax.text(0.04, 0.93, f"β={r.dose_slope:+.3f}\nr={r.dose_r:+.2f}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.5))
            except Exception:
                pass
            if ri == 0:
                ax.set_title(MODEL_LABEL[model], fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{TRAIT_LABEL[trait]}\nΔ P(act0)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylim(-1.05, 1.05)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Dose-response: does cue-induced shift scale with the payoff incentive?",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "x = signed payoff dose (>0 means act0 is the trait-preferred action; |x| = strength).  "
             "y = cue−baseline shift in P(act0).  Positive slope = trait-consistent steering.",
             ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_trait_steering_dose_response.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    beh = load_behavior()
    targets = trait_targets()
    pg = build_per_game(beh, targets)
    summary = summarize(pg)

    pg.to_csv(DATA_DIR / "trait_steering_proper_per_game.csv", index=False)
    summary.to_csv(DATA_DIR / "trait_steering_proper_summary.csv", index=False)

    fig_magnitude_direction(summary)
    fig_dose_response(pg, summary)

    print(apparatus_check(beh))
    print("\nROUND-1 SUMMARY (magnitude -> direction -> dose):")
    cols = ["model", "trait", "n_games", "magnitude", "magnitude_null",
            "dir_net", "dir_net_lo", "dir_net_hi", "directionality_index",
            "base_pref_mean", "dose_slope", "dose_slope_lo", "dose_slope_hi", "dose_slope_null"]
    show = summary[summary["round"] == 1][cols].copy()
    show = show.sort_values(["model", "trait"])
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(show.round(3).to_string(index=False))
    print("\nwrote analysis/_shared/datasets/trait_steering_proper_{per_game,summary}.csv")
    print("wrote analysis/block_a/figures/fig_trait_steering_{magnitude_direction,dose_response}.{png,pdf}")


if __name__ == "__main__":
    main()
