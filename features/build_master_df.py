#!/usr/bin/env python3
"""Human game master dataframe — Layer A Final prerequisite (SPEC §"Prerequisite").

Single source of truth that the whole of Layer A reads from, with the per-source
payoff representation made EXPLICIT so the scale provenance is enforced by data,
not by memory:

  * LLMs (this study)         -> canonical rank 8-vector {1,2,3,4}, payoff_multiplier=1.
  * Moore-Germano-Nagel (MGN) -> rank 8-vector {1,2,3,4}      (SAME scale as the LLMs).
  * Zhu-Griffiths             -> cardinal payoffs, rank-clean inclusion flag only.

Outputs (data/games/taxonomy/):
  human_game_master_per_canonical.csv  (144 rows)
  human_game_master_per_game.csv       ( 78 rows)

Build-time asserts (abort on failure):
  1. every MGN df_ros.game_vector is a permutation of {1,2,3,4} per 4-block.
  2. canonical_8vec (the LLM matrix) is a permutation of {1,2,3,4} per 4-block.
  3. the corrected one-shot LLM substrate has payoff_multiplier == 1 (read from config.json).

Citations (code keeps the nicknames; the manuscript must cite the sources):
  nagel / MGN  -> Moore, Germano & Nagel (2026), UPF Economics WP 1942
                  (Barcelona School of Economics WP 1571).
  griffiths    -> Zhu, Peterson, Enke & Griffiths (2025), Nature Human Behaviour
                  9, 2114-2120. doi:10.1038/s41562-025-02230-5.

CPU only; deterministic; reads only metadata + one-shot config.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


from analysis.probe_common import game_meta, parse_8vec  # noqa: E402
from analysis.layer_a.src import corrected_substrate as CS  # noqa: E402
from strategic_anatomy.config import game_features_csv, human_refs_root, repo_root, taxonomy_dir

ROOT = repo_root()


TAX_DIR = taxonomy_dir()
ROS = human_refs_root() / "raw" / "nagel" / "df_ros.csv"
EQUIV_GAME = TAX_DIR / "equivalence_per_game.csv"
FEATURES = game_features_csv()
SUBSTRATE = CS.CORRECTED_ROOT

OUT_CANON = TAX_DIR / "human_game_master_per_canonical.csv"
OUT_GAME = TAX_DIR / "human_game_master_per_game.csv"

MODELS = ("qwen", "qwen_instruct", "llama31_instruct", "gptoss")


# --------------------------------------------------------------------------- #
# build-time asserts (scale provenance is enforced by data, never by memory)
# --------------------------------------------------------------------------- #
def _is_perm_1234(vec8) -> bool:
    v = [int(round(float(x))) for x in vec8]
    return len(v) == 8 and sorted(v[:4]) == [1, 2, 3, 4] and sorted(v[4:]) == [1, 2, 3, 4]


def _norm_vec_str(s) -> str:
    """'[4,3,2,1,4,3,2,1]' or '4,3,2,1,...' -> '4,3,2,1,4,3,2,1'."""
    return ",".join(str(int(round(float(x)))) for x in str(s).strip("[]").split(","))


def assert_mgn_ranks(ros: pd.DataFrame) -> None:
    bad = ros[~ros["game_vector"].apply(lambda s: _is_perm_1234(_norm_vec_str(s).split(",")))]
    if len(bad):
        raise AssertionError(
            f"MGN scale assert FAILED: {len(bad)} df_ros.game_vector rows are not "
            f"permutations of {{1,2,3,4}} (p={bad['p'].tolist()[:10]}). "
            "Do NOT reintroduce the ×2/{2,4,6,8} rescaling note — MGN is ranks {1,2,3,4}."
        )
    print(f"  [assert] MGN ranks: all {len(ros)} df_ros.game_vector are perms of {{1,2,3,4}}  OK")


def assert_canonical_ranks(m: pd.DataFrame) -> None:
    bad = m[~m["canonical_8vec"].apply(lambda s: _is_perm_1234(parse_8vec(s)))]
    if len(bad):
        raise AssertionError(
            f"LLM canonical_8vec assert FAILED for {len(bad)} games: {bad['game_code'].tolist()[:10]}"
        )
    print(f"  [assert] LLM ranks: all {len(m)} canonical_8vec are perms of {{1,2,3,4}}  OK")


def assert_llm_pm1() -> None:
    cfg = CS.assert_complete_pm1()
    print(f"  [assert] LLM payoff_multiplier == 1 across all {len(cfg)} corrected config.json  OK")


# --------------------------------------------------------------------------- #
# per-canonical master (144 rows)
# --------------------------------------------------------------------------- #
def build_per_canonical() -> pd.DataFrame:
    m = game_meta().copy()  # equiv (+ canonical_8vec, nagel_*, clean_grif_*) merged with features
    assert_canonical_ranks(m)

    ros = pd.read_csv(ROS)
    assert_mgn_ranks(ros)
    ros_gv = ros.set_index("p")["game_vector"].apply(_norm_vec_str)
    ros_pt = ros.set_index("p")["p_t"]

    # explicit per-source payoff representation
    m["llm_rank_8vec"] = m["canonical_8vec"]                       # the matrix the LLMs saw
    m["mgn_rank_8vec"] = m["nagel_p_id"].map(ros_gv)               # the matrix MGN subjects saw
    m["mgn_p_paired"] = m["nagel_p_id"].map(ros_pt)
    m["payoff_scale_llm"] = "rank_1to4_pm1"
    m["payoff_scale_mgn"] = "rank_1to4"
    m["payoff_scale_griffiths"] = "cardinal"

    # Nagel axis flip: nagel_frac_choose_act0_canonical is P(act0), NOT P(canonical).
    c1 = m["canonical_action_p1"].astype(int)
    frac = m["nagel_frac_choose_act0_canonical"].astype(float)
    m["nagel_pcanon"] = np.where(c1 == 0, frac, 1.0 - frac)

    # Griffiths: cardinal, separate axis -> keep only the rank-clean inclusion flag
    m["grif_rank_clean"] = m["clean_grif_n_games"].fillna(0).astype(int) > 0

    if m["mgn_rank_8vec"].isna().any():
        raise AssertionError("mgn_rank_8vec has NaNs — nagel_p_id -> df_ros.p join is incomplete")

    # sanity: the flip must move the mean off the raw 0.5-ish act0 rate toward ~0.75
    print(f"  [sanity] mean nagel_frac_act0={frac.mean():.3f}  ->  mean nagel_pcanon={m['nagel_pcanon'].mean():.3f}"
          f"   (canon==1 share {float((c1 == 1).mean()):.3f})")
    return m


# --------------------------------------------------------------------------- #
# per-game master (78 rows, player-swap collapsed)
# --------------------------------------------------------------------------- #
def build_per_game(canon: pd.DataFrame) -> pd.DataFrame:
    eg = pd.read_csv(EQUIV_GAME)
    feat = pd.read_csv(FEATURES)[["game_code", "canonical_action_p1", "complexity_score",
                                  "num_pure_ne", "iesds_depth", "dominance_profile"]]
    ca = feat.set_index("game_code")["canonical_action_p1"]

    eg = eg.merge(feat.add_suffix("_x"), left_on="bruns_name_x", right_on="game_code_x", how="left")

    # canonical-aligned act0 rates are already in nagel_p1/nagel_p2 (per perspective);
    # flip each side onto its own canonical action axis.
    c1x = eg["bruns_name_x"].map(ca).astype("Int64")
    c1y = eg["bruns_name_y"].map(ca).astype("Int64")
    eg["nagel_pcanon_x"] = np.where(c1x == 0, eg["nagel_p1"], 1.0 - eg["nagel_p1"])
    eg["nagel_pcanon_y"] = np.where(c1y == 0, eg["nagel_p2"], 1.0 - eg["nagel_p2"])
    eg["grif_rank_clean"] = eg["clean_grif_n"].fillna(0).astype(int) > 0
    eg["payoff_scale_llm"] = "rank_1to4_pm1"
    eg["payoff_scale_mgn"] = "rank_1to4"
    eg["payoff_scale_griffiths"] = "cardinal"
    return eg


def main() -> int:
    TAX_DIR.mkdir(parents=True, exist_ok=True)
    print("building human game master ...")
    assert_llm_pm1()
    canon = build_per_canonical()
    game = build_per_game(canon)

    canon.to_csv(OUT_CANON, index=False)
    game.to_csv(OUT_GAME, index=False)
    print(f"\nwrote {OUT_CANON}  ({len(canon)} rows, {canon.shape[1]} cols)")
    print(f"wrote {OUT_GAME}  ({len(game)} rows, {game.shape[1]} cols)")

    # quick provenance summary
    print("\n=== payoff-scale provenance (per_canonical) ===")
    print(f"  LLM  : {canon['payoff_scale_llm'].iloc[0]}  (e.g. {canon['llm_rank_8vec'].iloc[0]})")
    print(f"  MGN  : {canon['payoff_scale_mgn'].iloc[0]}  (e.g. {canon['mgn_rank_8vec'].iloc[0]})")
    print(f"  Grif : {canon['payoff_scale_griffiths'].iloc[0]}  "
          f"({int(canon['grif_rank_clean'].sum())}/144 canonicals rank-clean)")
    assert len(canon) == 144, f"expected 144 canonical rows, got {len(canon)}"
    assert len(game) == 78, f"expected 78 per-game rows, got {len(game)}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
