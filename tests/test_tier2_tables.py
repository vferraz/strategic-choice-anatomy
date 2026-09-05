"""Tier-2 checks: regenerate from the deposit and compare against the committed tables.

Everything here carries ``@pytest.mark.tier2`` and is excluded from the default suite
(``make test`` runs ``-m "not tier2 and not gpu"``). ``make verify`` runs exactly these.

They encode the regenerate-and-compare checks that were first done by hand, so they re-run
mechanically instead of being re-derived. Comparison rule, from
REPRODUCING.md: exact sha256 is expected; where float formatting differs, parsed values must
agree at ``rtol = 1e-9``. A genuine numeric difference is a bug -- a port error or an
environment sensitivity -- not something to widen the tolerance for.

Nothing here writes into ``data/results``: the geometry checks compute in memory, and the
builder check runs with ``SCA_RESULTS_ROOT`` pointed at a tmp directory.

Requires ``SCA_DATA_ROOT`` (or a populated ``data_heavy/``); every test skips without it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategic_anatomy.config import gptoss_recap_root, layer_b_cache_root, repo_root

pytestmark = pytest.mark.tier2

RTOL = 1e-9
COMMITTED = repo_root() / "data" / "results"
FUSION = COMMITTED / "layer_b" / "fusion"
RECAP_CACHE = layer_b_cache_root() / "recap_baseline"
BASELINE_CACHE = layer_b_cache_root() / "baseline"


def _need(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not present at {path}; set SCA_DATA_ROOT or run "
                    f"`python scripts/download_data.py`")


@pytest.fixture(scope="session")
def recap_cache() -> Path:
    """The uniform-site recap cache, built once if the deposit is present but it is not."""
    _need(gptoss_recap_root(), "gptoss_recap deposit component")
    if not (RECAP_CACHE / "meta_gptoss.parquet").exists():
        subprocess.run(
            [sys.executable, "analysis/layer_b/fusion_associative/recap_cache.py"],
            cwd=repo_root(), check=True,
        )
    return RECAP_CACHE


def assert_close(regen: pd.DataFrame, committed: pd.DataFrame, keys, numeric, boolean=()):
    """Row-align on ``keys`` and compare every named column at RTOL."""
    a = committed.sort_values(list(keys)).reset_index(drop=True)
    b = regen.sort_values(list(keys)).reset_index(drop=True)
    assert len(a) == len(b), f"row count {len(b)} != committed {len(a)}"
    for c in numeric:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert (np.isnan(x) == np.isnan(y)).all(), f"{c}: NaN pattern differs"
        m = ~np.isnan(x)
        delta = np.abs(x[m] - y[m])
        # Relative where the committed value is non-zero, absolute where it sits at zero
        # (a relative tolerance is meaningless around machine epsilon).
        ref = np.abs(x[m])
        rel = delta.copy()
        nz = ref > 0
        rel[nz] = delta[nz] / ref[nz]
        assert rel.max(initial=0.0) <= RTOL, f"{c}: max rel {rel.max(initial=0.0):.3e} > {RTOL}"
    for c in boolean:
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), f"{c}: differs"


GEOM = ["angle_deg", "null_lo", "null_med", "null_hi", "perm_p", "gain_r", "dec_cohen_d"]


# --------------------------------------------------------------------- gate 0: the cache
def test_recap_cache_invariants(recap_cache):
    """576 rows / 144 games / cb0-3; mask 541 = 525 pure + 16 stated; 35 imputed excluded."""
    meta = pd.read_parquet(recap_cache / "meta_gptoss.parquet")
    assert len(meta) == 576
    assert meta.game_code.nunique() == 144
    assert sorted(meta.counterbalance_id.unique()) == [0, 1, 2, 3]
    keep = meta[meta.use_in_neural_target == 1]
    assert len(keep) == 541
    assert keep.prob_source.value_counts().to_dict() == {"pure": 525, "stated": 16}
    assert meta[meta.use_in_neural_target != 1].prob_source.unique().tolist() == ["default_uniform"]


def test_recap_layer0_anchor_is_invariant(recap_cache):
    """The uniform capture site is one shared token, so its layer-0 embedding is identical
    across every row. This is the invariant that makes the site 'uniform' at all, and the
    reason layer 0 is a degenerate 90-degree row rather than a measurement."""
    X = np.load(recap_cache / "X_gptoss_l0.npy")
    assert np.max(np.abs(X - X[0])) == 0.0


# ------------------------------------------------- gates 1, 2, 4: residual-stream geometry
@pytest.mark.parametrize("belief,table", [("uniform", "fusion_depth_table.csv"),
                                          ("empirical", "fusion_depth_empirical.csv")])
def test_fusion_gptoss_uniform_site(recap_cache, belief, table):
    """The GPT-OSS block of both depth tables regenerates from the recap substrate."""
    from analysis.layer_b.fusion_associative.build_fusion_figures import run_recap
    regen = run_recap(belief)
    regen = regen[regen.layer > 0]          # L0 belongs to the sensitivity table only
    committed = pd.read_csv(FUSION / table)
    committed = committed[committed.model == "gptoss"]
    assert set(committed.row_scope) == {"uniform_transition_policy"}
    assert set(committed.n_obs) == {541} and set(committed.n_games) == {144}
    assert_close(regen, committed, ["layer"], GEOM, ["sig"])


def test_fusion_sensitivity_uniform_block(recap_cache):
    """The sensitivity table keeps layer 0: a degenerate 90/90/90/90 row with p = 1.0 and
    empty diagnostics. It must be emitted, not skipped -- it also consumes a permutation
    block, so dropping it would shift every later layer's p-value."""
    from analysis.layer_b.fusion_associative.build_fusion_figures import run_recap
    regen = run_recap("uniform")
    committed = pd.read_csv(FUSION / "gptoss_capture_site_sensitivity.csv")
    committed = committed[committed.row_scope == "uniform_transition_policy"]
    assert len(committed) == 37
    assert_close(regen, committed, ["layer"], GEOM, ["sig"])
    l0 = regen[regen.layer == 0].iloc[0]
    assert (l0.angle_deg, l0.null_lo, l0.null_med, l0.null_hi, l0.perm_p) == (90.0, 90.0, 90.0, 90.0, 1.0)
    assert np.isnan(l0.gain_r) and np.isnan(l0.dec_cohen_d)


# ---------------------------------------------------------------- gate 3: router-gate space
def test_router_fusion_uniform_site(recap_cache):
    from analysis.layer_b.fusion_associative.oss_router_fusion import run_recap as router_run
    regen = pd.concat([router_run("uniform"), router_run("empirical")], ignore_index=True)
    committed = pd.read_csv(FUSION / "oss_router_fusion.csv")
    assert len(committed) == 28 and set(committed.n) == {541}
    assert_close(regen, committed, ["belief", "layer"],
                 ["angle_deg", "null_lo", "null_med", "null_hi", "perm_p"], ["sig"])


# ------------------------------------------------------- gate 5: the recruitment rename
def test_recruitment_geometry_is_the_fusion_table_renamed():
    """recruitment_geometry_depth.csv is the uniform depth table under other column names --
    for every model, not just GPT-OSS. Pure table algebra, so it needs no deposit."""
    fus = pd.read_csv(FUSION / "fusion_depth_table.csv")
    rec = pd.read_csv(COMMITTED / "layer_b" / "recruitment" / "recruitment_geometry_depth.csv")
    rename = {"angle_deg": "angle_decision_incentive_deg", "null_med": "null_median_deg",
              "null_lo": "null_lo_deg", "null_hi": "null_hi_deg", "sig": "below_null"}
    a = fus.rename(columns=rename).sort_values(["model", "layer"]).reset_index(drop=True)
    b = rec.sort_values(["model", "layer"]).reset_index(drop=True)
    assert len(a) == len(b) == 276
    assert_close(b, a, ["model", "layer"],
                 ["angle_decision_incentive_deg", "null_median_deg", "null_lo_deg",
                  "null_hi_deg", "perm_p", "n_obs", "n_games"], ["below_null"])
    assert (b.n_perm == 200).all()


# ------------------------------------------ the restored incentive columns on the baseline meta
def test_baseline_cache_carries_incentive_targets():
    """delta1c / delta2c / stim_control_cell00 must be on the cached baseline meta.

    They were dropped from the cache schema at some point after 2026-07-10, which broke all
    three `rebuild/` scripts on any freshly built cache. This
    guards the restoration: the columns must be present, and must equal the single authority
    for them -- rebuild_missing_tables._targets() -- joined per game.
    """
    _need(BASELINE_CACHE / "meta_gptoss.parquet", "Layer-B residual cache")
    from analysis.layer_b.rebuild.rebuild_missing_tables import _targets
    tgt = _targets()
    for model in ("qwen_instruct", "qwen", "llama31_instruct", "gptoss"):
        meta = pd.read_parquet(BASELINE_CACHE / f"meta_{model}.parquet")
        for col in ("delta1c", "delta2c", "stim_control_cell00"):
            assert col in meta.columns, f"{model}: cache meta lost {col}"
            expected = meta.game_code.map(tgt[col]).to_numpy(float)
            assert (expected == meta[col].to_numpy(float)).all(), f"{model}: {col} disagrees with _targets()"
        # delta1c is the continuous canonical-signed gap, not its sign -- the distinction that
        # A.5 had to settle against the 2026-07-10 oracle.
        assert meta.delta1c.nunique() > 3, "delta1c looks like a sign, not the continuous gap"
        assert meta.stim_control_cell00.value_counts().to_dict() == {1: 192, 2: 192, 3: 192}, \
            f"{model}: stimulus control is not balanced across the 576 rows"


# ------------------------------------------- blocker 1: the b1_* family from a fresh cache
B1 = ["b1_decodability.csv", "b1_equilibrium_structure.csv",
      "b1_crystallization.csv", "b1_crystallization_summary.csv"]


def test_b1_tables_regenerate(tmp_path):
    """The four tables that a commit_type writer/reader desync used to make unbuildable.

    Needs the Layer-B residual cache, which ``make tables-layer-b`` builds first (~90 s,
    ~4.5 GB); this test does not build it, it skips.
    """
    _need(BASELINE_CACHE / "meta_gptoss.parquet", "Layer-B residual cache")
    scratch = tmp_path / "results"
    scratch.mkdir()
    env = dict(os.environ, SCA_RESULTS_ROOT=str(scratch))
    for script in ("build_decodability.py", "build_crystallization.py"):
        subprocess.run([sys.executable, f"analysis/layer_b/{script}"],
                       cwd=repo_root(), env=env, check=True)
    for name in B1:
        assert_frames_equal_csv(scratch / "layer_b" / name, COMMITTED / "layer_b" / name)


def assert_frames_equal_csv(regen: Path, committed: Path) -> None:
    a, b = pd.read_csv(committed), pd.read_csv(regen)
    assert list(a.columns) == list(b.columns), f"{regen.name}: columns differ"
    numeric = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])
               and pd.api.types.is_numeric_dtype(b[c])]
    other = [c for c in a.columns if c not in numeric]
    assert_close(b, a, [c for c in a.columns if c in ("model", "layer", "target", "metric")] or list(a.columns[:1]),
                 numeric)
    for c in other:
        assert a[c].astype(str).tolist() == b[c].astype(str).tolist(), f"{regen.name}: {c} differs"
