#!/usr/bin/env python3
"""Regenerate ``data/MANIFEST.json``.

The manifest has four blocks:

``files``
    Every git-tracked file under ``data/``, with sha256 and byte size. This is what
    ``scripts/download_data.py --verify-committed`` checks a clone against.

``provenance``
    The builder -> frozen-table chain, carried forward from the paper's reproducibility
    manifest with paths rewritten to this repository's layout, plus the locked seeds. This
    is the record of *how* the committed tables were produced.

``human_raw_sources``
    The two participant-level exports we are **not** allowed to redistribute, recorded by
    path, sha256 and citation only. A user who obtains their own copy can prove it is
    byte-identical to ours, which is what makes the human lambda fit reproducible without
    redistributing anything.

``deposit``
    The Zenodo components, their sizes, and their checksums. Sizes are measured from a
    local copy when one is reachable; the record id and per-tar checksums are filled in
    when the deposit is published.

Usage::

    uv run python scripts/dev/build_manifest.py            # write data/MANIFEST.json
    uv run python scripts/dev/build_manifest.py --check    # verify, exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from strategic_anatomy.config import data_root, repo_root

MANIFEST_VERSION = "strategic-choice-anatomy-data-1"

# Carried forward from paper/paper_NHB/REPRODUCIBILITY_MANIFEST_v4p4.json (audit 2026-07-11,
# private repo head 2e65c16c4a509947d3e76e481cdc08fb10b4a6b1). Paths rewritten to this
# repository's layout; the builder sha256 values are the ones recorded at that audit and refer
# to the private-repo files, so they are labelled `source_sha256` rather than being recomputed
# against the ported copies (the ports differ by import rewrites -- see oss_migration/).
PROVENANCE = {
    "figures_reference": {
        "_note": (
            "The 11 PDFs in data/results/figures_reference/ are the paper's final renders. "
            "Rebuild them with scripts/dev/tier1_figures.sh and compare with "
            "scripts/dev/compare_figures.py. PDF bytes never match (matplotlib stamps a "
            "creation date), and text layout shifts across matplotlib minor versions, so "
            "compare rasterised pixels and match the renderer below for an exact comparison."
        ),
        "rendered_with_matplotlib": "3.10.8",
        "rerendered_2026_08_28": (
            "fig_steering_causal.pdf and fig_layerB_main_v2.pdf were re-rendered from this "
            "repository's code under matplotlib 3.10.8 as part of the q=0.5 construct-identity "
            "correction (docs/METHODS.md 6.4): the steering figure now defaults to the released "
            "q05 tables, and the Layer-B figure dropped a stale GPT-OSS annotation. Both rebuilds "
            "are pixel-identical to the corresponding paper images regenerated in the source "
            "repository, and both reproduce pixel-identically from a clean rebuild here."
        ),
        "reproduces_on_that_version": [
            "fig_layerB_main_v2", "fig_fusion_combined",
            "fig_steering_causal", "fig_layerC_paper",
            "fig2_trait_steering",
        ],
        "known_to_differ": {
            "fig_nullspread_geometry_nhb": (
                "The committed reference render was produced from the 2026-07-04 generation of "
                "fusion_depth_table.csv; rebuilding from that table reproduces the reference PDF "
                "pixel-for-pixel (0.000%), while rebuilding from the shipped table differs by "
                "6.99% of pixels. Verified against paper_NHB_v5p2_MANUAL_v6.tex: every other "
                "fusion claim in the manuscript (GPT-OSS 39.0 deg p=0.28, pointwise p<0.05 at "
                "layers 1-2 only, own-belief L1-17 + isolated L19 with 33.6 deg p=0.16, and dense "
                "onsets L56/L56/L37) reproduces exactly from the SHIPPED tables. Only the "
                "supplementary rank-correlation range -0.78..-0.98 traces to the older table; the "
                "shipped tables give -0.79..+0.46. angle_deg is bit-identical across all three "
                "table generations for every dense model -- only the Monte-Carlo permutation null "
                "band moved (SEED=20260702, N_PERM=200, rng consumed sequentially across layers). "
                "So the released code and data are self-consistent; the stale artefact is that one "
                "supplementary figure."
            ),
        },
    },
    "seeds": {
        "layer_b_bootstrap": 20260520,
        "layer_b_cv": 0,
        "recruitment_bridge_bootstrap": 20260627,
        "token_probe_lens": 0,
        "layer_c_cluster_bootstrap": 0,
        "permutation_direction_seeds_q05": [0, 1, 2],
        "permutation_direction_seeds_empirical": [1, 2, 3],
        "permutation_seed_rejection_rule": (
            "slots start a priori from integers 0,1,2; a candidate is accepted only if "
            "|corr(y_perm, y)| <= 0.15 at game level per model, else replaced by the next "
            "integer. Three fixed seeds license no p-value. The two incentive arms were screened "
            "against different targets and so ended on different seeds: on the pre-correction "
            "empirical-belief target seed 0 was rejected, giving 1/2/3; on the released q=0.5 "
            "target all of 0/1/2 pass, and that is the seed set behind "
            "data/results/steering/smalldose_summary_q05/perm_null_results.csv. The two seed sets "
            "are NOT matched, so old-vs-new permutation comparisons are not like-for-like "
            "(docs/METHODS.md 6.4)."
        ),
        "mixed_strategy_draw": "sha256(game, cb, player) -> Bernoulli(stated_p_act0)",
    },
    "game_features": {
        "builder": {
            "path": "features/compute_game_features.py",
            "source_sha256": "6f77052191fb990f5bf9d0650a9a1589865a86f68d0536031bfa6561d2a2d806",
        },
        "frozen_table": {
            "path": "data/games/game_features.csv",
            "source_sha256": "8d6d072bb6a6bb0a68a883d95c016e1687d8ac074c87e4aab7e20da83bf800a6",
            "rows": 144,
            "columns": 45,
        },
        "verification": "The restored builder reproduced the frozen CSV byte-for-byte at audit.",
    },
    "griffiths_cardinal_transformation": {
        "source_citation_key": "zhu2025capturing",
        "frozen_mapping": {
            "path": "data/games/taxonomy/griffiths_to_canonical.csv",
            "source_sha256": "4002000edb4a3431511f6244875ebbc4ed7084acf460c6674579edd8bc4be75f",
            "rows": 2416,
            "columns": 15,
        },
        "active_analysis_builder": {
            "path": "features/build_unified_pairs.py",
            "source_sha256": "c711ec0e7caba9009baa28b4f16546def3d26a028a356fc404a59f66f7a5f7ab",
        },
        "frozen_analysis_table": {
            "path": "data/human_refs/unified_pairs.parquet",
            "source_sha256": "660ba99e6f8d6e2e7aaa5555d9e0220a1932528c5652744fb06f0f2ceb0bf0f5",
            "rows_all_sources": 7466,
            "columns": 92,
            "rank_clean_griffiths_role_rows": 1664,
            "rank_clean_griffiths_games": 832,
            "canonical_perspectives_covered": 126,
            "player_swap_paired_structures_covered": 69,
        },
        "transformation_rule": (
            "Exclude an entire game_id if either role has tied payoff ranks; retain cardinal "
            "matrices even when cardinal and ordinal level-1 predictions differ; orient each role "
            "with swap_sr/swap_sc; evaluate theory on the oriented cardinal matrix."
        ),
    },
    "moore_nagel_session_provenance": {
        "source_citation_key": "moore2026similarity",
        "canonical_mapping": {
            "path": "data/games/taxonomy/human_game_master_per_canonical.csv",
            "source_sha256": "a63a840e4d513b00a33fcb19bf28de8043f800573dcc78b2cd50c7f4357340f7",
        },
        "primary_generator": {
            "path": "analysis/layer_a/src/human_lambda_mgn.py",
            "source_sha256": "b8dcd43ec75479a2a6a1b9d43ad4b303b16b4c44c67a624c116ebe0a5eeae501",
        },
        "primary_output": {
            "path": "data/results/layer_a/f3_human_lambda.csv",
            "source_sha256": "ad2ff2f721c140d2442f2f694573100be7a38cdc826f4197266dbad2c366079a",
        },
        "duplicate_session_robustness_generator": {
            "path": "analysis/layer_a/src/human_dedup_robustness.py",
            "source_sha256": "e96a2a0ebd7adca77c05232ed9c9f937ba72b24ffbb2a2c66f3a8606d4078c18",
        },
        "duplicate_session_robustness_output": {
            "path": "data/results/layer_a/human_dedup_robustness.csv",
            "source_sha256": "6533cbdda99418a31dc0763d550aefd9f215002bb63c2694489bf5185a1d521c",
            "decision_difference_between_duplicate_sessions": "68/144",
            "maximum_absolute_lambda_change": 0.006833,
            "maximum_absolute_complexity_rho_change": 0.001446,
        },
        "retention_rule": (
            "Retain both complete sessions; use session as the analysis and clustering unit; "
            "report leave-either-session-out sensitivity."
        ),
    },
    "probe_protocols": {
        "decodability_and_crystallisation": {
            "final_layer_builder": {
                "path": "analysis/layer_b/build_decodability.py",
                "folds": 5,
                "pca_components": 64,
                "bootstrap_resamples": 2000,
                "cv_seed": 0,
                "bootstrap_seed": 20260520,
            },
            "depth_curve_builder": {
                "path": "analysis/layer_b/build_crystallization.py",
                "folds": 5,
                "pca_components": 64,
                "bootstrap_resamples": 1000,
                "cv_seed": 0,
                "bootstrap_seed": 20260520,
            },
        },
        "recruitment_bridge": {
            "builder_path": "analysis/layer_b/rebuild/rebuild_bridge_variants.py",
            "bootstrap_helper": "analysis/layer_b/rebuild/rebuild_lib.py::boot_ci",
            "folds": 5,
            "dimensionality_reduction": "none",
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 20260627,
        },
        "token_probe_lens": {
            "builder_path": "analysis/layer_c/probe_bridge.py",
            "folds": 3,
            "maximum_pca_components": 120,
            "bootstrap_resamples": 800,
            "seed": 0,
        },
        "layer_c_statistics": {
            "builder_path": "analysis/layer_c/stats_layerc.py",
            "bootstrap_resamples": 1000,
            "cluster_unit": "game_code",
            "seed": 0,
        },
    },
}

# NOT redistributable. Recorded so a user can verify their own copy is the file we used.
HUMAN_RAW_SOURCES = {
    "_note": (
        "These participant-level exports are NOT included in this repository or the data "
        "deposit; they belong to the source studies' authors. The sha256 values let you "
        "confirm that a copy you obtain independently is byte-identical to the one used "
        "here, which is what makes the human lambda fit reproducible. See "
        "data/human_refs/raw/README.md."
    ),
    "moore_nagel": {
        "citation": (
            "Moore, Germano & Nagel (2026), working paper - the 144-perspective 2x2 catalogue "
            "and associated human choice data."
        ),
        "expected_path": "data/human_refs/raw/nagel/normalized_data_20241612.csv",
        "sha256": "2a45093b60eab9af14ed48239b3644cd7644b381a9049467534f16544092e5fc",
        "complete_sessions": 451,
        "unique_participants": 450,
        "also_required": [
            "data/human_refs/raw/nagel/df_ros.csv",
            "data/human_refs/raw/nagel/perspective_info.csv",
        ],
        "needed_by": [
            "analysis/layer_a/src/human_lambda_mgn.py",
            "analysis/layer_a/src/human_dedup_robustness.py",
            "analysis/layer_a/build_rule_classification.py",
            "analysis/layer_a/figscripts/fig3_qre_to_levelk.py (figS_model_selection)",
            "analysis/layer_a/figscripts/fig_behaviour_merged_2x3.py",
            "analysis/layer_a/figscripts/fig3b_v2_bounded_rationality_map.py",
            "features/build_master_df.py",
        ],
    },
    "zhu_griffiths": {
        "citation": (
            "Zhu, J.-Q., Peterson, J. C., Enke, B. & Griffiths, T. L. (2025). Capturing the "
            "complexity of human strategic decision-making with machine learning. Nature Human "
            "Behaviour 9, 2114-2120."
        ),
        "expected_path": "data/human_refs/raw/griffiths/games2p2k_main griffith emke.csv",
        "sha256": "16c066df11df7563efb4176a2111a071103f9057f247e654609a1c4c7ed6b9d8",
        "redistribution": (
            "obtain from the cited dataset/article subject to its redistribution terms"
        ),
        "needed_by": ["features/build_unified_pairs.py"],
    },
}

# Deposit components. Sizes are measured from a local copy when one is present; the record id
# and the per-tar sha256 values are filled in at publication time.
DEPOSIT_COMPONENTS = [
    ("substrate", "substrate/{model}/{game}/", "results.parquet, acts.npz, config.json, _DONE; gptoss adds router.npz"),
    ("gptoss_recap", "gptoss_recap/{game}/", "uniform-site recapture; adds genids.npz"),
    ("layerc", "layerc/{model}/{game}/", "tokens.parquet - per-token logit-lens scores"),
    ("layerc_bridge_residuals", "layerc_bridge_residuals/{model}/{game}/", "resid.npy + meta.parquet"),
    ("steering", "steering/{smalldose,smalldose_q05,perm,perm_q05,directions}/",
     "steering parquets and fitted directions; the released incentive arm is the q05 one "
     "(smalldose_q05, perm_q05, directions/akata_q05{,_perp,_perm}), the unsuffixed roots are "
     "the pre-correction empirical-belief arm kept as history"),
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _keep(root: Path, p: Path) -> bool:
    """The manifest covers every distributed file under data/, and nothing else."""
    if not p.is_file() or p.is_symlink():
        return False
    rel = p.relative_to(root).as_posix()
    if rel == "data/MANIFEST.json":              # the manifest cannot contain its own hash
        return False
    if rel.startswith("data/human_refs/raw/") and rel != "data/human_refs/raw/README.md":
        return False                              # third-party raw data is never distributed
    if p.name == ".DS_Store" or "__pycache__" in p.parts:
        return False
    return True


def tracked_data_files(root: Path) -> list[Path]:
    """Every distributed file under data/.

    Uses git (which honours .gitignore) when this is a working clone, and falls back to a
    filesystem walk otherwise — a Zenodo software archive or a plain tarball has no .git,
    and `--check` has to work there too.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "data"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        candidates = [root / rel.strip() for rel in listed if rel.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        candidates = list((root / "data").rglob("*"))

    return sorted(p for p in candidates if _keep(root, p))


def dir_size(path: Path) -> int:
    """Total bytes under ``path``, following symlinked directories.

    Path.rglob does not descend into a symlinked subdirectory, and a developer's
    ``data_heavy/`` is typically a tree of symlinks at exactly that depth, so walk it
    explicitly with os.walk(followlinks=True) instead.
    """
    import os
    if not path.exists():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=True):
        for name in filenames:
            f = Path(dirpath) / name
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def build(root: Path) -> dict:
    files = {}
    for p in tracked_data_files(root):
        rel = p.relative_to(root).as_posix()
        files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    deposit = {
        "record_id": None,
        "doi": None,
        "license": "CC-BY-4.0",
        "_note": (
            "record_id, doi and per-component sha256 are filled in when the Zenodo deposit is "
            "published. Sizes below are measured from a local copy of the deposit tree where "
            "one was reachable at manifest build time; 0 means not measured here."
        ),
        "components": [],
    }
    local = data_root()
    for name, layout, contents in DEPOSIT_COMPONENTS:
        component_root = local / name if name != "steering" else local / "steering"
        deposit["components"].append({
            "name": name,
            "layout": layout,
            "contents": contents,
            "archive": f"{name}.tar.gz",
            "sha256": None,
            "bytes_uncompressed": dir_size(component_root),
        })

    return {
        "manifest_version": MANIFEST_VERSION,
        "hash_algorithm": "SHA-256",
        "generator": "scripts/dev/build_manifest.py",
        "files": files,
        "provenance": PROVENANCE,
        "human_raw_sources": HUMAN_RAW_SOURCES,
        "deposit": deposit,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed manifest still matches the tree; exit 1 on drift")
    args = ap.parse_args(argv)

    root = repo_root()
    out = root / "data" / "MANIFEST.json"
    fresh = build(root)

    if args.check:
        if not out.exists():
            print(f"FAIL {out} does not exist", file=sys.stderr)
            return 1
        old = json.loads(out.read_text())
        drift = []
        for rel, rec in fresh["files"].items():
            prev = old.get("files", {}).get(rel)
            if prev is None:
                drift.append(f"  new       {rel}")
            elif prev["sha256"] != rec["sha256"]:
                drift.append(f"  changed   {rel}")
        for rel in old.get("files", {}):
            if rel not in fresh["files"]:
                drift.append(f"  removed   {rel}")
        if drift:
            print(f"MANIFEST drift ({len(drift)} entries):", file=sys.stderr)
            print("\n".join(drift[:40]), file=sys.stderr)
            return 1
        total = sum(r["bytes"] for r in fresh["files"].values())
        print(f"MANIFEST OK — {len(fresh['files'])} files, {total/1e6:.1f} MB")
        return 0

    out.write_text(json.dumps(fresh, indent=2, sort_keys=False) + "\n")
    total = sum(r["bytes"] for r in fresh["files"].values())
    print(f"wrote {out}")
    print(f"  files:   {len(fresh['files'])} ({total/1e6:.1f} MB)")
    for c in fresh["deposit"]["components"]:
        size = c["bytes_uncompressed"]
        print(f"  deposit: {c['name']:24s} {size/1e9:.2f} GB" if size else
              f"  deposit: {c['name']:24s} (not measured locally)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
