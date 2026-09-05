# Reproducing

What each result needs, and the exact command to produce it. Three tiers:

| Tier | What | Needs |
|---|---|---|
| **1** | Rebuild figures from committed tables | this clone |
| **2** | Regenerate the table families below from the substrate | the deposit (~21 GB), CPU |
| **3** | Re-collect the substrate | GPU, days |

```bash
uv venv --python 3.11
uv pip install -e ".[analysis,dev]"
make test
```

---

## Tier 1 — from a clean clone

```bash
make figures-tier1     # the 6 figures that need nothing else
make verify-data       # 227 committed files against data/MANIFEST.json
```

### Per-figure requirements

Measured, not asserted: `NO_HEAVY=1 scripts/dev/tier1_figures.sh` hides the deposit and reports
which figures still build.

| Figure | Producer | Needs |
|---|---|---|
| `fig2_trait_steering` | `analysis/layer_a/figscripts/fig2_trait_steering.py` | committed tables only |
| `fig_layerB_main_v2` | `analysis/layer_b/fig_layerB_main_v2.py` | committed tables only |
| `fig_fusion_combined` | `analysis/layer_b/fusion_associative/fig_fusion_combined.py` | committed tables only |
| `fig_nullspread_geometry_nhb` | `analysis/layer_b/fusion_associative/fig_nullspread_nhb.py` | committed tables only |
| `fig_steering_causal` | `analysis/steering/fig_steering_causal_paper.py` | committed tables only |
| `fig_layerC_paper` | `analysis/layer_c/fig_layerC_paper.py` | committed tables only |
| `figS_attribution` | `analysis/layer_a/build_attribution.py` | + deposit `substrate/` |
| `fig_router_bottleneck` | `analysis/layer_b/router_bottleneck_analysis.py` | + deposit `substrate/` (gptoss router) |
| `fig_token_heatmap_3panel` | `analysis/layer_c/fig_token_heatmap_3panel.py` | + deposit `layerc/` |
| `fig_behaviour_merged_2x3` | `analysis/layer_a/figscripts/fig_behaviour_merged_2x3.py` | + deposit `substrate/` **and raw human data** |
| `figS_model_selection` | `analysis/layer_a/figscripts/fig3_qre_to_levelk.py` | + deposit `substrate/` **and raw human data** |

So: **6 from a clean clone, 9 with the deposit, 11 with the deposit plus the raw human export.**

Run one on its own with `uv run python <producer>`. Output lands in `analysis/<layer>/figures/`
with the same filename as the reference render in `data/results/figures_reference/`.

Figure 1 of the paper is a concept schematic designed externally; there is no code claim for it.

### Comparing your rebuild against ours

```bash
make figures-diff        # rasterises both sides and reports differing pixels
```

**PDF bytes will never match.** matplotlib stamps a creation date into every file, and text
layout shifts between matplotlib minor versions — enough to move a few percent of pixels and to
change the canvas by a pixel or two. So compare rendered pixels, not bytes.

The reference PDFs were rendered with **matplotlib 3.10.8**. On that version, **five of the six**
Tier-1 figures reproduce: `fig_layerB_main_v2`, `fig_fusion_combined`, `fig_steering_causal` and
`fig_layerC_paper` come back **pixel-identical**, and `fig2_trait_steering` matches to 0.08% of
pixels (sub-pixel text placement). Measured 2026-08-28 with all six rebuilt on 3.10.8; the pinned
environment ships a newer matplotlib, so `make figures-diff` there will report a few percent of
differing pixels on every figure from text metrics alone.

`fig_steering_causal.pdf` and `fig_layerB_main_v2.pdf` were **re-rendered on 2026-08-28** as part of
the q = 0.5 construct-identity correction (`docs/METHODS.md` §6.4): the steering figure now defaults
to the corrected `smalldose_summary_q05` tables, and the Layer-B figure dropped a stale annotation.
Both new references are pixel-identical to the corresponding figures in the paper.

One differs, and we know why:

| Figure | Status |
|---|---|
| `fig_nullspread_geometry_nhb` | The committed reference render was produced from an **older** `null_spread_vs_geometry.csv`. Rebuilding from that older table reproduces the reference PDF exactly (0.000%); rebuilding from the shipped table differs by 6.99% of pixels. The **shipped table is the correct one** — its `null_spread`/`null_med` agree with the shipped `fusion_depth_table.csv` to 1.8e-15, which the older copy does not. So the released code and data are self-consistent and the reference PDF is stale. |

This is recorded in `data/MANIFEST.json` under `provenance.figures_reference` so the status
travels with the data rather than living only here.

If you want to check the *numbers* rather than the rendering, that is `make verify-data` (every
committed table against its sha256) and `make verify` (regenerate from the deposit and compare).

### The two figures needing raw human data

The human precision λ is fitted on **individual choices**, not on aggregate frequencies, so the
committed per-game aggregates cannot substitute. Obtain
`normalized_data_20241612.csv` (and `df_ros.csv`, `perspective_info.csv`) from Moore, Germano &
Nagel and place them in `data/human_refs/raw/nagel/`. Verify your copy:

```bash
shasum -a 256 data/human_refs/raw/nagel/normalized_data_20241612.csv
# compare against human_raw_sources.moore_nagel.sha256 in data/MANIFEST.json
```

Full instructions and the list of every consumer: `data/human_refs/raw/README.md`.

---

## Tier 2 — regenerate the tables

```bash
make download-data                       # or: export SCA_DATA_ROOT=/path/to/deposit
python scripts/download_data.py --list   # components and sizes
make tables                              # the four families (coverage below)
make verify                              # hash-compare against the committed copies
```

Per family:

| Family | Command | Regenerates |
|---|---|---|
| Layer A | `make tables-layer-a` | `data/results/layer_a/*` — `f1_*`, `f2_*`, `f3_*`, `b_*`, `s_*` |
| Layer B | `make tables-layer-b` | `data/results/layer_b/*` incl. `fusion/`, `rebuild/`, `recruitment/` |
| Layer C | `make tables-layer-c` | `data/results/layer_c/*` |
| Steering | `make tables-steering` | `data/results/steering/*` (pre-correction arm; the released q05 tables have their own commands, below) |

Layer B starts with `build_residual_cache.py`, a one-time ~90 s pass that materialises
`$SCA_DATA_ROOT/layer_b_cache/` (~4.5 GB) from `substrate/`. It is deliberately not in the deposit —
it is derived, and rebuilding is faster than downloading. It is followed by
`fusion_associative/recap_cache.py`, a ~3 s pass that builds the GPT-OSS uniform-site cache
(`layer_b_cache/recap_baseline/`, 576 × 37 × 2880) from the `gptoss_recap` component — the substrate
the amended GPT-OSS geometry is read on (`docs/METHODS.md`, uniform-site amendment).

### What `make tables` does and does not cover

**`make tables` regenerates 61 of the 152 committed files under `data/results/`, not all of them.**
Measured 2026-08-28 by seeding a scratch copy with a sentinel mtime and counting what the documented
chains actually rewrote:

| Family | Committed | Regenerated by `make tables-<family>` |
|---|---:|---:|
| layer_a | 35 | 14 |
| layer_b | 77 | 34 |
| layer_c | 12 | 9 |
| steering | 28 | 4 (+7 more via the explicit q05 commands below) |

Committed-but-uncovered examples, with why: the whole `f3_*` family (`f3_lambda.csv`,
`f3_model_selection_cv.csv`) comes from `src/fig3_behavioral_model.py`, reached only through
`figscripts/fig3_qre_to_levelk.py`, which needs the user-supplied raw human export and is in no
`make` chain; `f1_alignment_matrix.csv`, `f1_rationality_by_class.csv`, `f1_human_corr.csv` and
`f2_trait_aim.csv` are written by figure scripts rather than builders; `b_rule_fit.csv` comes from
`build_rule_classification.py`, which is not in the chain. These are reproducible — they are simply
not reached by `make`. Run their producer directly, and set `SCA_RESULTS_ROOT` to a scratch copy
first if you do not want to overwrite the committed tree.

Each family's chain, in order:

```
layer_a:   build_data_layer → build_regime_rationality → build_trait_steering_oneshot
           → validate_foundation → build_attribution
layer_b:   build_residual_cache → build_decodability → build_crystallization
           → validate_foundation → fusion_associative/recap_cache
           → fusion_associative/oss_router_fusion → fusion_associative/build_fusion_figures
           → recruitment/build_all → rebuild/rebuild_bridge_variants
           (the last step is a patch over three of build_all's tables, so it runs last)
layer_c:   compute_layerc → stats_layerc
steering:  analyze_smalldose → smalldose_final_summary
           (released q05 arm: smalldose_final_summary → perm_null_analysis
            → apparatus_geometry_q05 — explicit commands below)
```

**The released steering arm is the corrected q = 0.5 one** (`docs/METHODS.md` §6.4), and
`make tables-steering` regenerates the *pre-correction* tables, because those are what
`analyze_smalldose.py` and the analyzers' defaults point at. Regenerate the released arm explicitly:

```bash
uv run python analysis/steering/smalldose_final_summary.py \
    --in-root "$SCA_DATA_ROOT/steering/smalldose_q05" --delta-suffix _q05 \
    --out data/results/steering/smalldose_summary_q05

uv run python analysis/steering/perm_null_analysis.py \
    --perm-root "$SCA_DATA_ROOT/steering/perm_q05" \
    --game-slopes data/results/steering/smalldose_summary_q05/game_slopes.csv \
    --out data/results/steering/smalldose_summary_q05/perm_null_results.csv

uv run python analysis/steering/apparatus_geometry_q05.py     # writes into results_root()
```

All three reproduce the committed q05 tables **byte-for-byte**. The first two carry a
construct-identity guard: point them at q05 inputs while leaving `--out` at its default and they
exit non-zero rather than overwrite the pre-correction tables — which is the defect §6.4 exists to
prevent. `apparatus_geometry_q05.py` has no `--out`; set `SCA_RESULTS_ROOT` to a scratch copy to
run it without touching `data/results/`. It also reads `$SCA_DATA_ROOT/substrate/` and refits
`d_inc` from scratch, so it takes a few minutes and aborts if the refit does not reproduce the
shipped direction at cos ≥ 0.999.

**Comparison rule.** Exact sha256 match is expected. Where float formatting differs, compare parsed
values at `rtol = 1e-9`. A genuine numeric difference is a bug — a port error or an environment
sensitivity — not something to widen the tolerance for. `make verify` encodes these checks as
`pytest -m tier2` (`tests/test_tier2_tables.py`): the uniform-site cache invariants, the four
GPT-OSS geometry gates, the recruitment rename identity, and the four `b1_*` tables. Every one of
them skips with a stated reason when `SCA_DATA_ROOT` is unset, so it is safe to run from a bare clone.

**Environment pins that matter for byte-reproduction.** `scikit-learn` is pinned to **1.8.0** in the
`[analysis]` extra: four attribution tables (`s_attribution_shap`, `s_shap_beeswarm`,
`s_shap_delta1_dependence`, `s_variance_partition`) are sha256-identical on 1.8.0 and differ on 1.9.0
through SHAP tie ordering. The pin exists for byte-level reproduction, not correctness.
`matplotlib` is deliberately **not** pinned; 3.10.8 is recorded as the reference-render version for
`make figures-diff` (see above), and figures are compared by pixels rather than bytes.

The two `validate_foundation.py` scripts are canary gates: they assert documented values (e.g. the
Layer-A P(canonical)/λ table) and exit non-zero on any mismatch. Run them first if a downstream
table disagrees.

---

## Tier 3 — re-collect the substrate

Needs a DGX-Spark-class GPU and days of compute. Read
[`docs/METHODS.md` §8](docs/METHODS.md) **before** starting — four documented gotchas will silently
change your numbers:

1. the gpt-oss harmony template is **date-stamped** (prompts are not byte-reproducible across days);
2. a deliberate **one-block off-by-one** between capture key and steer hook, kept for comparability;
3. **kernel-path determinism** — the gpt-oss environment is mandatory, not a preference;
4. the **locked seeds** (§8.4).

Set up the environment with `scripts/setup/setup_spark.sh` (dense) and
`scripts/setup/setup_gptoss_env.sh` (MXFP4); see [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md).

```bash
export SCA_DATA_ROOT=/path/with/room     # ~21 GB of output
bash scripts/launchers/collect_dense.sh       # preflight gate -> 3 dense models × 144 games
bash scripts/launchers/collect_gptoss.sh      # preflight gate -> harmony substrate -> recapture
bash scripts/launchers/collect_layerc.sh      # token attribution -> B<->C bridge residuals
bash scripts/launchers/steer_smalldose.sh     # smoke gate -> small-dose chain
bash scripts/launchers/steer_perm.sh          # smoke gate -> permutation matched control
```

Five launchers, one per pipeline arm, each gated: none starts a GPU-day of work before proving the
mechanism on a bounded run. Add `--smoke-only` to run just that gate and stop — the preflight for
the collection arms, a 1-game smoke plus its deterministic validator for the steering arms:

```bash
bash scripts/launchers/collect_dense.sh --smoke-only
```

`scripts/launchers/LAUNCH.md` documents the order, per-arm durations, resume semantics and
logging. Every arm is resumable — `--skip-existing` continues from the per-game `_DONE` sentinels,
so a reboot costs you at most one game. Re-run the same command; there is no separate resume script.

Measured costs: dense ≈ 170 s/game (qwen) and ≈ 100 s/game (llama), 144 games × 3 models;
steering ≈ 4.7 s/row → ≈ 15.6 h/model at 18,144 rows/model.

The preflights (`collection/preflight_dense.py`, `collection/preflight_gptoss.py`) run the mechanics
on two games and print a table. The collection launchers run them as their gate, so
`collect_dense.sh --smoke-only` is the same check with the chain suppressed.

---

## Checking your environment

```bash
make check      # tests + committed-data verify + hygiene + manifest, all without the deposit
make gate       # additionally checks the collection/steering surface — needs the [gpu] extra
```

If a Tier-2 table disagrees with the committed copy, check in this order:

1. Are you on Python 3.11 with the locked dependency set (`uv sync`)? See
   [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md).
2. Does `make verify-data` pass — i.e. is the *committed* table itself intact?
3. Does the relevant `validate_foundation.py` pass?
4. Is `SCA_DATA_ROOT` pointing at a complete deposit? `python scripts/download_data.py
   --verify-only`.
