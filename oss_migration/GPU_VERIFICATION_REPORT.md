# Phase 6 Part B — GPU verification report

**Machine:** the DGX Spark collection box · **Date:** 2026-09-07 · **Branch:** `phase6-spark`
**Source of truth (read-only):** the private `neural-llms-gametheory` working copy @ `4a05270`

> **Internal document.** `oss_migration/` is migration bookkeeping, not part of the published
> repository — the release plan §5 excludes operational/PI-workflow docs. Remove or exclude this
> directory before the repo is made public (release plan §9, Phase 6 hygiene pass).

Covers `oss_migration/PHASE6_gpu_machine.md` steps 1–7 and the SPRINT_q05_gap Part B obligation.

---

## 0. Headline

| Item | Result |
|---|---|
| Part A's "q05 code not recoverable from git history" | **Falsified** — it is `eaded0c` on `origin/main` (§1) |
| q05 extractor CLI provenance proof | **PASS** (§2) |
| `[gpu]` / `[gptoss]` envs build from the new repo's pins | **PASS** (§3) |
| §8.1 Spark-only artifacts staged + checksummed | **PASS**, nothing missing (§4) |
| Test suite | 54/54 base; tier-2 **8 passed / 1 failed** (cross-platform tolerance, §5) |
| Bounded GPU smokes | **a, b, c, d PASS — gate met.** e unverified; 6 defects found + fixed (§6) |

---

## 1. Correction: the q05 producing code IS in git history

`PHASE6_gpu_machine.md` Part A (2026-08-28) concluded:

> "The producing code is **not recoverable from git history** … `git_commit 8d8370e…` … is **not an
> ancestor of HEAD**, and its `extract_causal_directions_oneshot.py` contains **no `q05` string**.
> … **The Spark working tree is the only source.**"

Verified on the Spark after `git fetch origin`:

| Check | Result |
|---|---|
| `git merge-base --is-ancestor 8d8370e HEAD` | **true** — `8d8370e` is `HEAD~2` |
| `q05` count in the extractor at `8d8370e` | **0** — Part A correct |
| `q05` count at `eaded0c` / at `HEAD` | **7** / **7** |
| q05 manifest `created_at` | `2026-08-22T17:01:42+00:00` |
| `eaded0c` commit time | `2026-08-22T17:44:37+00:00` (**+42 min 55 s**) |
| Spark working tree | clean; no stash; `main` == `origin/main` == `4a05270` |

The directions were extracted from a dirty tree on top of `8d8370e` — hence the recorded commit and
the absent `q05` string. 43 minutes later the same edits were committed as `eaded0c`
*"q=0.5 construct-identity steering redo"* and pushed. **Part A was reasoning from the Mac**, whose
checkout predated that push; from there `8d8370e` genuinely was not an ancestor of HEAD.

Consequences: the fallback (reimplement `--belief q05` as a PI-signed change) was **not needed**;
the port below is a faithful application of a pushed commit, not a reconstruction. The Mac's
canonical `oss_migration/PHASE6_gpu_machine.md` still carries the falsified conclusion and should
be amended. Full evidence: `~/phase6_docs/spark_state/14_partA_correction_proof.txt` and
`RECONCILIATION.md`.

There were **no uncommitted q05 tool edits to recover.** What was genuinely unpreserved on the
Spark — one untracked source file and the inventory of the gitignored run outputs — is preserved on
branch `spark-q05-tools` (`f18d651`) in the private repo.

---

## 2. q05 port and the proof obligation

### What was ported (from `eaded0c`, six files)

| Release file | Capability |
|---|---|
| `steering/extract_directions.py` | `--belief {empirical,q05}`; q = 0.5 for **both** players' incentives; `incentive_belief` recorded in the manifest; q05 writes a **suffixed** `delta_tables_{model}_q05.csv` so it can never overwrite the empirical table (the defect `eaded0c` itself fixed) |
| `steering/build_perm_directions.py` | `--dirs / --dirs-perp / --out / --delta-suffix / --models`; permuted control refit on the q05 target; belief + `dirs_root` recorded in the manifest |
| `steering/run_smalldose.py` | `--dirs-root`, `--dirs-perp` |
| `steering/run_perm.py` | `--dirs-perm` |
| `steering/build_perp_directions.py` | **new** — ℓ-orthogonalized `d_inc` / `d_choice_perp`, from `build_q05_perp_directions.py` |

Applied verbatim modulo the established rewrite classes: roots resolved through
`strategic_anatomy.config` (`steering_root()`, `results_root()`, `manifests_root()`,
`substrate_root()`), **no `sys.path` manipulation** anywhere in `steering/`, and the existing guards
preserved (notably the prompt_hash gate pairing generation moves to the residual substrate).

> Release plan §4.4 states that letter-orthogonalization "lives inside `run_akata_steer_smalldose.py`
> + `build_perm_directions.py` — no separate file to hunt for". That is no longer true for the q05
> arm: `eaded0c` introduced a standalone builder. §4.4 is outdated on this point.

### Bug found and fixed during the port

`build_perm_directions.py` carried a phase-4 `__main__`-block `ArgumentParser` that took no
arguments, purely so `--help` printed help. With a real parser now inside `main()`, that stub parsed
**first** and rejected every new option (`error: unrecognized arguments: --delta-suffix _q05`). The
stub is removed and its docstring/epilog moved onto the real parser. Audited the rest of the tree
for the same pattern: only `collection/preflight_dense.py` and `collection/preflight_gptoss.py` use
it, and both genuinely take no options — no shadowing there.

### Proof: does the ported CLI reproduce the shipped q05 directions?

Per Part A the obligation is **provenance for the extractor CLI**, not re-deriving the numbers
(`analysis/steering/apparatus_geometry_q05.py` already reproduced them on the Mac at cos = 1.000000).

Invocation — reproducing the shipped run's configuration (`decoder_source: substrate_slot_argmax`,
so no `--moves-root`; `data_root: output/oneshot_akata_main/{model}`):

```bash
SCA_DATA_ROOT=~/sca_data_root SCA_RESULTS_ROOT=~/sca_scratch/results \
.venv/bin/python steering/extract_directions.py \
    --belief q05 --models qwen,qwen_instruct,llama31_instruct \
    --out-root ~/sca_scratch/proof_q05          # scratch — never over data/
```

Preflight: **144/144 games, 80 capture layers (0–79), all three models.**

Reference for the comparison is the **release-committed**
`data/manifests/directions/akata_q05/{model}/manifest.json`, which carries `vector_sha256` per
direction — a stricter check than cosine and independent of the Spark's npz. Those committed
manifests are byte-identical (`diff` clean) to the Spark's local copies.

**Result: PASS.** The stated obligation — `cos = 1.000000` for `d_inc` at L65/L79 across 3 models —
is met, and met at the stricter sha256 level:

| model | `steer_layers` | sha256 vs committed manifest | `d_inc_l65` | `d_inc_l79` |
|---|---|---|---|---|
| `qwen` | `[30, 65, 75, 79]` | 797 / 799 | cos 1.000000, sha256 ✓ | cos 1.000000, sha256 ✓ |
| `qwen_instruct` | `[30, 65, 75, 79]` | **799 / 799** | cos 1.000000, sha256 ✓ | cos 1.000000, sha256 ✓ |
| `llama31_instruct` | `[30, 50, 65, 79]` | **799 / 799** | cos 1.000000, sha256 ✓ | cos 1.000000, sha256 ✓ |

`incentive_belief`, `n_games_substrate` (144) and `steer_layers` match the committed manifests for
all three models. The regenerated npz carry 801 arrays vs the manifest's 799 hashed vectors; the two
extras are the `layers` and `steer_layers` index arrays, which the shipped npz also carry — not a
discrepancy.

**The two `qwen` exceptions are float32 last-ULP noise, not a difference in the fit:**

| key | cos(regenerated, shipped) | max abs elementwise Δ | diagnostics |
|---|---|---|---|
| `d_opp_l1` | `1.000000000000` | `3.836e-16` | identical (`auc_groupkfold = 0.6011431383911067`) |
| `d_opp_l3` | `1.000000000000` | `1.966e-16` | identical (`auc_groupkfold = 0.5651249463793001`) |

A Δ of ~2–4e-16 is one float32 ULP at a coordinate of magnitude ~6e-9 — i.e. a handful of
near-zero components of the residual rounding differently under a different BLAS/numpy build
(collection ran `numpy 2.4.4`; this env resolves `2.5.3`). The per-direction AUC diagnostics are
**bit-identical**, so the regression itself is identical. `d_opp` is also not a steered direction in
the released small-dose arm (`h1_dinc` steers `d_inc`, `h2_choice` steers `d_choice_perp`); it is
`h3_oppinc`, which belongs to the superseded saturated grid. Only 4 of 80 `d_opp` keys are estimable
at all (l1, l2, l3, l79); the other two of those four are bit-exact.

This is **not** a stop condition: the documented trigger is `cos != 1.000`, and cosine is 1.000000
to twelve decimal places.

### Independent corroboration

The regenerated `delta_tables_qwen_q05.csv` is **byte-identical** (sha256
`4323726127c4308e…`, 145 lines) to the copy committed at
`data/results/steering/summary/delta_tables_qwen_q05.csv`. That table is the q = 0.5 incentive
construct itself, so the ported `--belief q05` path reproduces the shipped target exactly,
independently of the direction fit.

### Override wiring (runtime-verified, no model load)

With `run()` stubbed, the CLI overrides reach the direction loader and the roots they name open
with the expected keys:

```
defaults  : akata | akata_perp | akata_perm
after CLI : akata_q05 | akata_q05_perp | akata_q05_perm
  akata_q05        801 keys, has d_inc_l65
  akata_q05_perp   160 keys, has d_inc_l65
  akata_q05_perm     7 keys, has d_inc_perm0_l65
```

---

## 3. Environments

Both GPU environments were created **from this repo's pins**, on the new clone:

```bash
uv venv --python 3.12 .venv         && uv pip install -e ".[analysis,gpu,dev]"
uv venv --python 3.12 .venv_gptoss  && uv pip install -e ".[analysis,gpu,gptoss,dev]"
```

Both installs succeeded and reproduce the live collection environments exactly:

| | new `[gpu]` env | live `.venv` | new `[gptoss]` env | live `.venv_gptoss` |
|---|---|---|---|---|
| python | 3.12.3 | 3.12.3 | 3.12.3 | 3.12.3 |
| torch | 2.11.0+cu130 | 2.11.0+cu130 | 2.11.0+cu130 | 2.11.0+cu130 |
| transformers | 5.5.0 | 5.5.0 | 5.5.0 | 5.5.0 |
| kernels | — | — | 0.12.3 | 0.12.3 |
| openai-harmony | — | — | 0.0.8 | 0.0.8 |
| triton | 3.6.0 | 3.6.0 | 3.6.0 | 3.6.0 |
| `torch.cuda.is_available()` | True (NVIDIA GB10) | — | True (NVIDIA GB10) | — |

`pyproject.toml` `[gpu]`/`[gptoss]` changed from floors to exact pins; `uv.lock` regenerated
(torch 2.13.0→2.11.0, transformers 5.15.0→5.5.0, triton 3.7.1→3.6.0).

### Deviation: Python 3.12 for the GPU arms, not 3.11

`PHASE6_gpu_machine.md` step 3 and `.python-version` say 3.11. The pinned `torch 2.11.0` ships on
this platform as a **`cp312` `manylinux_2_28_aarch64` wheel** and does not install on 3.11 — so
pinning the versions the science actually ran on requires 3.12 for `[gpu]` and `[gptoss]`.
`.python-version` stays 3.11 because the default path (Tier 1/2, CI) is the analysis arm.
Documented in `docs/ENVIRONMENTS.md`; `README.md` was also corrected, since it previously showed the
GPU extra being installed into a 3.11 venv, which now fails.

### Deviation: base libraries stay floored

`numpy` / `pandas` / `pyarrow` / `scipy` are shared with `[analysis]` and legitimately differ per
arm — analysis was verified on macOS at `numpy 2.4.6 / pandas 3.0.5 / pyarrow 25.0.1`, the Spark's
live collection env ran `2.4.4 / 3.0.2 / 23.0.1`, and a fresh 3.12 build here resolves
`2.5.3 / 3.0.5`. Hard-pinning them in the shared `dependencies` block would make the arms mutually
unsatisfiable for no scientific gain; the Tier-2 hash tests guard the numerics instead. Full freezes
of both live environments: `~/phase6_docs/spark_state/env_venv_{dense,gptoss}_freeze.txt`.

### Hardware

| | |
|---|---|
| GPU | 1× NVIDIA GB10 (DGX Spark) |
| Unified memory | ~130 GB (121 GiB visible) |
| Driver | 580.173.02 |
| CUDA | 13.0 (V13.0.88) |
| Arch / OS | aarch64, Linux 6.17 |

---

## 4. Spark-only artifact sync (PHASE6 step 1 / plan §8.1)

**Every §8.1 path existed on this machine — no release-blocking gap.** Staged into `_deposit/`
(added to `.gitignore`; it was not covered before) and checksummed to
`_deposit/CHECKSUMS.partial.sha256`.

| Deposit path | Source (old repo) | Files | Size |
|---|---|---|---|
| `steering/directions/akata/` | `analysis/block_b/tables/causal_oneshot/directions_akata/` | 7 | 61 M |
| `steering/directions/akata_perp/` | `…/directions_akata_perp/` | 6 | 388 K |
| `steering/directions/akata_perm/` | `…/directions_akata_perm/` | 6 | 580 K |
| `steering/perm/` | `output/oneshot_akata_steer_perm/` | 3 parquet | 92 K |
| `steering/saturated/` | `output/oneshot_akata_steer/` | 6 parquet | 216 K |
| | **total** | **28** | **62 M** |

Every staged file was byte-compared against its source (`diff -r` / `cmp`): **all identical**.
`sha256sum -c` on the manifest passes.

The small `manifest.json`s were additionally copied into **tracked**
`data/manifests/directions/{akata,akata_perp,akata_perm}/` (plus `akata/extract_summary.json`) —
the release previously shipped only the `akata_q05*` set.

**q05 heavy data was deliberately not synced.** `PHASE6_gpu_machine.md` states it is already on the
Mac, and the byte-identical committed q05 manifests corroborate the provenance chain. Whether the
Mac holds the npz *payloads* is not determinable from this machine and is not asserted here.

---

## 5. Test suite

All runs used `SCA_DATA_ROOT=~/sca_data_root` (the symlink farm, §5.1) in the `[gpu]` env.

| run | result |
|---|---|
| `pytest -q -m "not tier2 and not gpu"` | **54 passed**, 9 deselected, 14 warnings, 7.48 s |
| `pytest -q -m tier2` (before the residual cache existed) | **7 passed, 2 skipped**, 54 deselected, 13.44 s |
| `pytest -q -m tier2` (after building the cache) | **8 passed, 1 failed**, 0 skipped, 25m53s |
| `--help` sweep over every collection/steering entry point | **26 / 26 pass** in the declaring env |

The 2 skips were `test_recap_cache_invariants` and `test_recap_layer0_anchor_is_invariant`, which
need the Layer-B residual cache. That cache was then built and **is present** at
`$SCA_DATA_ROOT/layer_b_cache` (5.2 GB: `baseline/`, `cues/`, `recap_baseline/`), rebuilt from the
substrate in 27 s of wall time across the three models
(`qwen` 9 s, `llama31_instruct` 12 s, `gptoss` 6 s; 576 baseline rows × 8192 dims × 81 layers for the
dense models, 4032 cue rows at the final layer across 7 conditions). The re-run with the cache present
collected all 9: **8 passed, 1 failed**, 0 skipped, 25m53s. Both recap invariants — including the
46,080-array integrity check — passed.

#### The one tier-2 failure — `test_b1_tables_regenerate`

```
AssertionError: auc: max rel 1.448e-04 > 1e-09
```

`RTOL = 1e-9` is effectively bit-identity. The regenerated AUC column agrees with the committed
`b1_decodability.csv` to about four significant figures, not nine. That table was built on the Mac;
this run regenerated it here:

| | Spark (this env) | Mac (§1 above) |
|---|---|---|
| platform | aarch64 Linux | macOS arm64 |
| python | 3.12.3 | 3.11.14 |
| numpy | **2.5.3** | 2.4.6 |
| scipy | **1.18.1** | 1.17.1 |
| scikit-learn | 1.8.0 | 1.8.0 (pinned, identical) |

**Deliberately not fixed.** The documented stop condition says a tolerance failure is likely an
environment-pin difference, to report exact versions and *not* chase it by editing science code;
loosening `RTOL` or touching the builder would be exactly that. `PHASE6_gpu_machine.md` step 5 asks
this run to "report any float-tolerance differences vs the Mac", so this is the check working, not a
defect. No reported claim moves — the decode AUCs are unchanged at reporting precision
(`canonical_action = 0.866`, `sign_delta1c = 0.869`, …).

### The symlink farm (§5.1)

`$SCA_DATA_ROOT=~/sca_data_root` maps the deposit layout onto the private repo's research layout:

| deposit path | → |
|---|---|
| `substrate/` | `output/oneshot_akata_main/` (already `{model}/{game}/`) |
| `gptoss_recap/`, `layerc/`, `layerc_bridge_residuals/` | the matching `output/oneshot_akata_*` roots |
| `steering/{smalldose,smalldose_q05,perm,perm_q05,saturated}` | the matching `output/oneshot_akata_steer*` roots |
| `steering/directions/akata{,_perp,_perm}{,_q05,_q05_perp,_q05_perm}` | `analysis/block_b/tables/causal_oneshot/directions_*` |
| `layer_b_cache/` | **a real, writable directory on scratch — deliberately not a symlink** |

That last row follows the PHASE6 §4 warning: `build_residual_cache.py` and
`fusion_associative/recap_cache.py` are *writers*, and the Phase-3 recipe pointed a writable name at
read-only source material. Read-only roots stay symlinks; nothing in this session wrote into the
private repo, verified by `git status` there remaining clean throughout.

---

## 6. Bounded GPU smokes

Run serially, `--smoke-only` throughout, `SCA_DATA_ROOT=~/sca_data_root`,
`SCA_RESULTS_ROOT=~/sca_scratch/results`. Every arm writes only to
`$SCA_DATA_ROOT/{akata_preflight, smoke/<arm>}` — new real directories, so no smoke can write into
the private repo.

| smoke | result | wall |
|---|---|---|
| a. `collect_dense.sh --smoke-only` | **PASS** | 10m09s |
| b. `collect_gptoss.sh --smoke-only` | **PASS** | 10m37s |
| c. `steer_smalldose.sh --smoke-only` | **PASS — 20/20 assertions** | 27m45s |
| d. `steer_perm.sh --smoke-only` | **PASS — 10/10 assertions** | 15m12s |
| e. `collect_layerc.sh --smoke-only` | **FAILED ×3 on real defects; all fixed; never completed** | — |
| f. `--help` sweep, both GPU envs | **PASS — 26/26** | — |

**The PHASE6 gate condition ("smokes a–c PASS against their built-in validators") is met**, and arm d
passed as well. **Arm e is unverified** (§6.1): its four blocking defects are fixed and committed, but
this session never observed it complete end-to-end, and no claim is made that it does.

### d. Permutation-control gate — expected vs observed

**10/10 PASS, 0 FAIL**, `SMOKE_VALIDATION_PASS`, on `h1_dinc` with variants `perm0`/`perm1`/`perm2`.
Same three properties as arm c: dose-0 identical across layers/variants (nunique per cb
`{0:1, 1:1, 2:1, 3:1}`), dose-0 slot pref equal to the substrate capture at 4 dp, dose-0 realized
letter equal to the substrate `move_letter`; injection alive at max |Δpref| = 0.2718; parse rate
1.000; 168 rows. It loads `directions_akata_perm`, so it also exercises the npz staged into
`_deposit/` (§4) rather than merely checksumming it.

### 6.1 Arm e, and the six defects the smokes surfaced

Arm e failed three times on three *different* real defects, each fixed in turn; the fourth attempt
got past all of them into `generate_layerc.py` and was interrupted mid-model-load. **Unverified.**

None of the six below is reachable by `pytest`, a plain import, or the `--help` sweep — every one
fails only when the code path executes. That is what these smokes are for.

| # | file | defect | severity |
|---|---|---|---|
| 1 | `collection/layerc/model_head.py` | missing `from pathlib import Path` → `NameError` L52 | **Layer-C arm dead** |
| 2 | " | missing `import json` → `NameError` L64 | " |
| 3 | " | missing `import numpy as np` → `NameError` L127 (L120/124 are annotations, lazy under `from __future__ import annotations`) | " |
| 4 | " | `@dataclass` dropped from `ModelHead` → `TypeError: ModelHead() takes no arguments` | " |
| 5 | `analysis/layer_a/decision_state.py` | missing `import pandas as pd` → `NameError` L34 | call-time failure |
| 6 | `collection/genutils.py` | `_make_move_stopper` ends at its `ClassDef`, never returns `StoppingCriteriaList([_MoveStop()])` — returns `None` | latent |

Defects 1–4 meant **both** Layer-C collectors (`generate_layerc.py:161` and
`capture_bridge_residuals.py:178` call `load_model_head`) could never have run in this repository.
Defect 6 is latent: nothing here calls `_make_move_stopper` today, but wired up as the private repo
wires it (`validate_readout_oneshot.py:282`) the caller silently loses early stopping and pays the
full `max_new_tokens` budget on every generation.

All six share one root cause: modules whose docstrings say symbols were "extracted verbatim" from
the private repo carried the bodies but not the enclosing imports, and in one case not the decorator.
`dataclass` was imported but unused, so an unused-import check would not have caught #4 either.

Found systematically rather than one error at a time:

* an AST scan for names used but never bound — now **0 unresolved** across the tracked tree;
* a symbol-by-symbol diff of every module claiming a verbatim extraction against its cited source —
  now clean for all seven, with one intended exception (`collection/model_setup.py::_setup_model`
  carries the deliberate `src.run_sim_spark` → `strategic_anatomy.runtime` rewrite).

### 6.2 Launcher defect: the missing settle

`LAUNCH.md` §Conventions asserted that **every** launcher opens with a `pgrep` guard followed by a
settle delay. True of the three collection launchers; false of `steer_smalldose.sh` and
`steer_perm.sh` — which §Recommended order runs back-to-back, both as `--smoke-only` and as the two
~2-day chains.

Measured here: a 75 GB 8-bit load starting ~1 s after the previous arm released 75 GB drives the box
into reclaim and degrades from **128 s for the whole load** to **14.79 s/shard** — a ~4 h projection
for two minutes of work, ~114× on identical work. Re-running the same arm behind a settle: 963 shards
in 2:04, model ready in 128 s.

Both arms now carry the same guard and 90 s settle as their siblings, and `LAUNCH.md` records why so
the delay is not optimised away. The guards use the `[r]un_perm\.py` bracket form deliberately: a
bare pattern matches the searching process's own command line and deadlocks.

### a. Dense preflight — expected vs observed

Expected: render → greedy generate → J/P parse → residual at the `A: Option` slot at all layers,
2 games × 4 cb × 3 dense models; non-zero exit means the mechanism is broken.

Observed: exit 0. **24/24 cells** (`qwen`, `qwen_instruct`, `llama31_instruct` × `AsBa`, `CmDl` × cb0–3).
8-bit load verified structurally (`Linear8bitLt` at `model.layers.0.self_attn.q_proj`);
`llama31_instruct` loaded in 126 s with J/P token ids `{'J': 622, 'P': 393}`;
**`n_resid_layers = 81` on every row**. Wrote `akata_preflight/dense_preflight.csv`,
`PREFLIGHT_DENSE_DONE`.

### b. GPT-OSS preflight — expected vs observed

Expected: harmony template applies, per-cell classify runs, `router.npz` keys present; aborts up
front if `kernels` is missing.

Observed: exit 0, **8/8 cells**, `commit_present = True` on every one. The `kernels` guard never
tripped — the pinned `kernels 0.12.3` / `triton 3.6.0` stack is live. `quantization_config: None`
(native MXFP4), **router 14 layers, residual 36 layers, top_k 4**. gpt-oss-120b loaded in 404 s
occupying **65.4 GB** (the doc's "65 GB of weights"); the GB10 reported **130.7 GB free / 130.7 GB
total**, corroborating the ~130 GB unified figure in §3. Generation ran to natural EOS,
208–1216 new tokens per cell. `PREFLIGHT_GPTOSS_DONE`.

### c. Small-dose steering gate — expected vs observed

Expected (doc): "the documented PASS bar is 18/18 incl. dose-0 bit-identity vs capture".

Observed: **20/20 PASS, 0 FAIL** — the validator runs 10 assertions per mode across `h1_dinc` and
`h2_choice`, i.e. two more than the doc records. `SMOKE_VALIDATION_PASS`.

| assertion (both modes) | observed |
|---|---|
| exactly one parquet; 168 rows; 24 cells × 7 doses | PASS |
| variants `main` / `random` / `main_perp` present | PASS |
| all status OK; parse rate ≥ 0.90 | **1.000** both modes |
| **dose-0 identical across layers/variants** | nunique per cb = `{0:1, 1:1, 2:1, 3:1}` |
| **dose-0 slot pref == substrate capture (4 dp)** | PASS |
| **dose-0 realized letter == substrate `move_letter`** | PASS |
| injection alive (max \|Δpref\| at dose ≠ 0) | 0.3911 (`h1_dinc`), 0.1398 (`h2_choice`) |

So on this machine the steering harness reproduces the released capture bit-for-bit at zero dose and
demonstrably moves the model off it at non-zero dose. Qwen-72B 8-bit occupied 75.4 GB; throughput
**4.58 s/row** (`h1_dinc`) and **4.45 s/row** (`h2_choice`) against the documented ≈ 4.7 s/row.

### Independent determinism checks (not part of the launchers' gates)

Both collection arms were checked against the *released* substrate, not just internally:

| arm | check | result |
|---|---|---|
| dense | preflight `prompt_sha` vs released `prompt_hash` | **24/24 match** |
| gpt-oss | `pin_prompt_to_stored` vs stored `prompt_hash` | **16/16 reproduced** |

The gpt-oss prompts do **not** match by direct comparison, and must not: the harmony chat template
injects a live `Current date:` line, so a prompt built today differs from the original by
construction. The reproduction path is `collection/recapture_gptoss_transition.py::pin_prompt_to_stored`,
which brute-forces the date candidates until `prompt_sha256` equals the stored hash and **fails the
row rather than regenerating it** if none match. All 16 rows checked (4 games × 4 cb) pinned
successfully, splitting **8× `2026-06-25` / 8× `2026-06-26`** — matching the code's note that the
original collection crossed UTC midnight. `transformers 5.5.0` therefore reproduces the original
harmony template byte-for-byte on this machine.

### To finish arms d and e

```bash
export SCA_DATA_ROOT=~/sca_data_root SCA_RESULTS_ROOT=~/sca_scratch/results
bash scripts/launchers/steer_perm.sh     --smoke-only   # needs directions_akata_perm (staged)
bash scripts/launchers/collect_layerc.sh --smoke-only
.venv/bin/python -m pytest -q -m tier2                  # 9 collected, cache present
```

## 7. Open items

- **D1** in `oss_migration/SPRINT_q05_gap.md` (ship both arms' heavy data, or the q05 heavy data
  plus the empirical tables only) — untouched. Both arms' parquets are staged so either choice is
  available; shipping is Vinicius's call.
- The Mac's `oss_migration/PHASE6_gpu_machine.md` Part A note needs the §1 correction.
- Release plan §4.4's "no separate perp file to hunt for" is outdated (§2).
- `c902bcd`, named as the Mac tip, resolves in no reachable repo (local, `origin`, or the release
  repo). Reconciliation commands for the Mac are in `~/phase6_docs/spark_state/RECONCILIATION.md`.
- Deposit staging is `_deposit/` on this machine only; Phase 7 assembles and uploads it.
- **Arm e (`collect_layerc`) is unverified.** Its four blocking defects are fixed; nobody has yet
  watched it run to completion. One command, ~4 minutes:
  `SCA_DATA_ROOT=~/sca_data_root SCA_RESULTS_ROOT=~/sca_scratch/results bash scripts/launchers/collect_layerc.sh --smoke-only`
- **The tier-2 `auc` tolerance (§5)** is an open decision, not a bug: accept a documented
  cross-platform tolerance for `b1_decodability`, or regenerate the committed table on the
  collection machine. Not decided here.
- **Provenance commit `1f47050` resolves nowhere** — not locally, not on the private remote
  (`gh api` → 422) — yet **11 release modules cite it** as the source of their verbatim extractions
  (`model_head.py`, `genutils.py`, `layerc_spec.py`, `model_setup.py`, `prompting.py`,
  `decision_state.py`, `steer_core.py`, `extract_directions.py` and three `layer_a/frozen/` figure
  scripts). The §6.1 extraction diffs were therefore run against the private repo's current working
  tree (`4a05270`), not the cited commit. Worth resolving before release: these citations are the
  release's provenance record.
- Four background-task terminations during this session were **not diagnosed**. Distinct from §6.2:
  the machine was healthy at each (>110 GB free, no kernel OOM at those timestamps) and the durations
  had no pattern (48 min, ~2 min, ~1 min, ~3 min). Recorded as unexplained rather than attributed.
