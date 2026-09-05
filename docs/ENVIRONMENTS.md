# Environments

Three environments, one per reproducibility tier. Only the first is needed to rebuild figures and
regenerate tables.

| Environment | Extra | Needed for | Platform |
|---|---|---|---|
| **analysis** | `[analysis,dev]` | Tier 1 (figures) and Tier 2 (tables) | any; developed on macOS arm64 and Linux |
| **gpu** | `[analysis,gpu,dev]` | Tier 3 dense collection and steering | Linux + CUDA |
| **gpt-oss** | `[analysis,gpu,gptoss,dev]` | Tier 3 GPT-OSS collection and router capture | Linux + CUDA |

`uv.lock` in the repository root pins the full resolved dependency graph. `.python-version` pins
**3.11**.

---

## 1. Analysis environment (CPU)

```bash
uv venv --python 3.11
uv pip install -e ".[analysis,dev]"
```

`[dev]` supplies `pytest`, so `make test` works; installing `[analysis]` alone gives you the
libraries but not the test runner.

The reference environment this release was verified in — Python **3.11.14**, macOS arm64:

| package | version |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| pyarrow | 25.0.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 (**pinned**, see below) |
| matplotlib | 3.11.1 |
| statsmodels | 0.14.6 |
| shap | 0.51.0 |
| nashpy | 0.0.43 |
| adjustText | 1.4.0 |

Reproduce the exact set with `uv sync` against the committed `uv.lock`, or inspect it with
`uv pip freeze`.

> **Float tolerance.** Tier-2 regeneration is checked against the committed tables by hash where
> possible, and otherwise on parsed values at `rtol = 1e-9`. Different BLAS builds can move the last
> couple of digits; a genuine numeric difference is a bug, not a tolerance question. See
> `REPRODUCING.md`.

### The two version sensitivities, and what was done about each

**scikit-learn is pinned to `==1.8.0`** — the only `==` pin in the project. Four committed Layer-A
tables reproduce byte-for-byte on 1.8.0 and differ on 1.9.0: `s_attribution_shap.csv`,
`s_shap_beeswarm.csv`, `s_shap_delta1_dependence.csv`, `s_variance_partition.csv`. The mechanism is
SHAP tie ordering, not a numerical error, and the differences are small (largest observed:
`auc_drop` in `s_variance_partition.csv`, 7.7% relative). Verified 2026-08-28: under 1.8.0 all four
are sha256-identical to the committed copies, on numpy 2.4.6 / pandas 3.0.5 — i.e. the sensitivity is
scikit-learn's alone and does not travel with the rest of the stack. **The pin exists for byte-level
reproduction of the released tables, not for correctness**; a newer scikit-learn gives materially the
same answers, and anyone extending this work should feel free to unpin it.

**matplotlib is deliberately not pinned.** Reference figure PDFs were rendered on **3.10.8**; the
locked environment ships 3.11.1. PDF bytes never match anyway (matplotlib stamps a creation date),
and text metrics shift between minor versions, so figures are compared by pixels rather than bytes —
`make figures-diff`. 3.10.8 is recorded here as the reference-render version so a mismatch can be
attributed rather than investigated. See `REPRODUCING.md` for the per-figure pixel results.

---

## 2. GPU environment (dense collection and steering)

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[analysis,gpu,dev]"
```

Adds `torch`, `transformers`, `accelerate`, and `bitsandbytes` (Linux only — there are no macOS
arm64 wheels, so the dependency is marked `sys_platform == 'linux'` and 8-bit loading is
unavailable on macOS).

Reference: `torch 2.13.0`, `transformers 5.15.0`.

The three dense models load 8-bit via `collection/model_setup.py::_setup_model`, which installs the
CPU-first bitsandbytes patch and disables the HF allocator warmup. A raw `from_pretrained`
OOM-kills the 8-bit load on the reference hardware — use `_setup_model`, not `from_pretrained`.

`scripts/setup/setup_spark.sh` provisions this environment on a DGX-Spark-class machine, and
`scripts/setup/smoke_spark.sh` is its smoke test.

> **The injection-depth convention is fixed across transformers versions.** `METHODS.md` §8.2
> documents a deliberate one-block off-by-one between the capture key and the steer hook. It is
> **kept** for comparability across all arms. If your `transformers` version changes hook
> semantics, the correct action is to reproduce the documented offset, not to "fix" it.

---

## 3. GPT-OSS environment (MXFP4)

```bash
uv venv --python 3.11 .venv_gptoss
uv pip install -e ".[analysis,gpu,gptoss,dev]"
```

Adds `kernels`, `openai-harmony`, and `triton` (Linux). `scripts/setup/setup_gptoss_env.sh`
provisions it.

**This environment is not optional for GPT-OSS.** A generic analysis or GPU environment lacks the
`kernels` package, MXFP4-fallbacks, and **changes the router code path** — see `METHODS.md` §8.3 and
HC-7. The collectors hard-error rather than run in that state:

```
kernels missing — run under the gpt-oss env (docs/ENVIRONMENTS.md; docs/METHODS.md HC-7).
```

`--allow-no-kernels` bypasses the guard, but results produced that way are **not** comparable to
the released substrate. The `capture_mode` actually used is stamped into every `router.npz`.

CPU-only analysis of already-saved GPT-OSS arrays may use the analysis environment, as long as it
does not import or instantiate the model.

---

## 4. Hardware

The released substrate was collected on a single **NVIDIA GB10 (DGX Spark), ~130 GB unified
memory**, memory-bandwidth-bound. Measured throughput:

| stage | cost |
|---|---|
| dense substrate | ≈ 170 s/game (qwen), ≈ 100 s/game (llama) — 144 games × 3 models |
| gpt-oss substrate | dominated by generation to natural EOS at up to 4,096 new tokens |
| steering, small-dose arm | ≈ 4.7 s/row → ≈ 15.6 h/model, 18,144 rows/model |

Tier 3 is days of GPU time. Tier 1 and Tier 2 need no GPU at all.

---

## 5. Exact GPU pins

The `[gpu]` and `[gptoss]` extras carry lower bounds, and `uv.lock` pins the resolved analysis
graph. The **exact** versions used on the collection machine — including the CUDA build, driver, and
the MXFP4 kernel stack — are not resolvable from this checkout; they are recorded in
`oss_migration/GPU_VERIFICATION_REPORT.md` when the GPU-side verification runs.

If you are re-collecting, pin from that report rather than from the lower bounds here: the
numerics guidance in `METHODS.md` §8.3 exists precisely because the kernel path is not
interchangeable.
