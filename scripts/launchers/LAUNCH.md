# Tier-3 launchers

Five gated launchers — one per pipeline arm — for re-collecting the substrate and re-running the
causal steering arms on a single-GPU machine. These are what produced the released deposit.

**Tier 3 only.** If you want figures or tables, you do not need any of this — see
[`../../REPRODUCING.md`](../../REPRODUCING.md).

## Before you start

1. Read [`../../docs/METHODS.md` §8](../../docs/METHODS.md). Four documented gotchas will silently
   change your numbers: the date-stamped gpt-oss harmony template, the deliberate one-block
   off-by-one injection depth, kernel-path determinism, and the locked seeds.
2. Provision the environments — [`../../docs/ENVIRONMENTS.md`](../../docs/ENVIRONMENTS.md).
   `scripts/setup/setup_spark.sh` for the dense stack, `scripts/setup/setup_gptoss_env.sh` for the
   MXFP4 stack. **GPT-OSS will not run correctly in the generic environment.**
3. Point `SCA_DATA_ROOT` at a disk with ~25 GB free:

   ```bash
   export SCA_DATA_ROOT=/mnt/big/sca_data
   ```

   Everything — substrate, activations, run logs — lands under it. Unset, it defaults to
   `<repo>/data_heavy`.

## The five arms

| Launcher | Runs | Env | Rough duration | Writes under `$SCA_DATA_ROOT` |
|---|---|---|---|---|
| `collect_dense.sh` | `preflight_dense` gate → qwen → qwen_instruct → llama31_instruct (`--chat`), 144 games each | `.venv` | ~2–3 days | `substrate/{model}/` |
| `collect_gptoss.sh` | `preflight_gptoss` gate → harmony substrate (144 games) → uniform-site recapture | `.venv_gptoss` | ~1.5–2 days | `substrate/gptoss/`, `gptoss_recap/` |
| `collect_layerc.sh` | Layer-C token attribution (3 models) → Layer-B↔C bridge residuals (3 models) | `.venv` | ~6–8 h | `layerc/{model}/`, `layerc_bridge_residuals/{model}/` |
| `steer_smalldose.sh` | 1-game smoke → deterministic gate → small-dose chain (3 models) | `.venv` | ~2 days | `steering/smalldose/` |
| `steer_perm.sh` | 1-game smoke → deterministic gate → permutation matched-control chain (3 models) | `.venv` | ~2 days | `steering/perm/` |

Durations are for a single DGX-Spark-class GPU and assume nothing else is competing for it.

### Every arm is gated

No launcher starts a GPU-day of work before proving the mechanism on a bounded run first. The
collection arms gate on their preflight (two games, prints a table); the steering arms gate on a
1-game smoke followed by a deterministic validator whose PASS bar includes dose-0 bit-identity
against the capture run. **A failed gate means the chain is not started at all.**

Run just the gate and stop — this is the bounded-smoke entry point, and it is what a GPU
verification pass should use:

```bash
bash scripts/launchers/collect_dense.sh     --smoke-only
bash scripts/launchers/collect_gptoss.sh    --smoke-only
bash scripts/launchers/collect_layerc.sh    --smoke-only     # 1 game of layerc + bridge
bash scripts/launchers/steer_smalldose.sh   --smoke-only
bash scripts/launchers/steer_perm.sh        --smoke-only
```

The three arms that write smoke output to a scratch root also take `--smoke-root DIR`
(default `$SCA_DATA_ROOT/smoke/<arm>`). `--help` on any launcher prints its own header.

### Every arm is resumable

`--skip-existing` continues from the per-game `_DONE` sentinels, so a reboot costs at most one
game; the steering runners skip any mode whose parquet already exists in the out-root. **Re-run the
same command** — there is no separate resume script.

## Conventions

Every launcher:

- **finds the repo root itself** (`git rev-parse --show-toplevel`) — run it from anywhere in a clone;
- uses the **project virtualenv interpreter**, not a system `python3`. The editable install puts the
  repo root on `sys.path`, which is how `collection/` and `steering/` resolve each other; a system
  interpreter has no `.pth` and the imports fail;
- writes to `$SCA_DATA_ROOT/run_logs/<chain>/`;
- **retries** a failed model up to 3× and continues to the next one — one model failing never blocks
  the others;
- opens with a `pgrep` wait loop, matched against the shipped entry-point names, so two GPU jobs
  never overlap on a single-GPU box, followed by a settle delay that lets CUDA memory actually free
  before the next model loads.

> **Do not remove the settle.** Until 2026-09-07 `steer_smalldose.sh` and `steer_perm.sh` were the
> two arms that lacked it, because they were written to be launched individually — yet the
> recommended order below runs them back-to-back. On the reference GB10 that starts a 75 GB 8-bit
> load about a second after the previous arm releases 75 GB, and the box goes into reclaim: the
> load degrades from **128 s** to **~14.8 s/shard**, a ~4 h projection for work that takes two
> minutes. Both arms now carry the same guard and a 90 s settle as the three collection
> launchers. If you chain arms from your own script, keep the same gap between them.

## Recommended order

```bash
# 1. Behaviour + internal state
bash scripts/launchers/collect_dense.sh
bash scripts/launchers/collect_gptoss.sh     # waits for the dense chain to finish

# 2. Token attribution and the bridge
bash scripts/launchers/collect_layerc.sh

# 3. Steering — fit the directions first, then run the gated arms
uv run python steering/extract_directions.py       # -> steering/directions/akata/
uv run python steering/build_perm_directions.py    # -> steering/directions/akata_perm/
bash scripts/launchers/steer_smalldose.sh
bash scripts/launchers/steer_perm.sh
```

Detach a long chain and let it survive your shell:

```bash
setsid nohup bash scripts/launchers/collect_dense.sh >/dev/null 2>&1 </dev/null &
```

Monitor:

```bash
tail -f "$SCA_DATA_ROOT"/run_logs/oneshot_akata/dense_chain.log
```

## Costs

| Stage | Measured |
|---|---|
| dense substrate | ≈ 170 s/game (qwen), ≈ 100 s/game (llama) × 144 games × 3 models |
| gpt-oss substrate | dominated by generation to natural EOS, up to 4,096 new tokens/cell |
| small-dose steering | ≈ 4.7 s/row → ≈ 15.6 h/model, 18,144 rows/model |

## Not shipped

The **saturated-dose** steering arm is not part of this release. Its 1-D dose grid turned out to be
letter-saturated, so the strategic question was never tested by it; see
[`../../docs/METHODS.md` §6.1](../../docs/METHODS.md). `collect_layerc.sh` stops after its own stage
and prints what to start next rather than auto-launching that arm, which is what the private repo's
chains did.
