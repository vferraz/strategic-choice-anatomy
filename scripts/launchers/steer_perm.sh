#!/bin/bash
# Permutation matched-control arm: 1-game smoke -> deterministic gate -> 3-model chain.
#
# Merged from run_perm_chain.sh during the launcher collapse, with the smoke gate added so
# this arm is launched the same way as the small-dose one. The gate mirrors the small-dose
# gate exactly: a 1-game qwen run to a scratch root, then steering/smoke_perm.py, whose
# documented PASS bar includes dose-0 bit-identity against the capture run. On any failure the
# chain is NOT started.
#
# Mode h1_dinc, variants perm0/perm1/perm2, layers 65+79, doses [-0.25..0.25], 54-game sample.
# Resumable: the runner skips any (model,mode) whose parquet already exists in the out-root.
# DO NOT launch before the geometry gate and the permuted directions
# (steering/build_perm_directions.py) have passed — see docs/METHODS.md §6.3.
#
# Usage:
#   bash scripts/launchers/steer_perm.sh                  # gate, then the 3-model chain
#   bash scripts/launchers/steer_perm.sh --smoke-only     # gate only, then exit (bounded)
#   bash scripts/launchers/steer_perm.sh --smoke-root DIR
# Detach with: setsid nohup bash scripts/launchers/steer_perm.sh >/dev/null 2>&1 </dev/null &
# Env: .venv (dense stack). Writes $SCA_DATA_ROOT/steering/perm/.
set -u
# Repo root without requiring git: a Zenodo software archive or a plain tarball has no
# .git, and every path below is repo-relative. Prefer git when it is available.
# Absolute path to this script, resolved BEFORE the cd below so --help can still read it.
SELF="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "${ROOT_DIR:-}" ] || ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$ROOT_DIR"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT

SMOKE_ONLY=0
SMOKE_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke-only) SMOKE_ONLY=1; shift ;;
    --smoke-root) SMOKE_ROOT="${2:?--smoke-root needs a directory}"; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$SELF"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
: "${SMOKE_ROOT:=$SCA_DATA_ROOT/smoke/perm}"

LOG="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOG"
CH="$LOG/perm_chain.log"

# Wait for any GPU job (a collection run, the other steering arm) to clear, then settle.
# Guards matched against the shipped entry-point names, as in the three collection launchers.
# Without this, LAUNCH.md's recommended order -- steer_smalldose.sh then steer_perm.sh -- starts a
# 75 GB 8-bit load within a second of the previous arm releasing 75 GB. Measured on the reference
# GB10: the load degrades from 128 s to ~14.8 s/shard (a ~4 h projection) as the box goes into
# reclaim. Verified 2026-09-07.
until ! pgrep -f '[r]un_smalldose\.py|[r]un_perm\.py|[r]un_saturated\.py|[g]enerate_dense_substrate|[g]enerate_gptoss_substrate|[g]enerate_layerc' >/dev/null; do sleep 20; done
sleep 90

# ------------------------------------------------------------------------------- smoke gate
echo "$(date -u +%FT%TZ) smoke start (1 game, qwen) -> $SMOKE_ROOT" >> "$CH"
.venv/bin/python -u steering/run_perm.py \
  --model qwen --n-games 1 --out-root "$SMOKE_ROOT" >> "$LOG/steer_perm_smoke.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) smoke exit=$rc" >> "$CH"
if [ $rc -ne 0 ]; then
  echo "$(date -u +%FT%TZ) SMOKE FAILED - chain NOT started" >> "$CH"
  exit 1
fi

.venv/bin/python -u steering/smoke_perm.py \
  --smoke-root "$SMOKE_ROOT" >> "$CH" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "$(date -u +%FT%TZ) SMOKE VALIDATION FAILED - chain NOT started" >> "$CH"
  exit 1
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
  echo "$(date -u +%FT%TZ) smoke PASS; --smoke-only -> stopping here" >> "$CH"
  echo "perm smoke PASS -> $SMOKE_ROOT"
  exit 0
fi
echo "$(date -u +%FT%TZ) smoke PASS -> running the full chain" >> "$CH"

# ------------------------------------------------------------------------- 3-model chain
for M in qwen qwen_instruct llama31_instruct; do
  echo "$(date -u +%FT%TZ) start $M" >> "$CH"
  .venv/bin/python -u steering/run_perm.py --model "$M" \
    >> "$LOG/steer_perm_$M.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $M exit=$rc" >> "$CH"
  if [ $rc -ne 0 ]; then
    echo "$(date -u +%FT%TZ) ABORT chain ($M failed)" >> "$CH"
    exit 1
  fi
done
echo "$(date -u +%FT%TZ) PERM_CHAIN_DONE" >> "$CH"
