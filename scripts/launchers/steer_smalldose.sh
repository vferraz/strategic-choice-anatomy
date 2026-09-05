#!/bin/bash
# Small-dose steering arm: 1-game smoke -> deterministic 18/18 gate -> 3-model chain.
#
# Merged from run_smalldose_smoke_then_chain.sh and run_smalldose_chain.sh during the launcher
# collapse. The gate is unchanged: a 1-game qwen run to a scratch root, then
# steering/smoke_smalldose.py, whose documented PASS bar includes dose-0 bit-identity against
# the capture run. On any failure the chain is NOT started.
#
# The merged chain now runs in-process instead of being detached with setsid by the gate — the
# two halves are one file, so detach the whole launcher if you want it to survive your shell:
#   setsid nohup bash scripts/launchers/steer_smalldose.sh >/dev/null 2>&1 </dev/null &
# Resumable: the runner skips any mode whose parquet already exists in the out-root.
#
# Usage:
#   bash scripts/launchers/steer_smalldose.sh                  # gate, then the 3-model chain
#   bash scripts/launchers/steer_smalldose.sh --smoke-only     # gate only, then exit (bounded)
#   bash scripts/launchers/steer_smalldose.sh --smoke-root DIR
# Env: .venv (dense stack). Writes $SCA_DATA_ROOT/steering/smalldose/.
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
: "${SMOKE_ROOT:=$SCA_DATA_ROOT/smoke/smalldose}"

LOG="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOG"
CH="$LOG/smalldose_chain.log"

# ------------------------------------------------------------------------------- smoke gate
echo "$(date -u +%FT%TZ) smoke start (1 game, qwen) -> $SMOKE_ROOT" >> "$CH"
.venv/bin/python -u steering/run_smalldose.py \
  --model qwen --n-games 1 --out-root "$SMOKE_ROOT" >> "$LOG/steer_smalldose_smoke.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) smoke exit=$rc" >> "$CH"
if [ $rc -ne 0 ]; then
  echo "$(date -u +%FT%TZ) SMOKE FAILED - chain NOT started" >> "$CH"
  exit 1
fi

.venv/bin/python -u steering/smoke_smalldose.py \
  --smoke-root "$SMOKE_ROOT" >> "$CH" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "$(date -u +%FT%TZ) SMOKE VALIDATION FAILED - chain NOT started" >> "$CH"
  exit 1
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
  echo "$(date -u +%FT%TZ) smoke PASS; --smoke-only -> stopping here" >> "$CH"
  echo "smalldose smoke PASS -> $SMOKE_ROOT"
  exit 0
fi
echo "$(date -u +%FT%TZ) smoke PASS -> running the full chain" >> "$CH"

# ------------------------------------------------------------------------- 3-model chain
for M in qwen qwen_instruct llama31_instruct; do
  echo "$(date -u +%FT%TZ) start $M" >> "$CH"
  .venv/bin/python -u steering/run_smalldose.py --model "$M" \
    >> "$LOG/steer_smalldose_$M.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $M exit=$rc" >> "$CH"
  if [ $rc -ne 0 ]; then
    echo "$(date -u +%FT%TZ) ABORT chain ($M failed)" >> "$CH"
    exit 1
  fi
done
echo "$(date -u +%FT%TZ) SMALLDOSE_CHAIN_DONE" >> "$CH"
