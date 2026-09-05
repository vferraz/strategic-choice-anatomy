#!/bin/bash
# Dense substrate arm: preflight gate -> qwen -> qwen_instruct -> llama31_instruct(--chat).
#
# Merged from run_akata_dense_chain.sh during the launcher collapse; the GPU-serialization
# guard, the settle delay and the per-model retry/resume loop are preserved verbatim. The
# preflight was previously a manual step documented in LAUNCH.md — it is now the gate, so a
# broken mechanism costs minutes instead of a GPU-day.
#
# Hardened for unattended overnight running:
#   - settle delay after gpt-oss exits (let CUDA memory free -> avoid orphan-OOM on the qwen 8-bit load)
#   - per-model retry up to 3x on non-zero exit (resumes via --skip-existing on _DONE; no rework)
#   - one model failing never blocks the others
# Resume after a Spark reboot: just re-run this script -> --skip-existing continues.
#
# Usage:
#   bash scripts/launchers/collect_dense.sh                 # preflight gate, then the full chain
#   bash scripts/launchers/collect_dense.sh --smoke-only    # preflight only, then exit (bounded)
# Env: .venv (dense stack). Writes $SCA_DATA_ROOT/substrate/{model}/.
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
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke-only) SMOKE_ONLY=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$SELF"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOGDIR"
CHAIN_LOG="$LOGDIR/dense_chain.log"
EP=collection/generate_dense_substrate.py

log () { echo "$(date -u +%FT%TZ) $*" >> "$CHAIN_LOG"; }

log "chain start; waiting for gpt-oss to free the GPU..."
# Guard matched against the shipped entry-point name, so two GPU jobs never overlap.
until ! pgrep -f '[g]enerate_gptoss_substrate' >/dev/null; do sleep 60; done
log "gpt-oss gone; settling 90s for CUDA memory to free"
sleep 90
log "GPU free -> dense preflight gate"

# Preflight gate: mechanics on two games, no options (see collection/preflight_dense.py).
# Non-zero exit means the mechanism is broken -> never start a GPU-day chain behind it.
.venv/bin/python -u collection/preflight_dense.py >> "$LOGDIR/preflight_dense.log" 2>&1
rc=$?
log "preflight_dense exit=$rc"
if [ "$rc" -ne 0 ]; then
  log "PREFLIGHT FAILED - chain NOT started"
  echo "preflight_dense FAILED (exit $rc); see $LOGDIR/preflight_dense.log" >&2
  exit 1
fi
if [ "$SMOKE_ONLY" -eq 1 ]; then
  log "preflight PASS; --smoke-only -> stopping here"
  echo "preflight_dense PASS"
  exit 0
fi
log "preflight PASS -> starting DENSE chain"

run_model () {  # $1 = python args, $2 = model name
  local tries=0 rc=1
  while [ "$tries" -lt 3 ]; do
    tries=$((tries + 1))
    log "START $2 (try $tries)"
    .venv/bin/python -u "$EP" $1 --skip-existing >> "$LOGDIR/dense_full_$2.log" 2>&1
    rc=$?
    log "END $2 try $tries (exit $rc)"
    [ "$rc" -eq 0 ] && break
    log "$2 exited $rc; retry in 120s (resumes via --skip-existing)"
    sleep 120
  done
  [ "$rc" -ne 0 ] && log "$2 FAILED after 3 tries — continuing to next model"
}

run_model "--model qwen"                    qwen
run_model "--model qwen_instruct"           qwen_instruct
run_model "--model llama31_instruct --chat" llama31_instruct
log "DENSE_CHAIN_DONE"
