#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Full DENSE Akata collection chain: qwen -> qwen_instruct -> llama31_instruct(--chat), 144 games each.
# Waits for the gpt-oss smoke to free the single GPU, then runs the 3 models serially.
# Hardened for unattended overnight running:
#   - settle delay after gpt-oss exits (let CUDA memory free -> avoid orphan-OOM on the qwen 8-bit load)
#   - per-model retry up to 3x on non-zero exit (resumes via --skip-existing on _DONE; no rework)
#   - one model failing never blocks the others
# Resume after a Spark reboot: just re-run this script (or the per-model command) -> --skip-existing continues.
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOGDIR"
CHAIN_LOG="$LOGDIR/dense_chain.log"
EP=collection/generate_dense_substrate.py

log () { echo "$(date -u +%FT%TZ) $*" >> "$CHAIN_LOG"; }

log "chain start; waiting for gpt-oss smoke to free the GPU..."
until ! pgrep -f '[g]enerate_oneshot_akata_gptoss' >/dev/null; do sleep 60; done
log "gpt-oss gone; settling 90s for CUDA memory to free"
sleep 90
log "GPU free -> starting DENSE chain"

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
