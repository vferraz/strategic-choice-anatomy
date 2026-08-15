#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Full gpt-oss Akata collection: waits for the DENSE chain to finish, then runs the validated
# harmony entrypoint over all 144 games -> "$SCA_DATA_ROOT"/substrate/gptoss/.
# Hardened: +90s GPU settle after dense; retry up to 3x on crash; resumable (--skip-existing on _DONE,
# so the 2 smoke games AsAs/AsBa are skipped). Runs under .venv_gptoss (kernels). ~1.3-1.5 days.
# Resume after reboot: re-run this script (or the python --skip-existing directly).
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOGDIR"
CHAIN_LOG="$LOGDIR/gptoss_chain.log"
DENSE_LOG="$LOGDIR/dense_chain.log"
EP=collection/generate_gptoss_substrate.py
log () { echo "$(date -u +%FT%TZ) $*" >> "$CHAIN_LOG"; }

log "waiting for the DENSE chain to finish before collecting gpt-oss..."
until grep -aq 'DENSE_CHAIN_DONE' "$DENSE_LOG" 2>/dev/null || ! pgrep -f '[r]un_akata_dense_chain' >/dev/null; do
  sleep 120
done
log "dense done; settling 90s for CUDA memory to free"
sleep 90
log "starting gpt-oss full collection (.venv_gptoss)"

tries=0; rc=1
while [ "$tries" -lt 3 ]; do
  tries=$((tries + 1))
  log "START gptoss (try $tries)"
  .venv_gptoss/bin/python -u "$EP" --skip-existing >> "$LOGDIR/gptoss_full.log" 2>&1
  rc=$?
  log "END gptoss try $tries (exit $rc)"
  [ "$rc" -eq 0 ] && break
  log "gptoss exited $rc; retry in 180s (resumes via --skip-existing)"
  sleep 180
done
log "GPTOSS_CHAIN_DONE (exit $rc)"
