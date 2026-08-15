#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Opportunistic Layer B<->C bridge capture, WITHOUT killing steering.
# Waits until steering has stopped ON ITS OWN (chain + python both gone), then runs the Akata bridge
# capture (3 dense models, baseline-only, ~25 min), then RESUMES steering if it was incomplete.
# It never kills anything — it only acts in a GPU-free window.
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata_layerc; mkdir -p "$LOGDIR"
ORCH="$LOGDIR/bridge_then_steer.log"
EP=collection/layerc/capture_bridge_residuals.py
log(){ echo "$(date -u +%FT%TZ) $*" >> "$ORCH"; }

log "armed; waiting for steering to free the GPU (no kill)"
# GPU genuinely free = steering chain gone AND steer python gone AND no other akata GPU job
until ! pgrep -f '[r]un_akata_steer_chain' >/dev/null \
   && ! pgrep -f '[r]un_akata_steer\.py' >/dev/null \
   && ! pgrep -f '[g]enerate_oneshot_akata' >/dev/null \
   && ! pgrep -f '[c]apture_residuals_bridge' >/dev/null; do sleep 120; done
sleep 90
log "GPU free -> Akata bridge capture (3 dense models, baseline-only)"

runb(){ local n=0 rc=1; while [ $n -lt 2 ]; do n=$((n+1)); log "START bridge $1 $2 (try $n)"
  .venv/bin/python -u "$EP" --model "$1" $2 --load_8bit --skip-existing >> "$LOGDIR/bridge_$1.log" 2>&1; rc=$?
  log "END bridge $1 try $n (exit $rc)"; [ $rc -eq 0 ] && break; sleep 30; done; }

runb qwen ""
runb qwen_instruct ""
runb llama31_instruct "--chat"
log "bridge done -> resuming steering if incomplete"

N=$(ls "$SCA_DATA_ROOT"/steering/saturated/steer_*_*.parquet 2>/dev/null | wc -l)
if [ "$N" -lt 12 ]; then
  setsid nohup bash scripts/experiment1/run_akata_steer_chain.sh >/dev/null 2>&1 < /dev/null &
  log "steering relaunched (resumes per-mode); steer parquets=$N/12"
else
  log "steering already complete ($N/12) — no resume needed"
fi
log "ORCHESTRATOR_DONE"
