#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Layer B<->C bridge residual capture (Akata substrate), 3 dense models, baseline-only.
# NO steering afterward (steering is being redesigned). Resumable via --skip-existing.
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata_layerc; mkdir -p "$LOGDIR"
CHAIN="$LOGDIR/bridge_chain.log"; EP=collection/layerc/capture_bridge_residuals.py
log(){ echo "$(date -u +%FT%TZ) $*" >> "$CHAIN"; }
until ! pgrep -f '[r]un_akata_steer.py|[c]apture_residuals_bridge_akata|[g]enerate_oneshot_akata' >/dev/null; do sleep 30; done
sleep 30
log "GPU free -> bridge capture (3 dense models)"
runb(){ local n=0 rc=1; while [ $n -lt 3 ]; do n=$((n+1)); log "START $1 $2 (try $n)"
  .venv/bin/python -u "$EP" --model "$1" $2 --load_8bit --skip-existing >> "$LOGDIR/bridge_$1.log" 2>&1; rc=$?
  log "END $1 try $n (exit $rc)"; [ $rc -eq 0 ] && break; sleep 60; done; }
runb qwen ""
runb qwen_instruct ""
runb llama31_instruct "--chat"
log "BRIDGE_CHAIN_DONE (no steering launched — redesign pending)"
