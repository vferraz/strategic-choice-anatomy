#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Continuation: waits for the qwen Layer C run to finish, then qwen_instruct + llama(--chat),
# then RESTARTS the dense steering chain (resumable per-mode). Run detached.
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata_layerc
CHAIN="$LOGDIR/layerc_chain.log"
EP=collection/layerc/generate_layerc.py
log(){ echo "$(date -u +%FT%TZ) $*" >> "$CHAIN"; }

# wait for the already-running qwen layerc to finish
log "waiting for qwen layerc to finish..."
until ! pgrep -f '[g]enerate_oneshot_akata_layerc.py --model qwen --' >/dev/null; do sleep 60; done
log "qwen done; continuing"

runm(){ local n=0 rc=1; while [ $n -lt 3 ]; do n=$((n+1)); log "START $1 $2 (try $n)"
  .venv/bin/python -u "$EP" --model "$1" $2 --skip-existing >> "$LOGDIR/layerc_$1.log" 2>&1; rc=$?
  log "END $1 try $n (exit $rc)"; [ $rc -eq 0 ] && break; sleep 60; done; }

runm qwen_instruct ""
runm llama31_instruct "--chat"
log "LAYERC_CHAIN_DONE -> restarting steering"
setsid nohup bash scripts/experiment1/run_akata_steer_chain.sh >/dev/null 2>&1 < /dev/null &
log "steering restarted"
