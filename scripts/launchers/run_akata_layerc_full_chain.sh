#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Full Layer C token-attribution collection (dense): qwen -> qwen_instruct -> llama(--chat), 144 games
# each (--skip-existing keeps the 2 smoke games). Then AUTO-RESUME the dense steering chain, which
# resumes per-mode (qwen skips the finished h0 -> h1_dinc/h2/h3, then qwen_instruct, then llama).
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata_layerc
CHAIN="$LOGDIR/layerc_chain.log"; mkdir -p "$LOGDIR"
EP=collection/layerc/generate_layerc.py
log(){ echo "$(date -u +%FT%TZ) $*" >> "$CHAIN"; }

# wait for any GPU job (the smoke, a stray steer) to clear, then settle
until ! pgrep -f '[g]enerate_oneshot_akata_layerc|[r]un_akata_steer\.py|[g]enerate_oneshot_akata' >/dev/null; do sleep 20; done
sleep 60
log "GPU free -> Layer C full (3 dense models)"

runm(){ local n=0 rc=1; while [ $n -lt 3 ]; do n=$((n+1)); log "START layerc $1 $2 (try $n)"
  .venv/bin/python -u "$EP" --model "$1" $2 --skip-existing >> "$LOGDIR/layerc_$1.log" 2>&1; rc=$?
  log "END layerc $1 try $n (exit $rc)"; [ $rc -eq 0 ] && break; sleep 60; done
  return $rc; }

runm qwen ""
runm qwen_instruct ""
runm llama31_instruct "--chat"
log "LAYERC ALL DONE -> auto-resuming steering"
setsid nohup bash scripts/experiment1/run_akata_steer_chain.sh >/dev/null 2>&1 < /dev/null &
log "steering chain restarted (resumes per-mode: qwen skips h0)"
