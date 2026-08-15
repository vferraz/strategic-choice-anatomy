#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Keep the gpt-oss uniform-site recapture alive until complete. The box hard-locks every few
# hours; --skip-existing resumes per game so a crash costs time, never data. Detached.
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
EP=collection/recapture_gptoss_transition.py
LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata
BLOG="$LOGDIR/gptoss_recap_full.log"
SUP="$LOGDIR/gptoss_recap_supervisor.log"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$SUP"; }
donec(){ ls "$SCA_DATA_ROOT"/gptoss_recap 2>/dev/null | grep -cv '\.'; }
running(){ pgrep -f "[r]ecapture_gptoss_transition.py" >/dev/null; }

# gpt-oss needs ~65 GB. A dead process does not release its memory instantly; relaunching too
# soon loads into memory that is not free yet -> CUDA OOM mid-weight-load -> crash-loop.
# Gate every relaunch on MemAvailable, and never relaunch while a python still holds the GPU.
NEED_MB=90000
mem_avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
gpu_busy(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }
wait_for_mem(){
  for _ in $(seq 1 60); do            # up to 10 min
    if [ "$(mem_avail)" -ge "$NEED_MB" ] && ! gpu_busy; then return 0; fi
    sleep 10
  done
  log "WARN: memory still not free after 10 min (avail=$(mem_avail)MB) -> relaunching anyway"
}

log "recap keepalive armed ($(donec)/144 games done)"
while [ "$(donec)" -lt 144 ]; do
  if ! running; then
    sleep 30
    [ "$(donec)" -ge 144 ] && break
    wait_for_mem
    log "recap not running & incomplete ($(donec)/144, avail=$(mem_avail)MB) -> relaunch"
    setsid nohup .venv_gptoss/bin/python -u "$EP" --skip-existing \
      >> "$BLOG" 2>&1 < /dev/null &
    sleep 60
  fi
  sleep 120
done
log "recap COMPLETE ($(donec)/144) -> GPTOSS_RECAP_KEEPALIVE_DONE"
