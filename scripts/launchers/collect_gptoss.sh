#!/bin/bash
# GPT-OSS arm: preflight gate -> harmony substrate (144 games) -> uniform-site recapture.
#
# Merged from run_akata_gptoss_chain.sh and run_gptoss_recap_keepalive.sh during the launcher
# collapse. Preserved verbatim: the wait on the dense chain, the 90s settle, the substrate
# retry loop, and the recapture supervisor's memory/GPU gating — the box hard-locks every few
# hours on long MXFP4 runs, and relaunching before the 65 GB of weights are actually free
# turns a crash into a crash-loop. Both stages resume per game via --skip-existing, so a lock
# costs time, never data.
#
# Runs under .venv_gptoss (the MXFP4 `kernels` stack). The generic .venv silently falls back
# and is wrong for every command here — see docs/ENVIRONMENTS.md.
#
# Usage:
#   bash scripts/launchers/collect_gptoss.sh                # preflight gate, substrate, recapture
#   bash scripts/launchers/collect_gptoss.sh --smoke-only   # preflight only, then exit (bounded)
# Writes $SCA_DATA_ROOT/substrate/gptoss/ and $SCA_DATA_ROOT/gptoss_recap/.
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
CHAIN_LOG="$LOGDIR/gptoss_chain.log"
DENSE_LOG="$LOGDIR/dense_chain.log"
EP=collection/generate_gptoss_substrate.py
RECAP_EP=collection/recapture_gptoss_transition.py
log () { echo "$(date -u +%FT%TZ) $*" >> "$CHAIN_LOG"; }

# ---------------------------------------------------------------- preflight gate
if [ "$SMOKE_ONLY" -eq 0 ]; then
  log "waiting for the DENSE chain to finish before collecting gpt-oss..."
  until grep -aq 'DENSE_CHAIN_DONE' "$DENSE_LOG" 2>/dev/null || ! pgrep -f '[c]ollect_dense' >/dev/null; do
    sleep 120
  done
  log "dense done; settling 90s for CUDA memory to free"
  sleep 90
fi

log "gpt-oss preflight gate (.venv_gptoss)"
.venv_gptoss/bin/python -u collection/preflight_gptoss.py >> "$LOGDIR/preflight_gptoss.log" 2>&1
rc=$?
log "preflight_gptoss exit=$rc"
if [ "$rc" -ne 0 ]; then
  log "PREFLIGHT FAILED - collection NOT started"
  echo "preflight_gptoss FAILED (exit $rc); see $LOGDIR/preflight_gptoss.log" >&2
  exit 1
fi
if [ "$SMOKE_ONLY" -eq 1 ]; then
  log "preflight PASS; --smoke-only -> stopping here"
  echo "preflight_gptoss PASS"
  exit 0
fi
log "preflight PASS -> starting gpt-oss full collection (.venv_gptoss)"

# ---------------------------------------------------------------- substrate (144 games)
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
log "GPTOSS_SUBSTRATE_DONE (exit $rc)"
if [ "$rc" -ne 0 ]; then
  log "substrate incomplete -> NOT starting the recapture"
  exit 1
fi

# ---------------------------------------------------------------- uniform-site recapture
# Supervisor preserved from the keepalive: relaunch until all 144 games are present, gating
# every relaunch on MemAvailable and an idle GPU.
BLOG="$LOGDIR/gptoss_recap_full.log"
SUP="$LOGDIR/gptoss_recap_supervisor.log"
rlog(){ echo "$(date -u +%FT%TZ) $*" >> "$SUP"; }
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
  rlog "WARN: memory still not free after 10 min (avail=$(mem_avail)MB) -> relaunching anyway"
}

rlog "recap supervisor armed ($(donec)/144 games done)"
log "starting uniform-site recapture"
while [ "$(donec)" -lt 144 ]; do
  if ! running; then
    sleep 30
    [ "$(donec)" -ge 144 ] && break
    wait_for_mem
    rlog "recap not running & incomplete ($(donec)/144, avail=$(mem_avail)MB) -> relaunch"
    setsid nohup .venv_gptoss/bin/python -u "$RECAP_EP" --skip-existing \
      >> "$BLOG" 2>&1 < /dev/null &
    sleep 60
  fi
  sleep 120
done
rlog "recap COMPLETE ($(donec)/144)"
log "GPTOSS_CHAIN_DONE"
