#!/bin/bash
# Layer-C arm: token-attribution capture -> Layer-B<->C bridge residuals, 3 dense models each.
#
# Merged from run_akata_layerc_full_chain.sh and run_akata_bridge_chain.sh during the launcher
# collapse. Preserved verbatim: both pgrep GPU-serialization guards (matched against the
# shipped entry-point names), both settle delays, and both 3-try retry loops. Every stage is
# resumable via --skip-existing, so a reboot costs at most one game.
#
# The private repo auto-resumed the SATURATED-dose steering chain after Layer C. That arm was
# stopped and is not part of this release (docs/METHODS.md §6.1), so this stops and says what
# to start rather than silently running a different experiment.
#
# Usage:
#   bash scripts/launchers/collect_layerc.sh                 # layerc (3 models) then bridge (3 models)
#   bash scripts/launchers/collect_layerc.sh --smoke-only    # 1 game of each, qwen, scratch root
#   bash scripts/launchers/collect_layerc.sh --smoke-only --smoke-root DIR
# Env: .venv (dense stack). Writes $SCA_DATA_ROOT/layerc/{model}/ and
# $SCA_DATA_ROOT/layerc_bridge_residuals/{model}/.
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
: "${SMOKE_ROOT:=$SCA_DATA_ROOT/smoke/layerc}"

LOGDIR="$SCA_DATA_ROOT"/run_logs/oneshot_akata_layerc
mkdir -p "$LOGDIR"
CHAIN="$LOGDIR/layerc_chain.log"
EP=collection/layerc/generate_layerc.py
BRIDGE_EP=collection/layerc/capture_bridge_residuals.py
log(){ echo "$(date -u +%FT%TZ) $*" >> "$CHAIN"; }

# wait for any GPU job (a stray steer, a collection run) to clear, then settle.
# Guards matched against the shipped entry-point names.
until ! pgrep -f '[g]enerate_layerc|[r]un_saturated\.py|[g]enerate_dense_substrate|[g]enerate_gptoss_substrate' >/dev/null; do sleep 20; done
sleep 60

# ---------------------------------------------------------------------------- bounded smoke
if [ "$SMOKE_ONLY" -eq 1 ]; then
  mkdir -p "$SMOKE_ROOT"
  log "smoke: 1 game of layerc + bridge (qwen) -> $SMOKE_ROOT"
  .venv/bin/python -u "$EP" --model qwen --max-games 1 \
    --out-root "$SMOKE_ROOT/layerc" >> "$LOGDIR/layerc_smoke.log" 2>&1
  rc=$?
  log "smoke layerc exit=$rc"
  [ "$rc" -ne 0 ] && { echo "layerc smoke FAILED (exit $rc); see $LOGDIR/layerc_smoke.log" >&2; exit 1; }
  .venv/bin/python -u "$BRIDGE_EP" --model qwen --load_8bit --max-games 1 \
    --out-root "$SMOKE_ROOT/bridge" >> "$LOGDIR/bridge_smoke.log" 2>&1
  rc=$?
  log "smoke bridge exit=$rc"
  [ "$rc" -ne 0 ] && { echo "bridge smoke FAILED (exit $rc); see $LOGDIR/bridge_smoke.log" >&2; exit 1; }
  log "LAYERC_SMOKE_PASS"
  echo "layerc + bridge smoke PASS -> $SMOKE_ROOT"
  exit 0
fi

# ------------------------------------------------------------------ layer C, 3 dense models
log "GPU free -> Layer C full (3 dense models)"
runm(){ local n=0 rc=1; while [ $n -lt 3 ]; do n=$((n+1)); log "START layerc $1 $2 (try $n)"
  .venv/bin/python -u "$EP" --model "$1" $2 --skip-existing >> "$LOGDIR/layerc_$1.log" 2>&1; rc=$?
  log "END layerc $1 try $n (exit $rc)"; [ $rc -eq 0 ] && break; sleep 60; done
  return $rc; }

runm qwen ""
runm qwen_instruct ""
runm llama31_instruct "--chat"
log "LAYERC ALL DONE"

# ------------------------------------------------------- bridge residuals, 3 dense models
# Second guard: the bridge stage waits for its own predecessors, exactly as the standalone
# bridge chain did, so this is safe to run against a box that picked up other work meanwhile.
until ! pgrep -f '[r]un_saturated.py|[c]apture_bridge_residuals|[g]enerate_dense_substrate|[g]enerate_gptoss_substrate|[g]enerate_layerc' >/dev/null; do sleep 30; done
sleep 30
log "GPU free -> bridge capture (3 dense models)"
runb(){ local n=0 rc=1; while [ $n -lt 3 ]; do n=$((n+1)); log "START $1 $2 (try $n)"
  .venv/bin/python -u "$BRIDGE_EP" --model "$1" $2 --load_8bit --skip-existing >> "$LOGDIR/bridge_$1.log" 2>&1; rc=$?
  log "END $1 try $n (exit $rc)"; [ $rc -eq 0 ] && break; sleep 60; done; }
runb qwen ""
runb qwen_instruct ""
runb llama31_instruct "--chat"
log "BRIDGE_CHAIN_DONE"

log "next: start the steering arm yourself -> scripts/launchers/steer_smalldose.sh"
echo "LAYERC + BRIDGE DONE. Next: bash scripts/launchers/steer_smalldose.sh"
