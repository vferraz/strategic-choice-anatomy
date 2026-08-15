#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Gate-then-launch: 1-game qwen smoke to a scratch root -> deterministic validation
# (validate_smalldose_smoke.py) -> on PASS detach the full 3-model chain and exit.
# On any failure the chain is NOT started.
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOG="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOG"
CH="$LOG/smalldose_chain.log"
SMOKE_ROOT="${1:?usage: run_smalldose_smoke_then_chain.sh <scratch-smoke-root>}"

echo "$(date -u +%FT%TZ) smoke start (1 game, qwen) -> $SMOKE_ROOT" >> "$CH"
.venv/bin/python -u steering/run_smalldose.py \
  --model qwen --n-games 1 --out-root "$SMOKE_ROOT" >> "$LOG/steer_smalldose_smoke.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) smoke exit=$rc" >> "$CH"
if [ $rc -ne 0 ]; then
  echo "$(date -u +%FT%TZ) SMOKE FAILED - chain NOT started" >> "$CH"
  exit 1
fi

.venv/bin/python -u steering/smoke_smalldose.py \
  --smoke-root "$SMOKE_ROOT" >> "$CH" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "$(date -u +%FT%TZ) SMOKE VALIDATION FAILED - chain NOT started" >> "$CH"
  exit 1
fi

echo "$(date -u +%FT%TZ) smoke PASS -> detaching full chain" >> "$CH"
setsid nohup bash scripts/experiment1/run_smalldose_chain.sh </dev/null >/dev/null 2>&1 &
echo "$(date -u +%FT%TZ) chain detached (pid $!)" >> "$CH"
exit 0
