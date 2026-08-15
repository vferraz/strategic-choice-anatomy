#!/bin/bash
# Ported from the private repo's scripts/experiment1/ during the open-source migration.
# Retry / skip-existing / chaining logic is byte-identical; only the repo root, the log
# root and the entry-point paths changed. Run from anywhere inside a clone.
# The interpreter must be the project venv: the editable install puts the repo root on
# sys.path, which is how collection/ and steering/ resolve each other.
# Permutation-null (matched-control) steering chain: qwen -> qwen_instruct -> llama31_instruct.
# Mode h1_dinc, variants perm0/perm1/perm2, layers 65+79, doses [-0.25..0.25], 54-game wave sample.
# Resumable: the runner skips any (model,mode) whose parquet already exists in the out-root.
# DO NOT launch until the PI verifies the Step-0 gate + Step-1 directions + smoke and gives GO.
# Launch (detached) with:  setsid bash scripts/experiment1/run_perm_chain.sh &
set -u
cd "$(git rev-parse --show-toplevel)"
# Heavy data lives outside the repo; override with SCA_DATA_ROOT (see strategic_anatomy/config.py).
: "${SCA_DATA_ROOT:=$PWD/data_heavy}"
export SCA_DATA_ROOT
LOG="$SCA_DATA_ROOT"/run_logs/oneshot_akata
mkdir -p "$LOG"
CH="$LOG/perm_chain.log"
for M in qwen qwen_instruct llama31_instruct; do
  echo "$(date -u +%FT%TZ) start $M" >> "$CH"
  .venv/bin/python -u steering/run_perm.py --model "$M" \
    >> "$LOG/steer_perm_$M.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $M exit=$rc" >> "$CH"
  if [ $rc -ne 0 ]; then
    echo "$(date -u +%FT%TZ) ABORT chain ($M failed)" >> "$CH"
    exit 1
  fi
done
echo "$(date -u +%FT%TZ) PERM_CHAIN_DONE" >> "$CH"
