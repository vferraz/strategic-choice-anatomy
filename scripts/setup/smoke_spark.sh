#!/usr/bin/env bash
# NOTE: this script documents the ORIGINAL DGX-Spark-class environment used for the
# published collection. It is kept for reproducibility reference (Tier 3); exact pinned
# versions are in docs/ENVIRONMENTS.md. Paths and venv names are the originals.
# smoke_spark.sh — Incremental smoke tests on DGX Spark
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

echo "=== DGX Spark smoke tests ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Test 1: Unit tests (no GPU)
echo ""
echo "--- Test 1: Unit tests ---"
python -m pytest tests/ -v

# Test 2: Tiny model, bf16 (pipeline sanity)
echo ""
echo "--- Test 2: Qwen2.5-0.5B, bf16, 2 rounds ---"
python src/run_sim_spark.py \
    --model_name Qwen/Qwen2.5-0.5B \
    --rounds 2 --game PdPd \
    --layer all --use_ln_f_all \
    --trait_p1 none --trait_p2 none \
    --seed 42 --run_tag smoke_bf16

# Test 3: Small model, 8-bit (quantization + probing)
echo ""
echo "--- Test 3: Qwen2.5-7B, 8-bit, 2 rounds ---"
python src/run_sim_spark.py \
    --model_name Qwen/Qwen2.5-7B \
    --rounds 2 --game PdPd \
    --layer all --use_ln_f_all \
    --load_8bit \
    --trait_p1 none --trait_p2 none \
    --seed 42 --run_tag smoke_8bit

echo ""
echo "=== All smoke tests passed ==="
