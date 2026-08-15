#!/usr/bin/env bash
# NOTE: this script documents the ORIGINAL DGX-Spark-class environment used for the
# published collection. It is kept for reproducibility reference (Tier 3); exact pinned
# versions are in docs/ENVIRONMENTS.md. Paths and venv names are the originals.
# setup_gptoss_env.sh — Isolated env for GPT-OSS diagnostics on Spark
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV_DIR=".venv_gptoss"

echo "=== GPT-OSS Spark env setup ==="
echo "Repo:      $(pwd)"
echo "Venv:      $VENV_DIR"
echo "Python:    $(python3 --version)"
echo "GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Venv already exists."
fi

source "$VENV_DIR/bin/activate"

echo "[2/4] Upgrading packaging tools..."
pip install --upgrade pip wheel "setuptools<82"

echo "[3/4] Installing GPT-OSS runtime deps..."
pip install -U torch transformers accelerate
pip install -U kernels openai-harmony

echo "[4/4] Printing installed versions..."
python - <<'PY'
import sys
import torch
import accelerate
import transformers
import triton

try:
    import kernels
    kernels_version = getattr(kernels, "__version__", "unknown")
except Exception:
    kernels_version = "unavailable"

try:
    import openai_harmony
    harmony_version = getattr(openai_harmony, "__version__", "unknown")
except Exception:
    harmony_version = "unavailable"

print("=== Installed versions ===")
print("python       ", sys.version.split()[0])
print("torch        ", torch.__version__)
print("transformers ", transformers.__version__)
print("accelerate   ", accelerate.__version__)
print("triton       ", triton.__version__)
print("kernels      ", kernels_version)
print("openai_harmony", harmony_version)
print("cuda avail   ", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu          ", props.name)
    print("gpu mem      ", f"{props.total_memory / 1e9:.1f} GB")
PY

echo "=== GPT-OSS env ready ==="
