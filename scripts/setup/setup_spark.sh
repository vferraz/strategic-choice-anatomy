#!/usr/bin/env bash
# NOTE: this script documents the ORIGINAL DGX-Spark-class environment used for the
# published collection. It is kept for reproducibility reference (Tier 3); exact pinned
# versions are in docs/ENVIRONMENTS.md. Paths and venv names are the originals.
# setup_spark.sh — One-time environment setup on NVIDIA DGX Spark
set -euo pipefail

echo "=== DGX Spark environment setup ==="

cd "$(dirname "$0")/../.."

# System info
echo "[1/4] System diagnostics..."
uname -m
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 --version

# Create venv with system site packages (in case NVIDIA provides torch)
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[2/4] Venv already exists."
fi
source "$VENV_DIR/bin/activate"

# Install deps
echo "[3/4] Installing packages..."
pip install --upgrade pip
pip install torch transformers accelerate bitsandbytes
pip install numpy pandas pyarrow nashpy scipy matplotlib

# Verify
echo "[4/4] Verifying..."
python -c "
import sys, torch, transformers, accelerate
print(f'Python:       {sys.version}')
print(f'torch:        {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU avail:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU:          {props.name}')
    print(f'GPU mem:      {props.total_memory / 1e9:.1f} GB')
    print(f'Compute cap:  {props.major}.{props.minor}')
print(f'transformers: {transformers.__version__}')
print(f'accelerate:   {accelerate.__version__}')
try:
    import bitsandbytes as bnb
    print(f'bitsandbytes: {bnb.__version__}')
except Exception as e:
    print(f'bitsandbytes: FAILED ({e})')
"

echo "=== Setup complete ==="
