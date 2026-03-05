#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup_a4000_gradient.sh
# Optional env vars:
#   PYTHON_BIN=python3.12 bash setup_a4000_gradient.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 1
fi

echo "[1/7] Creating virtual environment: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/7] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[3/7] Installing CUDA-enabled PyTorch (cu128)"
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "[4/7] Installing CuPy + project dependencies"
python -m pip install cupy-cuda12x numpy pandas openpyxl

echo "[5/7] Verifying Python packages"
python - <<'PY'
import importlib
mods = ["torch", "cupy", "numpy", "pandas", "openpyxl"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m}: {getattr(mod, '__version__', 'ok')}")
PY

echo "[6/7] Verifying GPU access (Torch + CuPy)"
python - <<'PY'
import torch
import cupy as cp

print("torch version:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("torch cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch gpu:", torch.cuda.get_device_name(0))

print("cupy version:", cp.__version__)
count = cp.cuda.runtime.getDeviceCount()
print("cupy device count:", count)
if count > 0:
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(name, bytes):
        name = name.decode()
    print("cupy gpu:", name)
PY

echo "[7/7] NVIDIA driver summary"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; ensure NVIDIA drivers are installed in this runtime."
fi

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
