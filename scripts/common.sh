#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export LP_NUM_THREADS=2
if [ "$MUJOCO_GL" = glfw ]; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
  export LIBGL_ALWAYS_SOFTWARE=0
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="$ROOT/logs/matplotlib"
export HF_HOME="$HOME/.cache/codex-so101-hf"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$(cat "$ROOT/python-path.txt")"
cd "$ROOT"
