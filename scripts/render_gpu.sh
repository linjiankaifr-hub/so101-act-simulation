#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
export MUJOCO_GL=glfw
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LIBGL_ALWAYS_SOFTWARE=0
export LP_NUM_THREADS=2
timeout 100 "$PYTHON" scripts/benchmark_render.py
