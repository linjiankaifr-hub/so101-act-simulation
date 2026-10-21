#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
for backend in egl glfw; do
  export MUJOCO_GL="$backend"
  timeout 40 "$PYTHON" scripts/benchmark_render.py || true
done
