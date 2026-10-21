#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
export LP_NUM_THREADS=2
"$PYTHON" scripts/benchmark_render.py
