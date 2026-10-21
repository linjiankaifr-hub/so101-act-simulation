#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PYTHON" -m pytest "$ROOT/tests" -q "$@"
