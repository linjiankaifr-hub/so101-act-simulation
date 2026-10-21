#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
while [ ! -s "$ROOT/data/base/manifest.json" ]; do sleep 3; done
sleep 1
"$PYTHON" scripts/export_lerobot.py --name base
