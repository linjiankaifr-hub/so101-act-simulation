#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
while [ ! -s "$ROOT/data/base/manifest.json" ]; do sleep 3; done
sleep 1
"$PYTHON" scripts/train_act.py --name base --datasets base --steps 6000 --batch-size 8
