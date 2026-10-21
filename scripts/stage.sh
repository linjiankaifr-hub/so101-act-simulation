#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
task="${1:-status}"
name="${2:-base}"
count="${3:-100}"
case "$task" in
status) "$PYTHON" scripts/status.py ;;
test) "$PYTHON" -m pytest tests -q ;;
evaluate) "$PYTHON" scripts/evaluate_act.py --name "$name" --split test --episodes "$count" --output-tag "manual-$(date +%Y%m%d-%H%M%S)" ;;
collect)
  case "$name" in
  base) "$PYTHON" scripts/collect.py --name base --count "$count" --start 10000 ;;
  random) "$PYTHON" scripts/collect.py --name random --count 20 --start 11000 ;;
  targeted) "$PYTHON" scripts/collect.py --name targeted --count 20 --seeds "$ROOT/results/targeted_seeds.json" ;;
  esac
  ;;
train)
  if [ "$name" = base ]; then
    "$PYTHON" scripts/train_act.py --name base --datasets base --steps 6000
  else
    "$PYTHON" scripts/train_act.py --name "$name" --datasets base "$name" --steps 6000
  fi
  ;;
full) "$PYTHON" scripts/pipeline.py ;;
ros)
  checkpoint="$ROOT/checkpoints/$name/deploy"
  exec bash "$ROOT/scripts/ros.sh" --checkpoint "$checkpoint"
  ;;
*) printf 'Unknown task: %s\n' "$task";exit 2 ;;
esac
