#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=47
exec "$PYTHON" scripts/ros_bridge.py "$@"
