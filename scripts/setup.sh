#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export UV_CACHE_DIR="$HOME/.cache/codex-so101-uv"
export UV_SYSTEM_CERTS=true UV_HTTP_TIMEOUT=120
UV="$HOME/.venvs/codex-dppo-tools/bin/uv"
ENV_DIR="$HOME/.venvs/codex-so101-act"
if [ ! -x "$ENV_DIR/bin/python" ]; then "$UV" venv --python /usr/bin/python3.12 "$ENV_DIR"; fi
PY="$ENV_DIR/bin/python"
printf '%s\n' "$PY" > "$ROOT/python-path.txt"
"$UV" pip install --python "$PY" --index-url https://pypi.tuna.tsinghua.edu.cn/simple mujoco==3.3.7 numpy==2.2.6 scipy pytest matplotlib imageio imageio-ffmpeg h5py pillow
"$UV" pip install --python "$PY" --index-url https://pypi.tuna.tsinghua.edu.cn/simple torch==2.7.1 torchvision==0.22.1
"$UV" pip install --python "$PY" --index-url https://pypi.tuna.tsinghua.edu.cn/simple lerobot==0.5.1 torch==2.7.1 torchvision==0.22.1 'setuptools<80'
"$UV" pip freeze --python "$PY" > "$ROOT/requirements-lock.txt"
