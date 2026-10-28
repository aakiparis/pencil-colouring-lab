#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
# Ephemeral local environment; model weights alone live on the shared filesystem.
python -m venv --system-site-packages /tmp/pencil-inference-venv
/tmp/pencil-inference-venv/bin/python -m pip install -r /app/requirements.txt
exec /tmp/pencil-inference-venv/bin/python /app/benchmark_worker.py "$@"
