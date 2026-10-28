#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 TOKENIZERS_PARALLELISM=false
ENV_DIR=/tmp/pencil-lora-evaluation-venv
python -m venv --system-site-packages "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --no-compile -r /app/requirements.txt
exec "$ENV_DIR/bin/python" /app/eval_worker.py
