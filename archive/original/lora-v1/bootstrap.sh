#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
python - <<'CHECK_DISK'
import json,shutil
u=shutil.disk_usage("/tmp");print(json.dumps({"local_environment_disk":{"total_bytes":u.total,"free_bytes":u.free}}),flush=True)
CHECK_DISK
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
# Isolate training dependencies from the active inference environment.
ENV_DIR=/tmp/pencil-lora-v1-venv
python -m venv --system-site-packages "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --no-compile -r /app/requirements.txt
exec "$ENV_DIR/bin/python" /app/train_worker.py "$@"
