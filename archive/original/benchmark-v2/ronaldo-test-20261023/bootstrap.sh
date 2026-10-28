#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 TOKENIZERS_PARALLELISM=false
BENCH_ENV=/tmp/pencil-lora-benchmark-venv
python -m venv --system-site-packages "$BENCH_ENV"
"$BENCH_ENV/bin/python" -m pip install --no-compile -r /app/requirements.txt
exec "$BENCH_ENV/bin/python" /app/benchmark_worker.py "$@"
