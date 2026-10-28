#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
BENCH_ENV=/cache/benchmark-v2-venv
if [ "${AWAIT_SHARED_ENVIRONMENT:-0}" = "1" ]; then
 for attempt in $(seq 1 120); do
  if [ -f /cache/benchmark-v2-environment-ready ]; then break; fi
  sleep 5
 done
 test -f /cache/benchmark-v2-environment-ready || { echo 'Shared environment preparation did not finish'; exit 1; }
fi
if [ ! -f /cache/benchmark-v2-environment-ready ]; then
 python -m venv --system-site-packages "$BENCH_ENV"
 "$BENCH_ENV/bin/python" -m pip install 'https://github.com/huggingface/diffusers/archive/b800f828f3951c4d206e154411bcbfa693cd322e.tar.gz' 'transformers==5.20.1' 'accelerate==1.15.0' 'huggingface-hub==2.2.0' 'boto3==1.43.114' 'Pillow==13.0.0' 'ftfy==6.3.1' 'sentencepiece==0.2.2' 'protobuf==6.32.1'
 "$BENCH_ENV/bin/python" - <<'PY'
from diffusers import QwenImage21Pipeline, ModularPipeline
from transformers import Qwen3_5ForConditionalGeneration
print('Benchmark environment imports verified', flush=True)
PY
 touch /cache/benchmark-v2-environment-ready
fi
exec "$BENCH_ENV/bin/python" /app/benchmark_worker.py "$@"
