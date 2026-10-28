#!/bin/bash
set -euo pipefail
export HF_HOME=/cache/huggingface
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
if [ ! -f /cache/environment-ready-v2 ]; then
 python -m venv --system-site-packages /cache/venv
 /cache/venv/bin/python -m pip install 'diffusers==0.37.0' 'transformers==4.57.3' 'accelerate==1.10.1' 'boto3==1.40.25' 'Pillow==11.3.0' 'ftfy==6.3.1' 'sentencepiece==0.2.2' 'protobuf==6.32.1'
 /cache/venv/bin/python - <<'PY'
import inspect
import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
assert "zero_cond_t" in inspect.signature(QwenImageTransformer2DModel.__init__).parameters
print("Environment verified", torch.__version__, flush=True)
PY
 touch /cache/environment-ready-v2
fi
exec /cache/venv/bin/python /app/worker.py "$@"
