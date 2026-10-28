#!/bin/bash
set -euo pipefail
# httpx2 uses the bounded Brotli decoder API; the public container has an older system copy.
/cache/benchmark-v2-venv/bin/python -m pip install 'brotli==1.2.0'
exec /bin/bash /app/benchmark_bootstrap.sh "$@"
