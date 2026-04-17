#!/usr/bin/env bash
# Run the demo script inside the repo venv. Extra args are forwarded.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_activate.sh
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" python scripts/demo.py "$@"
