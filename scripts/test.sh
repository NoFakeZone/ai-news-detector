#!/usr/bin/env bash
# Run the pytest suite inside the repo venv. Extra args are forwarded to pytest.
# Examples:
#   scripts/test.sh
#   scripts/test.sh tests/test_punctuation_ratio.py
#   scripts/test.sh -k per_word
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_activate.sh
python -m pytest -v "$@"
