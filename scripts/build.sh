#!/usr/bin/env bash
# Build sdist + wheel into ./dist using the repo venv.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_activate.sh
python -m pip install --quiet --upgrade build
rm -rf dist
python -m build
