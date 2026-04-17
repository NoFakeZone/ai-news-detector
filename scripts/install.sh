#!/usr/bin/env bash
# Install the package in editable mode with dev dependencies.
# Creates venv/ on first run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_activate.sh
python -m pip install -e ".[dev]"
