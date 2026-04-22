#!/usr/bin/env bash
# One-shot environment bootstrap:
#   1. create venv/
#   2. activate it
#   3. install the package with dev dependencies
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Create venv
if [ -d "venv" ]; then
    echo "venv/ already exists — reusing."
else
    echo "Creating venv/..."
    python -m venv venv
fi

# 2. Activate it
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash / MSYS)
    # shellcheck disable=SC1091
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # Linux / macOS
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    echo "error: could not find venv activate script under venv/Scripts or venv/bin" >&2
    exit 1
fi

# 3. Install packages
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

echo
echo "Environment ready. To activate it in your own shell:"
echo "  source venv/Scripts/activate   # Windows (Git Bash)"
echo "  source venv/bin/activate       # Linux / macOS"
