#!/usr/bin/env bash
# Source this from other scripts to activate the repo-local venv.
# Creates the venv on first use.
# Usage:  source "$(dirname "$0")/_activate.sh"
# Assumes the caller has already cd'd to the repo root.

if [ ! -d "venv" ]; then
    echo "venv/ not found — creating..."
    python -m venv venv
fi

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
