#!/usr/bin/env bash
set -euo pipefail

echo "=== LBP 2D Simulator ==="

# ── Locate uv ────────────────────────────────────────────────────────────────
UV_FALLBACK="$HOME/.local/bin"

if ! command -v uv &>/dev/null; then
    if [ -x "$UV_FALLBACK/uv" ]; then
        export PATH="$UV_FALLBACK:$PATH"
    else
        echo "uv not found. Installing (requires internet, one-time only)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$UV_FALLBACK:$PATH"
    fi
fi

# ── Install packages once; reinstall only when requirements.txt changes ──────
cd "$(dirname "$0")"

STAMP=".venv/.req_stamp"
NEEDS_INSTALL=0

if [ ! -x ".venv/bin/python" ] && [ ! -x ".venv/Scripts/python.exe" ]; then
    NEEDS_INSTALL=1
elif [ ! -f "$STAMP" ] || [ "requirements.txt" -nt "$STAMP" ]; then
    NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" = "1" ]; then
    echo "Installing packages (one-time or requirements changed)..."
    if [ ! -d ".venv" ]; then
        uv venv --python 3.11 .venv
    fi
    VENV_PY=".venv/bin/python"
    [ -x ".venv/Scripts/python.exe" ] && VENV_PY=".venv/Scripts/python.exe"
    uv pip install --python "$VENV_PY" -r requirements.txt
    touch "$STAMP"
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Starting simulator..."
echo
PYTHON=".venv/bin/python"
[ -x ".venv/Scripts/python.exe" ] && PYTHON=".venv/Scripts/python.exe"
"$PYTHON" LBPSimulator.py
