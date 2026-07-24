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

# ── Run ──────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
echo "Starting simulator..."
echo
uv run --python 3.11 --with-requirements requirements.txt python BraitenbergSimulator.py
