#!/usr/bin/env bash
set -euo pipefail

# install.sh — standalone bootstrapper: download this file alone (no git, no
# prior clone) and run it. It fetches the simulator into INSTALL_DIR on first
# run, then hands off to start.sh there. Re-running later just launches the
# existing install — updates after that go through the app's own update
# button, not this script.
#
# Downloaded files aren't executable by default: run it as
#   bash install.sh
# or make it executable first:
#   chmod +x install.sh && ./install.sh

echo "=== LBP Simulator Installer ==="
echo

REPO_ZIP="https://github.com/fchampalimaud/cf.lbp/archive/refs/heads/main.zip"
DEFAULT_DIR="$HOME/LBP-Simulator"
read -rp "Install location [$DEFAULT_DIR]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_DIR}"
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"   # expand a leading ~ (read doesn't do this itself)
INSTALL_DIR="${INSTALL_DIR%/}"
echo

if [ -f "$INSTALL_DIR/LBPSimulator.py" ]; then
    echo "Found an existing install at $INSTALL_DIR."
else
    echo "Downloading the simulator to $INSTALL_DIR (one-time, needs internet)..."

    TMP_ZIP="$(mktemp -t lbp_simulator.XXXXXX).zip"
    if command -v curl &>/dev/null; then
        curl -LsSf "$REPO_ZIP" -o "$TMP_ZIP"
    elif command -v wget &>/dev/null; then
        wget -q "$REPO_ZIP" -O "$TMP_ZIP"
    else
        echo "Need 'curl' or 'wget' to download. Install one and try again."
        exit 1
    fi

    TMP_EXTRACT="$(mktemp -d -t lbp_simulator_extract.XXXXXX)"
    if command -v unzip &>/dev/null; then
        unzip -q "$TMP_ZIP" -d "$TMP_EXTRACT"
    elif command -v python3 &>/dev/null; then
        python3 -m zipfile -e "$TMP_ZIP" "$TMP_EXTRACT"
    else
        echo "Need 'unzip' or 'python3' to extract the download. Install one and try again."
        exit 1
    fi
    rm -f "$TMP_ZIP"

    REPO_ROOT="$(find "$TMP_EXTRACT" -mindepth 1 -maxdepth 1 -type d)"
    mkdir -p "$INSTALL_DIR"
    cp -R "$REPO_ROOT/simulation/." "$INSTALL_DIR/"
    rm -rf "$TMP_EXTRACT"

    echo "Installed to $INSTALL_DIR."
    echo
fi

cd "$INSTALL_DIR"
bash start.sh
