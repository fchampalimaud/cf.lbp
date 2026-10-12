#!/usr/bin/env bash
set -euo pipefail

# lbp_install.sh — standalone bootstrapper: download this file alone (no git, no
# prior clone) and run it. It fetches the simulator into LBP_DIR/app on first
# run, then hands off to start.sh there (your own sessions/networks/… land in
# the sibling LBP_DIR/lbp_files, created by the app itself). Re-running later
# just launches the existing install — updates after that go through the
# app's own update button, not this script.
#
# Downloaded files aren't executable by default: run it as
#   bash lbp_install.sh
# or make it executable first:
#   chmod +x lbp_install.sh && ./lbp_install.sh

echo "=== LBP Simulator Installer ==="
echo

REPO_ZIP="https://github.com/fchampalimaud/cf.lbp/archive/refs/heads/main.zip"
DEFAULT_DIR="$HOME/LBPSimulator"
read -rp "Install location (press Enter to use $DEFAULT_DIR): " LBP_DIR
LBP_DIR="${LBP_DIR:-$DEFAULT_DIR}"
LBP_DIR="${LBP_DIR/#\~/$HOME}"   # expand a leading ~ (read doesn't do this itself)
LBP_DIR="${LBP_DIR%/}"
APP_DIR="$LBP_DIR/app"
echo

if [ -f "$APP_DIR/LBPSimulator.py" ]; then
    echo "Found an existing install at $LBP_DIR."
else
    echo "Downloading the simulator to $LBP_DIR (one-time, needs internet)..."

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
    mkdir -p "$APP_DIR"
    cp -R "$REPO_ROOT/simulation/." "$APP_DIR/"
    rm -rf "$TMP_EXTRACT"

    echo "Installed to $LBP_DIR."
    echo
fi

echo "To run the simulator again later, without this installer, run start.sh in:"
echo "  $APP_DIR"
echo
echo "Your own sessions, networks and brains are saved in:"
echo "  $LBP_DIR/lbp_files"
echo

cd "$APP_DIR"
bash start.sh
