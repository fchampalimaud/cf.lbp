#!/usr/bin/env bash
set -euo pipefail

# lbp_install_v0.21.0.sh — standalone bootstrapper: download this file alone (no git, no
# prior clone) and run it. It fetches the simulator into LBP_DIR/app on first
# run, then hands off to start.sh there (your own sessions/networks/… land in
# the sibling LBP_DIR/lbp_files, created by the app itself). Re-running it
# later launches the existing install, offering to update it first if it is
# older than this installer (lbp_files is never touched).
#
# Downloaded files aren't executable by default: run it as
#   bash lbp_install_v0.21.0.sh
# or make it executable first:
#   chmod +x lbp_install_v0.21.0.sh && ./lbp_install_v0.21.0.sh

VERSION="0.21.0"
echo "=== LBP Simulator Installer v$VERSION ==="
echo

REPO_ZIP="https://github.com/fchampalimaud/cf.lbp/archive/refs/heads/main.zip"
DEFAULT_DIR="$HOME/LBPSimulator"
read -rp "Install location (press Enter to use $DEFAULT_DIR): " LBP_DIR
LBP_DIR="${LBP_DIR:-$DEFAULT_DIR}"
LBP_DIR="${LBP_DIR/#\~/$HOME}"   # expand a leading ~ (read doesn't do this itself)
LBP_DIR="${LBP_DIR%/}"
APP_DIR="$LBP_DIR/app"
echo

version_lt() {   # true if $1 < $2, comparing dotted numbers
    awk -v a="$1" -v b="$2" 'BEGIN { n = split(a, x, "."); m = split(b, y, ".")
        for (i = 1; i <= (n > m ? n : m); i++) { if (x[i]+0 < y[i]+0) exit 0; if (x[i]+0 > y[i]+0) exit 1 }
        exit 1 }'
}

DO_DOWNLOAD=1
if [ -f "$APP_DIR/LBPSimulator.py" ]; then
    OLD_VERSION="$( { head -n1 "$APP_DIR/VERSION" 2>/dev/null || true; } | tr -d '[:space:]')"
    OLD_VERSION="${OLD_VERSION:-0}"
    echo "Found an existing install at $LBP_DIR, version $OLD_VERSION."
    DO_DOWNLOAD=0
    if version_lt "$OLD_VERSION" "$VERSION"; then
        read -rp "Update it to v$VERSION? Your own files in lbp_files are kept. [Y/n] " ANSWER
        case "$ANSWER" in [nN]*) ;; *) DO_DOWNLOAD=1 ;; esac
        echo
    fi
fi

if [ "$DO_DOWNLOAD" = 1 ]; then
    echo "Downloading the simulator to $LBP_DIR (needs internet)..."

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
