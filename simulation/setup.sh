#!/usr/bin/env bash
set -euo pipefail

echo ""
echo " ============================================"
echo "  LBP Simulator - Setup"
echo " ============================================"
echo ""

# ----- Find Python -----
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo " [ERROR] Python not found."
    echo "         macOS:  brew install python"
    echo "         Ubuntu: sudo apt install python3"
    echo "         Fedora: sudo dnf install python3"
    exit 1
fi

# ----- Check version >= 3.10 -----
PYMAJ=$($PYTHON -c "import sys; print(sys.version_info.major)")
PYMIN=$($PYTHON -c "import sys; print(sys.version_info.minor)")
PYVER=$($PYTHON -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')")

if [ "$PYMAJ" -lt 3 ] || { [ "$PYMAJ" -eq 3 ] && [ "$PYMIN" -lt 10 ]; }; then
    echo " [ERROR] Python 3.10+ required, found $PYVER"
    exit 1
fi
echo " [OK] Python $PYVER"

# ----- pip -----
if ! $PYTHON -m pip --version &>/dev/null; then
    echo " [WARN] pip missing - installing via ensurepip..."
    $PYTHON -m ensurepip --upgrade
fi
echo " [OK] pip"

# ----- Core dependencies -----
echo ""
echo " Installing core dependencies..."
$PYTHON -m pip install -r requirements.txt
echo " [OK] Core dependencies installed"

# ----- Optional: MuJoCo -----
echo ""
read -rp " Install MuJoCo physics backend? [y/N]: " OPT_MJ
if [[ "${OPT_MJ,,}" == "y" ]]; then
    $PYTHON -m pip install mujoco
    echo " [OK] MuJoCo installed"
fi

# ----- Optional: WebEngine -----
read -rp " Install PySide6-WebEngine (inline help viewer)? [y/N]: " OPT_WE
if [[ "${OPT_WE,,}" == "y" ]]; then
    $PYTHON -m pip install PySide6-WebEngine
    echo " [OK] PySide6-WebEngine installed"
fi

echo ""
echo " ============================================"
echo "  Setup complete!"
echo "  Run:  $PYTHON LBPSimulator.py"
echo " ============================================"
echo ""
