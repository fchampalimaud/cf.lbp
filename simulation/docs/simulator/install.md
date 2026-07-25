# Installing the Simulator

## Requirements

- **Python 3.10 or later**
- **pip**
- A terminal (macOS / Linux) or Command Prompt / PowerShell (Windows)

No GPU is needed. Everything runs on CPU.

---

## Quick setup

After cloning the repo, run the setup script — it checks Python, installs all dependencies, and offers the optional extras.

=== "Windows"

    Double-click `setup.bat`, or run in Command Prompt / PowerShell:

    ```
    setup.bat
    ```

=== "macOS / Linux"

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

The script handles the steps below automatically. Read on only if something goes wrong or you prefer a manual install.

---

## Step-by-step

---

## 0. Get Python and pip

If you already have Python 3.10+ and pip, skip to [step 1](#1-get-the-code).

=== "Windows"

    Download the installer from [python.org/downloads](https://www.python.org/downloads/).  
    During setup, tick **"Add Python to PATH"** before clicking Install.

    Verify in a new Command Prompt or PowerShell window:

    ```
    python --version
    pip --version
    ```

    If `pip` is missing, run:

    ```
    python -m ensurepip --upgrade
    ```

=== "macOS"

    macOS ships an old Python 2. Install Python 3 with [Homebrew](https://brew.sh):

    ```bash
    brew install python
    ```

    This also installs `pip3`. Verify:

    ```bash
    python3 --version
    pip3 --version
    ```

    Use `python3` and `pip3` in place of `python` and `pip` throughout these instructions.

=== "Linux"

    Most distros ship Python 3 — check with `python3 --version`. Install pip if absent:

    ```bash
    # Debian / Ubuntu
    sudo apt install python3-pip

    # Fedora
    sudo dnf install python3-pip
    ```

    Use `python3` and `pip3` in place of `python` and `pip` throughout these instructions.

---

## 1. Get the code

```bash
git clone https://github.com/fchampalimaud/cf.lbp.git
cd cf.lbp/simulation
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies installed:

| Package | Purpose |
|---|---|
| `PySide6` | Qt GUI framework |
| `pyqtgraph` | Real-time oscilloscope plots |
| `numpy` | Sensor and weight matrix math |
| `torch` | Neural layer dynamics (CPU) |

---

## 3. Run

```bash
python LBPSimulator.py
```

The window opens with the robot centred in an empty arena.

---

## Optional dependencies

### In-app documentation viewer

The **?** help button in layer and weight dialogs opens documentation in a browser by default. To render it inline instead, install the WebEngine:

```bash
pip install PySide6-WebEngine
```

### MuJoCo physics

To use MuJoCo as the physics backend (multi-body robots, contact dynamics):

```bash
pip install mujoco
```

---

## Platform notes

=== "Windows"

    Use Command Prompt or PowerShell. The `.bat` launcher in the folder can also be used:
    ```
    start.bat
    ```

=== "macOS / Linux"

    A shell launcher is provided:
    ```bash
    ./start.sh
    ```

---

## Updating

```bash
git pull
pip install -r requirements.txt
```

No database migrations or build steps needed — the simulator is pure Python.
