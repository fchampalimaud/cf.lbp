# Installing the Simulator

## 1. Get the code

```bash
git clone https://github.com/fchampalimaud/cf.lbp.git
cd cf.lbp/simulation
```

## 2. Run

=== "Windows"

    ```
    start.bat
    ```

=== "macOS / Linux"

    ```bash
    ./start.sh
    ```

First run installs [uv](https://docs.astral.sh/uv/) if needed, then uses it to install the right
Python version and all dependencies into an isolated environment, and launches the simulator.
No manual Python install required. Later runs reuse the same environment and start instantly.

---

## Manual install

Prefer to manage Python yourself instead of using `start.bat`/`start.sh`? You'll need **Python
3.10+** and **pip**.

=== "Windows"

    Download the installer from [python.org/downloads](https://www.python.org/downloads/).
    During setup, tick **"Add Python to PATH"** before clicking Install.

=== "macOS"

    ```bash
    brew install python
    ```

    Use `python3` and `pip3` in place of `python` and `pip` below.

=== "Linux"

    ```bash
    sudo apt install python3-pip   # Debian / Ubuntu — use dnf/pacman/etc. on other distros
    ```

    Use `python3` and `pip3` in place of `python` and `pip` below.

Then install dependencies and run:

```bash
pip install -r requirements.txt
python LBPSimulator.py
```

---

## Optional extras

### In-app documentation viewer

The **?** help button in layer and weight dialogs opens documentation in a browser by default. To render it inline instead, install the WebEngine:

```bash
pip install PySide6-WebEngine
```

### Session video recording

The Session tab's video capture needs `imageio`:

```bash
pip install imageio imageio-ffmpeg
```

---

## Updating

```bash
git pull
```

Re-run `start.bat`/`start.sh` (or `pip install -r requirements.txt` for a manual install) to pick up any new dependencies. No database migrations or build steps needed — the simulator is pure Python.
