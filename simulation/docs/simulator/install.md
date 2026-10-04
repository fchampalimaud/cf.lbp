# Installing the Simulator

## 1. Get the code

Open a terminal in the folder where you want to install the simulator, then:

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

**From the app** — Session tab → **Check for updates** (the simulator also checks quietly at start-up; the button then reads **Update to vX**). It downloads the latest version from the public repository, replaces the app's own files and restarts. Your files are not touched.

**With git** — if you cloned the repository:

```bash
git pull
```

Either way, re-run `start.bat`/`start.sh` (or `pip install -r requirements.txt` for a manual install) to pick up any new dependencies. No database migrations or build steps needed — the simulator is pure Python.

## Your files

Everything you create — sessions, networks, worlds, motifs, brains, logs, videos — is saved in **`~/LBPSimulator`** (the `LBPSimulator` folder in your home folder), never inside the app's folder, so updates can't touch it. Session tab → **My files** shows the folder, opens it, or moves it somewhere else. The sessions and networks that ship with the simulator (Tutorials, Demos, Default) are marked 🔒 and are read-only; saving one writes your own copy. If you used the simulator before this folder existed, your files are moved there automatically the first time a new version starts.
