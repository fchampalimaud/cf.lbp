@echo off
setlocal enabledelayedexpansion

echo === LBP 2D Simulator ===

:: ── Check if uv is already on PATH ──────────────────────────────────────────
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 goto :run

:: ── Check common install locations (after a previous install) ────────────────
for %%p in ("%USERPROFILE%\.local\bin" "%USERPROFILE%\.cargo\bin") do (
    if exist "%%~p\uv.exe" (
        set "PATH=%%~p;%PATH%"
        goto :run
    )
)

:: ── uv not found — download and install it ───────────────────────────────────
echo uv not found. Installing (requires internet, one-time only)...
powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
if %ERRORLEVEL% neq 0 (
    echo.
    echo Failed to install uv. Check your internet connection and try again.
    pause
    exit /b 1
)

:: Add the default install location to PATH for this session
set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

:run
echo Starting simulator...
echo.
cd /d "%~dp0"
uv run --python 3.11 --with-requirements requirements.txt python LBPSimulator.py

:: Keep the window open if something went wrong
if %ERRORLEVEL% neq 0 (
    echo.
    echo The simulator exited with an error. See the message above.
    pause
)
