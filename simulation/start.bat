@echo off
setlocal enabledelayedexpansion

echo === LBP 2D Simulator ===

:: ── Create shortcuts with the LBP icon (first run only): Desktop + this folder ──
powershell -ExecutionPolicy Bypass -NoProfile -Command ^
    "$shell = New-Object -ComObject WScript.Shell; foreach ($path in @((Join-Path $shell.SpecialFolders('Desktop') 'LBP Simulator.lnk'), (Join-Path '%~dp0' 'LBPSimulator.lnk'))) { if (-not (Test-Path $path)) { $s = $shell.CreateShortcut($path); $s.TargetPath = '%~f0'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = '%~dp0docs\assets\icon.ico'; $s.Save() } }" >nul 2>nul

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
cd /d "%~dp0"

:: ── Install packages once; reinstall only when requirements.txt changes ──────
set "STAMP=.venv\.req_stamp"
set "NEEDS_INSTALL=0"
if not exist ".venv\Scripts\python.exe" set "NEEDS_INSTALL=1"
if "%NEEDS_INSTALL%"=="0" (
    :: compare requirements.txt modification time against the stamp file
    for %%f in (requirements.txt) do set "REQ_TIME=%%~tf"
    for %%f in ("%STAMP%") do set "STAMP_TIME=%%~tf"
    if not exist "%STAMP%" set "NEEDS_INSTALL=1"
)
if "%NEEDS_INSTALL%"=="0" (
    :: stamp older than requirements.txt → reinstall
    if "!REQ_TIME!" GTR "!STAMP_TIME!" set "NEEDS_INSTALL=1"
)

if "%NEEDS_INSTALL%"=="1" (
    echo Installing packages (one-time or requirements changed)...
)
if "%NEEDS_INSTALL%"=="1" if not exist ".venv\Scripts\python.exe" (
    uv venv --python 3.11 .venv
    if %ERRORLEVEL% neq 0 (
        echo.
        echo Failed to create virtual environment. Make sure uv can find Python 3.11.
        pause
        exit /b 1
    )
)
if "%NEEDS_INSTALL%"=="1" (
    uv pip sync --python ".venv\Scripts\python.exe" requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo.
        echo Package installation failed. See above for details.
        pause
        exit /b 1
    )
    echo. > "%STAMP%"
)

echo Starting simulator...
echo.
".venv\Scripts\python.exe" LBPSimulator.py

:: Keep the window open if something went wrong
if %ERRORLEVEL% neq 0 (
    echo.
    echo The simulator exited with an error. See the message above.
    pause
)
