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
:: No parenthesised blocks here: a ")" inside one ends it early and cmd aborts.
if not exist ".venv\Scripts\python.exe" goto :install
fc /b requirements.txt ".venv\requirements.txt" >nul 2>nul
if errorlevel 1 goto :install
goto :launch

:install
echo Installing packages - first run, or the requirements changed...
if exist ".venv\Scripts\python.exe" goto :sync
call uv venv --python 3.11 .venv
if errorlevel 1 goto :install_failed
:sync
call uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto :install_failed
copy /y requirements.txt ".venv\requirements.txt" >nul
goto :launch

:install_failed
echo.
echo Package installation failed. See the message above.
pause
exit /b 1

:launch
echo Starting simulator...
echo.
".venv\Scripts\python.exe" LBPSimulator.py
if errorlevel 1 goto :sim_failed
exit /b 0

:sim_failed
echo.
echo The simulator exited with an error. See the message above.
pause
exit /b 1
