@echo off
setlocal enabledelayedexpansion
title LBP Simulator Setup

echo.
echo  ============================================
echo   LBP Simulator - Setup
echo  ============================================
echo.

:: ----- Find Python -----
set PYTHON=
for %%c in (python py python3) do (
    if not defined PYTHON (
        %%c --version >nul 2>&1 && set PYTHON=%%c
    )
)
if not defined PYTHON (
    echo  [ERROR] Python not found.
    echo         Download from https://www.python.org/downloads/
    echo         Tick "Add Python to PATH" during installation.
    echo.
    pause & exit /b 1
)

:: ----- Check version >= 3.10 -----
for /f "tokens=2" %%v in ('!PYTHON! --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (set PYMAJ=%%a & set PYMIN=%%b)
if !PYMAJ! LSS 3 goto :badver
if !PYMAJ! EQU 3 if !PYMIN! LSS 10 goto :badver
echo  [OK] Python !PYVER!
goto :checkpip

:badver
echo  [ERROR] Python 3.10+ required, found !PYVER!
echo         Download a newer version from https://www.python.org/downloads/
pause & exit /b 1

:checkpip
:: ----- pip -----
!PYTHON! -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  [WARN] pip missing - installing via ensurepip...
    !PYTHON! -m ensurepip --upgrade
)
echo  [OK] pip

:: ----- Core dependencies -----
echo.
echo  Installing core dependencies...
!PYTHON! -m pip install -r requirements.txt
if errorlevel 1 (
    echo  [ERROR] Dependency installation failed.
    pause & exit /b 1
)
echo  [OK] Core dependencies installed

:: ----- Optional: MuJoCo -----
echo.
set /p OPT_MJ= Install MuJoCo physics backend? [y/N]:
if /i "!OPT_MJ!"=="y" (
    !PYTHON! -m pip install mujoco
    echo  [OK] MuJoCo installed
)

:: ----- Optional: WebEngine -----
set /p OPT_WE= Install PySide6-WebEngine (inline help viewer)? [y/N]:
if /i "!OPT_WE!"=="y" (
    !PYTHON! -m pip install PySide6-WebEngine
    echo  [OK] PySide6-WebEngine installed
)

echo.
echo  ============================================
echo   Setup complete!
echo   Run:  !PYTHON! LBPSimulator.py
echo  ============================================
echo.
pause
