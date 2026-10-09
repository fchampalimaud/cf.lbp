@echo off
setlocal enabledelayedexpansion

:: install.bat — standalone bootstrapper: download this file alone (no git, no
:: prior clone) and double-click it. It fetches the simulator into INSTALL_DIR
:: on first run, then hands off to start.bat there. Re-running later (e.g. the
:: next time you double-click it) just launches the existing install — updates
:: after that go through the app's own update button, not this script.

echo === LBP Simulator Installer ===
echo.

set "REPO_ZIP=https://github.com/fchampalimaud/cf.lbp/archive/refs/heads/main.zip"
set "DEFAULT_DIR=%USERPROFILE%\LBP-Simulator"
set "INSTALL_DIR="
set /p "INSTALL_DIR=Install location [%DEFAULT_DIR%]: "
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%DEFAULT_DIR%"
call set "INSTALL_DIR=%INSTALL_DIR%"
if "%INSTALL_DIR:~-1%"=="\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
echo.

if exist "%INSTALL_DIR%\LBPSimulator.py" (
    echo Found an existing install at "%INSTALL_DIR%".
    goto :run
)

echo Downloading the simulator to "%INSTALL_DIR%" (one-time, needs internet)...
set "TMP_ZIP=%TEMP%\lbp_simulator_%RANDOM%.zip"
set "TMP_EXTRACT=%TEMP%\lbp_simulator_extract_%RANDOM%"

powershell -ExecutionPolicy Bypass -NoProfile -Command "Invoke-WebRequest -Uri '%REPO_ZIP%' -OutFile '%TMP_ZIP%'"
if %ERRORLEVEL% neq 0 (
    echo.
    echo Download failed. Check your internet connection and try again.
    pause
    exit /b 1
)

powershell -ExecutionPolicy Bypass -NoProfile -Command "Expand-Archive -Path '%TMP_ZIP%' -DestinationPath '%TMP_EXTRACT%' -Force"
if %ERRORLEVEL% neq 0 (
    echo.
    echo Could not extract the download. Try again, or report this at
    echo https://github.com/fchampalimaud/cf.lbp/issues
    pause
    exit /b 1
)
del "%TMP_ZIP%" >nul 2>nul

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
for /d %%d in ("%TMP_EXTRACT%\*") do set "REPO_ROOT=%%d"
xcopy /e /i /y "!REPO_ROOT!\simulation\*" "%INSTALL_DIR%\" >nul
rmdir /s /q "%TMP_EXTRACT%" >nul 2>nul

echo Installed to "%INSTALL_DIR%".
echo.

:run
cd /d "%INSTALL_DIR%"
call start.bat
