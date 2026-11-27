@echo off
setlocal enabledelayedexpansion

:: lbp_install_v0.23.3.bat — standalone bootstrapper: download this file alone (no git, no
:: prior clone) and double-click it. It fetches the simulator into LBP_DIR\app
:: on first run, then hands off to start.bat there (your own sessions/networks/…
:: land in the sibling LBP_DIR\lbp_files, created by the app itself). Re-running
:: it later launches the existing install, offering to update it first if it is
:: older than this installer (lbp_files is never touched).

set "VERSION=0.23.3"
echo === LBP Simulator Installer v%VERSION% ===
echo.

set "REPO_ZIP=https://github.com/fchampalimaud/cf.lbp/archive/refs/heads/main.zip"
set "DEFAULT_DIR=%USERPROFILE%\LBPSimulator"
set "LBP_DIR="
echo Install location (press Enter to use %DEFAULT_DIR%):
set /p "LBP_DIR=> "
if "%LBP_DIR%"=="" set "LBP_DIR=%DEFAULT_DIR%"
call set "LBP_DIR=%LBP_DIR%"
if "%LBP_DIR:~-1%"=="\" set "LBP_DIR=%LBP_DIR:~0,-1%"
set "APP_DIR=%LBP_DIR%\app"
echo.

if not exist "%APP_DIR%\LBPSimulator.py" goto :download
set "OLD_VERSION=0.0"
if exist "%APP_DIR%\VERSION" set /p OLD_VERSION=<"%APP_DIR%\VERSION"
echo Found an existing install at "%LBP_DIR%", version !OLD_VERSION!.
powershell -NoProfile -Command "if ([version]'!OLD_VERSION!' -lt [version]'%VERSION%') { exit 1 }" >nul 2>nul
if not errorlevel 1 goto :run
set "UPDATE=Y"
set /p "UPDATE=Update it to v%VERSION%? Your own files in lbp_files are kept. [Y/n] "
if /i "!UPDATE!"=="n" goto :run
echo.

:download
echo Downloading the simulator to "%LBP_DIR%" (needs internet)...
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

if not exist "%APP_DIR%" mkdir "%APP_DIR%"
for /d %%d in ("%TMP_EXTRACT%\*") do set "REPO_ROOT=%%d"
xcopy /e /i /y "!REPO_ROOT!\simulation\*" "%APP_DIR%\" >nul
rmdir /s /q "%TMP_EXTRACT%" >nul 2>nul

echo Installed to "%LBP_DIR%".
echo.

:run
echo To run the simulator again later, without this installer, double-click
echo LBPSimulator in:
echo   "%APP_DIR%"
echo.
echo Your own sessions, networks and brains are saved in:
echo   "%LBP_DIR%\lbp_files"
echo.
cd /d "%APP_DIR%"
call start.bat
