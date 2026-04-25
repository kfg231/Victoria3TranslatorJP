@echo off
setlocal EnableDelayedExpansion

:: ========================================
:: Admin Elevation
:: ========================================
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting admin privileges...
    powershell -Command "Start-Process -FilePath '%~0' -Verb RunAs"
    exit /b
)

echo ============================================
echo   Victoria 3 Translator - SETUP
echo ============================================
echo.

cd /d "%~dp0"

:: ========================================
:: [1/4] Check Python
:: ========================================
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found.
    echo  Please install Python 3.10 or higher:
    echo    https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo  Python found.
echo.

:: ========================================
:: [2/4] Install Dependencies
:: ========================================
if not exist "requirements.txt" (
    echo  [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

echo [2/4] Installing packages...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to install packages.
    pause
    exit /b 1
)
echo.
echo  Installation complete.
echo.

:: ========================================
:: [3/4] Set DeepSeek API Key
:: ========================================
echo [3/4] Configure DeepSeek API Key
echo.

set "EXISTING_DS="
for /f "tokens=2,*" %%a in ('reg query "HKCU\Environment" /v DEEPSEEK_API_KEY 2^>nul') do set "EXISTING_DS=%%b"

if defined EXISTING_DS (
    echo  Environment variable DEEPSEEK_API_KEY is already set.
    echo.
    set /p "CHANGE_DS=  Change it? (y/N): "
    if /i "!CHANGE_DS!" neq "y" (
        echo  Using existing key.
        goto :openrouter_setup
    )
)

echo.
echo  Enter your DeepSeek API Key.
echo  (Get one at: https://platform.deepseek.com/)
echo  Press Enter to skip.
echo.
set /p "DS_API_KEY=  DeepSeek API Key: "

if not "!DS_API_KEY!"=="" (
    setx DEEPSEEK_API_KEY "!DS_API_KEY!" >nul 2>&1
    if errorlevel 1 (
        echo  [WARNING] Failed to set DeepSeek environment variable.
    ) else (
        echo  DeepSeek API Key has been set successfully.
    )
) else (
    echo  Skipped.
)

:: ========================================
:: [4/4] Set OpenRouter API Key
:: ========================================
:openrouter_setup
echo.
echo [4/4] Configure OpenRouter API Key
echo.

set "EXISTING_OR="
for /f "tokens=2,*" %%a in ('reg query "HKCU\Environment" /v OPENROUTER_API_KEY 2^>nul') do set "EXISTING_OR=%%b"

if defined EXISTING_OR (
    echo  Environment variable OPENROUTER_API_KEY is already set.
    echo.
    set /p "CHANGE_OR=  Change it? (y/N): "
    if /i "!CHANGE_OR!" neq "y" (
        echo  Using existing key.
        goto :setup_done
    )
)

echo.
echo  Enter your OpenRouter API Key.
echo  (Get one at: https://openrouter.ai/keys)
echo  Press Enter to skip.
echo.
set /p "OR_API_KEY=  OpenRouter API Key: "

if not "!OR_API_KEY!"=="" (
    setx OPENROUTER_API_KEY "!OR_API_KEY!" >nul 2>&1
    if errorlevel 1 (
        echo  [WARNING] Failed to set OpenRouter environment variable.
    ) else (
        echo  OpenRouter API Key has been set successfully.
    )
) else (
    echo  Skipped.
)

:setup_done
echo.
echo ============================================
echo   SETUP COMPLETE.
echo   You can now launch the app using run.bat
echo ============================================
echo.
pause
