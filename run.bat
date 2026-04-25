@echo off
setlocal
cd /d "%~dp0"
python -m vic3_translator.main
if errorlevel 1 pause
