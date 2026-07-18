@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto :noenv
start "" ".venv\Scripts\pythonw.exe" run_workstation.py
exit /b 0
:noenv
echo [ERROR] .venv not found. Please run environment setup first.
pause
exit /b 1
