@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if exist ".venv-web\Scripts\python.exe" (
  ".venv-web\Scripts\python.exe" launch_web.py
) else (
  python launch_web.py
)
if errorlevel 1 pause
endlocal
