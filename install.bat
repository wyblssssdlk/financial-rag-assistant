@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv-web\Scripts\python.exe" (
  py -3.11 -m venv .venv-web
  if errorlevel 1 (
    echo 请先安装 Python 3.11，然后重试。
    pause
    exit /b 1
  )
)
".venv-web\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo 安装未完成，请检查上方错误和网络连接。
) else (
  echo 安装完成，请双击 start.bat 启动。
)
pause
endlocal
