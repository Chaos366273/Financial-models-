@echo off
setlocal
cd /d "%~dp0"

set "PY=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PY%" (
  echo Python 3.11 not found at %PY%
  exit /b 1
)

echo Installing dissertation-models dependencies into Python 3.11...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo pip install failed.
  exit /b 1
)
echo.
echo Done. You can now run: run_model.bat all
exit /b 0
