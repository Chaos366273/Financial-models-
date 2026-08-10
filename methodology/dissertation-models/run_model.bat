@echo off
setlocal
cd /d "%~dp0"

REM Use Python 3.11 (same as PPO). Default "python" on this PC is 3.13 without deps.
set "PY=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PY%" (
  echo Python 3.11 not found at:
  echo   %PY%
  echo Edit run_model.bat and set PY= to your Python path.
  exit /b 1
)

if "%~1"=="" (
  echo Usage: run_model.bat ^<model_id^>
  echo Example: run_model.bat ff5
  echo          run_model.bat all
  echo.
  echo First time: run install_deps.bat
  exit /b 1
)

"%PY%" run_all.py --model %1 --export outputs\lab_results_bundle.json
if errorlevel 1 (
  echo.
  echo Run failed. If you see ModuleNotFoundError, run: install_deps.bat
  exit /b 1
)
echo.
echo Done. In Dissertation Lab: Settings -^> Import results bundle
echo Bundle: %~dp0outputs\lab_results_bundle.json
exit /b 0
