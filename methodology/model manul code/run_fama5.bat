@echo off
REM Fama-French Use B walk-forward (default) — Python 3.11 preferred
set PY311=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe
cd /d "%~dp0"

if exist "%PY311%" (
  set PY=%PY311%
) else (
  set PY=python
)

if "%~1"=="" (
  echo Running FF5 walk-forward Use B + H1 tables...
  "%PY%" fama5.py --walk-forward
) else (
  "%PY%" fama5.py %*
)

echo.
echo Outputs: fama french\
echo Dissertation table: fama french\dissertation_table_h1.md
pause
