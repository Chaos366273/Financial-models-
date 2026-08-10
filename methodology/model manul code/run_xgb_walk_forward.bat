@echo off
REM Rolling walk-forward: baseline (19 feats) + sentiment (19 + sentiment cols)
REM Outputs -> final test results\xgboost 2\
REM From repo root:
REM   & ".\methodology\model manul code\run_xgb_walk_forward.bat"
REM Baseline only:  run_xgb_walk_forward.bat --baseline-only
REM Sentiment only: run_xgb_walk_forward.bat --sentiment-only

set PY311=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe

if exist "%PY311%" (
  "%PY311%" "%~dp0xgb_walk_forward.py" %*
) else (
  python "%~dp0xgb_walk_forward.py" %*
)

pause
