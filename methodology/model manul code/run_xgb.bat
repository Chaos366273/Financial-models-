@echo off

REM Use Python 3.11 (same as dissertation-models) if available
REM Default (no flags) = PRIMARY frozen B_drop_bottom9 (19 feats) + val stability.
REM From repo root:
REM   & ".\methodology\model manul code\run_xgb.bat"
REM Secondary A_top16:   run_xgb.bat --parsimony-a
REM Feature robustness:  run_xgb.bat --feature-robustness
REM Sector diagnose:     run_xgb.bat --diagnose-sector
REM Final one-shot:      run_xgb.bat --final-test
REM   -> all test logs/CSVs/lock go to: final test results\xgboost\
REM Skip stability:      run_xgb.bat --no-stability

set PY311=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe

if exist "%PY311%" (
  "%PY311%" "%~dp0xgb_return.py" %*
) else (
  python "%~dp0xgb_return.py" %*
)

pause
