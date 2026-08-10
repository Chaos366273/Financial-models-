@echo off

REM Use Python 3.11 (same as dissertation-models) if 3.13 packages missing

set PY311=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe

if exist "%PY311%" (

  "%PY311%" "%~dp0lstm_return.py"

) else (

  python "%~dp0lstm_return.py"

)

pause

