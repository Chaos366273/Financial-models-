@echo off

set PY311=C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe

if exist "%PY311%" (

  "%PY311%" "%~dp0fetch_yahoo_data.py" %*

) else (

  python "%~dp0fetch_yahoo_data.py" %*

)

pause

