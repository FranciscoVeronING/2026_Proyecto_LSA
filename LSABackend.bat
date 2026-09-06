@echo off
cd /d "%~dp0"
where python >nul 2>&1
if %errorlevel%==0 (
  python run_backend.py %*
) else (
  py -3 run_backend.py %*
)
if errorlevel 1 pause
