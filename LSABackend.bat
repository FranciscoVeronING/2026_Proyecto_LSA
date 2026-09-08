@echo off
cd /d "%~dp0"

if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" run_backend.py %*
    goto :end
  )
)

if exist "D:\miniconda3\envs\lsa_extension\python.exe" (
  "D:\miniconda3\envs\lsa_extension\python.exe" run_backend.py %*
  goto :end
)

where python >nul 2>&1
if %errorlevel%==0 (
  python run_backend.py %*
  goto :end
)

py -3 run_backend.py %*

:end
if errorlevel 1 pause
