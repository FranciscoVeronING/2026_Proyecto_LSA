@echo off
setlocal
cd /d "%~dp0\.."
echo Packaging ILSA on CPU (classifier; translates against Cloud Run).

set "CONDA=D:\miniconda3\Scripts\conda.exe"
set "ENV=%~dp0_env_cpu"
set "PY=%ENV%\python.exe"

if not exist "%CONDA%" (
  echo conda is not at D:\miniconda3
  exit /b 1
)

if not exist "%PY%" (
  echo Creating CPU packaging env: packaging\_env_cpu
  "%CONDA%" create -p "%ENV%" python=3.10 --yes
  if errorlevel 1 exit /b 1
)

echo Installing Torch CPU + PyInstaller
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
"%PY%" -c "import torch; v=torch.__version__; print('torch', v); raise SystemExit(0 if '+cpu' in v else 1)"
if errorlevel 1 (
  echo The env does not have Torch CPU. Delete packaging\_env_cpu and run again.
  exit /b 1
)
"%PY%" -m pip install -r packaging\requirements_exe.txt
if errorlevel 1 exit /b 1

"%PY%" -m PyInstaller --noconfirm --clean packaging\LSABackend.spec
if errorlevel 1 (
  echo PyInstaller failed.
  exit /b 1
)

if not exist "extension\bin" mkdir "extension\bin"
if exist "extension\bin\ILSA.zip" del /f /q "extension\bin\ILSA.zip"

echo Compressing dist\LSABackend -^> extension\bin\ILSA.zip
powershell -NoProfile -Command "Push-Location 'dist\LSABackend'; tar.exe -a -c -f '..\..\extension\bin\ILSA.zip' *; Pop-Location"
if errorlevel 1 (
  echo Could not create extension\bin\ILSA.zip
  exit /b 1
)

echo.
echo Done.
echo  1. dist\LSABackend\ILSA.exe
echo  2. extension\bin\ILSA.zip
echo Then: powershell -File packaging\upload_ilsa_zip.ps1
endlocal
exit /b 0
