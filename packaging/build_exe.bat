@echo off
setlocal
cd /d "%~dp0\.."
echo Empaquetando ILSA (ventana + API). Requiere: pip install pyinstaller
pyinstaller --noconfirm --clean packaging\LSABackend.spec
if errorlevel 1 (
  echo Fallo PyInstaller.
  pause
  exit /b 1
)

if not exist "extension\bin" mkdir "extension\bin"
if exist "extension\bin\ILSA.zip" del /f /q "extension\bin\ILSA.zip"

echo Comprimiendo dist\LSABackend -^> extension\bin\ILSA.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\LSABackend\*' -DestinationPath 'extension\bin\ILSA.zip' -Force"
if errorlevel 1 (
  echo No se pudo crear extension\bin\ILSA.zip
  pause
  exit /b 1
)

echo.
echo Listo.
echo  1. dist\LSABackend\ILSA.exe  (carpeta completa, no separes el exe)
echo  2. extension\bin\ILSA.zip     (lo que descarga el boton de la extension)
echo Recarga la extension en chrome://extensions para que el boton vea el zip.
pause
