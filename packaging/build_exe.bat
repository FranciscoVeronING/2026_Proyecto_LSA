@echo off
setlocal
cd /d "%~dp0\.."
echo Empaquetando LSABackend (onedir). Requiere PyInstaller:
echo   pip install pyinstaller
pyinstaller --noconfirm --clean packaging\LSABackend.spec
echo.
echo Listo: dist\LSABackend\LSABackend.exe
echo Deja esa carpeta junta (modelos + exe) y corre el exe antes de usar la extension.
pause
