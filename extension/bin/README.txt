El botón «Descargar ILSA» de la extensión sirve el zip que genere:

  packaging\build_exe.bat

Ese script deja acá `ILSA.zip` (ILSA.exe + runtime CPU, sin el GGUF).
No versionar el zip.

El traductor (Llama 3.2 1B, ~770 MB) no va en este archivo. ILSA lo baja
la primera vez que se abre, desde el GitHub Release `ilsa-llama-1b`, a
%LOCALAPPDATA%\ILSA\models\
