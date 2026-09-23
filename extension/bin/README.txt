El botón «Descargar ILSA» de la extensión sirve el zip que genere:

  packaging\build_exe.bat

Ese script deja acá `ILSA.zip` (ILSA.exe + clasificador CPU, sin GGUF).
No versionar el zip.

El traductor corre aparte: `python run_semantic_server.py` + ngrok.
ILSA baja la URL de `semantic_url.txt` en el release `ilsa-llama-1b`. Ver docs/SEMANTICO_NUBE.md.
