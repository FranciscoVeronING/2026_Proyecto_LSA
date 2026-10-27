# Poses del avatar

Esta rama publica los archivos `.pose` del avatar. El nombre de cada archivo en `poses/`, sin extensión, es la glosa. El servicio escucha en `0.0.0.0`. El puerto sale de `PORT` y, si no está definida, es `8080`.

```text
app/config.py        rutas, límites y tipos de archivo
app/fingerspell.py   deletreo
app/main.py          aplicación y arranque
poses/               glosas
```

Si falta `poses/` al arrancar, el proceso termina y no abre el puerto. Acepta `GET`, `HEAD` y `OPTIONS` desde cualquier origen. El navegador puede leer el encabezado `X-Count`.

## Archivos

`GET /health` responde `200` con `{"status": "ok"}`. No comprueba si una glosa existe.

`GET /tposes/{GLOSA}.pose` devuelve ese archivo. Si existe, responde `200` con `Content-Type: application/octet-stream`. Si no existe, responde `404`. No deletrea el nombre que falta.

La misma carpeta también publica `.json`, `.gltf`, `.bin` y `.fbx` cuando se agregan ahí. Sus tipos son `application/json`, `model/gltf+json` y `application/octet-stream`.

## Glosas

ILSA arma el nombre antes de pedir el archivo:

```text
/tposes/{GLOSA}.pose
```

1. Pasar la glosa a mayúsculas.
2. Quitar el sufijo de género `_A`, `_a` o `_RA`. `ABOGADO_A` queda `ABOGADO`. `PROFESOR_RA` queda `PROFESOR`. `EL_ELLA` queda `EL`.
3. Sacar las tildes de las vocales: `Á É Í Ó Ú Ü` pasan a `A E I O U U`. `CAFÉ` queda `CAFE`.
4. Borrar espacios, `_`, guiones, paréntesis y cualquier carácter que no sea letra o número.
5. Si una seña tiene más de un significado, concatenar las glosas sin separador.

| Significado | Archivo |
| --- | --- |
| Hoy / ahora | `HOYAHORA.pose` |
| Anual / por período | `ANUALPORPERIODO.pose` |
| Sprite / 7Up | `SPRITE7UP.pose` |
| Camarero / mozo | `CAMAREROMOZO.pose` |
| Países Bajos / Holanda | `PAISESBAJOSHOLANDA.pose` |
| Época / período | `EPOCAPERIODO.pose` |

`PERIODO.pose` es otra grabación y se pide aparte. Dentro de una palabra, `Ñ` pasa a `N`: `ESPAÑA` queda `ESPANA` y `MAÑANA` queda `MANANA`.

Dos sustituciones valen solo cuando son la glosa completa:

- La letra `Ñ` se pide como `ENIE.pose`. La letra `N` sigue en `N.pose`.
- La glosa `PAPA`, como parentesco, se pide como `PADRE.pose`. `PAPA.pose` es la papa como alimento. `PAPEL` y `PAPASFRITAS` no cambian.

Los nombres de `poses/` son ASCII. Un archivo con `Ñ` o con tilde hace fallar la importación del contenedor en Cloud Run.

## Deletreo

`GET /fingerspell` pide solo letras, en el orden enviado. Las letras van en `letters`, separadas por comas o repetidas:

```text
GET /fingerspell?letters=H,O,L,A
GET /fingerspell?letters=CH,I,L,E
GET /fingerspell?letters=N&letters=ENIE
```

Las letras válidas son `A`–`Z`, `CH` y `ENIE`. `Ñ` se resuelve como `ENIE`. Una palabra, un número u otra glosa responde `400`.

El máximo es 20 letras. El abecedario tiene 27 letras más `CH`, así que una solicitud no puede pedirlas todas. Si se pasa el límite, responde `400` con `received` y `limit`.

Si todas existen, responde `200` con `multipart/mixed`. Cada parte es el `.pose`, con `Content-Disposition` (`H.pose`), `X-Index` desde 0 y `X-Letter` (`H`, `CH` o `ENIE`). `X-Count` indica cuántas volvieron. Si falta una letra, responde `404` y no entrega un deletreo incompleto. Los errores JSON usan `detail` y, cuando corresponde, `letter`.

## Ejecución

Python 3.10 o superior, desde la raíz:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

Las dependencias son FastAPI `0.115.12` y Uvicorn `0.34.2`.

La imagen usa `python:3.10-slim`, instala las dependencias y copia `app/` y `poses/`. El arranque es `python -m app.main`.

```powershell
docker build -t tpose-server .
docker run --rm -p 8080:8080 tpose-server
```

Cloud Run, desde la raíz, en `southamerica-east1` y el proyecto `ilsa-511622`:

```powershell
gcloud run deploy tpose-server --source . --region southamerica-east1 --port 8080 --allow-unauthenticated --project ilsa-511622
```

Si `gcloud` tiene guardado el número de proyecto, hace falta `--project ilsa-511622` o `gcloud config set project ilsa-511622`.
