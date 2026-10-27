# Avatar pose files

This branch serves the avatar `.pose` files. Each filename in `poses/`, without the extension, is the gloss. The service listens on `0.0.0.0`. The port comes from `PORT` and defaults to `8080`.

```text
app/config.py        paths, limits, and media types
app/fingerspell.py   fingerspelling
app/main.py          application and startup
poses/               gloss files
```

If `poses/` is missing at startup, the process exits and does not open the port. It accepts `GET`, `HEAD`, and `OPTIONS` from any origin. Browsers can read the `X-Count` header.

## Files

`GET /health` returns `200` with `{"status": "ok"}`. It does not check whether a gloss exists.

`GET /tposes/{GLOSS}.pose` returns that file. A match returns `200` with `Content-Type: application/octet-stream`. A missing file returns `404`. The server does not fingerspell the missing name.

The same directory also serves `.json`, `.gltf`, `.bin`, and `.fbx` when those files are added. Their types are `application/json`, `model/gltf+json`, and `application/octet-stream`.

## Glosses

The caller builds the name before requesting the file:

```text
/tposes/{GLOSS}.pose
```

1. Uppercase the gloss.
2. Drop a gender suffix `_A`, `_a`, or `_RA`. `ABOGADO_A` becomes `ABOGADO`. `PROFESOR_RA` becomes `PROFESOR`. `EL_ELLA` becomes `EL`.
3. Remove vowel accents: `Á É Í Ó Ú Ü` become `A E I O U U`. `CAFÉ` becomes `CAFE`.
4. Remove spaces, `_`, hyphens, parentheses, and any character that is not a letter or digit.
5. If one sign has more than one meaning, concatenate the glosses with no separator.

| Meaning | File |
| --- | --- |
| Today / now | `HOYAHORA.pose` |
| Annual / per period | `ANUALPORPERIODO.pose` |
| Sprite / 7Up | `SPRITE7UP.pose` |
| Waiter | `CAMAREROMOZO.pose` |
| Netherlands / Holland | `PAISESBAJOSHOLANDA.pose` |
| Era / period | `EPOCAPERIODO.pose` |

`PERIODO.pose` is a separate recording. Inside a word, `Ñ` becomes `N`: `ESPAÑA` becomes `ESPANA` and `MAÑANA` becomes `MANANA`.

Two replacements apply only when they are the whole gloss:

- The letter `Ñ` is requested as `ENIE.pose`. The letter `N` stays `N.pose`.
- The kinship gloss `PAPA` is requested as `PADRE.pose`. `PAPA.pose` is the food item. `PAPEL` and `PAPASFRITAS` stay unchanged.

Filenames in `poses/` are ASCII. A filename with `Ñ` or an accent makes Cloud Run fail while importing the container.

## Fingerspelling

`GET /fingerspell` requests letters only, in the order sent. Letters go in `letters`, separated by commas or repeated:

```text
GET /fingerspell?letters=H,O,L,A
GET /fingerspell?letters=CH,I,L,E
GET /fingerspell?letters=N&letters=ENIE
```

Valid letters are `A`–`Z`, `CH`, and `ENIE`. `Ñ` resolves to `ENIE`. A word, a number, or any other gloss returns `400`.

The limit is 20 letters. The alphabet has 27 letters plus `CH`, so one call cannot request all of them. Over the limit, the response is `400` with `received` and `limit`.

When every letter exists, the response is `200` with `multipart/mixed`. Each part is the `.pose`, with `Content-Disposition` (`H.pose`), `X-Index` starting at 0, and `X-Letter` (`H`, `CH`, or `ENIE`). `X-Count` is how many parts came back. If one letter file is missing, the response is `404` and no partial spelling is returned. JSON errors use `detail` and, when relevant, `letter`.

## Run

Python 3.10 or newer, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

Dependencies are FastAPI `0.115.12` and Uvicorn `0.34.2`.

The image uses `python:3.10-slim`, installs the dependencies, and copies `app/` and `poses/`. Startup is `python -m app.main`.

```powershell
docker build -t tpose-server .
docker run --rm -p 8080:8080 tpose-server
```

Cloud Run, from the repository root, region `southamerica-east1`, project `ilsa-511622`:

```powershell
gcloud run deploy tpose-server --source . --region southamerica-east1 --port 8080 --allow-unauthenticated --project ilsa-511622
```

If `gcloud` has the project number stored, pass `--project ilsa-511622` or run `gcloud config set project ilsa-511622`.
