import mimetypes
import os
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

mimetypes.add_type("application/json", ".json")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/octet-stream", ".bin")
mimetypes.add_type("application/octet-stream", ".fbx")
mimetypes.add_type("application/octet-stream", ".pose")

BASE_DIR = Path(__file__).resolve().parent
POSES_DIR = BASE_DIR / "poses"
MAX_LETRAS_POR_SOLICITUD = 20
LETRAS_DELETREO = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"ENIE", "CH"}

app = FastAPI(title="Avatar T-Pose Assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Cantidad"],
)


def letras_pedidas(valores: list[str]) -> list[str]:
    letras = []
    for valor in valores:
        letras.extend(parte.strip() for parte in valor.split(","))
    return [letra for letra in letras if letra]


def canon_letra(token: str) -> str:
    letra = token.strip().upper()
    if letra in {"Ñ", "ENIE"}:
        return "ENIE"
    return letra


def respuesta_deletreo(items: list[tuple[str, Path]]) -> Response:
    boundary = f"deletreo{uuid.uuid4().hex}"
    cuerpo = bytearray()
    for indice, (letra, path) in enumerate(items):
        encabezado = (
            f"--{boundary}\r\n"
            "Content-Type: application/octet-stream\r\n"
            f'Content-Disposition: attachment; filename="{path.name}"\r\n'
            f"X-Indice: {indice}\r\n"
            f"X-Letra: {letra}\r\n"
            "\r\n"
        )
        cuerpo.extend(encabezado.encode("ascii"))
        cuerpo.extend(path.read_bytes())
        cuerpo.extend(b"\r\n")
    cuerpo.extend(f"--{boundary}--\r\n".encode("ascii"))
    return Response(
        content=bytes(cuerpo),
        media_type=f"multipart/mixed; boundary={boundary}",
        headers={"X-Cantidad": str(len(items))},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fingerspell")
def fingerspell(letters: list[str] = Query(..., min_length=1)):
    pedidas = letras_pedidas(letters)
    if not pedidas:
        return JSONResponse(
            status_code=400,
            content={"detalle": "La solicitud no incluye letras."},
        )
    if len(pedidas) > MAX_LETRAS_POR_SOLICITUD:
        return JSONResponse(
            status_code=400,
            content={
                "detalle": (
                    f"Se pueden pedir hasta {MAX_LETRAS_POR_SOLICITUD} letras "
                    "por solicitud."
                ),
                "recibidas": len(pedidas),
                "limite": MAX_LETRAS_POR_SOLICITUD,
            },
        )

    archivos = []
    for token in pedidas:
        letra = canon_letra(token)
        if letra not in LETRAS_DELETREO:
            return JSONResponse(
                status_code=400,
                content={
                    "detalle": f"{token} no es una letra del deletreo.",
                    "letra": token,
                },
            )
        path = POSES_DIR / f"{letra}.pose"
        if not path.is_file():
            return JSONResponse(
                status_code=404,
                content={
                    "detalle": f"No existe el archivo de la letra {letra}.",
                    "letra": letra,
                },
            )
        archivos.append((letra, path))

    return respuesta_deletreo(archivos)


if not POSES_DIR.is_dir():
    raise RuntimeError(f"No se encontró la carpeta de T-Pose: {POSES_DIR}")

app.mount("/tposes", StaticFiles(directory=str(POSES_DIR)), name="tposes")

port = int(os.environ.get("PORT", 8080))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
