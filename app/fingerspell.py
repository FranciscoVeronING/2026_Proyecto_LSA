import uuid
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response

from app.config import FINGERSPELL_LETTERS, MAX_LETTERS_PER_REQUEST, POSES_DIR


def parse_letters(values: list[str]) -> list[str]:
    letters = []
    for value in values:
        letters.extend(part.strip() for part in value.split(","))
    return [letter for letter in letters if letter]


def canonical_letter(token: str) -> str:
    letter = token.strip().upper()
    if letter in {"Ñ", "ENIE"}:
        return "ENIE"
    return letter


def multipart_response(items: list[tuple[str, Path]]) -> Response:
    boundary = f"fingerspell{uuid.uuid4().hex}"
    body = bytearray()
    for index, (letter, path) in enumerate(items):
        header = (
            f"--{boundary}\r\n"
            "Content-Type: application/octet-stream\r\n"
            f'Content-Disposition: attachment; filename="{path.name}"\r\n'
            f"X-Index: {index}\r\n"
            f"X-Letter: {letter}\r\n"
            "\r\n"
        )
        body.extend(header.encode("ascii"))
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    return Response(
        content=bytes(body),
        media_type=f"multipart/mixed; boundary={boundary}",
        headers={"X-Count": str(len(items))},
    )


def fingerspell(letters: list[str] = Query(..., min_length=1)):
    requested = parse_letters(letters)
    if not requested:
        return JSONResponse(
            status_code=400,
            content={"detail": "La solicitud no incluye letras."},
        )
    if len(requested) > MAX_LETTERS_PER_REQUEST:
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    f"Se pueden pedir hasta {MAX_LETTERS_PER_REQUEST} letras "
                    "por solicitud."
                ),
                "received": len(requested),
                "limit": MAX_LETTERS_PER_REQUEST,
            },
        )

    files = []
    for token in requested:
        letter = canonical_letter(token)
        if letter not in FINGERSPELL_LETTERS:
            return JSONResponse(
                status_code=400,
                content={"detail": f"{token} no es una letra del deletreo.", "letter": token},
            )
        path = POSES_DIR / f"{letter}.pose"
        if not path.is_file():
            return JSONResponse(
                status_code=404,
                content={"detail": f"No existe el archivo de la letra {letter}.", "letter": letter},
            )
        files.append((letter, path))
    return multipart_response(files)


def register(app: FastAPI) -> None:
    app.get("/fingerspell")(fingerspell)
