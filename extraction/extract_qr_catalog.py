"""Paso 1 del diccionario: PDF del Señario CAS → catálogo JSON (glosa + URL).

Los QR del Señario no están embebidos como JPEG: son dibujos vectoriales
(miles de cuadrados). Extraer las fotos del PDF no trae el código. Este
script rasteriza cada página (escala ×3) y ZXing (lector de códigos de
barras, incluido QR) lee todos los códigos de la página.

Flujo:

1. Busca `extraction/*.pdf` (volumen 1 = el más chico / nombre «Libro»;
   volumen 2 = el que contiene «2» en el nombre).
2. Por página: pixmap RGB → `zxingcpp.read_barcodes`.
3. Asocia cada QR al texto más cercano *debajo* de su centro (en el vol. 2
   el nombre está más lejos: se busca en una franja vertical de ±120 pt).
4. Filtra ruido (números de página, «Señario II», instrucciones de escaneo).
5. Clasifica: `sign` (YouTube de una seña), `playlist`, `note` (bit.ly).
6. Escribe `extraction/out/qr_catalog.json`.

No hace falta un PNG por seña: el JSON ya es glosa + URL.

  py -3.11 extraction/extract_qr_catalog.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import zxingcpp

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
CATALOG_PATH = OUT_DIR / "qr_catalog.json"

NOISE = {
    "señario",
    "senario",
    "ii",
    "vol",
    "volumen",
    "cas",
    "lsa",
    "notas",
    "culturales",
    "categoría",
    "categoria",
    "comodidad",
    "podés",
    "podes",
    "acceder",
    "lista",
    "reproducción",
    "reproduccion",
    "pasos",
    "escaneá",
    "escanea",
    "código",
    "codigo",
    "enlace",
    "video",
    "videos",
    "listo",
    "mirá",
    "mira",
    "simple",
    "simples",
    "cómo",
    "como",
    "hago",
    "para",
    "ver",
    "los",
    "al",
    "ya",
    "de",
    "la",
    "el",
    "en",
    "tu",
}

PLAYLIST_HINTS = ("reproducción", "reproduccion", "lista de reproducción", "playlist")


def find_pdfs() -> list[Path]:
    """PDF sueltos en extraction/, ordenados por tamaño (vol. 1 suele ser el menor)."""
    pdfs = sorted(ROOT.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if not pdfs:
        raise FileNotFoundError(f"No hay PDF en {ROOT}")
    return pdfs


def volume_name(path: Path) -> str:
    """Etiqueta `vol1` / `vol2` a partir del nombre de archivo."""
    name = path.name.lower()
    if "volumen 1" in name or "volumen1" in name or "libro" in name:
        return "vol1"
    if "2" in name:
        return "vol2"
    return path.stem


def is_noise_token(token: str) -> bool:
    """True si la palabra no puede ser glosa (página, CAS, instrucciones, etc.)."""
    cleaned = re.sub(r"[•·|]+", "", token).strip()
    if not cleaned:
        return True
    if cleaned.isdigit():
        return True
    if cleaned.lower() in NOISE:
        return True
    if cleaned.startswith("http"):
        return True
    if re.fullmatch(r"\d+", cleaned):
        return True
    return False


def qr_center_pdf(result: zxingcpp.Result, scale: float) -> tuple[float, float]:
    """Centro del QR en puntos PDF (divide por la escala del pixmap)."""
    pos = result.position
    xs = [pos.top_left.x, pos.top_right.x, pos.bottom_left.x, pos.bottom_right.x]
    ys = [pos.top_left.y, pos.top_right.y, pos.bottom_left.y, pos.bottom_right.y]
    return ((min(xs) + max(xs)) / 2.0) / scale, ((min(ys) + max(ys)) / 2.0) / scale


def page_words(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """Palabras de la página con su centro (x, y), sin tokens de ruido."""
    words = []
    for item in page.get_text("words"):
        x0, y0, x1, y1, text = item[:5]
        if is_noise_token(text):
            continue
        words.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, text))
    return words


def label_for_qr(cx: float, cy: float, words: list[tuple[float, float, str]]) -> str:
    """Glosa: palabras debajo del QR, alineadas en X (±120 pt) y en la misma línea."""
    below = []
    for wx, wy, text in words:
        if wy <= cy - 2:
            continue
        if abs(wx - cx) > 120:
            continue
        below.append((wy - cy, abs(wx - cx), wx, text))
    if not below:
        return ""
    below.sort()
    nearest_dy = below[0][0]
    line = [(wx, text) for dy, _dx, wx, text in below if dy <= nearest_dy + 10]
    line.sort()
    return " ".join(text for _wx, text in line).strip()


def decode_page(page: pymupdf.Page, scale: float) -> list[zxingcpp.Result]:
    """Rasteriza la página y devuelve todos los QR. `scale=3` es el default usable."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return list(zxingcpp.read_barcodes(bgr, formats=zxingcpp.BarcodeFormat.QRCode))


def is_sign_url(url: str) -> bool:
    return any(host in url.lower() for host in ("youtu.be", "youtube.com"))


def extract_pdf(path: Path, scale: float = 3.0) -> list[dict]:
    """Recorre un volumen y arma filas {volume, page, label, url, kind, x, y}."""
    volume = volume_name(path)
    doc = pymupdf.open(path)
    rows: list[dict] = []
    print(f"\n{path.name}: {doc.page_count} páginas")
    for page_index in range(doc.page_count):
        page = doc[page_index]
        try:
            codes = decode_page(page, scale)
        except Exception as error:
            print(f"  página {page_index + 1}: error {error}")
            continue
        if not codes:
            continue
        words = page_words(page)
        page_text = page.get_text("text").lower()
        playlist = any(hint in page_text for hint in PLAYLIST_HINTS)
        print(f"  página {page_index + 1}: {len(codes)} QR")
        seen_urls = set()
        for code in codes:
            url = (code.text or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            cx, cy = qr_center_pdf(code, scale)
            label = label_for_qr(cx, cy, words)
            kind = "playlist" if playlist and "list" in url.lower() or playlist and not label else "sign"
            if playlist and not label:
                kind = "playlist"
            if not is_sign_url(url) and "bit.ly" in url.lower():
                kind = "note"
            rows.append(
                {
                    "volume": volume,
                    "pdf": path.name,
                    "page": page_index + 1,
                    "label": label,
                    "url": url,
                    "kind": kind,
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                }
            )
    doc.close()
    return rows


def slug(label: str, fallback: str) -> str:
    if not label:
        return fallback
    text = label.strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for pdf in find_pdfs():
        rows.extend(extract_pdf(pdf))

    used = {}
    for row in rows:
        base = slug(row["label"], f"{row['volume']}_p{row['page']}")
        count = used.get(base, 0)
        used[base] = count + 1
        row["id"] = base if count == 0 else f"{base}_{count + 1}"

    CATALOG_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    signs = [r for r in rows if r["kind"] == "sign"]
    labeled = [r for r in signs if r["label"]]
    print(f"\nTotal QR: {len(rows)}")
    print(f"Señas: {len(signs)} (con nombre: {len(labeled)})")
    print(f"Sin nombre: {len(signs) - len(labeled)}")
    print(f"Otros (playlist/nota): {len(rows) - len(signs)}")
    print(f"Catálogo: {CATALOG_PATH}")
    unnamed = [r for r in signs if not r["label"]][:12]
    if unnamed:
        print("Ejemplos sin etiqueta:")
        for row in unnamed:
            print(f"  {row['volume']} p{row['page']} {row['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
