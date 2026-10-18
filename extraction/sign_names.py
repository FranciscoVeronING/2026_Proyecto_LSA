"""Nombres canónicos de señas: sin vol1_/vol2_ y sin glosas vecinas pegadas.

El catálogo QR juntó a veces dos tarjetas (`F G`, `ABECEDARIO A`).
La seña del archivo es la primera letra / primera glosa útil.

  vol1_F G            → F
  vol1_ABECEDARIO A   → A
  vol1_HOLA           → HOLA
  vol2_Agua           → Agua
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POSE_DIR = ROOT / "out" / "poses"
VIDEO_DIR = ROOT / "out" / "videos"
CATALOG = ROOT / "out" / "qr_catalog.json"

LETTER_TOKENS = set("abcdefghijklmnopqrstuvwxyz") | {"ch", "ñ"}
SAFE_CHARS = re.compile(r'[<>:"/\\|?*]+')


def fold_keep_enye(text: str) -> str:
    mapped: list[str] = []
    for char in unicodedata.normalize("NFKD", text):
        if char.lower() == "ñ" or char == "\u0303":
            if char == "\u0303" and mapped and mapped[-1] == "n":
                mapped[-1] = "ñ"
            elif char.lower() == "ñ":
                mapped.append("ñ")
            continue
        if unicodedata.combining(char):
            continue
        mapped.append(char.lower())
    return "".join(mapped).strip()


def strip_volume(stem: str) -> str:
    name = stem
    while True:
        for prefix in ("vol1_", "vol2_"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        else:
            return name


def pretty_letter(token: str) -> str:
    key = fold_keep_enye(token)
    if key == "ñ":
        return "Ñ"
    if key == "ch":
        return "CH"
    if len(key) == 1:
        return key.upper()
    return token.strip()


def is_letter_token(token: str) -> bool:
    return fold_keep_enye(token) in LETTER_TOKENS


def fold_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", fold_keep_enye(text)).strip()


def fold_loose(text: str) -> str:
    """Compara nombres ignorando espacios y puntuación: Pañuelo (x) == Pañuelo(x)."""
    return re.sub(r"[^a-z0-9ñ]+", "", fold_keep_enye(text))


def sanitize_stem(raw: str) -> str:
    text = SAFE_CHARS.sub("_", str(raw)).strip().strip(".")
    return re.sub(r"\s+", " ", text)


def desired_stem(stem: str) -> str:
    """Nombre sin vol1_/vol2_. Si el QR pegó dos letras (`F G`), deja la primera."""
    name = strip_volume(stem).strip()
    if not name:
        return stem
    parts = [part for part in re.split(r"\s+", name) if part]
    folded = [fold_keep_enye(part) for part in parts]
    if folded and folded[0] == "abecedario":
        letters = [part for part, key in zip(parts[1:], folded[1:]) if key in LETTER_TOKENS]
        if letters:
            return pretty_letter(letters[0])
    if parts and all(key in LETTER_TOKENS for key in folded):
        return pretty_letter(parts[0])
    return sanitize_stem(name)


def list_stems(directory: Path, suffixes: set[str]) -> list[str]:
    if not directory.exists():
        return []
    return [path.stem for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]


def match_stem(queries: list[str], stems: list[str]) -> str | None:
    """Enlaza un id/label del catálogo con el stem real en disco (nombres ya editados a mano)."""
    if not stems:
        return None
    by_fold: dict[str, list[str]] = {}
    by_loose: dict[str, list[str]] = {}
    for stem in stems:
        by_fold.setdefault(fold_spaces(stem), []).append(stem)
        by_loose.setdefault(fold_loose(stem), []).append(stem)
    ranked = sorted(stems, key=lambda item: len(fold_spaces(item)), reverse=True)

    def unique(bucket: dict[str, list[str]], key: str) -> str | None:
        hits = bucket.get(key) or []
        return hits[0] if len(hits) == 1 else None

    for query in queries:
        text = strip_volume(str(query or "")).strip()
        if not text:
            continue
        found = unique(by_fold, fold_spaces(text)) or unique(by_loose, fold_loose(text))
        if found:
            return found
        parts = text.split()
        if len(parts) >= 2 and fold_keep_enye(parts[0]) == fold_keep_enye(parts[1]):
            collapsed = " ".join(parts[1:])
            found = unique(by_fold, fold_spaces(collapsed)) or unique(by_loose, fold_loose(collapsed))
            if found:
                return found
        hay = fold_spaces(text)
        for stem in ranked:
            prefix = fold_spaces(stem)
            if hay == prefix or hay.startswith(prefix + " "):
                return stem
    return None


def catalog_disk_stem(row: dict) -> str:
    """Stem para extraer/reanudar: el archivo que ya está en poses/ o videos/, sin vol1_."""
    queries = [str(row.get("id") or ""), str(row.get("label") or "")]
    found = match_stem(queries, list_stems(POSE_DIR, {".pose"}))
    if found:
        return found
    found = match_stem(queries, list_stems(VIDEO_DIR, {".mp4", ".webm", ".mkv", ".m4v"}))
    if found:
        return found
    raw = sanitize_stem(row.get("id") or row.get("label") or f"p{row.get('page', 0)}")
    if not raw:
        raw = f"p{row.get('page', 0)}"
    return strip_volume(raw)


def find_cached_video(stem: str) -> Path | None:
    """mp4 cuyo nombre coincide con el stem, o lo contiene como prefijo (DADOS vs DADOS DAMAS)."""
    if not VIDEO_DIR.exists():
        return None
    suffixes = {".mp4", ".webm", ".mkv", ".m4v"}
    for path in VIDEO_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes and path.stem == stem:
            return path
    hit = match_stem([stem], list_stems(VIDEO_DIR, suffixes))
    if hit:
        for path in VIDEO_DIR.iterdir():
            if path.is_file() and path.suffix.lower() in suffixes and path.stem == hit:
                return path
    return None


def unique_mapping(stems: list[str]) -> dict[str, str]:
    """old_stem → new_stem. Si dos archivos piden A, el segundo queda A_2."""
    ranked: list[tuple[int, str]] = []
    for stem in stems:
        name = strip_volume(stem)
        parts = [part for part in re.split(r"\s+", name) if part]
        folded = [fold_keep_enye(part) for part in parts]
        if len(parts) == 1 and folded and folded[0] in LETTER_TOKENS:
            rank = 0
        elif folded and folded[0] == "abecedario":
            rank = 1
        elif parts and all(key in LETTER_TOKENS for key in folded):
            rank = 2
        else:
            rank = 3
        ranked.append((rank, stem))
    ranked.sort(key=lambda item: (item[0], item[1].lower()))

    taken: set[str] = set()
    mapping: dict[str, str] = {}
    for _rank, stem in ranked:
        want = desired_stem(stem)
        name = strip_volume(stem)
        parts = [part for part in re.split(r"\s+", name) if part]
        folded = [fold_keep_enye(part) for part in parts]
        candidates = [want]
        if _rank == 2 and len(parts) >= 2:
            second = pretty_letter(parts[1])
            if second not in candidates:
                candidates.append(second)
        chosen = None
        for candidate in candidates:
            if candidate not in taken:
                chosen = candidate
                break
        if chosen is None:
            index = 2
            chosen = f"{want}_{index}"
            while chosen in taken:
                index += 1
                chosen = f"{want}_{index}"
        taken.add(chosen)
        mapping[stem] = chosen
    return mapping


def rename_files(directory: Path, mapping: dict[str, str], suffixes: tuple[str, ...]) -> int:
    if not directory.exists():
        return 0
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    if not files:
        return 0
    tmp_moves: list[tuple[Path, Path, Path]] = []
    for index, path in enumerate(files):
        new_stem = mapping.get(path.stem)
        if not new_stem or new_stem == path.stem:
            continue
        dest = path.with_name(new_stem + path.suffix)
        tmp = path.with_name(f"__ren_{index:04d}__{path.suffix}")
        tmp_moves.append((path, tmp, dest))
    for src, tmp, _dest in tmp_moves:
        src.rename(tmp)
    renamed = 0
    for _src, tmp, dest in tmp_moves:
        if dest.exists():
            raise FileExistsError(f"Destino ocupado: {dest.name}")
        tmp.rename(dest)
        renamed += 1
    return renamed


def update_catalog(mapping: dict[str, str]) -> None:
    if not CATALOG.exists():
        return
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))
    changed = 0
    for row in rows:
        old = f"{row.get('volume', 'vol')}_{row.get('id') or row.get('label') or ''}"
        old = SAFE_CHARS.sub("_", old)
        new = mapping.get(old)
        if not new:
            raw = str(row.get("id") or row.get("label") or "")
            new = mapping.get(raw) or desired_stem(raw)
        if new and row.get("id") != new:
            row["id"] = new
            changed += 1
    CATALOG.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catálogo: {changed} id actualizados")


def original_catalog_ids() -> list[tuple[dict, str, str]]:
    """(row, old_file_stem, original_id) como estaban antes del rename."""
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))
    used: dict[str, int] = {}
    out: list[tuple[dict, str, str]] = []
    for row in rows:
        volume = str(row.get("volume") or "vol")
        label = str(row.get("label") or "").strip()
        fallback = f"{volume}_p{row.get('page', 0)}"
        base = label if label else fallback
        base = re.sub(r'[\\/:*?"<>|]+', "_", base)
        base = re.sub(r"\s+", " ", base).strip()
        count = used.get(base, 0)
        used[base] = count + 1
        orig_id = base if count == 0 else f"{base}_{count + 1}"
        old_stem = f"{volume}_{orig_id}"
        out.append((row, old_stem, orig_id))
    return out


def revert() -> None:
    """Vuelve poses/videos/catálogo a vol1_F G, vol1_HOLA, etc."""
    entries = original_catalog_ids()
    old_stems = [old for row, old, _orig in entries if row.get("kind") == "sign" and row.get("url")]
    mapping = unique_mapping(old_stems)
    reverse = {new: old for old, new in mapping.items()}
    current = {path.stem for path in POSE_DIR.glob("*.pose")} if POSE_DIR.exists() else set()
    missing = sorted(stem for stem in current if stem not in reverse)
    print(f"Revertir {len(current)} .pose; sin mapeo: {len(missing)}")
    if missing:
        print("  ", ", ".join(missing[:20]))
    n_pose = rename_files(POSE_DIR, reverse, (".pose",))
    n_vid = rename_files(VIDEO_DIR, reverse, (".mp4", ".webm", ".mkv", ".m4v"))
    print(f"Restaurados: {n_pose} .pose, {n_vid} videos")

    rows = json.loads(CATALOG.read_text(encoding="utf-8"))
    orig_by_url = {row.get("url"): orig for row, _old, orig in entries}
    changed = 0
    for row in rows:
        orig = orig_by_url.get(row.get("url"))
        if orig and row.get("id") != orig:
            row["id"] = orig
            changed += 1
    CATALOG.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catálogo: {changed} id restaurados")


def main() -> None:
    import sys

    if "--revert" in sys.argv:
        revert()
        return
    print("El rename automático está desactivado. Para volver a vol1_/vol2_:")
    print("  py -3.11 extraction/sign_names.py --revert")


if __name__ == "__main__":
    main()
