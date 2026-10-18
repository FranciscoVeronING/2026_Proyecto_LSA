"""Arma una frase de glosas en memoria para el canvas de Meet.

No escribe mp4. Reusa el compositor (`render_pose_pair.prepare_sign` y
`compose_sequence`): trim, park, EMA, canvas y puentes. El navegador dibuja
los frames con `canvas.captureStream`.

Los `.pose` viven en `extraction/out/poses/` (no están en git).
"""

from __future__ import annotations

import importlib.util
import re
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
POSE_DIR = ROOT / "out" / "poses"
_MAX_SIGNS = 24
_MAX_FRAMES = 1200
_TOKEN = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_DIGITS = {
    "0": "CERO",
    "1": "UNO",
    "2": "DOS",
    "3": "TRES",
    "4": "CUATRO",
    "5": "CINCO",
    "6": "SEIS",
    "7": "SIETE",
    "8": "OCHO",
    "9": "NUEVE",
}

# Contornos de FaceMesh (sin iris: Holistic tiene 468 puntos, el iris empieza en 468).
# Óvalo, cejas, ojos, nariz y labios. La teselación completa no se lee en medio Meet.
_FACE_EDGES = (
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19),
    (19, 94), (94, 2), (98, 97), (97, 2), (2, 326), (326, 327), (327, 294),
    (294, 278), (278, 344), (344, 440), (440, 275), (275, 4), (4, 45),
    (45, 220), (220, 115), (115, 48), (48, 64), (64, 98),
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 409),
    (409, 291), (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
    (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415),
    (415, 308),
)
# Hombros, codos, muñecas, caderas. Sin cara de pose, sin puntas y sin piernas.
_POSE_IDX = (11, 12, 13, 14, 15, 16, 23, 24)
_FACE_START = 33
_LEFT_HAND = 33 + 468
_RIGHT_HAND = _LEFT_HAND + 21

_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()
_RP = None


def _fail(error: str, missing: list[str] | None = None) -> dict:
    return {
        "ok": False,
        "error": error,
        "missing": missing or [],
        "frames": [],
    }


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text or "")


def _load_rp():
    global _RP
    if _RP is not None:
        return _RP
    path = ROOT / "render_pose_pair.py"
    spec = importlib.util.spec_from_file_location("render_pose_pair", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RP = module
    return module


def _args(rp):
    import argparse

    return argparse.Namespace(
        fps=rp.DEFAULT_FPS,
        trim_mode=rp.DEFAULT_TRIM_MODE,
        trim=rp.DEFAULT_TRIM,
        missing_conf=rp.MISSING_CONF_MIN,
        missing_xy=rp.MISSING_XY_PX,
        fill_missing=rp.FILL_MISSING,
        fill_max_gap=rp.FILL_MAX_GAP,
        park_idle=rp.PARK_IDLE_HANDS,
        ema=rp.DEFAULT_EMA,
        center=rp.CENTER_ON_SHOULDERS,
        canvas_width=rp.CANVAS_WIDTH,
        canvas_height=rp.CANVAS_HEIGHT,
        shoulder_px=rp.CANVAS_SHOULDER_PX,
        shoulder_y=rp.CANVAS_SHOULDER_Y,
        hide_legs=rp.HIDE_LEGS,
        bridge_min=rp.BRIDGE_MIN_FRAMES,
        bridge_max=rp.BRIDGE_MAX_FRAMES,
        bridge_dist_short=rp.BRIDGE_DIST_SHORT_PX,
        bridge_dist_long=rp.BRIDGE_DIST_LONG_PX,
        easing=rp.BRIDGE_EASING,
        bridge_fixed=rp.BRIDGE_FIXED_FRAMES,
    )


def _append_sign(found: list[tuple[str, Path]], query: str, path: Path) -> bool:
    if len(found) >= _MAX_SIGNS:
        return False
    found.append((query, path))
    return True


def _spell_fallback(rp, query: str) -> list[str]:
    """Una palabra sin clip se parte en letras. Una letra sola no se vuelve a partir."""
    try:
        letters = rp.expand_spell(query)
    except ValueError:
        return []
    if len(letters) < 2:
        return []
    return letters


def _resolve_chain(rp, tokens: list[str]) -> tuple[list[tuple[str, Path]], list[str]]:
    """Junta tokens de a lo más largo (BUENAS TARDES). Si no hay clip, deletrea."""
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    index = 0
    while index < len(tokens):
        if len(found) >= _MAX_SIGNS:
            missing.extend(tokens[index:])
            break
        matched = False
        for size in (3, 2, 1):
            if index + size > len(tokens):
                continue
            raw = " ".join(tokens[index : index + size])
            query = _DIGITS.get(raw, raw)
            try:
                path = rp.resolve_sign(query)
            except FileNotFoundError:
                continue
            if not _append_sign(found, query, path):
                missing.append(query)
            index += size
            matched = True
            break
        if matched:
            continue
        query = _DIGITS.get(tokens[index], tokens[index])
        letters = _spell_fallback(rp, query)
        if not letters:
            missing.append(tokens[index])
            index += 1
            continue
        for letter in letters:
            if len(found) >= _MAX_SIGNS:
                missing.append(letter)
                continue
            try:
                path = rp.resolve_sign(letter)
            except FileNotFoundError:
                missing.append(letter)
                continue
            _append_sign(found, letter, path)
        index += 1
    return found, missing


def _prepare(rp, path: Path, args):
    key = str(path.resolve())
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    pose, _filled, _parked = rp.prepare_sign(path, args)
    with _LOCK:
        _CACHE[key] = pose
    return pose


def _face_layout() -> tuple[list[int], list[list[int]]]:
    order: list[int] = []
    seen: dict[int, int] = {}
    edges: list[list[int]] = []
    for a, b in _FACE_EDGES:
        for point in (a, b):
            if point not in seen:
                seen[point] = len(order)
                order.append(point)
        edges.append([seen[a], seen[b]])
    return order, edges


def _pack(data: np.ndarray, conf: np.ndarray, indices: list[int] | tuple[int, ...]) -> list[list[int]]:
    if not indices:
        return [[] for _ in range(len(data))]
    sel = np.asarray(indices, dtype=np.int32)
    pts = np.rint(data[:, 0, sel, :2]).astype(np.int32)
    bad = conf[:, 0, sel] < 0.15
    pts[bad] = -1
    flat = pts.reshape(len(pts), -1)
    return flat.tolist()


def sequence_for_glosses(text: str) -> dict:
    """Glosas → frames dibujables. `ok: false` si no hay diccionario o no matchea nada."""
    tokens = _tokens(text)
    if not tokens:
        return _fail("No llegaron glosas.")
    if not any(POSE_DIR.glob("*.pose")):
        return _fail(
            "Falta el diccionario en extraction/out/poses (no está en git).",
            missing=tokens,
        )
    try:
        rp = _load_rp()
    except ImportError as exc:
        return _fail(
            f"Falta una librería del compositor ({exc}). Hace falta Python 3.11 con pose-format y opencv-python.",
            missing=tokens,
        )
    except Exception as exc:
        return _fail(f"No se pudo cargar el compositor: {exc}", missing=tokens)

    found, missing = _resolve_chain(rp, tokens)
    if not found:
        return _fail("Ninguna glosa tiene clip.", missing=missing)

    args = _args(rp)
    poses = []
    names: list[str] = []
    for query, path in found:
        try:
            poses.append(_prepare(rp, path, args))
            names.append(rp.label_for(query, path))
        except Exception:
            missing.append(query)
    if not poses:
        return _fail("No se pudo preparar ninguna glosa.", missing=missing)

    pose, _labels, _bridges = rp.compose_sequence(poses, names, args)
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if len(data) > _MAX_FRAMES:
        data = data[:_MAX_FRAMES]
        conf = conf[:_MAX_FRAMES]
    if len(data) == 0:
        return _fail("El compositor no devolvió frames.", missing=missing)
    face_order, face_edges = _face_layout()
    face_abs = [ _FACE_START + i for i in face_order ]
    hand = list(range(21))
    return {
        "ok": True,
        "fps": float(pose.body.fps or args.fps),
        "w": int(args.canvas_width),
        "h": int(args.canvas_height),
        "shoulder": float(args.shoulder_px),
        "shoulder_y": float(args.shoulder_y),
        "played": names,
        "missing": missing,
        "face_edges": face_edges,
        "frames": [
            {
                "p": row_p,
                "f": row_f,
                "l": row_l,
                "r": row_r,
            }
            for row_p, row_f, row_l, row_r in zip(
                _pack(data, conf, _POSE_IDX),
                _pack(data, conf, face_abs),
                _pack(data, conf, [ _LEFT_HAND + i for i in hand ]),
                _pack(data, conf, [ _RIGHT_HAND + i for i in hand ]),
            )
        ],
    }
