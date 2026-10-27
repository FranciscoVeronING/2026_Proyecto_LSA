"""In-memory gloss phrase for the Meet canvas (no mp4)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from poseclip.compose import SETTINGS, compose_sequence, prepare_sign

_MAX_FRAMES = 1200

# Contornos de FaceMesh (sin iris: Holistic tiene 468 puntos, el iris empieza en 468).
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
_POSE_IDX = (11, 12, 13, 14, 15, 16, 23, 24)
_FACE_START = 33
_LEFT_HAND = 33 + 468
_RIGHT_HAND = _LEFT_HAND + 21

_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()


def _fail(error: str, missing: list[str] | None = None) -> dict:
    return {
        "ok": False,
        "error": error,
        "missing": missing or [],
        "frames": [],
    }


def _prepare(path: Path):
    key = str(path.resolve())
    with _LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    pose = prepare_sign(path, SETTINGS)
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
    return pts.reshape(len(pts), -1).tolist()


def sequence_from_files(found: list[tuple[str, Path]], missing: list[str] | None = None) -> dict:
    """`.pose` clips downloaded from the T-pose server → canvas frames."""
    missing = list(missing or [])
    if not found:
        return _fail("No gloss has a clip.", missing=missing)

    poses = []
    names: list[str] = []
    for query, path in found:
        try:
            poses.append(_prepare(path))
            names.append(str(query))
        except Exception:
            missing.append(query)
    if not poses:
        return _fail("Could not prepare any gloss.", missing=missing)

    pose = compose_sequence(poses, SETTINGS)
    data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)
    conf = np.ma.filled(pose.body.confidence, 0.0).astype(np.float32)
    if len(data) > _MAX_FRAMES:
        data = data[:_MAX_FRAMES]
        conf = conf[:_MAX_FRAMES]
    if len(data) == 0:
        return _fail("The compositor returned no frames.", missing=missing)
    face_order, face_edges = _face_layout()
    face_abs = [_FACE_START + i for i in face_order]
    hand = list(range(21))
    return {
        "ok": True,
        "fps": float(pose.body.fps or SETTINGS.fps),
        "w": int(SETTINGS.canvas_width),
        "h": int(SETTINGS.canvas_height),
        "shoulder": float(SETTINGS.shoulder_px),
        "shoulder_y": float(SETTINGS.shoulder_y),
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
                _pack(data, conf, [_LEFT_HAND + i for i in hand]),
                _pack(data, conf, [_RIGHT_HAND + i for i in hand]),
            )
        ],
    }
