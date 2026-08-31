import argparse
import csv
import os
import re
import time
import json
from datetime import datetime
from threading import Thread, Lock
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

import config as cfg
from model_arch import TinySkeletonClassifier
from utils import (
    get_anchor_and_scale,
    normalize_spatial_points,
    compute_landmark_hand_motion,
    sequence_buffer_to_model_input,
    mirror_landmarks_for_left_handed,
)

# =============================================================================
# UTILIDADES
# =============================================================================
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX


class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_vector = None

    def update(self, new_vector):
        if self.prev_vector is None:
            self.prev_vector = new_vector
            return new_vector
        smoothed = (self.alpha * new_vector) + ((1 - self.alpha) * self.prev_vector)
        self.prev_vector = smoothed
        return smoothed

    def reset(self):
        self.prev_vector = None


def prepare_input_tensor(buffer_list, device, target_frames=None):
    """Convierte buffer temporal a tensor (1, max_frames, features)."""
    target_frames = target_frames or cfg.MAX_FRAMES
    matrix = sequence_buffer_to_model_input(buffer_list, target_frames=target_frames)
    if matrix.shape[0] != target_frames:
        return None
    tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)


def should_start_recording(capture_mode, hands_present, is_moving, consecutive_hands_frames):
    if not hands_present:
        return False
    if capture_mode == "dynamic":
        return is_moving
    if capture_mode == "static":
        return consecutive_hands_frames >= cfg.STATIC_HANDS_FRAMES_TO_START
    static_ready = consecutive_hands_frames >= cfg.STATIC_HANDS_FRAMES_TO_START
    return is_moving or static_ready


def extract_normalized_vector(results, left_handed: bool):
    """Construye vector (225,) desde resultados MediaPipe; espeja si es zurdo."""
    anchor, scale = get_anchor_and_scale(results.pose_landmarks)
    raw_pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )
    raw_lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    raw_rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    norm_pose = normalize_spatial_points(raw_pose, anchor, scale)
    norm_lh = normalize_spatial_points(raw_lh, anchor, scale)
    norm_rh = normalize_spatial_points(raw_rh, anchor, scale)
    vector = np.concatenate([norm_pose, norm_lh, norm_rh])

    if left_handed:
        vector = mirror_landmarks_for_left_handed(vector, pose_dim=cfg.POSE_DIM)
    return vector


def fit_text(text, max_w, scale=0.45):
    out = text
    while out and cv2.getTextSize(out, UI_FONT, scale, 1)[0][0] > max_w:
        out = out[:-1]
    return out


# =============================================================================
# CATALOGO DE MODELOS
# =============================================================================
def resolve_model_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "model")
    if os.path.isdir(candidate):
        return candidate
    return os.path.normpath(os.path.join(here, cfg.MODEL_SAVE_DIR))


def _safe_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_arch_from_folder(folder):
    """Fallback: '128HD 4H 2L' y carpetas tipo 16_Frames / 32_frames_a."""
    name = os.path.basename(folder)
    parent = os.path.basename(os.path.dirname(folder))
    blob = f"{parent} {name}"
    out = {}
    m = re.search(r"(\d+)\s*HD", blob, re.I)
    if m:
        out["hidden_dim"] = int(m.group(1))
    m = re.search(r"(?<![A-Za-z0-9])(\d+)H(?![A-Za-z])", blob)
    if m:
        out["num_heads"] = int(m.group(1))
    m = re.search(r"(?<![A-Za-z0-9])(\d+)L(?![A-Za-z])", blob)
    if m:
        out["num_layers"] = int(m.group(1))
    m = re.search(r"(\d+)\s*[_-]?\s*[Ff]rames", blob)
    if m:
        out["max_frames"] = int(m.group(1))
    return out


def infer_arch_from_checkpoint(pth_path):
    """Lee hidden_dim, num_layers y num_classes del .pth (más fiable que metrics.json)."""
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    hidden = int(state["conv_extractor.0.weight"].shape[0])
    n_cls = int(state["classification_head.weight"].shape[0])
    layer_ids = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("transformer.layers.")
    }
    n_layers = (max(layer_ids) + 1) if layer_ids else cfg.NUM_LAYERS
    compatible = "attention_pool.weight" in state
    return {
        "hidden_dim": hidden,
        "num_layers": n_layers,
        "num_classes": n_cls,
        "compatible": compatible,
    }


def _guess_num_heads(hidden_dim, parsed=None, metrics=None, metrics_trustworthy=False):
    if metrics_trustworthy and metrics and metrics.get("num_heads") is not None:
        return int(metrics["num_heads"])
    if parsed and parsed.get("num_heads") is not None:
        return int(parsed["num_heads"])
    if hidden_dim % 4 == 0 and hidden_dim <= 128:
        return 4
    if hidden_dim % 2 == 0:
        return 2
    return 1


def _load_arch_metrics(folder):
    metrics = _safe_json(os.path.join(folder, "metrics.json"))
    if metrics and "hidden_dim" in metrics:
        return metrics

    try:
        json_names = os.listdir(folder)
    except OSError:
        json_names = []
    for name in json_names:
        if not name.lower().endswith(".json") or "optuna" not in name.lower():
            continue
        data = _safe_json(os.path.join(folder, name))
        if not data or "best_params" not in data:
            continue
        bp = data["best_params"]
        return {
            "hidden_dim": int(bp.get("hidden_dim", cfg.HIDDEN_DIM)),
            "num_heads": int(bp.get("num_heads", cfg.NUM_HEADS)),
            "num_layers": int(bp.get("num_layers", cfg.NUM_LAYERS)),
            "dropout_rate": float(bp.get("dropout_rate", cfg.DROPOUT_RATE)),
            "max_frames": int(bp.get("max_frames", cfg.MAX_FRAMES)),
        }
    return _parse_arch_from_folder(folder)


def _load_class_mapping(folder, metrics, model_root):
    mapeo = _safe_json(os.path.join(folder, "mapeo_clases.json"))
    if isinstance(mapeo, dict) and mapeo:
        return {str(k): int(v) for k, v in mapeo.items()}

    classes = metrics.get("classes") if metrics else None
    if isinstance(classes, list) and classes:
        return {name: idx for idx, name in enumerate(classes)}

    mapeo = _safe_json(os.path.join(model_root, "mapeo_clases.json"))
    if isinstance(mapeo, dict) and mapeo:
        return {str(k): int(v) for k, v in mapeo.items()}
    return {name: idx for idx, name in enumerate(cfg.SIGN_CLASSES)}


def _make_labels(rel_id, metrics, archived):
    folder = os.path.basename(rel_id.replace("\\", "/"))
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})_(.+)", folder)
    if m:
        tag = m.group(4).replace("model_", "").replace("_", " ")
        short = f"{m.group(2)}-{m.group(3)} {tag}"
    else:
        short = folder[:36]

    parts = []
    ncls = metrics.get("num_classes")
    if ncls:
        parts.append(f"{ncls}c")
    mf = metrics.get("max_frames")
    if mf:
        parts.append(f"{mf}f")
    hd = metrics.get("hidden_dim")
    nh = metrics.get("num_heads")
    nl = metrics.get("num_layers")
    if hd:
        parts.append(f"{hd}d")
    if nh is not None and nl is not None:
        parts.append(f"{nh}H{nl}L")
    acc = metrics.get("val_accuracy_top1_pct")
    if acc is not None:
        parts.append(f"val {acc:.0f}%")

    prefix = "arch/" if archived else ""
    detail = " ".join(parts)
    label = f"{prefix}{short} | {detail}" if detail else f"{prefix}{short}"
    return short.strip(), label


class ModelSpec:
    def __init__(
        self,
        spec_id,
        short,
        label,
        folder,
        pth_path,
        hidden_dim,
        num_heads,
        num_layers,
        dropout_rate,
        max_frames,
        class_to_idx,
        archived,
        num_classes=None,
        val_acc=None,
    ):
        self.id = spec_id
        self.short = short
        self.label = label
        self.folder = folder
        self.pth_path = pth_path
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_frames = max_frames
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = int(num_classes) if num_classes is not None else len(class_to_idx)
        self.archived = archived
        self.val_acc = val_acc


def discover_models(model_root):
    """Recorre src/model (incluye archivados) y arma un spec por carpeta con .pth."""
    specs = []
    if not os.path.isdir(model_root):
        return specs

    for dirpath, dirnames, filenames in os.walk(model_root):
        pth_files = [f for f in filenames if f.lower().endswith(".pth")]
        if not pth_files:
            continue
        preferred = [f for f in pth_files if "best" in f.lower()]
        pth_name = sorted(preferred or pth_files)[0]
        pth_path = os.path.join(dirpath, pth_name)

        rel = os.path.relpath(dirpath, model_root)
        if rel == ".":
            spec_id = os.path.splitext(pth_name)[0]
        else:
            spec_id = rel.replace("\\", "/")
        archived = spec_id.replace("\\", "/").startswith("archivados")

        try:
            ckpt = infer_arch_from_checkpoint(pth_path)
        except Exception as e:
            print(f"[!] No se pudo leer {pth_path}: {e}")
            continue
        if not ckpt.get("compatible", True):
            print(f"[!] Omitido (arquitectura previa, sin attention pool): {spec_id}")
            continue

        metrics = _load_arch_metrics(dirpath) or {}
        parsed = _parse_arch_from_folder(dirpath)
        metrics_ok = int(metrics.get("hidden_dim", -1)) == ckpt["hidden_dim"]

        hidden_dim = ckpt["hidden_dim"]
        num_layers = ckpt["num_layers"]
        num_heads = _guess_num_heads(hidden_dim, parsed, metrics, metrics_ok)
        dropout_rate = float(
            (metrics.get("dropout_rate") if metrics_ok else None) or cfg.DROPOUT_RATE
        )
        if metrics_ok and metrics.get("max_frames") is not None:
            max_frames = int(metrics["max_frames"])
        elif parsed.get("max_frames") is not None:
            max_frames = int(parsed["max_frames"])
        else:
            max_frames = cfg.MAX_FRAMES

        class_to_idx = _load_class_mapping(dirpath, metrics, model_root)
        if len(class_to_idx) != ckpt["num_classes"]:
            root_map = _safe_json(os.path.join(model_root, "mapeo_clases.json"))
            if isinstance(root_map, dict) and len(root_map) == ckpt["num_classes"]:
                class_to_idx = {str(k): int(v) for k, v in root_map.items()}

        label_metrics = {
            **metrics,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "max_frames": max_frames,
            "num_classes": ckpt["num_classes"],
        }
        short, label = _make_labels(spec_id, label_metrics, archived)
        specs.append(
            ModelSpec(
                spec_id=spec_id,
                short=short,
                label=label,
                folder=dirpath,
                pth_path=pth_path,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                max_frames=max_frames,
                class_to_idx=class_to_idx,
                archived=archived,
                num_classes=ckpt["num_classes"],
                val_acc=metrics.get("val_accuracy_top1_pct") if metrics_ok else None,
            )
        )

    specs.sort(key=lambda s: (s.archived, s.id.lower()), reverse=False)
    # Dentro de los activos, el más reciente primero (fecha en el nombre).
    active = [s for s in specs if not s.archived]
    archived = [s for s in specs if s.archived]
    active.sort(key=lambda s: s.id, reverse=True)
    archived.sort(key=lambda s: s.id)
    return active + archived


def default_model_index(catalog):
    for i, spec in enumerate(catalog):
        if not spec.archived:
            return i
    return 0 if catalog else -1


def build_classifier(spec, device):
    model = TinySkeletonClassifier(
        cfg.FRAME_FEATURES_DIM,
        spec.hidden_dim,
        num_heads=spec.num_heads,
        num_layers=spec.num_layers,
        num_classes=spec.num_classes,
        dropout_rate=spec.dropout_rate,
    ).to(device)
    state = torch.load(spec.pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# =============================================================================
# UI COMPONENTS
# =============================================================================
class Button:
    def __init__(self, x, y, w, h, text, callback_func=None):
        self.rect = (x, y, w, h)
        self.text = text
        self.callback = callback_func
        self.is_hover = False

    def set_rect(self, x, y, w, h):
        self.rect = (x, y, w, h)

    def update(self, mouse_x, mouse_y, clicked_event):
        x, y, w, h = self.rect
        self.is_hover = (x <= mouse_x <= x + w) and (y <= mouse_y <= y + h)
        if self.is_hover and clicked_event:
            if self.callback:
                self.callback()
            return True
        return False

    def draw(self, canvas, active=False):
        x, y, w, h = self.rect
        bg_color = (0, 200, 100) if active else ((80, 80, 80) if self.is_hover else (50, 50, 50))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
        text_size = cv2.getTextSize(self.text, UI_FONT, 0.5, 1)[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        cv2.putText(canvas, self.text, (tx, ty), UI_FONT, 0.5, (255, 255, 255), 1)


class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial_val, label):
        self.x, self.y, self.w, self.h = x, y, w, 20
        self.min_val, self.max_val, self.val = min_val, max_val, initial_val
        self.label = label
        self.dragging = False

    def update(self, mouse_x, mouse_y, is_m_down):
        hover = (self.x <= mouse_x <= self.x + self.w) and (self.y - 5 <= mouse_y <= self.y + self.h + 5)
        if hover and is_m_down:
            self.dragging = True
        if not is_m_down:
            self.dragging = False
        if self.dragging:
            ratio = max(0, min(mouse_x - self.x, self.w)) / self.w
            self.val = self.min_val + (self.max_val - self.min_val) * ratio

    def draw(self, canvas):
        display_val = f"{int(self.val)}" if self.max_val > 1 else f"{self.val:.2f}"
        cv2.putText(canvas, f"{self.label}: {display_val}", (self.x, self.y - 10), UI_FONT, 0.5, (200, 200, 200), 1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (40, 40, 40), -1)
        fill_w = int(self.w * (self.val - self.min_val) / (self.max_val - self.min_val))
        cv2.rectangle(canvas, (self.x, self.y), (self.x + fill_w, self.y + self.h), (0, 165, 255), -1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)


class Dropdown:
    ITEM_H = 24

    def __init__(self, x, y, w, h, options, selected=0, max_visible=8):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.options = options
        self.selected = max(0, min(selected, max(0, len(options) - 1)))
        self.open = False
        self.scroll = 0
        self.max_visible = max_visible
        self.changed = False

    def set_rect(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def _menu_geom(self):
        n = min(len(self.options), self.max_visible)
        menu_h = max(n, 1) * self.ITEM_H
        return self.x, self.y - menu_h, self.w, menu_h

    def update(self, mouse_x, mouse_y, clicked, wheel=0):
        self.changed = False
        if not self.options:
            return False

        on_box = (self.x <= mouse_x <= self.x + self.w) and (self.y <= mouse_y <= self.y + self.h)
        if clicked and on_box:
            self.open = not self.open
            return True

        if not self.open:
            return False

        mx, my, mw, mh = self._menu_geom()
        over_menu = (mx <= mouse_x <= mx + mw) and (my <= mouse_y <= my + mh)
        max_scroll = max(0, len(self.options) - self.max_visible)

        if wheel and (over_menu or on_box):
            self.scroll = max(0, min(max_scroll, self.scroll - int(wheel)))
            return True

        if clicked:
            if over_menu:
                idx = self.scroll + (mouse_y - my) // self.ITEM_H
                if 0 <= idx < len(self.options):
                    if idx != self.selected:
                        self.selected = idx
                        self.changed = True
                    self.open = False
                return True
            self.open = False
            return True
        return False

    def draw(self, canvas):
        bg = (70, 70, 70) if self.open else (50, 50, 50)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (200, 200, 200), 1)
        current = self.options[self.selected] if self.options else "(sin modelos)"
        cv2.putText(
            canvas,
            fit_text(current, self.w - 28, 0.42),
            (self.x + 8, self.y + self.h - 8),
            UI_FONT,
            0.42,
            (255, 255, 255),
            1,
        )
        cv2.putText(canvas, "^" if self.open else "v", (self.x + self.w - 16, self.y + self.h - 8), UI_FONT, 0.5, (200, 200, 200), 1)

        if not self.open or not self.options:
            return

        mx, my, mw, mh = self._menu_geom()
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (25, 25, 25), -1)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (0, 165, 255), 1)
        visible = self.options[self.scroll : self.scroll + self.max_visible]
        for i, opt in enumerate(visible):
            abs_i = self.scroll + i
            iy = my + i * self.ITEM_H
            if abs_i == self.selected:
                cv2.rectangle(canvas, (mx, iy), (mx + mw, iy + self.ITEM_H), (0, 90, 160), -1)
            cv2.putText(
                canvas,
                fit_text(opt, mw - 12, 0.4),
                (mx + 6, iy + self.ITEM_H - 7),
                UI_FONT,
                0.4,
                (255, 255, 255),
                1,
            )


def select_handedness_modal():
    """
    Modal inicial: elegir diestro o zurdo antes de abrir la cámara.
    Retorna 'right' o 'left'.
    """
    modal_w, modal_h = 520, 280
    canvas = np.zeros((modal_h, modal_w, 3), dtype=np.uint8)
    choice = {"value": None}

    def pick_right():
        choice["value"] = "right"

    def pick_left():
        choice["value"] = "left"

    btn_right = Button(70, 160, 160, 50, "DIESTRO", pick_right)
    btn_left = Button(290, 160, 160, 50, "ZURDO", pick_left)
    mouse = {"x": 0, "y": 0, "down": False, "clicked": False}

    def on_mouse(event, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["down"] = False
            mouse["clicked"] = True

    cv2.namedWindow("LSA DETECTOR - Configuracion")
    cv2.setMouseCallback("LSA DETECTOR - Configuracion", on_mouse)

    while choice["value"] is None:
        canvas[:] = (35, 35, 35)
        cv2.putText(canvas, "Mano dominante", (130, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "Selecciona antes de iniciar la camara", (70, 100), UI_FONT, 0.55, (180, 180, 180), 1)
        cv2.putText(canvas, "Teclas: D = diestro | Z = zurdo", (110, 130), UI_FONT, 0.5, (140, 140, 140), 1)
        btn_right.update(mouse["x"], mouse["y"], mouse["clicked"])
        btn_left.update(mouse["x"], mouse["y"], mouse["clicked"])
        btn_right.draw(canvas)
        btn_left.draw(canvas)
        cv2.imshow("LSA DETECTOR - Configuracion", canvas)
        mouse["clicked"] = False

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("d"), ord("D")):
            choice["value"] = "right"
        elif key in (ord("z"), ord("Z")):
            choice["value"] = "left"
        elif key == 27:
            cv2.destroyWindow("LSA DETECTOR - Configuracion")
            return None

    cv2.destroyWindow("LSA DETECTOR - Configuracion")
    label = "diestro" if choice["value"] == "right" else "zurdo"
    print(f"[*] Mano dominante: {label}")
    return choice["value"]


def select_models_modal(catalog, multi=True, prechecked=None):
    """
    Modal para elegir modelos. En --eval es multi-select.
    Retorna lista de ModelSpec o None si se cancela.
    """
    if not catalog:
        return None

    row_h = 28
    header_h = 100
    footer_h = 80
    visible = min(len(catalog), 14)
    modal_w, modal_h = 720, header_h + visible * row_h + footer_h
    canvas = np.zeros((modal_h, modal_w, 3), dtype=np.uint8)

    if prechecked is None:
        checked = [not spec.archived for spec in catalog]
        if not any(checked):
            checked = [True] + [False] * (len(catalog) - 1)
    else:
        wanted = set(prechecked)
        checked = [spec.id in wanted for spec in catalog]

    scroll = [0]
    done = {"ok": False, "cancel": False}
    mouse = {"x": 0, "y": 0, "clicked": False, "wheel": 0}

    list_y0 = header_h
    list_h = visible * row_h

    def on_mouse(event, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONUP:
            mouse["clicked"] = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
            mouse["wheel"] = 1 if delta > 0 else -1

    cv2.namedWindow("LSA DETECTOR - Modelos")
    cv2.setMouseCallback("LSA DETECTOR - Modelos", on_mouse)

    btn_all = Button(40, modal_h - 58, 110, 40, "TODOS")
    btn_none = Button(160, modal_h - 58, 110, 40, "NINGUNO")
    btn_ok = Button(modal_w - 280, modal_h - 58, 120, 40, "COMENZAR")
    btn_cancel = Button(modal_w - 150, modal_h - 58, 110, 40, "CANCELAR")

    while not done["ok"] and not done["cancel"]:
        canvas[:] = (35, 35, 35)
        title = "Modelos para eval (mismo try)" if multi else "Seleccionar modelo"
        cv2.putText(canvas, title, (30, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "Click para marcar. Rueda = scroll. Enter = comenzar. Esc = cancelar.",
            (30, 72),
            UI_FONT,
            0.45,
            (160, 160, 160),
            1,
        )

        max_scroll = max(0, len(catalog) - visible)
        if mouse["wheel"]:
            scroll[0] = max(0, min(max_scroll, scroll[0] - mouse["wheel"]))
            mouse["wheel"] = 0

        mx, my, clicked = mouse["x"], mouse["y"], mouse["clicked"]
        for i in range(visible):
            abs_i = scroll[0] + i
            if abs_i >= len(catalog):
                break
            spec = catalog[abs_i]
            y = list_y0 + i * row_h
            hover = (20 <= mx <= modal_w - 20) and (y <= my <= y + row_h)
            if hover and clicked:
                if multi:
                    checked[abs_i] = not checked[abs_i]
                else:
                    checked = [j == abs_i for j in range(len(catalog))]
                clicked = False
                mouse["clicked"] = False

            bg = (55, 70, 55) if checked[abs_i] else ((50, 50, 50) if hover else (42, 42, 42))
            cv2.rectangle(canvas, (20, y), (modal_w - 20, y + row_h - 2), bg, -1)
            mark = "[x]" if checked[abs_i] else "[ ]"
            color = (0, 220, 120) if checked[abs_i] else (200, 200, 200)
            cv2.putText(canvas, f"{mark}  {spec.label}", (32, y + 20), UI_FONT, 0.45, color, 1)

        n_sel = sum(1 for c in checked if c)
        cv2.putText(canvas, f"{n_sel} seleccionado(s)", (300, modal_h - 32), UI_FONT, 0.45, (180, 180, 180), 1)

        if btn_all.update(mx, my, mouse["clicked"]):
            checked[:] = [True] * len(checked)
        if btn_none.update(mx, my, mouse["clicked"]):
            checked[:] = [False] * len(checked)
        if btn_ok.update(mx, my, mouse["clicked"]) and n_sel > 0:
            done["ok"] = True
        if btn_cancel.update(mx, my, mouse["clicked"]):
            done["cancel"] = True

        btn_all.draw(canvas)
        btn_none.draw(canvas)
        btn_ok.draw(canvas, active=n_sel > 0)
        btn_cancel.draw(canvas)

        cv2.imshow("LSA DETECTOR - Modelos", canvas)
        mouse["clicked"] = False

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and n_sel > 0:
            done["ok"] = True
        elif key == 27:
            done["cancel"] = True
        elif key in (ord("a"), ord("A")):
            checked[:] = [True] * len(checked)
        elif key in (ord("n"), ord("N")):
            checked[:] = [False] * len(checked)

    cv2.destroyWindow("LSA DETECTOR - Modelos")
    if done["cancel"] or not done["ok"]:
        return None
    return [spec for spec, on in zip(catalog, checked) if on]


# =============================================================================
# EVAL CSV
# =============================================================================
class EvalSession:
    FIELDNAMES = [
        "timestamp",
        "eval_index",
        "expected_sign",
        "model",
        "top1",
        "conf1",
        "top2",
        "conf2",
        "top3",
        "conf3",
        "hit_top1",
        "hit_top3",
        "handedness",
        "capture_mode",
    ]

    def __init__(self, sign_list: list[str], csv_path: str, handedness: str, model_ids: list[str]):
        self.sign_list = sign_list
        self.csv_path = csv_path
        self.handedness = handedness
        self.model_ids = model_ids
        self.index = 0
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    @property
    def finished(self):
        return self.index >= len(self.sign_list)

    @property
    def expected_sign(self):
        if self.finished:
            return None
        return self.sign_list[self.index]

    def _write_row(self, model_id, top3, capture_mode, skipped=False):
        expected = self.expected_sign
        if skipped:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "eval_index": self.index + 1,
                "expected_sign": expected,
                "model": model_id,
                "top1": "SKIP",
                "conf1": "",
                "top2": "",
                "conf2": "",
                "top3": "",
                "conf3": "",
                "hit_top1": 0,
                "hit_top3": 0,
                "handedness": self.handedness,
                "capture_mode": capture_mode,
            }
        else:
            top_names = [t[0] for t in top3]
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "eval_index": self.index + 1,
                "expected_sign": expected,
                "model": model_id,
                "top1": top3[0][0] if len(top3) > 0 else "",
                "conf1": f"{top3[0][1]:.4f}" if len(top3) > 0 else "",
                "top2": top3[1][0] if len(top3) > 1 else "",
                "conf2": f"{top3[1][1]:.4f}" if len(top3) > 1 else "",
                "top3": top3[2][0] if len(top3) > 2 else "",
                "conf3": f"{top3[2][1]:.4f}" if len(top3) > 2 else "",
                "hit_top1": int(top3[0][0] == expected) if top3 else 0,
                "hit_top3": int(expected in top_names),
                "handedness": self.handedness,
                "capture_mode": capture_mode,
            }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

    def log_predictions(self, results_by_model: dict, capture_mode: str):
        if self.finished:
            return
        for model_id in self.model_ids:
            top3 = results_by_model.get(model_id, [])
            self._write_row(model_id, top3, capture_mode, skipped=False)
        self.index += 1

    def skip_current(self, capture_mode: str):
        if self.finished:
            return
        for model_id in self.model_ids:
            self._write_row(model_id, [], capture_mode, skipped=True)
        self.index += 1


# =============================================================================
# BACKEND E INFERENCIA
# =============================================================================
shared_state = {
    "inference_queue": deque(maxlen=5),
    "prediction": "...",
    "confidence": 0.0,
    "top3": [],
    "results_by_model": {},
    "last_inference_time": 0.0,
    "lock": Lock(),
    "running": True,
}


class InferenceWorker:
    def __init__(self, device):
        self.device = device
        self._reload_lock = Lock()
        self.entries = []

    def start(self):
        Thread(target=self.loop, args=(), daemon=True).start()

    def set_specs(self, specs):
        entries = []
        for spec in specs:
            try:
                model = build_classifier(spec, self.device)
                entries.append({"spec": spec, "model": model})
                print(
                    f"[*] Modelo listo: {spec.label}  "
                    f"({os.path.basename(spec.pth_path)}, {spec.max_frames} frames)"
                )
            except Exception as e:
                print(f"[!] No se pudo cargar {spec.id}: {e}")
        if not entries:
            print("[!] Ningun modelo se pudo cargar.")
        with self._reload_lock:
            self.entries = entries
        with shared_state["lock"]:
            shared_state["top3"] = []
            shared_state["prediction"] = "..."
            shared_state["confidence"] = 0.0
            shared_state["results_by_model"] = {}
        return [e["spec"] for e in entries]

    def _decode_top3(self, probs, idx_to_class):
        values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
        results = []
        for conf, idx in zip(values.tolist(), indices.tolist()):
            name = idx_to_class.get(idx, "desconocido")
            results.append((name, float(conf)))
        return results

    def loop(self):
        while shared_state["running"]:
            buffer = None
            with shared_state["lock"]:
                if len(shared_state["inference_queue"]) > 0:
                    buffer = shared_state["inference_queue"].popleft()

            if buffer is None:
                time.sleep(0.01)
                continue

            with self._reload_lock:
                entries = list(self.entries)
            if not entries:
                time.sleep(0.05)
                continue

            results_by_model = {}
            primary_top3 = []
            try:
                with torch.no_grad():
                    for i, entry in enumerate(entries):
                        spec = entry["spec"]
                        tensor = prepare_input_tensor(buffer, self.device, target_frames=spec.max_frames)
                        if tensor is None:
                            continue
                        logits = entry["model"](tensor)
                        probs = torch.softmax(logits, dim=1)[0]
                        top3 = self._decode_top3(probs, spec.idx_to_class)
                        results_by_model[spec.id] = top3
                        if i == 0:
                            primary_top3 = top3
                        print(f"{spec.short}: " + " | ".join(f"{n.upper()} ({c:.1%})" for n, c in top3))

                with shared_state["lock"]:
                    shared_state["results_by_model"] = results_by_model
                    shared_state["top3"] = primary_top3
                    if primary_top3:
                        shared_state["prediction"] = primary_top3[0][0].upper()
                        shared_state["confidence"] = primary_top3[0][1]
                    shared_state["last_inference_time"] = time.time()
            except Exception as e:
                print(f"Error inferencia: {e}")


class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            print("ERROR CRITICO: No se puede abrir la camara.")
            self.stopped = True
            return

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
            else:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# =============================================================================
# MAIN
# =============================================================================
mouse_state = {"x": 0, "y": 0, "down": False, "clicked": False, "wheel": 0}


def mouse_callback(event, x, y, flags, param):
    mouse_state["x"], mouse_state["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["down"] = False
        mouse_state["clicked"] = True
    elif event == cv2.EVENT_MOUSEWHEEL:
        delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
        mouse_state["wheel"] = 1 if delta > 0 else -1


def can_enqueue_inference(last_enqueue_time: float) -> bool:
    now = time.time()
    if now - last_enqueue_time < cfg.INFERENCE_COOLDOWN_SEC:
        return False
    with shared_state["lock"]:
        if now - shared_state["last_inference_time"] < cfg.INFERENCE_COOLDOWN_SEC:
            return False
    return True


def enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time_ref: list):
    if not can_enqueue_inference(last_enqueue_time_ref[0]):
        return False
    copied = [np.array(frame, copy=True) for frame in frames_temp_buffer]
    with shared_state["lock"]:
        shared_state["inference_queue"].append(copied)
    last_enqueue_time_ref[0] = time.time()
    return True


def draw_top3_panel(canvas, top3, y_start, threshold):
    for i, (name, conf) in enumerate(top3[:3]):
        color = (0, 255, 0) if i == 0 and conf >= threshold else (200, 200, 200)
        cv2.putText(
            canvas,
            f"{i + 1}. {name.upper()}  {conf:.0%}",
            (20, y_start + i * 22),
            UI_FONT,
            0.55,
            color,
            1,
        )


def draw_multi_model_panel(canvas, specs, results_by_model, y_start, threshold, expected_sign=None):
    for i, spec in enumerate(specs):
        top3 = results_by_model.get(spec.id, [])
        y = y_start + i * 22
        prefix = f"{spec.short}: "
        if not top3:
            cv2.putText(canvas, prefix + "...", (20, y), UI_FONT, 0.45, (120, 120, 120), 1)
            continue
        name, conf = top3[0]
        hit = expected_sign is not None and name == expected_sign
        if hit:
            color = (0, 255, 0)
        elif conf >= threshold:
            color = (0, 220, 180)
        else:
            color = (200, 200, 200)
        extras = "  ".join(f"{n} {c:.0%}" for n, c in top3[1:3])
        line = f"{prefix}{name.upper()} {conf:.0%}"
        if extras:
            line += f" | {extras}"
        cv2.putText(canvas, fit_text(line, 600, 0.45), (20, y), UI_FONT, 0.45, color, 1)


def union_sign_list(specs):
    seen = []
    known = set()
    for spec in specs:
        for name in spec.class_to_idx.keys():
            if name not in known:
                known.add(name)
                seen.append(name)
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(description="Inferencia LSA en tiempo real.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Modo evaluacion: recorre las senias y guarda CSV. Permite varios modelos en paralelo.",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Ruta del CSV de evaluacion (default: eval_94senias_<fecha>.csv en src/).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = resolve_model_root()
    catalog = discover_models(model_root)
    if not catalog:
        print(f"[!] No se encontraron .pth en {model_root}.")
        return

    print(f"[*] {len(catalog)} modelo(s) en {model_root}")
    for spec in catalog:
        print(f"    - {spec.label}")

    handedness = select_handedness_modal()
    if handedness is None:
        print("[!] Configuracion cancelada.")
        return

    eval_session = None
    if args.eval:
        selected = select_models_modal(catalog, multi=True)
        if not selected:
            print("[!] Eval cancelada: no hay modelos seleccionados.")
            return
        active_specs = selected
    else:
        idx = default_model_index(catalog)
        active_specs = [catalog[idx]]

    worker = InferenceWorker(device)
    active_specs = worker.set_specs(active_specs)
    if not active_specs:
        print("[!] Ningun modelo se pudo cargar.")
        return
    worker.start()

    if args.eval:
        sign_list = union_sign_list(active_specs)
        csv_path = args.eval_output or os.path.join(
            os.path.dirname(__file__),
            f"eval_{len(sign_list)}senias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        eval_session = EvalSession(sign_list, csv_path, handedness, [s.id for s in active_specs])
        print(f"[*] Modo eval activo. CSV: {csv_path}")
        print(f"[*] Modelos en paralelo: {', '.join(s.short for s in active_specs)}")
        print(f"[*] Senias a probar: {len(sign_list)}. Tecla 'n' = saltar senia.")

    n_models = len(active_specs)
    print("\n[*] Iniciando camara...")
    print(f"[*] Modo captura: {cfg.CAPTURE_MODE} | modelos: {n_models}")
    print(f"[*] Umbral confianza: {cfg.CONFIDENCE_THRESHOLD:.0%} | Cooldown: {cfg.INFERENCE_COOLDOWN_SEC}s")

    vs = WebcamStream(0).start()
    time.sleep(2.0)
    if vs.stopped:
        shared_state["running"] = False
        return

    cv2.namedWindow("LSA DETECTOR")
    cv2.setMouseCallback("LSA DETECTOR", mouse_callback)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    VID_W, VID_H = 640, 480

    btn_capture = Button(280, VID_H + 8, 105, 36, "CAPTURAR")
    btn_pause = Button(395, VID_H + 8, 90, 36, "PAUSA")
    btn_view = Button(495, VID_H + 8, 125, 36, "Esqueleto")
    btn_conf = Button(495, VID_H + 50, 125, 36, "Config")
    dropdown = Dropdown(
        20,
        VID_H + 155,
        460,
        30,
        [s.label for s in catalog],
        selected=default_model_index(catalog),
        max_visible=8,
    )

    slider_sens = Slider(150, 150, 340, 100, 5000, cfg.MOTION_PIXEL_THRESHOLD, "Sensibilidad (Pixeles)")
    slider_conf = Slider(150, 200, 340, 0.1, 1.0, cfg.CONFIDENCE_THRESHOLD, "Confianza Min")
    slider_still = Slider(150, 250, 340, 5, 40, cfg.STILL_FRAMES_LIMIT, "Corte por Silencio (Frames)")
    slider_static = Slider(150, 300, 340, 2, 15, cfg.STATIC_HANDS_FRAMES_TO_START, "Frames Manos (Estatico)")
    btn_save = Button(220, 380, 100, 40, "CERRAR")

    show_config = False
    show_landmarks = True
    paused = False
    capture_mode = cfg.CAPTURE_MODE
    left_handed = handedness == "left"

    smoother = LandmarkSmoother(alpha=0.6)
    frames_temp_buffer = []
    prev_gray = None
    prev_hand_vector = None
    motion_val = 0
    landmark_motion_val = 0.0
    consecutive_still_frames = 0
    consecutive_hands_frames = 0
    missing_hands_frames = 0
    last_enqueue_time = [0.0]
    pending_eval_after = [0.0]

    def clear_capture_state():
        nonlocal frames_temp_buffer, consecutive_still_frames, missing_hands_frames
        nonlocal consecutive_hands_frames, prev_hand_vector
        frames_temp_buffer = []
        consecutive_still_frames = 0
        missing_hands_frames = 0
        consecutive_hands_frames = 0
        prev_hand_vector = None
        smoother.reset()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = vs.read()
            if frame is None:
                break

            image = frame.copy()
            results = None
            current_vector = None
            hands_present = False
            is_moving_pixels = False
            is_moving = False
            is_recording = len(frames_temp_buffer) > 0

            if not paused:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is not None:
                    frame_delta = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    motion_val = cv2.countNonZero(thresh)
                    if motion_val > cfg.MOTION_PIXEL_THRESHOLD:
                        is_moving_pixels = True
                prev_gray = gray

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if show_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                hands_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)

                if hands_present:
                    current_vector = extract_normalized_vector(results, left_handed=left_handed)
                    landmark_motion_val = compute_landmark_hand_motion(
                        current_vector, prev_hand_vector, cfg.POSE_DIM
                    )
                    prev_hand_vector = current_vector.copy()
                    consecutive_hands_frames += 1
                else:
                    landmark_motion_val = 0.0
                    consecutive_hands_frames = 0

                is_moving = is_moving_pixels or landmark_motion_val > cfg.LANDMARK_MOTION_THRESHOLD
            else:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (VID_W, 50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
                cv2.putText(image, "PAUSA - landmarks no se captan", (90, 34), UI_FONT, 0.7, (0, 200, 255), 2)

            n_models = len(active_specs)
            if n_models <= 1:
                pred_h = 70
            else:
                pred_h = 22 * n_models + 4
            pred_y = VID_H + 88
            dropdown_y = pred_y + pred_h
            TOT_H = dropdown_y + 56
            dropdown.set_rect(20, dropdown_y, 460, 30)
            btn_conf.set_rect(495, VID_H + 50, 125, 36)

            wheel = mouse_state["wheel"]
            mouse_state["wheel"] = 0
            clicked = mouse_state["clicked"]
            dropdown_consumed = False
            if not show_config and not args.eval:
                dropdown_consumed = dropdown.update(mouse_state["x"], mouse_state["y"], clicked, wheel)
                if dropdown.changed:
                    spec = catalog[dropdown.selected]
                    loaded = worker.set_specs([spec])
                    if loaded:
                        active_specs = loaded
                        print(f"[*] Modelo activo: {spec.label}")
            clicked_ui = clicked and not dropdown_consumed and not (dropdown.open and not args.eval)

            if not paused:
                if not show_config and btn_capture.update(mouse_state["x"], mouse_state["y"], clicked_ui):
                    if len(frames_temp_buffer) >= cfg.MIN_CAPTURE_FRAMES:
                        if enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time):
                            pending_eval_after[0] = last_enqueue_time[0]
                        clear_capture_state()

                if not is_recording and should_start_recording(
                    capture_mode, hands_present, is_moving, consecutive_hands_frames
                ):
                    is_recording = True

                if is_recording:
                    if hands_present and current_vector is not None:
                        missing_hands_frames = 0
                        smooth_vector = smoother.update(current_vector)
                        frames_temp_buffer.append(smooth_vector)

                        if is_moving:
                            consecutive_still_frames = 0
                            cv2.circle(image, (30, 30), 10, (0, 255, 0), -1)
                        else:
                            consecutive_still_frames += 1
                            cv2.circle(image, (30, 30), 10, (0, 255, 255), -1)

                        if (
                            len(frames_temp_buffer) >= cfg.CAPTURE_BUFFER_SIZE
                            or consecutive_still_frames >= cfg.STILL_FRAMES_LIMIT
                        ):
                            if enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time):
                                pending_eval_after[0] = last_enqueue_time[0]
                            clear_capture_state()

                    else:
                        missing_hands_frames += 1
                        cv2.circle(image, (30, 30), 10, (0, 0, 255), -1)

                        if len(frames_temp_buffer) > 0:
                            frames_temp_buffer.append(frames_temp_buffer[-1])

                        if missing_hands_frames >= cfg.MISSING_HANDS_LIMIT:
                            if enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time):
                                pending_eval_after[0] = last_enqueue_time[0]
                            clear_capture_state()

            with shared_state["lock"]:
                top3 = list(shared_state["top3"])
                p_txt = shared_state["prediction"]
                c_val = shared_state["confidence"]
                last_inf_time = shared_state["last_inference_time"]
                results_by_model = dict(shared_state["results_by_model"])

            if (
                pending_eval_after[0] > 0
                and results_by_model
                and eval_session
                and not eval_session.finished
                and last_inf_time >= pending_eval_after[0]
            ):
                eval_session.log_predictions(results_by_model, capture_mode)
                pending_eval_after[0] = 0.0
                if eval_session.finished:
                    print(f"[*] Evaluacion completa. CSV: {eval_session.csv_path}")

            canvas = np.zeros((TOT_H, VID_W, 3), dtype="uint8")
            canvas[0:VID_H, 0:VID_W] = image
            cv2.rectangle(canvas, (0, VID_H), (VID_W, TOT_H), (30, 30, 30), -1)

            if not show_config:
                buf_len = len(frames_temp_buffer)
                prog = min(buf_len / cfg.CAPTURE_BUFFER_SIZE, 1.0)
                col_prog = (0, 255, 0) if prog >= 1.0 else (0, 255, 255)
                if consecutive_still_frames > 0:
                    col_prog = (0, 165, 255)
                if paused:
                    col_prog = (80, 80, 80)

                cv2.rectangle(canvas, (20, VID_H + 18), (20 + int(240 * prog), VID_H + 28), col_prog, -1)
                cv2.putText(
                    canvas,
                    f"Buffer: {buf_len} | Silencio: {consecutive_still_frames}/{cfg.STILL_FRAMES_LIMIT}",
                    (20, VID_H + 48),
                    UI_FONT,
                    0.42,
                    (150, 150, 150),
                    1,
                )

                hand_label = "ZURDO" if left_handed else "DIESTRO"
                cv2.putText(
                    canvas,
                    f"Mano: {hand_label} | Modo: {capture_mode.upper()}",
                    (20, VID_H + 68),
                    UI_FONT,
                    0.42,
                    (180, 180, 180),
                    1,
                )

                if eval_session and not eval_session.finished:
                    expected = eval_session.expected_sign
                    cv2.putText(
                        canvas,
                        f"EVAL {eval_session.index + 1}/{len(eval_session.sign_list)} -> {expected.upper()}",
                        (20, 20),
                        UI_FONT,
                        0.65,
                        (0, 200, 255),
                        2,
                    )
                elif eval_session and eval_session.finished:
                    expected = None
                    cv2.putText(canvas, "EVAL COMPLETA", (20, 20), UI_FONT, 0.65, (0, 255, 0), 2)
                else:
                    expected = None

                threshold = cfg.CONFIDENCE_THRESHOLD
                if n_models > 1:
                    draw_multi_model_panel(
                        canvas,
                        active_specs,
                        results_by_model,
                        pred_y + 16,
                        threshold,
                        expected_sign=expected,
                    )
                elif top3:
                    draw_top3_panel(canvas, top3, pred_y + 16, threshold)
                elif p_txt != "...":
                    color = (0, 255, 0) if c_val >= threshold else (0, 200, 255)
                    cv2.putText(canvas, p_txt, (20, pred_y + 28), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
                    cv2.putText(
                        canvas,
                        f"Confianza: {c_val:.1%}",
                        (20, pred_y + 56),
                        UI_FONT,
                        0.55,
                        (180, 180, 180),
                        1,
                    )
                else:
                    cv2.putText(canvas, "Esperando...", (20, pred_y + 28), UI_FONT, 0.9, (100, 100, 100), 2)

                help_txt = "q=salir | m=modo | p=pausa | n=saltar (eval)"
                cv2.putText(canvas, help_txt, (20, TOT_H - 10), UI_FONT, 0.42, (120, 120, 120), 1)

                if btn_view.update(mouse_state["x"], mouse_state["y"], clicked_ui):
                    show_landmarks = not show_landmarks
                btn_view.draw(canvas, active=show_landmarks)

                if btn_conf.update(mouse_state["x"], mouse_state["y"], clicked_ui):
                    show_config = True
                    dropdown.open = False
                btn_conf.draw(canvas)
                btn_capture.draw(canvas)

                if btn_pause.update(mouse_state["x"], mouse_state["y"], clicked_ui):
                    paused = not paused
                    if paused:
                        clear_capture_state()
                    else:
                        prev_gray = None
                    print("[*] PAUSA" if paused else "[*] Captura reanudada")
                btn_pause.text = "PLAY" if paused else "PAUSA"
                btn_pause.draw(canvas, active=paused)

                if args.eval:
                    cv2.putText(
                        canvas,
                        fit_text("Eval: " + " + ".join(s.short for s in active_specs), 460, 0.42),
                        (20, dropdown_y + 20),
                        UI_FONT,
                        0.42,
                        (180, 180, 180),
                    )
                else:
                    dropdown.draw(canvas)
            else:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (VID_W, TOT_H), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
                mx, my, mw, mh = 100, 100, 440, 340
                cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (50, 50, 50), -1)
                cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (0, 165, 255), 2)

                slider_sens.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_sens.draw(canvas)
                slider_conf.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_conf.draw(canvas)
                slider_still.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_still.draw(canvas)
                slider_static.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_static.draw(canvas)

                cv2.putText(
                    canvas,
                    f"Modo captura: {capture_mode} (tecla 'm' para alternar)",
                    (mx + 10, my + mh - 15),
                    UI_FONT,
                    0.45,
                    (200, 200, 200),
                    1,
                )

                if btn_save.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    cfg.MOTION_PIXEL_THRESHOLD = int(slider_sens.val)
                    cfg.CONFIDENCE_THRESHOLD = slider_conf.val
                    cfg.STILL_FRAMES_LIMIT = int(slider_still.val)
                    cfg.STATIC_HANDS_FRAMES_TO_START = int(slider_static.val)
                    show_config = False
                btn_save.draw(canvas)

            cv2.imshow("LSA DETECTOR", canvas)
            mouse_state["clicked"] = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                capture_mode = {"auto": "static", "static": "dynamic", "dynamic": "auto"}[capture_mode]
                print(f"[*] Modo captura cambiado a: {capture_mode}")
            if key == ord("p"):
                paused = not paused
                if paused:
                    clear_capture_state()
                else:
                    prev_gray = None
                print("[*] PAUSA" if paused else "[*] Captura reanudada")
            if key == ord("n") and eval_session and not eval_session.finished:
                eval_session.skip_current(capture_mode)
                print(f"[*] Saltada senia. Siguiente: {eval_session.expected_sign}")

    shared_state["running"] = False
    vs.stop()
    cv2.destroyAllWindows()

    if eval_session:
        print(
            f"[*] Evaluacion: {eval_session.index}/{len(eval_session.sign_list)} "
            f"registros en {eval_session.csv_path}"
        )


if __name__ == "__main__":
    main()
