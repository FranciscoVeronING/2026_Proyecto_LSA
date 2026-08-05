import argparse
import csv
import os
import time
import json
from datetime import datetime
from threading import Thread, Lock
from collections import deque
from queue import Queue, Empty

import pyttsx3
import cv2
import mediapipe as mp
import numpy as np
import torch

import model.classifier.config as cfg
from model.classifier.model_arch import TinySkeletonClassifier
from utils import (
    get_anchor_and_scale,
    normalize_spatial_points,
    compute_landmark_hand_motion,
    sequence_buffer_to_model_input,
    mirror_landmarks_for_left_handed,
)
from repeat_policy import RepeatGate, format_literal_utterance
from conversation_memory import ConversationMemory

# OJO: NO importar src.model.semantic.model acá.
# Unsloth/transformers pueden tumbar el arranque de la cámara.


# =============================================================================
# GLOSSES → LLM
# =============================================================================
def normalize_gloss(raw_name: str) -> str:
    """Classifier key --> normalized gloss (upper case)."""
    key = (raw_name or "").strip()
    mapped = cfg.GLOSS_NORMALIZER.get(key)
    if mapped:
        return mapped
    mapped = cfg.GLOSS_NORMALIZER.get(key.lower())
    if mapped:
        return mapped
    return key.upper()


class UtteranceBuffer:
    """
    Gather accepted glosses. When UTTERANCE_PAUSE_SEC passes without a new one,
    closes the utterance and submits it to the LLM.
    """

    def __init__(
        self,
        pause_sec: float,
        dedup_sec: float,
        min_confidence: float,
        max_letter_consecutive: int = 2,
    ):
        self.pause_sec = pause_sec
        self.min_confidence = min_confidence
        self.glosses = []
        self.repeat_gate = RepeatGate(
            dedup_sec=dedup_sec,
            max_letter_consecutive=max_letter_consecutive,
        )

    def try_add(self, gloss_raw, confidence, now):
        if confidence < self.min_confidence:
            return False
        gloss = normalize_gloss(gloss_raw)
        if not gloss or gloss == "DESCONOCIDO":
            return False
        if not self.repeat_gate.allow(gloss, now):
            return False
        self.glosses.append(gloss)
        return True

    def maybe_close(self, now):
        if not self.glosses or self.repeat_gate.last_at is None:
            return None
        if (now - self.repeat_gate.last_at) < self.pause_sec:
            return None
        closed = list(self.glosses)
        self.glosses.clear()
        self.repeat_gate.reset()
        return closed

    def pending_text(self) -> str:
        return " ".join(self.glosses) if self.glosses else ""


class SemanticWorker:
    """
    Async queue: list[str] → translate_glosses(str).
    Carga la LLM en un hilo; si falla, la cámara sigue sin traducción.
    """

    def __init__(self, enabled: bool = True, history_size: int = 10):
        self.enabled = enabled
        self.queue: Queue = Queue(maxsize=8)
        self.ready = False
        self._translate_glosses = None
        self.memory = ConversationMemory(maxlen=history_size)
        # NO llamar load_model_and_tokenizer() acá (bloquea / puede crashear).

    def start(self):
        if not self.enabled:
            print("[*] Semantic disabled (--no-llm). Only glosses will be displayed.")
            return
        Thread(target=self._bootstrap_and_loop, args=(), daemon=True).start()

    def _bootstrap_and_loop(self):
        try:
            # Compat: `python -m camera` (cwd=src) o `python -m src.camera` (raíz)
            try:
                from model.semantic.model import (
                    load_model_and_tokenizer,
                    translate_glosses,
                )
            except ImportError:
                from src.model.semantic.model import (
                    load_model_and_tokenizer,
                    translate_glosses,
                )

            load_model_and_tokenizer()
            self._translate_glosses = translate_glosses
            self.ready = True
            print("[*] Semantic worker ready (translate_glosses).")
        except Exception as e:
            print(f"[!] Could not initialize semantic: {e}")
            print("[!] Camera continues; glosses only (no LLM).")
            print("[!] Tip: desde la raiz del repo → python -m src.camera")
            print("[!]      o desde src/ → python -m camera")
            self._translate_glosses = None
            self.ready = False

        while shared_state["running"]:
            try:
                glosses = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if glosses is None:
                break
            self._handle(glosses)

    def _handle(self, glosses):
        joined = " ".join(glosses)
        literal = format_literal_utterance(glosses)
        if literal is not None:
            print(f"[*] Utterance closed → literal: {joined} => {literal}")
        else:
            print(f"[*] Utterance closed → LLM: {joined}")

        with shared_state["lock"]:
            shared_state["semantic_busy"] = True
            shared_state["last_utterance"] = joined
            shared_state["spanish_text"] = (
                literal if literal is not None else "Translating..."
            )

        # Deletreo / solo números: mostrar y decir sin pasar por la LLM
        text = literal if literal is not None else joined
        if literal is None and self._translate_glosses is not None:
            try:
                history = self.memory.as_messages()
                text = self._translate_glosses(joined, history_messages=history) or joined
            except Exception as e:
                print(f"[!] Error LLM: {e}")
                text = joined

        # Contexto cercano: también literales (deletreo / números)
        self.memory.add_signer(text, glosses=joined)

        with shared_state["lock"]:
            shared_state["spanish_text"] = text
            shared_state["semantic_busy"] = False
            shared_state["utterance_glosses"] = []
            shared_state["conversation_turns"] = len(self.memory.turns)

        if cfg.VOICE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 100)  # Adjust speech rate if needed
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[!] Error in voice synthesis: {e}")
        print(f"[*] Español: {text}")
        print(f"[*] Memoria conversación: {len(self.memory.turns)}/{self.memory.maxlen} turnos")

    def add_hearing(self, text: str):
        """API lista para UI/STT del oyente (aún sin captura en cámara)."""
        self.memory.add_hearing(text)
        with shared_state["lock"]:
            shared_state["conversation_turns"] = len(self.memory.turns)
        print(f"[*] Oyente registrado: {text}")

    def clear_conversation(self):
        self.memory.clear()
        with shared_state["lock"]:
            shared_state["conversation_turns"] = 0
        print("[*] Conversación reiniciada.")

    def submit(self, glosses):
        if not glosses:
            return
        if not self.enabled:
            joined = " ".join(glosses)
            text = format_literal_utterance(glosses) or joined
            print(f"[*] Utterance closed (without LLM): {joined} => {text}")
            self.memory.add_signer(text, glosses=joined)
            with shared_state["lock"]:
                shared_state["last_utterance"] = joined
                shared_state["spanish_text"] = text
                shared_state["utterance_glosses"] = []
                shared_state["conversation_turns"] = len(self.memory.turns)
            return
        try:
            self.queue.put_nowait(list(glosses))
        except Exception:
            print("[!] Semantic queue full; utterance discarded:", " ".join(glosses))


# =============================================================================
# UTILITIES
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


def prepare_input_tensor(buffer_list, device):
    """Transforms a list of buffers into a tensor (1, MAX_FRAMES, features)."""
    matrix = sequence_buffer_to_model_input(buffer_list)
    if matrix.shape[0] != cfg.MAX_FRAMES:
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
    """Constructs a (225,) vector from MediaPipe results; mirrors if left-handed."""
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


# =============================================================================
# UI COMPONENTS
# =============================================================================
class Button:
    def __init__(self, x, y, w, h, text, callback_func=None):
        self.rect = (x, y, w, h)
        self.text = text
        self.callback = callback_func
        self.is_hover = False

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


def select_handedness_modal():
    """
    Initial modal: choose right or left hand before opening the camera.
    Returns 'right' or 'left'.
    """
    modal_w, modal_h = 520, 280
    canvas = np.zeros((modal_h, modal_w, 3), dtype=np.uint8)
    choice = {"value": None}

    def pick_right():
        choice["value"] = "right"

    def pick_left():
        choice["value"] = "left"

    btn_right = Button(70, 160, 160, 50, "RIGHT", pick_right)
    btn_left = Button(290, 160, 160, 50, "LEFT", pick_left)
    mouse = {"x": 0, "y": 0, "down": False, "clicked": False}

    def on_mouse(event, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["down"] = False
            mouse["clicked"] = True

    cv2.namedWindow("LSA DETECTOR - Settings")
    cv2.setMouseCallback("LSA DETECTOR - Settings", on_mouse)

    while choice["value"] is None:
        canvas[:] = (35, 35, 35)
        cv2.putText(canvas, "Dominant Hand", (130, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "Select before starting the camera", (70, 100), UI_FONT, 0.55, (180, 180, 180), 1)
        cv2.putText(canvas, "Keys: D = right | Z = left", (110, 130), UI_FONT, 0.5, (140, 140, 140), 1)
        btn_right.update(mouse["x"], mouse["y"], mouse["clicked"])
        btn_left.update(mouse["x"], mouse["y"], mouse["clicked"])
        btn_right.draw(canvas)
        btn_left.draw(canvas)
        cv2.imshow("LSA DETECTOR - Settings", canvas)
        mouse["clicked"] = False

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("d"), ord("D")):
            choice["value"] = "right"
        elif key in (ord("z"), ord("Z")):
            choice["value"] = "left"
        elif key == 27:
            cv2.destroyWindow("LSA DETECTOR - Settings")
            return None

    cv2.destroyWindow("LSA DETECTOR - Settings")
    label = "right" if choice["value"] == "right" else "left"
    print(f"[*] Dominant hand: {label}")
    return choice["value"]


# =============================================================================
# EVAL CSV
# =============================================================================
class EvalSession:
    FIELDNAMES = [
        "timestamp",
        "eval_index",
        "expected_sign",
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

    def __init__(self, sign_list: list[str], csv_path: str, handedness: str):
        self.sign_list = sign_list
        self.csv_path = csv_path
        self.handedness = handedness
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

    def log_prediction(self, top3: list[tuple[str, float]], capture_mode: str):
        if self.finished:
            return

        expected = self.expected_sign
        top_names = [t[0] for t in top3]
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_index": self.index + 1,
            "expected_sign": expected,
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
        self.index += 1

    def skip_current(self, capture_mode: str):
        if self.finished:
            return
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_index": self.index + 1,
            "expected_sign": self.expected_sign,
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
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)
        self.index += 1


# =============================================================================
# BACKEND AND WORKERS
# =============================================================================
shared_state = {
    "inference_queue": deque(maxlen=5),
    "prediction": "...",
    "confidence": 0.0,
    "top3": [],
    "last_inference_time": 0.0,
    "utterance_glosses": [],
    "last_utterance": "",
    "spanish_text": "",
    "semantic_busy": False,
    "conversation_turns": 0,
    "lock": Lock(),
    "running": True,
}


class InferenceWorker:
    def __init__(self, idx_to_class, num_classes, device):
        self.idx_to_class = idx_to_class
        self.device = device
        self.model = None

        try:
            self.model = TinySkeletonClassifier(
                cfg.FRAME_FEATURES_DIM,
                cfg.HIDDEN_DIM,
                num_heads=cfg.NUM_HEADS,
                num_layers=cfg.NUM_LAYERS,
                num_classes=num_classes,
                dropout_rate=cfg.DROPOUT_RATE,
            ).to(self.device)

            ruta_modelo = os.path.join(cfg.MODEL_SAVE_DIR)
            self.model.load_state_dict(torch.load(ruta_modelo, map_location=self.device, weights_only=True))
            self.model.eval()
            print("[*] Modelo PyTorch cargado. Worker listo en", self.device)
        except Exception as e:
            print(f"[!] Error cargando modelo: {e}")
            print("[!] Si cambiaste la arquitectura, reentrená con train.py antes de usar la cámara.")

    def start(self):
        Thread(target=self.loop, args=(), daemon=True).start()

    def _decode_top3(self, probs):
        values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
        results = []
        for conf, idx in zip(values.tolist(), indices.tolist()):
            name = self.idx_to_class.get(idx, "desconocido")
            results.append((name, float(conf)))
        return results

    def loop(self):
        while shared_state["running"]:
            if self.model is None:
                time.sleep(1)
                continue

            input_tensor = None
            with shared_state["lock"]:
                if len(shared_state["inference_queue"]) > 0:
                    input_tensor = shared_state["inference_queue"].popleft()

            if input_tensor is not None:
                try:
                    with torch.no_grad():
                        logits = self.model(input_tensor)
                        probs = torch.softmax(logits, dim=1)[0]
                        top3 = self._decode_top3(probs)

                    with shared_state["lock"]:
                        shared_state["top3"] = top3
                        shared_state["prediction"] = top3[0][0].upper()
                        shared_state["confidence"] = top3[0][1]
                        shared_state["last_inference_time"] = time.time()

                    print(
                        " | ".join(f"{n.upper()} ({c:.1%})" for n, c in top3)
                    )
                except Exception as e:
                    print(f"Error inferencia: {e}")
            else:
                time.sleep(0.01)


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
mouse_state = {"x": 0, "y": 0, "down": False, "clicked": False}


def mouse_callback(event, x, y, flags, param):
    mouse_state["x"], mouse_state["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["down"] = False
        mouse_state["clicked"] = True


def can_enqueue_inference(last_enqueue_time: float) -> bool:
    now = time.time()
    if now - last_enqueue_time < cfg.INFERENCE_COOLDOWN_SEC:
        return False
    with shared_state["lock"]:
        if now - shared_state["last_inference_time"] < cfg.INFERENCE_COOLDOWN_SEC:
            return False
    return True


def enqueue_buffer_for_inference(frames_temp_buffer, device, last_enqueue_time_ref: list):
    if not can_enqueue_inference(last_enqueue_time_ref[0]):
        return False
    tensor = prepare_input_tensor(frames_temp_buffer, device)
    if tensor is not None:
        with shared_state["lock"]:
            shared_state["inference_queue"].append(tensor)
        last_enqueue_time_ref[0] = time.time()
        return True
    return False


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


def main():
    parser = argparse.ArgumentParser(description="Inferencia LSA en tiempo real.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Modo evaluacion: recorre las 91 senias y guarda CSV.",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Ruta del CSV de evaluacion (default: eval_91senias_<fecha>.csv en src/).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="No cargar la LLM: solo acumula glosas y muestra el listado al cerrar el enunciado.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ruta_mapeo = os.path.join(cfg.CLASSES_PATH)
    if not os.path.exists(ruta_mapeo):
        print(f"[!] No se encontro {ruta_mapeo}. Ejecuta el entrenamiento primero.")
        return

    with open(ruta_mapeo, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    sign_list = sorted(class_to_idx.keys())

    handedness = select_handedness_modal()
    if handedness is None:
        print("[!] Configuracion cancelada.")
        return

    eval_session = None
    if args.eval:
        csv_path = args.eval_output or os.path.join(
            os.path.dirname(__file__),
            f"eval_91senias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        eval_session = EvalSession(sign_list, csv_path, handedness)
        print(f"[*] Modo eval activo. CSV: {csv_path}")
        print(f"[*] Senias a probar: {len(sign_list)}. Tecla 'n' = saltar senia.")

    print("\n[*] Iniciando camara...")
    print(f"[*] Modo captura: {cfg.CAPTURE_MODE} | MAX_FRAMES: {cfg.MAX_FRAMES}")
    print(f"[*] Umbral confianza: {cfg.CONFIDENCE_THRESHOLD:.0%} | Cooldown: {cfg.INFERENCE_COOLDOWN_SEC}s")
    print(
        f"[*] Pausa enunciado→LLM: {cfg.UTTERANCE_PAUSE_SEC}s | "
        f"dedup other: {cfg.GLOSS_DEDUP_SEC}s | "
        f"letras max: {cfg.LETTER_MAX_CONSECUTIVE}"
    )

    vs = WebcamStream(0).start()
    time.sleep(2.0)
    if vs.stopped:
        return

    worker = InferenceWorker(idx_to_class, num_classes, device)
    worker.start()

    utterance_buffer = UtteranceBuffer(
        pause_sec=cfg.UTTERANCE_PAUSE_SEC,
        dedup_sec=cfg.GLOSS_DEDUP_SEC,
        min_confidence=cfg.CONFIDENCE_THRESHOLD,
        max_letter_consecutive=cfg.LETTER_MAX_CONSECUTIVE,
    )
    try:
        from model.semantic.config import CONVERSATION_HISTORY_SIZE
    except ImportError:
        from src.model.semantic.config import CONVERSATION_HISTORY_SIZE

    semantic_worker = SemanticWorker(
        enabled=not args.no_llm and not args.eval,
        history_size=CONVERSATION_HISTORY_SIZE,
    )
    semantic_worker.start()
    print(f"[*] Memoria conversación: últimos {CONVERSATION_HISTORY_SIZE} turnos (tecla 'c' = limpiar)")
    last_seen_inference_time = 0.0

    cv2.namedWindow("LSA DETECTOR")
    cv2.setMouseCallback("LSA DETECTOR", mouse_callback)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    VID_W, VID_H = 640, 480
    TOT_H = VID_H + 230

    btn_view = Button(520, VID_H + 20, 100, 40, "Esqueleto")
    btn_conf = Button(520, VID_H + 80, 100, 40, "Config")
    btn_capture = Button(400, VID_H + 20, 110, 40, "CAPTURAR")
    btn_voice = Button(520, VID_H + 140, 100, 40, "VOICE")

    slider_sens = Slider(150, 150, 340, 100, 5000, cfg.MOTION_PIXEL_THRESHOLD, "Sensibilidad (Pixeles)")
    slider_conf = Slider(150, 200, 340, 0.1, 1.0, cfg.CONFIDENCE_THRESHOLD, "Confianza Min")
    slider_still = Slider(150, 250, 340, 5, 40, cfg.STILL_FRAMES_LIMIT, "Corte por Silencio (Frames)")
    slider_static = Slider(150, 300, 340, 2, 15, cfg.STATIC_HANDS_FRAMES_TO_START, "Frames Manos (Estatico)")
    
    btn_save = Button(220, 380, 100, 40, "CERRAR")

    show_config = False
    show_landmarks = True
    capture_mode = cfg.CAPTURE_MODE
    voice_enabled = cfg.VOICE
    left_handed = handedness == "left"

    smoother = LandmarkSmoother(alpha=0.6)
    frames_temp_buffer = []
    prev_gray = None
    prev_hand_vector = None
    consecutive_still_frames = 0
    consecutive_hands_frames = 0
    missing_hands_frames = 0
    last_enqueue_time = [0.0]
    pending_eval_after = [0.0]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = vs.read()
            if frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            is_moving_pixels = False

            if prev_gray is not None:
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_val = cv2.countNonZero(thresh)
                if motion_val > cfg.MOTION_PIXEL_THRESHOLD:
                    is_moving_pixels = True
            prev_gray = gray

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if show_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            hands_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            is_recording = len(frames_temp_buffer) > 0

            current_vector = None
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

            if btn_capture.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                if len(frames_temp_buffer) >= cfg.MIN_CAPTURE_FRAMES:
                    if enqueue_buffer_for_inference(frames_temp_buffer, device, last_enqueue_time):
                        pending_eval_after[0] = last_enqueue_time[0]
                    frames_temp_buffer = []
                    consecutive_still_frames = 0
                    missing_hands_frames = 0
                    smoother.reset()

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
                        if enqueue_buffer_for_inference(frames_temp_buffer, device, last_enqueue_time):
                            pending_eval_after[0] = last_enqueue_time[0]
                        frames_temp_buffer = []
                        consecutive_still_frames = 0
                        smoother.reset()

                else:
                    missing_hands_frames += 1
                    cv2.circle(image, (30, 30), 10, (0, 0, 255), -1)

                    if len(frames_temp_buffer) > 0:
                        frames_temp_buffer.append(frames_temp_buffer[-1])

                    if missing_hands_frames >= cfg.MISSING_HANDS_LIMIT:
                        if enqueue_buffer_for_inference(frames_temp_buffer, device, last_enqueue_time):
                            pending_eval_after[0] = last_enqueue_time[0]
                        frames_temp_buffer = []
                        consecutive_still_frames = 0
                        missing_hands_frames = 0
                        smoother.reset()

            with shared_state["lock"]:
                top3 = list(shared_state["top3"])
                p_txt = shared_state["prediction"]
                c_val = shared_state["confidence"]
                last_inf_time = shared_state["last_inference_time"]
                spanish_text = shared_state["spanish_text"]
                semantic_busy = shared_state["semantic_busy"]

            if (
                not args.eval
                and last_inf_time > last_seen_inference_time
                and top3
            ):
                last_seen_inference_time = last_inf_time
                gloss_name, gloss_conf = top3[0][0], top3[0][1]
                if utterance_buffer.try_add(gloss_name, gloss_conf, last_inf_time):
                    with shared_state["lock"]:
                        shared_state["utterance_glosses"] = list(utterance_buffer.glosses)
                    print(
                        f"[*] Gloss added: {normalize_gloss(gloss_name)} | "
                        f"list=[{utterance_buffer.pending_text()}]"
                    )

            if not args.eval:
                closed = utterance_buffer.maybe_close(time.time())
                if closed is not None:
                    with shared_state["lock"]:
                        shared_state["utterance_glosses"] = []
                        shared_state["last_utterance"] = " ".join(closed)
                    semantic_worker.submit(closed)

            if (
                pending_eval_after[0] > 0
                and top3
                and eval_session
                and not eval_session.finished
                and last_inf_time >= pending_eval_after[0]
            ):
                eval_session.log_prediction(top3, capture_mode)
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

                cv2.rectangle(canvas, (20, VID_H + 20), (20 + int(200 * prog), VID_H + 30), col_prog, -1)
                cv2.putText(
                    canvas,
                    f"Buffer: {buf_len} | Silencio: {consecutive_still_frames}/{cfg.STILL_FRAMES_LIMIT}",
                    (230, VID_H + 28),
                    UI_FONT,
                    0.45,
                    (150, 150, 150),
                    1,
                )

                hand_label = "LEFT" if left_handed else "RIGHT"
                cv2.putText(
                    canvas,
                    f"Hand: {hand_label} | Mode: {capture_mode.upper()}",
                    (20, VID_H + 48),
                    UI_FONT,
                    0.42,
                    (180, 180, 180),
                    1,
                )

                if eval_session and not eval_session.finished:
                    expected = eval_session.expected_sign
                    cv2.putText(
                        canvas,
                        f"EVAL {eval_session.index + 1}/{len(sign_list)} -> {expected.upper()}",
                        (20, 20),
                        UI_FONT,
                        0.65,
                        (0, 200, 255),
                        2,
                    )
                elif eval_session and eval_session.finished:
                    cv2.putText(canvas, "EVAL COMPLETA", (20, 20), UI_FONT, 0.65, (0, 255, 0), 2)

                threshold = utterance_buffer.min_confidence

                if top3:
                    draw_top3_panel(canvas, top3, VID_H + 68, threshold)
                elif p_txt != "...":
                    color = (0, 255, 0) if c_val >= threshold else (0, 200, 255)
                    cv2.putText(canvas, p_txt, (20, VID_H + 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
                    cv2.putText(
                        canvas,
                        f"Confianza: {c_val:.1%}",
                        (20, VID_H + 118),
                        UI_FONT,
                        0.55,
                        (180, 180, 180),
                        1,
                    )
                else:
                    cv2.putText(canvas, "Waiting...", (20, VID_H + 90), UI_FONT, 0.9, (100, 100, 100), 2)

                gloss_line = utterance_buffer.pending_text() or "(vacio)"
                cv2.putText(
                    canvas,
                    f"Glosas: {gloss_line}",
                    (20, VID_H + 145),
                    UI_FONT,
                    0.5,
                    (0, 220, 255),
                    1,
                )
                es_prefix = "ES (waiting for LLM)..." if semantic_busy else "ES:"
                es_line = spanish_text if spanish_text else "-"
                if len(es_line) > 70:
                    es_line = es_line[:67] + "..."
                cv2.putText(
                    canvas,
                    f"{es_prefix} {es_line}",
                    (20, VID_H + 172),
                    UI_FONT,
                    0.5,
                    (180, 255, 180),
                    1,
                )

                cv2.putText(
                    canvas,
                    "q=exit | m=mode | n=skip (eval) | pause 2s = send to LLM",
                    (20, TOT_H - 8),
                    UI_FONT,
                    0.38,
                    (120, 120, 120),
                    1,
                )

                if btn_view.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    show_landmarks = not show_landmarks
                btn_view.draw(canvas, active=show_landmarks)

                if btn_conf.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    show_config = True
                btn_conf.draw(canvas)

                btn_voice.text = "VOICE ON" if voice_enabled else "VOICE OFF"
                if btn_voice.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    voice_enabled = not voice_enabled
                    cfg.VOICE = voice_enabled
                    print(f"[*] Voice enabled: {voice_enabled}")
                btn_voice.draw(canvas)

                btn_capture.draw(canvas)
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
                    f"Capture mode: {capture_mode} (key 'm' to toggle)",
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
                    utterance_buffer.min_confidence = cfg.CONFIDENCE_THRESHOLD
                    show_config = False
                btn_save.draw(canvas)

            cv2.imshow("LSA DETECTOR", canvas)
            mouse_state["clicked"] = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                capture_mode = {"auto": "static", "static": "dynamic", "dynamic": "auto"}[capture_mode]
                print(f"[*] Capture mode changed to: {capture_mode}")
            if key == ord("c"):
                semantic_worker.clear_conversation()
            if key == ord("n") and eval_session and not eval_session.finished:
                eval_session.skip_current(capture_mode)
                print(f"[*] Skipped sign. Next: {eval_session.expected_sign}")

    shared_state["running"] = False
    vs.stop()
    cv2.destroyAllWindows()

    if eval_session:
        print(f"[*] Evaluation: {eval_session.index}/{len(sign_list)} records in {eval_session.csv_path}")


if __name__ == "__main__":
    main()
