import argparse
import csv
import os
import time
import json
from datetime import datetime
from threading import Thread, Lock
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

from src.config import (
    CAPTURE_MODE, 
    CONFIDENCE_THRESHOLD, 
    INFERENCE_COOLDOWN_SEC, 
    MIN_CAPTURE_FRAMES, 
    POSE_DIM, 
    WEIGHTS_PATH, 
    CLASSES_MAP_JSON
)
from src.model.backbone import get_model
from src.utils import (
    get_anchor_and_scale,
    normalize_spatial_points,
    sequence_buffer_to_model_input,
    mirror_landmarks_for_left_handed,
)

# Configuración explícita de GPU para Keras / TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[*] GPU detectada y configurada: {gpus[0].name}")
    except RuntimeError as e:
        print(f"[!] Error al configurar GPU: {e}")
else:
    print("[!] ADVERTENCIA: No se detectó GPU. Se usará CPU.")

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


def prepare_input_tensor(buffer_list, target_frames=16):
    """Convierte buffer temporal a array NumPy con forma (1, target_frames, features)."""
    matrix = sequence_buffer_to_model_input(buffer_list)
    
    # Interpolar o submuestrear para asegurar exactamente los 16 frames deseados
    if matrix.shape[0] != target_frames:
        indices = np.linspace(0, matrix.shape[0] - 1, target_frames).astype(int)
        matrix = matrix[indices]

    tensor = np.expand_dims(matrix, axis=0).astype(np.float32)
    return tensor


def extract_normalized_vector(results, left_handed: bool = False):
    """
    Extrae y normaliza los landmarks de la cámara siguiendo la misma lógica 
    del script de preprocesamiento.
    """
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        if results.left_hand_landmarks
        else np.full((21, 3), np.nan)
    )
    pose = (
        np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])
        if results.pose_landmarks
        else np.full((33, 3), np.nan)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        if results.right_hand_landmarks
        else np.full((21, 3), np.nan)
    )

    all_landmarks = np.concatenate([lh, pose, rh], axis=0)
    all_landmarks = np.nan_to_num(all_landmarks, nan=0.0)

    anchor, scale = get_anchor_and_scale(results.pose_landmarks)
    raw_flat = all_landmarks.flatten()
    vector = normalize_spatial_points(raw_flat, anchor, scale)

    if left_handed:
        vector = mirror_landmarks_for_left_handed(vector, pose_dim=POSE_DIM)

    return vector


# =============================================================================
# COMPONENTES UI
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


def select_handedness_modal():
    modal_w, modal_h = 520, 280
    canvas = np.zeros((modal_h, modal_w, 3), dtype=np.uint8)
    choice = {"value": None}

    def pick_right(): choice["value"] = "right"
    def pick_left(): choice["value"] = "left"

    btn_right = Button(70, 160, 160, 50, "DIESTRO", pick_right)
    btn_left = Button(290, 160, 160, 50, "ZURDO", pick_left)
    mouse = {"x": 0, "y": 0, "down": False, "clicked": False}

    def on_mouse(event, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN: mouse["down"] = True
        elif event == cv2.EVENT_LBUTTONUP: mouse["down"] = False; mouse["clicked"] = True

    cv2.namedWindow("LSA DETECTOR - Configuracion")
    cv2.setMouseCallback("LSA DETECTOR - Configuracion", on_mouse)

    while choice["value"] is None:
        canvas[:] = (35, 35, 35)
        cv2.putText(canvas, "Mano dominante", (130, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, "Selecciona antes de iniciar la camara", (70, 100), UI_FONT, 0.55, (180, 180, 180), 1)
        btn_right.update(mouse["x"], mouse["y"], mouse["clicked"])
        btn_left.update(mouse["x"], mouse["y"], mouse["clicked"])
        btn_right.draw(canvas)
        btn_left.draw(canvas)
        cv2.imshow("LSA DETECTOR - Configuracion", canvas)
        mouse["clicked"] = False

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("d"), ord("D")): choice["value"] = "right"
        elif key in (ord("z"), ord("Z")): choice["value"] = "left"
        elif key == 27:
            cv2.destroyWindow("LSA DETECTOR - Configuracion")
            return None

    cv2.destroyWindow("LSA DETECTOR - Configuracion")
    return choice["value"]


# =============================================================================
# EVAL CSV
# =============================================================================
class EvalSession:
    FIELDNAMES = [
        "timestamp", "eval_index", "expected_sign",
        "top1", "conf1", "top2", "conf2", "top3", "conf3",
        "hit_top1", "hit_top3", "handedness", "capture_mode",
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
    def finished(self): return self.index >= len(self.sign_list)

    @property
    def expected_sign(self): return None if self.finished else self.sign_list[self.index]

    def log_prediction(self, top3: list[tuple[str, float]], capture_mode: str):
        if self.finished: return
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
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
        self.index += 1

    def skip_current(self, capture_mode: str):
        if self.finished: return
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_index": self.index + 1,
            "expected_sign": self.expected_sign,
            "top1": "SKIP", "conf1": "", "top2": "", "conf2": "", "top3": "", "conf3": "",
            "hit_top1": 0, "hit_top3": 0, "handedness": self.handedness, "capture_mode": capture_mode,
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
        self.index += 1


# =============================================================================
# WORKER DE INFERENCIA
# =============================================================================
shared_state = {
    "inference_queue": deque(maxlen=5),
    "prediction": "...",
    "confidence": 0.0,
    "top3": [],
    "last_inference_time": 0.0,
    "lock": Lock(),
    "running": True,
}


class KerasInferenceWorker:
    def __init__(self, idx_to_class, model_weights_path, num_classes):
        self.idx_to_class = idx_to_class
        self.model_weights_path = model_weights_path
        self.num_classes = num_classes
        self.model = None

    def start(self):
        Thread(target=self.loop, args=(), daemon=True).start()

    def _decode_top3(self, probs):
        top_indices = np.argsort(probs)[::-1][:3]
        results = []
        for idx in top_indices:
            name = self.idx_to_class.get(idx, "desconocido")
            results.append((name, float(probs[idx])))
        return results

    def _build_and_load_model(self):
        base_model = get_model()
        x = base_model.layers[-2].output
        outputs = layers.Dense(self.num_classes, activation='softmax', name='lsa_classifier_94')(x)
        model = keras.Model(inputs=base_model.input, outputs=outputs)
        model.load_weights(self.model_weights_path, by_name=True, skip_mismatch=True)
        print(f"[*] Pesos cargados exitosamente desde {self.model_weights_path}")
        return model

    def loop(self):
        try:
            self.model = self._build_and_load_model()
        except Exception as e:
            print(f"[!] Error al construir o cargar los pesos del modelo: {e}")
            return

        while shared_state["running"]:
            input_tensor = None
            with shared_state["lock"]:
                if len(shared_state["inference_queue"]) > 0:
                    input_tensor = shared_state["inference_queue"].popleft()

            if input_tensor is not None:
                try:
                    preds = self.model.predict(input_tensor, verbose=0)[0]
                    top3 = self._decode_top3(preds)

                    with shared_state["lock"]:
                        shared_state["top3"] = top3
                        shared_state["prediction"] = top3[0][0].upper()
                        shared_state["confidence"] = top3[0][1]
                        shared_state["last_inference_time"] = time.time()

                    print(" | ".join(f"{n.upper()} ({c:.1%})" for n, c in top3))
                except Exception as e:
                    print(f"Error en inferencia Keras: {e}")
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
            if not grabbed: self.stop()
            else: self.frame = frame

    def read(self): return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# =============================================================================
# MAIN
# =============================================================================
mouse_state = {"x": 0, "y": 0, "down": False, "clicked": False}


def mouse_callback(event, x, y, flags, param):
    mouse_state["x"], mouse_state["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN: mouse_state["down"] = True
    elif event == cv2.EVENT_LBUTTONUP: mouse_state["down"] = False; mouse_state["clicked"] = True


def can_enqueue_inference(last_enqueue_time: float) -> bool:
    now = time.time()
    if now - last_enqueue_time < INFERENCE_COOLDOWN_SEC: return False
    with shared_state["lock"]:
        if now - shared_state["last_inference_time"] < INFERENCE_COOLDOWN_SEC: return False
    return True


def enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time_ref: list):
    if not can_enqueue_inference(last_enqueue_time_ref[0]): return False
    tensor = prepare_input_tensor(frames_temp_buffer, target_frames=16)
    if tensor is not None:
        with shared_state["lock"]:
            shared_state["inference_queue"].append(tensor)
        last_enqueue_time_ref[0] = time.time()
        return True
    return False


def draw_top3_panel(canvas, top3, y_start, threshold):
    for i, (name, conf) in enumerate(top3[:3]):
        color = (0, 255, 0) if i == 0 and conf >= threshold else (200, 200, 200)
        cv2.putText(canvas, f"{i + 1}. {name.upper()}  {conf:.0%}", (20, y_start + i * 22), UI_FONT, 0.55, color, 1)


def main():
    parser = argparse.ArgumentParser(description="Inferencia Transfer Learning LSA en tiempo real (Keras).")
    parser.add_argument("--eval", action="store_true", help="Modo evaluacion: recorre las senias y guarda CSV.")
    parser.add_argument("--eval-output", default=None, help="Ruta del CSV de evaluacion.")
    args = parser.parse_args()

    with open(CLASSES_MAP_JSON, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
    sign_list = sorted(class_to_idx.keys())

    handedness = select_handedness_modal()
    if handedness is None: return

    eval_session = None
    if args.eval:
        csv_path = args.eval_output or os.path.join(
            os.path.dirname(__file__),
            f"eval_tf_91senias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        eval_session = EvalSession(sign_list, csv_path, handedness)
        print(f"[*] Modo eval activo. CSV: {csv_path}")

    worker = KerasInferenceWorker(idx_to_class, WEIGHTS_PATH, num_classes)
    worker.start()

    vs = WebcamStream(0).start()
    time.sleep(2.0)
    if vs.stopped: return

    cv2.namedWindow("LSA DETECTOR")
    cv2.setMouseCallback("LSA DETECTOR", mouse_callback)

    VID_W, VID_H = 640, 480
    TOT_H = VID_H + 175

    btn_view = Button(520, VID_H + 20, 100, 40, "Esqueleto")
    btn_capture = Button(400, VID_H + 20, 110, 40, "CAPTURAR")

    show_landmarks = True
    capture_mode = CAPTURE_MODE
    left_handed = handedness == "left"

    smoother = LandmarkSmoother(alpha=0.6)
    frames_temp_buffer = []
    last_enqueue_time = [0.0]
    pending_eval_after = [0.0]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = vs.read()
            if frame is None: break

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

            # -----------------------------------------------------------------
            # MÁQUINA DE ESTADOS: DETECCIÓN POR ENTRADA / SALIDA DE MANOS
            # -----------------------------------------------------------------
            if hands_present:
                # 1. Manos en pantalla: Grabar vector continuamente
                current_vector = extract_normalized_vector(results, left_handed=left_handed)
                smooth_vector = smoother.update(current_vector)
                frames_temp_buffer.append(smooth_vector)
                
                # Indicador de Grabación Activa (Punto Rojo)
                cv2.circle(image, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(image, f"GRABANDO: {len(frames_temp_buffer)} frames", (50, 35), UI_FONT, 0.6, (0, 0, 255), 2)
            else:
                # 2. Manos fuera de pantalla: Evaluar si recién se retiraron
                if len(frames_temp_buffer) > 0:
                    # Se terminó la seña -> Verificar si cumple la duración mínima
                    if len(frames_temp_buffer) >= MIN_CAPTURE_FRAMES:
                        if enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time):
                            pending_eval_after[0] = last_enqueue_time[0]
                    
                    # Limpiar buffer y suavizador para la próxima seña
                    frames_temp_buffer = []
                    smoother.reset()

            # Disparador manual con botón "CAPTURAR" por si se desea forzar el envío
            if btn_capture.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                if len(frames_temp_buffer) >= MIN_CAPTURE_FRAMES:
                    if enqueue_buffer_for_inference(frames_temp_buffer, last_enqueue_time):
                        pending_eval_after[0] = last_enqueue_time[0]
                    frames_temp_buffer = []
                    smoother.reset()

            # -----------------------------------------------------------------
            # ACTUALIZACIÓN DE ESTADO Y UI
            # -----------------------------------------------------------------
            with shared_state["lock"]:
                top3 = list(shared_state["top3"])
                last_inf_time = shared_state["last_inference_time"]

            if (pending_eval_after[0] > 0 and top3 and eval_session and not eval_session.finished and last_inf_time >= pending_eval_after[0]):
                eval_session.log_prediction(top3, capture_mode)
                pending_eval_after[0] = 0.0

            canvas = np.zeros((TOT_H, VID_W, 3), dtype="uint8")
            canvas[0:VID_H, 0:VID_W] = image
            cv2.rectangle(canvas, (0, VID_H), (VID_W, TOT_H), (30, 30, 30), -1)

            if eval_session and not eval_session.finished:
                cv2.putText(canvas, f"EVAL {eval_session.index + 1}/{len(sign_list)} -> {eval_session.expected_sign.upper()}", (20, 20), UI_FONT, 0.65, (0, 200, 255), 2)
            if top3: draw_top3_panel(canvas, top3, VID_H + 68, CONFIDENCE_THRESHOLD)

            btn_view.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"])
            btn_view.draw(canvas, active=show_landmarks)
            btn_capture.draw(canvas)

            cv2.imshow("LSA DETECTOR", canvas)
            mouse_state["clicked"] = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("n") and eval_session and not eval_session.finished:
                eval_session.skip_current(capture_mode)

    shared_state["running"] = False
    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()