"""
Inferencia LSA en tiempo real: cámara → landmarks → glosas → español.

Punto de entrada recomendado desde la raíz del repo:
    python run.py [--eval] [--no-llm]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Permite `python src/app/main.py` además de `python run.py`.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import mediapipe as mp
import numpy as np
import torch

import classifier.config as cfg
from app.capture import (
    LandmarkSmoother,
    WebcamStream,
    extract_normalized_vector,
    prepare_input_tensor,
    should_start_recording,
)
from app.eval_session import EvalSession
from app.state import shared_state, stop
from app.ui import (
    UI_FONT,
    Button,
    Slider,
    draw_top3_panel,
    mouse_callback,
    mouse_state,
    select_handedness_modal,
)
from app.utterance import UtteranceBuffer, normalize_gloss
from app.workers import InferenceWorker, SemanticWorker, VoiceWorker
from core.landmarks import compute_landmark_hand_motion
from semantic.config import CONVERSATION_HISTORY_SIZE

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia LSA en tiempo real.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Modo evaluacion: recorre todas las senias y guarda un CSV.",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Ruta del CSV de evaluacion (default: eval_senias_<fecha>.csv en la raiz).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="No cargar la LLM: solo acumula glosas y muestra el listado al cerrar.",
    )
    return parser.parse_args()


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
    if tensor is None:
        return False
    with shared_state["lock"]:
        shared_state["inference_queue"].append(tensor)
    last_enqueue_time_ref[0] = time.time()
    return True


def load_classes():
    """Devuelve (idx_to_class, sign_list) o (None, None) si falta el mapeo."""
    if not os.path.exists(cfg.CLASSES_PATH):
        print(f"[!] No se encontro {cfg.CLASSES_PATH}. Ejecuta el entrenamiento primero.")
        return None, None
    with open(cfg.CLASSES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class, sorted(class_to_idx.keys())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_to_class, sign_list = load_classes()
    if idx_to_class is None:
        return
    num_classes = len(idx_to_class)

    handedness = select_handedness_modal()
    if handedness is None:
        print("[!] Configuracion cancelada.")
        return

    eval_session = None
    if args.eval:
        csv_path = args.eval_output or str(
            REPO_ROOT / f"eval_senias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        eval_session = EvalSession(sign_list, csv_path, handedness)
        print(f"[*] Modo eval activo. CSV: {csv_path}")
        print(f"[*] Senias a probar: {len(sign_list)}. Tecla 'n' = saltar senia.")

    print("\n[*] Iniciando camara...")
    print(f"[*] Modo captura: {cfg.CAPTURE_MODE} | MAX_FRAMES: {cfg.MAX_FRAMES}")
    print(
        f"[*] Umbral confianza: {cfg.CONFIDENCE_THRESHOLD:.0%} | "
        f"Cooldown: {cfg.INFERENCE_COOLDOWN_SEC}s"
    )
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
    voice_worker = VoiceWorker()
    voice_worker.start()

    semantic_worker = SemanticWorker(
        enabled=not args.no_llm and not args.eval,
        history_size=CONVERSATION_HISTORY_SIZE,
        voice=voice_worker,
    )
    semantic_worker.start()
    print(
        f"[*] Memoria conversacional: ultimos {CONVERSATION_HISTORY_SIZE} turnos "
        "(tecla 'c' = limpiar)"
    )
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
                if cv2.countNonZero(thresh) > cfg.MOTION_PIXEL_THRESHOLD:
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
                    frames_temp_buffer.append(smoother.update(current_vector))

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
                conv_turns = shared_state["conversation_turns"]

            if not args.eval and last_inf_time > last_seen_inference_time and top3:
                last_seen_inference_time = last_inf_time
                gloss_name, gloss_conf = top3[0][0], top3[0][1]
                if utterance_buffer.try_add(gloss_name, gloss_conf, last_inf_time):
                    with shared_state["lock"]:
                        shared_state["utterance_glosses"] = list(utterance_buffer.glosses)
                    print(
                        f"[*] Glosa agregada: {normalize_gloss(gloss_name)} | "
                        f"lista=[{utterance_buffer.pending_text()}]"
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
                    f"Hand: {hand_label} | Mode: {capture_mode.upper()} | "
                    f"Contexto: {conv_turns}/{CONVERSATION_HISTORY_SIZE}",
                    (20, VID_H + 48),
                    UI_FONT,
                    0.42,
                    (180, 180, 180),
                    1,
                )

                if eval_session and not eval_session.finished:
                    cv2.putText(
                        canvas,
                        f"EVAL {eval_session.index + 1}/{len(sign_list)} -> "
                        f"{eval_session.expected_sign.upper()}",
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
                    cv2.putText(canvas, "Esperando...", (20, VID_H + 90), UI_FONT, 0.9, (100, 100, 100), 2)

                cv2.putText(
                    canvas,
                    f"Glosas: {utterance_buffer.pending_text() or '(vacio)'}",
                    (20, VID_H + 145),
                    UI_FONT,
                    0.5,
                    (0, 220, 255),
                    1,
                )

                es_prefix = "ES (traduciendo)..." if semantic_busy else "ES:"
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
                    "q=salir | m=modo | c=limpiar contexto | n=saltar (eval)",
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
                    print(f"[*] Voz: {'on' if voice_enabled else 'off'}")
                btn_voice.draw(canvas)

                btn_capture.draw(canvas)
            else:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0, 0), (VID_W, TOT_H), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
                mx, my, mw, mh = 100, 100, 440, 340
                cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (50, 50, 50), -1)
                cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (0, 165, 255), 2)

                for slider in (slider_sens, slider_conf, slider_still, slider_static):
                    slider.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                    slider.draw(canvas)

                cv2.putText(
                    canvas,
                    f"Capture mode: {capture_mode} (tecla 'm' para cambiar)",
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
                print(f"[*] Modo de captura: {capture_mode}")
            if key == ord("c"):
                semantic_worker.clear_conversation()
            if key == ord("n") and eval_session and not eval_session.finished:
                eval_session.skip_current(capture_mode)
                print(f"[*] Senia salteada. Siguiente: {eval_session.expected_sign}")

    stop()
    vs.stop()
    cv2.destroyAllWindows()

    if eval_session:
        print(
            f"[*] Evaluacion: {eval_session.index}/{len(sign_list)} registros "
            f"en {eval_session.csv_path}"
        )


if __name__ == "__main__":
    main()
