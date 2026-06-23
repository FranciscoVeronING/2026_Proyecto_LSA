import os
import cv2
import numpy as np
import mediapipe as mp
import time
import json
import torch
import torch.nn as nn
from threading import Thread, Lock
from collections import deque

import config as cfg
from utils import get_anchor_and_scale, normalize_spatial_points, uniform_subsampling

# =============================================================================
# ARQUITECTURA DEL MODELO (Necesaria para cargar los pesos)
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) 
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TinySkeletonClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_classes: int, dropout_rate: float):
        super().__init__()
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout1d(p=0.2), 
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_dropout = nn.Dropout(p=dropout_rate)
        self.classification_head = nn.Linear(hidden_dim, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        x = self.conv_extractor(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x_pooled = x.mean(dim=1)
        x_dropped = self.classifier_dropout(x_pooled)
        return self.classification_head(x_dropped)


# =============================================================================
# UTILIDADES
# =============================================================================
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
    """Convierte la lista temporal a un tensor compatible con PyTorch (1, 16, 225)"""
    if len(buffer_list) < 5: return None 
    
    sampled_matrix = uniform_subsampling(buffer_list, target_frames=cfg.MAX_FRAMES)
    
    tensor = torch.tensor(sampled_matrix, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)

# =============================================================================
# UI COMPONENTS
# =============================================================================
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX

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
            if self.callback: self.callback()
            return True
        return False

    def draw(self, canvas, active=False):
        x, y, w, h = self.rect
        bg_color = (0, 200, 100) if active else ((80, 80, 80) if self.is_hover else (50, 50, 50))
        cv2.rectangle(canvas, (x, y), (x+w, y+h), bg_color, -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (200, 200, 200), 1)
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
        if hover and is_m_down: self.dragging = True
        if not is_m_down: self.dragging = False
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

# =============================================================================
# BACKEND E INFERENCIA (PyTorch)
# =============================================================================
shared_state = {
    "inference_queue": deque(maxlen=5), 
    "prediction": "...",
    "confidence": 0.0,
    "lock": Lock(),
    "running": True
}

class InferenceWorker:
    def __init__(self, idx_to_class, num_classes, device):
        self.idx_to_class = idx_to_class
        self.device = device
        
        try:
            self.model = TinySkeletonClassifier(
                cfg.FRAME_FEATURES_DIM, cfg.HIDDEN_DIM, 
                num_heads=cfg.NUM_HEADS, num_layers=cfg.NUM_LAYERS, 
                num_classes=num_classes, dropout_rate=cfg.DROPOUT_RATE
            ).to(self.device)
            
            ruta_modelo = os.path.join(cfg.MODEL_SAVE_DIR, "tinyskeleton_best.pth")
            self.model.load_state_dict(torch.load(ruta_modelo, map_location=self.device, weights_only=True))
            self.model.eval() 
            print("[*] Modelo PyTorch cargado. Worker listo en", self.device)
        except Exception as e:
            print(f"[!] Error cargando modelo: {e}")
            self.model = None

    def start(self): Thread(target=self.loop, args=(), daemon=True).start()

    def loop(self):
        while shared_state["running"]:
            if self.model is None: time.sleep(1); continue
            
            input_tensor = None
            with shared_state["lock"]:
                if len(shared_state["inference_queue"]) > 0:
                    input_tensor = shared_state["inference_queue"].popleft()

            if input_tensor is not None:
                try:
                    with torch.no_grad():
                        # Inferencia
                        logits = self.model(input_tensor)
                        # Convertimos logits a probabilidades (0 a 1)
                        probs = torch.softmax(logits, dim=1)[0]
                        
                        best_idx = torch.argmax(probs).item()
                        conf = probs[best_idx].item()
                        
                        nombre_seña = self.idx_to_class.get(best_idx, "Desconocido")

                    if conf > cfg.CONFIDENCE_THRESHOLD:
                        print(f"🧠 {nombre_seña.upper()} ({conf:.1%})")
                        shared_state["prediction"] = nombre_seña.upper()
                        shared_state["confidence"] = conf
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
            print("ERROR CRÍTICO: No se puede abrir la cámara.")
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
            (grabbed, frame) = self.stream.read()
            if not grabbed: self.stop()
            else: self.frame = frame

    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release()

# =============================================================================
# MAIN
# =============================================================================
mouse_state = {"x": 0, "y": 0, "down": False, "clicked": False}
def mouse_callback(event, x, y, flags, param):
    mouse_state["x"], mouse_state["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN: mouse_state["down"] = True
    elif event == cv2.EVENT_LBUTTONUP: mouse_state["down"] = False; mouse_state["clicked"] = True


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Carga Dinámica de Diccionario de Clases ---
    ruta_mapeo = os.path.join(cfg.MODEL_SAVE_DIR, "mapeo_clases.json")
    if not os.path.exists(ruta_mapeo):
        print(f"[!] No se encontró {ruta_mapeo}. Ejecuta el entrenamiento primero.")
        return
        
    with open(ruta_mapeo, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
        
    # Invertimos el diccionario para buscar por índice en tiempo real
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    print("\n[*] Iniciando camara...")
    vs = WebcamStream(0).start()
    time.sleep(2.0)
    if vs.stopped: return

    worker = InferenceWorker(idx_to_class, num_classes, device)
    worker.start()
    
    cv2.namedWindow("LSA DETECTOR")
    cv2.setMouseCallback("LSA DETECTOR", mouse_callback)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    VID_W, VID_H = 640, 480
    TOT_H = VID_H + 150
    
    btn_view = Button(520, VID_H + 20, 100, 40, "Esqueleto")
    btn_conf = Button(520, VID_H + 80, 100, 40, "Config")
    
    slider_sens = Slider(150, 150, 340, 100, 5000, cfg.MOTION_PIXEL_THRESHOLD, "Sensibilidad (Pixeles)")
    slider_conf = Slider(150, 200, 340, 0.1, 1.0, cfg.CONFIDENCE_THRESHOLD, "Confianza Min")
    slider_still = Slider(150, 250, 340, 10, 60, cfg.STILL_FRAMES_LIMIT, "Corte por Silencio (Frames)") 
    
    btn_save = Button(220, 350, 100, 40, "CERRAR")

    show_config = False
    show_landmarks = True
    
    smoother = LandmarkSmoother(alpha=0.6)
    
    frames_temp_buffer = [] 
    prev_gray = None
    motion_val = 0
    consecutive_still_frames = 0 
    missing_hands_frames = 0 # Nuevo contador de ceguera

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            frame = vs.read()
            if frame is None: break
            
            # Detección de Movimiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            is_moving = False
            
            if prev_gray is not None:
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_val = cv2.countNonZero(thresh)
                if motion_val > cfg.MOTION_PIXEL_THRESHOLD:
                    is_moving = True
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

            # --- MÁQUINA DE ESTADOS INTELIGENTE PARA MICROPALSAS ---
            
            # 1. Condición de Disparo: Si no estamos grabando, iniciamos solo con movimiento activo
            if not is_recording and hands_present and is_moving:
                is_recording = True

            # 2. Si el estado es "Grabando", gestionamos la secuencia temporal
           # 2. Si el estado es "Grabando", gestionamos la secuencia temporal
            if is_recording:
                if hands_present:
                    missing_hands_frames = 0 # Reiniciamos la tolerancia de ceguera
                    
                    anchor, scale = get_anchor_and_scale(results.pose_landmarks)
                    raw_pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
                    raw_lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
                    raw_rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

                    norm_pose = normalize_spatial_points(raw_pose, anchor, scale)
                    norm_lh = normalize_spatial_points(raw_lh, anchor, scale)
                    norm_rh = normalize_spatial_points(raw_rh, anchor, scale)

                    frame_vector = np.concatenate([norm_pose, norm_lh, norm_rh])
                    smooth_vector = smoother.update(frame_vector)
                    frames_temp_buffer.append(smooth_vector)

                    if is_moving:
                        consecutive_still_frames = 0
                        cv2.circle(image, (30, 30), 10, (0, 255, 0), -1)  # Verde
                    else:
                        consecutive_still_frames += 1
                        cv2.circle(image, (30, 30), 10, (0, 255, 255), -1)  # Amarillo

                    if len(frames_temp_buffer) >= cfg.CAPTURE_BUFFER_SIZE or consecutive_still_frames >= cfg.STILL_FRAMES_LIMIT:
                        tensor = prepare_input_tensor(frames_temp_buffer, device)
                        if tensor is not None:
                            with shared_state["lock"]: shared_state["inference_queue"].append(tensor)
                        frames_temp_buffer = []
                        consecutive_still_frames = 0
                        smoother.reset()

                else:
                    # 3. MODO SUPERVIVENCIA: Las manos desaparecieron de repente
                    missing_hands_frames += 1
                    cv2.circle(image, (30, 30), 10, (0, 0, 255), -1) # Rojo: ¡Alerta de pérdida de rastreo!

                    if len(frames_temp_buffer) > 0:
                        # Rellenamos el tiempo perdido duplicando la última posición conocida
                        # Así evitamos un "teletransporte" en el tiempo cuando las manos vuelvan
                        frames_temp_buffer.append(frames_temp_buffer[-1])

                    # Solo cortamos y analizamos si se agota la paciencia
                    if missing_hands_frames >= cfg.MISSING_HANDS_LIMIT:
                        tensor = prepare_input_tensor(frames_temp_buffer, device)
                        if tensor is not None:
                            print(f"👋 Corte definitivo por retiro de manos ({len(frames_temp_buffer)} frames)")
                            with shared_state["lock"]: shared_state["inference_queue"].append(tensor)
                        
                        frames_temp_buffer = []
                        consecutive_still_frames = 0
                        missing_hands_frames = 0
                        smoother.reset()

            # --- 4. Render UI ---
            canvas = np.zeros((TOT_H, VID_W, 3), dtype="uint8")
            canvas[0:VID_H, 0:VID_W] = image
            cv2.rectangle(canvas, (0, VID_H), (VID_W, TOT_H), (30, 30, 30), -1)
            
            if not show_config:
                p_txt = shared_state["prediction"]
                c_val = shared_state["confidence"]
                
                buf_len = len(frames_temp_buffer)
                prog = min(buf_len / cfg.CAPTURE_BUFFER_SIZE, 1.0)
                col_prog = (0, 255, 0) if prog >= 1.0 else (0, 255, 255)
                if consecutive_still_frames > 0: col_prog = (0, 165, 255) 
                
                cv2.rectangle(canvas, (20, VID_H+20), (20 + int(200*prog), VID_H+30), col_prog, -1)
                cv2.putText(canvas, f"Buffer: {buf_len} | Silencio: {consecutive_still_frames}/{cfg.STILL_FRAMES_LIMIT}", (230, VID_H+28), UI_FONT, 0.5, (150,150,150), 1)

                mot_ratio = min(motion_val / (cfg.MOTION_PIXEL_THRESHOLD * 2), 1.0)
                mot_col = (0, 0, 255) if is_moving else (100, 100, 100)
                cv2.rectangle(canvas, (20, VID_H+45), (20 + int(100*mot_ratio), VID_H+50), mot_col, -1)
                cv2.putText(canvas, f"Mov: {motion_val} px", (130, VID_H+48), UI_FONT, 0.4, (200,200,200), 1)

                if p_txt != "...":
                    cv2.putText(canvas, p_txt, (20, VID_H + 90), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(canvas, f"Confianza: {c_val:.1%}", (20, VID_H + 120), UI_FONT, 0.6, (180, 180, 180), 1)
                else:
                    cv2.putText(canvas, "Esperando...", (20, VID_H + 90), UI_FONT, 1, (100, 100, 100), 2)
                
                cv2.putText(canvas, "Presione 'q' para salir", (VID_W - 200, TOT_H - 10), UI_FONT, 0.5, (100, 100, 100), 1)

                if btn_view.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    show_landmarks = not show_landmarks
                btn_view.draw(canvas, active=show_landmarks)

                if btn_conf.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    show_config = True
                btn_conf.draw(canvas)
            else:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (0,0), (VID_W, TOT_H), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
                mx, my, mw, mh = 100, 100, 440, 300
                cv2.rectangle(canvas, (mx, my), (mx+mw, my+mh), (50,50,50), -1)
                cv2.rectangle(canvas, (mx, my), (mx+mw, my+mh), (0, 165, 255), 2)
                
                slider_sens.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_sens.draw(canvas)
                
                slider_conf.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_conf.draw(canvas)
                
                slider_still.update(mouse_state["x"], mouse_state["y"], mouse_state["down"])
                slider_still.draw(canvas)
                
                if btn_save.update(mouse_state["x"], mouse_state["y"], mouse_state["clicked"]):
                    cfg.MOTION_PIXEL_THRESHOLD = int(slider_sens.val)
                    cfg.CONFIDENCE_THRESHOLD = slider_conf.val
                    cfg.STILL_FRAMES_LIMIT = int(slider_still.val)
                    show_config = False
                btn_save.draw(canvas)

            cv2.imshow("LSA DETECTOR", canvas)
            mouse_state["clicked"] = False
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    shared_state["running"] = False
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()