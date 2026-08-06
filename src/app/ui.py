"""Widgets y dibujo sobre el canvas de OpenCV."""

import cv2
import numpy as np


UI_FONT = cv2.FONT_HERSHEY_SIMPLEX

mouse_state = {"x": 0, "y": 0, "down": False, "clicked": False}


def mouse_callback(event, x, y, flags, param):
    mouse_state["x"], mouse_state["y"] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["down"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["down"] = False
        mouse_state["clicked"] = True


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
        hover = (self.x <= mouse_x <= self.x + self.w) and (
            self.y - 5 <= mouse_y <= self.y + self.h + 5
        )
        if hover and is_m_down:
            self.dragging = True
        if not is_m_down:
            self.dragging = False
        if self.dragging:
            ratio = max(0, min(mouse_x - self.x, self.w)) / self.w
            self.val = self.min_val + (self.max_val - self.min_val) * ratio

    def draw(self, canvas):
        display_val = f"{int(self.val)}" if self.max_val > 1 else f"{self.val:.2f}"
        cv2.putText(
            canvas,
            f"{self.label}: {display_val}",
            (self.x, self.y - 10),
            UI_FONT,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (40, 40, 40), -1)
        fill_w = int(self.w * (self.val - self.min_val) / (self.max_val - self.min_val))
        cv2.rectangle(canvas, (self.x, self.y), (self.x + fill_w, self.y + self.h), (0, 165, 255), -1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)


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


def select_handedness_modal():
    """
    Modal inicial: elegir mano dominante antes de abrir la cámara.
    Devuelve 'right', 'left' o None si se cancela con ESC.
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
    print(f"[*] Mano dominante: {choice['value']}")
    return choice["value"]
