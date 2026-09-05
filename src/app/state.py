"""
Estado compartido entre el hilo de la cámara y los workers.

Un único diccionario protegido por `lock`. Los workers escriben resultados
(predicción, texto en español) y el loop de la cámara los lee para dibujar.
`running` es la señal de apagado: al ponerse en False todos los hilos salen.
"""

from collections import deque
from threading import Lock


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
    "semantic_model": "",
    "conversation_turns": 0,
    "lock": Lock(),
    "running": True,
}


def is_running() -> bool:
    return bool(shared_state["running"])


def stop() -> None:
    shared_state["running"] = False
