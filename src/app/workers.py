"""
Hilos de fondo: clasificador, traductor semántico y voz.

Los tres siguen el mismo patrón: una cola de entrada, un hilo daemon que la
consume, y escritura de resultados en `shared_state` bajo lock. Así el loop de
la cámara nunca se bloquea esperando a la GPU ni al sintetizador de voz.
"""

import time
from queue import Queue, Empty
from threading import Thread

import pyttsx3
import torch

import classifier.config as cfg
from app.state import shared_state, is_running
from classifier.arch import TinySkeletonClassifier
from core.conversation_memory import ConversationMemory
from core.repeat_policy import format_literal_utterance
from semantic.config import USE_CONVERSATION_HISTORY


class InferenceWorker:
    """Corre el clasificador de señas sobre los tensores que encola la cámara."""

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

            self.model.load_state_dict(
                torch.load(cfg.WEIGHTS_PATH, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            print(f"[*] Clasificador cargado en {self.device}")
        except Exception as e:
            print(f"[!] Error cargando el clasificador: {e}")
            print("[!] Si cambiaste la arquitectura, reentrená antes de usar la cámara.")

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
        while is_running():
            if self.model is None:
                time.sleep(1)
                continue

            input_tensor = None
            with shared_state["lock"]:
                if len(shared_state["inference_queue"]) > 0:
                    input_tensor = shared_state["inference_queue"].popleft()

            if input_tensor is None:
                time.sleep(0.01)
                continue

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

                print(" | ".join(f"{n.upper()} ({c:.1%})" for n, c in top3))
            except Exception as e:
                print(f"[!] Error de inferencia: {e}")


class VoiceWorker:
    """
    Síntesis de voz en su propio hilo.

    pyttsx3 bloquea hasta terminar de hablar; si eso corriera en el hilo
    semántico, la traducción siguiente esperaría a que termine el audio.
    El engine se crea dentro del hilo que lo usa (requisito de COM en Windows).
    """

    def __init__(self, rate: int = 100):
        self.rate = rate
        self.queue: Queue = Queue(maxsize=4)

    def start(self):
        Thread(target=self._loop, args=(), daemon=True).start()

    def _loop(self):
        while is_running():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if text is None:
                break
            self._speak(text)

    def _speak(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[!] Error en la síntesis de voz: {e}")

    def say(self, text):
        if not cfg.VOICE or not text:
            return
        try:
            self.queue.put_nowait(text)
        except Exception:
            print(f"[!] Cola de voz llena; audio descartado: {text}")


class SemanticWorker:
    """
    Cola asíncrona: list[str] de glosas → oración en español.

    La LLM se carga dentro del hilo. Si la carga falla, la cámara sigue
    funcionando y solo se muestran las glosas.
    """

    def __init__(self, enabled: bool = True, history_size: int = 10, voice=None):
        self.enabled = enabled
        self.queue: Queue = Queue(maxsize=8)
        self.ready = False
        self._translate_glosses = None
        self.memory = ConversationMemory(maxlen=history_size)
        self.voice = voice

    def start(self):
        if not self.enabled:
            print("[*] Traductor desactivado. Solo se muestran glosas.")
            return
        Thread(target=self._bootstrap_and_loop, args=(), daemon=True).start()

    def _bootstrap_and_loop(self):
        try:
            from semantic.translator import load_model_and_tokenizer, translate_glosses

            load_model_and_tokenizer()
            self._translate_glosses = translate_glosses
            self.ready = True
            print("[*] Traductor semántico listo.")
        except Exception as e:
            print(f"[!] No se pudo iniciar el traductor: {e}")
            print("[!] La cámara sigue; solo glosas, sin LLM.")
            print("[!] Tip: pip install -r requirements.txt")
            self._translate_glosses = None
            self.ready = False

        while is_running():
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
            print(f"[*] Enunciado cerrado → literal: {joined} => {literal}")
        else:
            print(f"[*] Enunciado cerrado → LLM: {joined}")

        with shared_state["lock"]:
            shared_state["semantic_busy"] = True
            shared_state["last_utterance"] = joined
            shared_state["spanish_text"] = literal if literal is not None else "Traduciendo..."

        # Deletreo o solo números: se muestra y se dice sin pasar por la LLM.
        text = literal if literal is not None else joined
        if literal is None and self._translate_glosses is not None:
            try:
                history = self.memory.as_messages() if USE_CONVERSATION_HISTORY else None
                text = self._translate_glosses(joined, history_messages=history) or joined
            except Exception as e:
                print(f"[!] Error de la LLM: {e}")
                text = joined

        # Los literales también entran al contexto: son parte de la conversación.
        self.memory.add_signer(text, glosses=joined)

        with shared_state["lock"]:
            shared_state["spanish_text"] = text
            shared_state["semantic_busy"] = False
            shared_state["utterance_glosses"] = []
            shared_state["conversation_turns"] = len(self.memory.turns)

        if self.voice is not None:
            self.voice.say(text)
        print(f"[*] Español: {text}")
        print(f"[*] Contexto: {len(self.memory.turns)}/{self.memory.maxlen} turnos")

    def add_hearing(self, text: str):
        """Listo para la UI/STT del oyente (todavía sin captura en la cámara)."""
        self.memory.add_hearing(text)
        with shared_state["lock"]:
            shared_state["conversation_turns"] = len(self.memory.turns)
        print(f"[*] Turno del oyente registrado: {text}")

    def clear_conversation(self):
        """Corta el contexto que ve la LLM; el log de la sesión se conserva."""
        self.memory.clear()
        with shared_state["lock"]:
            shared_state["conversation_turns"] = 0
        print(
            "[*] Contexto conversacional limpiado "
            f"(log de sesión: {len(self.memory.session_log)} turnos)."
        )

    def submit(self, glosses):
        if not glosses:
            return
        if not self.enabled:
            joined = " ".join(glosses)
            text = format_literal_utterance(glosses) or joined
            print(f"[*] Enunciado cerrado (sin LLM): {joined} => {text}")
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
            print(f"[!] Cola semántica llena; enunciado descartado: {' '.join(glosses)}")
