"""Servicio semántico compartido (una instancia LLM, cola por sala)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Optional

from core.conversation_memory import ConversationMemory
from core.repeat_policy import format_literal_utterance
from semantic.config import CONVERSATION_HISTORY_SIZE, USE_CONVERSATION_HISTORY


@dataclass
class SemanticJob:
    room_id: str
    participant_id: str
    glosses: list[str]


class SharedSemanticService:
    """Traduce glosas a español; una LLM, cola global, memoria por sala."""

    def __init__(
        self,
        enabled: bool = True,
        on_result: Optional[Callable[[str, str, str, str], None]] = None,
    ):
        self.enabled = enabled
        self.ready = False
        self._translate_glosses = None
        self._queue: Queue[SemanticJob] = Queue(maxsize=32)
        self._running = True
        self._on_result = on_result
        self._memories: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._bootstrap_and_loop, daemon=True)
        self._thread.start()

    def get_memory(self, room_id: str) -> ConversationMemory:
        with self._lock:
            if room_id not in self._memories:
                self._memories[room_id] = ConversationMemory(maxlen=CONVERSATION_HISTORY_SIZE)
            return self._memories[room_id]

    def add_hearing(self, room_id: str, text: str) -> None:
        memory = self.get_memory(room_id)
        memory.add_hearing(text)

    def clear_context(self, room_id: str) -> None:
        memory = self.get_memory(room_id)
        memory.clear()

    def submit(self, room_id: str, participant_id: str, glosses: list[str]) -> None:
        if not glosses:
            return
        if not self.enabled:
            joined = " ".join(glosses)
            text = format_literal_utterance(glosses) or joined
            memory = self.get_memory(room_id)
            memory.add_signer(text, glosses=joined)
            if self._on_result:
                self._on_result(room_id, participant_id, joined, text)
            return
        try:
            self._queue.put_nowait(
                SemanticJob(room_id=room_id, participant_id=participant_id, glosses=list(glosses))
            )
        except Exception:
            print(f"[!] Cola semántica llena: {' '.join(glosses)}")

    def _bootstrap_and_loop(self) -> None:
        if self.enabled:
            try:
                from semantic.translator import load_model_and_tokenizer, translate_glosses

                load_model_and_tokenizer()
                self._translate_glosses = translate_glosses
                self.ready = True
                print("[*] Traductor semántico web listo.")
            except Exception as e:
                print(f"[!] No se pudo iniciar traductor web: {e}")
                self._translate_glosses = None

        while self._running:
            try:
                job = self._queue.get(timeout=0.2)
            except Empty:
                continue
            self._handle(job)

    def _handle(self, job: SemanticJob) -> None:
        joined = " ".join(job.glosses)
        literal = format_literal_utterance(job.glosses)
        memory = self.get_memory(job.room_id)

        text = literal if literal is not None else joined
        if literal is None and self._translate_glosses is not None:
            try:
                history = memory.as_messages() if USE_CONVERSATION_HISTORY else None
                text = self._translate_glosses(joined, history_messages=history) or joined
            except Exception as e:
                print(f"[!] Error LLM web: {e}")
                text = joined

        memory.add_signer(text, glosses=joined)
        if self._on_result:
            self._on_result(job.room_id, job.participant_id, joined, text)

    def stop(self) -> None:
        self._running = False
