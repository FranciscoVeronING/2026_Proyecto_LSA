"""Memoria conversacional para interpretación LSA ↔ persona oyente."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Deque, List, Optional, Sequence


ROLE_SIGNER = "signer"
ROLE_HEARING = "hearing"


@dataclass
class Turn:
    """Un turno de la conversación."""

    role: str  # signer | hearing
    text: str
    glosses: Optional[str] = None
    ts: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Ventana deslizante de turnos para contexto de la LLM + log de sesión.

    - window: últimos `maxlen` turnos (lo que ve el modelo)
    - session: historial completo de la sesión (auditoría / futuro UI)
    - signer: interpretación LSA → español (glosas opcionales)
    - hearing: texto o voz del oyente (STT/UI pendiente)

    Thread-safe: el hilo semántico escribe mientras la UI puede limpiar.
    """

    def __init__(self, maxlen: int = 10):
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self.maxlen = maxlen
        self._lock = RLock()
        self._window: Deque[Turn] = deque(maxlen=maxlen)
        self._session: List[Turn] = []

    def _append(self, turn: Turn) -> None:
        with self._lock:
            self._window.append(turn)
            self._session.append(turn)

    def add_signer(self, text: str, glosses: Optional[str] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        glosses_clean = (glosses or "").strip() or None
        self._append(Turn(role=ROLE_SIGNER, text=text, glosses=glosses_clean))

    def add_hearing(self, text: str) -> None:
        """Reservado para captura por texto o STT del oyente."""
        text = (text or "").strip()
        if not text:
            return
        self._append(Turn(role=ROLE_HEARING, text=text))

    def clear(self) -> None:
        """Corta el contexto que ve la LLM; conserva el log de la sesión."""
        with self._lock:
            self._window.clear()

    def reset_session(self) -> None:
        """Descarta también el log completo (nueva sesión de trabajo)."""
        with self._lock:
            self._window.clear()
            self._session.clear()

    @property
    def turns(self) -> Sequence[Turn]:
        with self._lock:
            return tuple(self._window)

    @property
    def session_log(self) -> Sequence[Turn]:
        with self._lock:
            return tuple(self._session)

    def as_messages(self) -> List[dict]:
        """
        Historial en formato chat (user/assistant) para apply_chat_template.

        - signer: user=Glosas / assistant=español
        - hearing: user=Persona oyente / assistant=ack corto (mantiene alternancia)
        """
        messages: List[dict] = []
        for turn in self.turns:
            if turn.role == ROLE_SIGNER:
                user_content = (
                    f"Glosas: {turn.glosses}" if turn.glosses else f"LSA: {turn.text}"
                )
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": turn.text})
            else:
                messages.append(
                    {"role": "user", "content": f"Persona oyente: {turn.text}"}
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "(mensaje del oyente registrado como contexto)",
                    }
                )
        return messages
