"""stdout/logging → cola del panel de registros."""

from __future__ import annotations

import logging
import queue
import sys


class QueueWriter:

    encoding = "utf-8"
    errors = "replace"
    closed = False

    def __init__(self, q: queue.Queue):
        self.q = q
        self._orig = sys.__stdout__

    def write(self, msg: str) -> int:
        if not msg:
            return 0
        self.q.put(msg)
        if self._orig:
            try:
                self._orig.write(msg)
            except Exception:
                pass
        return len(msg)

    def flush(self) -> None:
        if self._orig:
            try:
                self._orig.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self._orig and hasattr(self._orig, "fileno"):
            return self._orig.fileno()
        raise OSError("no fileno")


class QueueLogHandler(logging.Handler):

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record) + "\n")
        except Exception:
            pass
