"""Servicio de inferencia compartido (una instancia del clasificador)."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Optional

import torch

import classifier.config as cfg
from classifier.arch import TinySkeletonClassifier


@dataclass
class InferenceJob:
    room_id: str
    participant_id: str
    tensor: torch.Tensor


class SharedInferenceService:
    """Cola global de inferencia con un solo modelo TinySkeleton."""

    def __init__(self, idx_to_class: dict, num_classes: int, device: torch.device):
        self.idx_to_class = idx_to_class
        self.device = device
        self.model: Optional[TinySkeletonClassifier] = None
        self.ready = False
        self._queue: Queue[InferenceJob] = Queue(maxsize=64)
        self._callback: Optional[Callable[[str, str, list], None]] = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

        try:
            self.model = TinySkeletonClassifier(
                cfg.FRAME_FEATURES_DIM,
                cfg.HIDDEN_DIM,
                num_heads=cfg.NUM_HEADS,
                num_layers=cfg.NUM_LAYERS,
                num_classes=num_classes,
                dropout_rate=cfg.DROPOUT_RATE,
            ).to(device)
            self.model.load_state_dict(
                torch.load(cfg.WEIGHTS_PATH, map_location=device, weights_only=True)
            )
            self.model.eval()
            self.ready = True
            print(f"[*] Clasificador web cargado en {device}")
        except Exception as e:
            print(f"[!] Error cargando clasificador web: {e}")

        self._thread.start()

    def set_callback(self, callback: Callable[[str, str, list], None]) -> None:
        self._callback = callback

    def enqueue(self, room_id: str, participant_id: str, tensor: torch.Tensor) -> None:
        try:
            self._queue.put_nowait(
                InferenceJob(room_id=room_id, participant_id=participant_id, tensor=tensor)
            )
        except Exception:
            print(f"[!] Cola de inferencia llena; descartado {participant_id}")

    def _decode_top3(self, probs) -> list[tuple[str, float]]:
        values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
        results = []
        for conf, idx in zip(values.tolist(), indices.tolist()):
            name = self.idx_to_class.get(idx, "desconocido")
            results.append((name, float(conf)))
        return results

    def _loop(self) -> None:
        while self._running:
            if self.model is None:
                time.sleep(1)
                continue
            try:
                job = self._queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                with torch.no_grad():
                    logits = self.model(job.tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    top3 = self._decode_top3(probs)

                cb = self._callback
                if cb:
                    cb(job.room_id, job.participant_id, top3)
            except Exception as e:
                print(f"[!] Error inferencia web: {e}")

    def stop(self) -> None:
        self._running = False
