"""
Interpretation session: classifier + gloss buffer + translator.

The extension crops signs (Holistic + capture.js) and sends utterance end.
Landmarks are not extracted here: classify, accumulate, translate.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if getattr(sys, "frozen", False):
    _root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent / "_internal"))
    for _lib in (_root / "torch" / "lib", _root):
        if _lib.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_lib))
            except OSError:
                pass

import torch

import classifier.config as cfg
from core.utterance import UtteranceBuffer, normalize_gloss
from backend.landmarks_payload import LandmarkSmoother, frames_to_matrix, vector_from_frame
from classifier.arch import load_classifier_bundle
from core.repeat_policy import (
    collapse_literal_runs,
    format_literal_utterance,
    resolve_letter_lookalikes,
    simple_lexical_spanish,
)
from semantic.config import DEFAULT_MODEL_ID


class SignerSession:
    def __init__(
        self,
        enable_semantic: bool = True,
        semantic_model_id: str | None = None,
        left_handed: bool = False,
    ):
        # Translator ping runs on a thread: loading it here would mute /health.
        self.lock = threading.RLock()
        self._semantic_gate = threading.Lock()
        self.enable_semantic = enable_semantic
        self._wanted_model_id = semantic_model_id or DEFAULT_MODEL_ID
        self.left_handed = bool(left_handed)
        self.device = torch.device("cpu")
        print(
            "[backend] Classifier on CPU. Translator: remote server. "
            f"Hand: {'left' if self.left_handed else 'right'}."
        )
        self.model, self.idx_to_class = load_classifier_bundle(self.device)
        print(f"[backend] Classifier: {len(self.idx_to_class)} signs on {self.device}")
        self.buffer = UtteranceBuffer(
            pause_sec=cfg.UTTERANCE_PAUSE_SEC,
            min_confidence=cfg.CONFIDENCE_THRESHOLD,
            max_letter_consecutive=cfg.LETTER_MAX_CONSECUTIVE,
        )
        self.last_enqueue_time = 0.0
        self.last_inference_time = 0.0
        self.top3: list[tuple[str, float]] = []
        self.spanish_text = ""
        self.last_utterance = ""
        self.semantic_busy = False
        self.semantic_ready = False
        self.semantic_error = ""
        self._translate_glosses = None
        self._current_model_id = ""
        if enable_semantic:
            threading.Thread(target=self._bootstrap_semantic, daemon=True).start()

    def _bootstrap_semantic(self):
        try:
            self._ensure_remote()
            if not self.semantic_ready:
                raise RuntimeError(self.semantic_error or "El traductor Cloud Run no responde")
            print("[backend] Remote translator ready.")
        except Exception as e:
            with self.lock:
                self.semantic_error = str(e)
                self.semantic_ready = False
            print(f"[backend] Remote translator unavailable: {e}")

    def _ensure_remote(self) -> None:
        """Ping Cloud Run again if the boot thread has not left a client yet."""
        from semantic.remote import configured, ping, translate_sign

        with self.lock:
            if self._translate_glosses is not None:
                return
        if not self.enable_semantic:
            return
        if not configured():
            with self.lock:
                self.semantic_error = "Falta la URL del traductor semántico"
                self.semantic_ready = False
            return
        if not ping():
            with self.lock:
                self.semantic_error = "El traductor Cloud Run no responde"
                self.semantic_ready = False
            return
        with self._semantic_gate:
            with self.lock:
                self._translate_glosses = translate_sign
                self._current_model_id = "remote"
                self.semantic_ready = True
                self.semantic_error = ""

    def reset_session(self, left_handed: bool):
        with self.lock:
            self.left_handed = bool(left_handed)
            self.buffer = UtteranceBuffer(
                pause_sec=cfg.UTTERANCE_PAUSE_SEC,
                min_confidence=cfg.CONFIDENCE_THRESHOLD,
                max_letter_consecutive=cfg.LETTER_MAX_CONSECUTIVE,
            )
            self.top3 = []
            self.spanish_text = ""
            self.last_utterance = ""
            self.semantic_busy = False
            self.last_enqueue_time = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "left_handed": self.left_handed,
                "classifier_ready": self.model is not None,
                "num_classes": len(self.idx_to_class),
                "semantic_ready": self.semantic_ready,
                "semantic_error": self.semantic_error,
                "semantic_model": self._current_model_id,
                "semantic_busy": self.semantic_busy,
                "top3": [{"gloss": n, "confidence": c} for n, c in self.top3],
                "glosses": list(self.buffer.glosses),
                "pending_text": self.buffer.pending_text(),
                "spanish": self.spanish_text,
                "last_utterance": self.last_utterance,
                "device": str(self.device),
            }

    def note_activity(self):
        with self.lock:
            self.buffer.note_signing_activity(time.time())

    def ingest_sign(self, frames: list[dict], client: dict | None = None) -> dict[str, Any]:
        # The sign is already cropped. Here: EMA → (16, 225) → TinySkeleton.
        t0 = time.perf_counter()
        now = time.time()
        client = client or {}
        n = len(frames or [])
        with_pose = sum(1 for f in frames or [] if f.get("pose"))
        with_lh = sum(1 for f in frames or [] if f.get("left_hand"))
        with_rh = sum(1 for f in frames or [] if f.get("right_hand"))
        print(
            f"[backend] /sign received: {n} frames "
            f"(pose={with_pose} left_hand={with_lh} right_hand={with_rh})"
        )

        with self.lock:
            cool = cfg.INFERENCE_COOLDOWN_SEC
            if now - self.last_enqueue_time < cool or now - self.last_inference_time < cool:
                print("[backend] /sign rejected: cooldown")
                from backend.metrics import log_event

                log_event(
                    "sign",
                    accepted=False,
                    reason="cooldown",
                    n_frames=n,
                    capture_ms=client.get("capture_ms"),
                    holistic_ms=client.get("holistic_ms"),
                )
                return {"accepted": False, "reason": "cooldown", **self.snapshot()}
            if n < cfg.MIN_CAPTURE_FRAMES:
                print(
                    f"[backend] /sign rejected: too_short "
                    f"({n} < min {cfg.MIN_CAPTURE_FRAMES})"
                )
                from backend.metrics import log_event

                log_event(
                    "sign",
                    accepted=False,
                    reason="too_short",
                    n_frames=n,
                    capture_ms=client.get("capture_ms"),
                    holistic_ms=client.get("holistic_ms"),
                )
                return {"accepted": False, "reason": "too_short", **self.snapshot()}

        t_prep = time.perf_counter()
        smoother = LandmarkSmoother(alpha=0.6)
        vectors = []
        for frame in frames:
            vec = vector_from_frame(frame, left_handed=self.left_handed)
            vectors.append(smoother.update(vec))

        matrix = frames_to_matrix(vectors)
        prepare_ms = (time.perf_counter() - t_prep) * 1000
        print(
            f"[backend] /sign tensor: entrada={n} "
            f"tras muestreo={matrix.shape} (esperado {cfg.MAX_FRAMES}x{matrix.shape[1] if matrix.ndim == 2 else '?'})"
        )
        if matrix.shape[0] != cfg.MAX_FRAMES:
            print("[backend] /sign rejected: bad_tensor")
            from backend.metrics import log_event

            log_event("sign", accepted=False, reason="bad_tensor", n_frames=n)
            return {"accepted": False, "reason": "bad_tensor", **self.snapshot()}
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_inf = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probs, k=min(3, probs.shape[0]))
            top3 = []
            for conf, idx in zip(values.tolist(), indices.tolist()):
                name = self.idx_to_class.get(idx, "desconocido")
                top3.append((name, float(conf)))
        infer_ms = (time.perf_counter() - t_inf) * 1000

        added = False
        activity = False
        gloss = normalize_gloss(top3[0][0]) if top3 else ""
        with self.lock:
            self.last_enqueue_time = now
            self.last_inference_time = time.time()
            self.top3 = top3
            if top3:
                prev_act = self.buffer.last_activity_at
                added = self.buffer.try_add(top3[0][0], top3[0][1], self.last_inference_time)
                activity = self.buffer.last_activity_at != prev_act

        print(
            "[backend] /sign top3: "
            + " | ".join(f"{n.upper()} ({c:.1%})" for n, c in top3)
            + (f" → added {gloss}" if added else " → not added (repeat/threshold)")
        )
        snap = self.snapshot()
        total_ms = (time.perf_counter() - t0) * 1000
        conf1 = float(top3[0][1]) if top3 else None
        conf2 = float(top3[1][1]) if len(top3) > 1 else None
        conf3 = float(top3[2][1]) if len(top3) > 2 else None
        conf_margin = round(conf1 - conf2, 4) if conf1 is not None and conf2 is not None else None
        reject_reason = ""
        if not added:
            if conf1 is None or conf1 < cfg.CONFIDENCE_THRESHOLD:
                reject_reason = "confidence"
            elif gloss == "DESCONOCIDO":
                reject_reason = "confidence"
            else:
                reject_reason = "repeat"
        snap.update({
            "accepted": True,
            "added": added,
            "activity": activity,
            "gloss": gloss or (top3[0][0] if top3 else None),
            "confidence": conf1,
            "reject_reason": reject_reason,
            "ms": round(total_ms, 1),
        })
        from backend.metrics import log_event
        from utils.benchmark_logger import record_sign_row

        sign_fields = {
            "accepted": True,
            "added": added,
            "reason": reject_reason or None,
            "n_frames": n,
            "gloss": top3[0][0] if top3 else "",
            "conf1": conf1,
            "conf2": conf2,
            "conf3": conf3,
            "conf_margin": conf_margin,
            "classifier_ms": round(infer_ms, 1),
            "prepare_ms": round(prepare_ms, 1),
            "total_ms": round(total_ms, 1),
            "capture_ms": client.get("capture_ms"),
            "holistic_ms": client.get("holistic_ms"),
            "jpeg_ms": client.get("jpeg_ms"),
            "time_to_record_ms": client.get("time_to_record_ms"),
            "n_raw": client.get("n_raw"),
            "n_sent": client.get("n_sent"),
            "payload_kb": client.get("payload_kb"),
            "left_handed": self.left_handed,
        }
        log_event("sign", **sign_fields)
        record_sign_row(sign_fields)
        print(
            f"[backend] /sign response: accepted=True added={added} "
            f"gloss={snap.get('gloss')} pending={snap.get('pending_text')!r}"
        )
        return snap

    def close_utterance(self, pause_ms: float | None = None) -> dict[str, Any]:
        with self.lock:
            closed = list(self.buffer.glosses)
            self.buffer.glosses.clear()
            self.buffer.repeat_gate.reset()
            self.buffer.last_activity_at = None
            if not closed:
                print("[backend] /utterance/end: no pending glosses")
                snap = self.snapshot()
                snap.update({"closed": False, "glosses": [], "spanish": self.spanish_text})
                return snap
            self.semantic_busy = True
            self.last_utterance = " ".join(closed)

        print(f"[backend] /utterance/end received: {closed!r}")
        t0 = time.perf_counter()
        text, via, remote_ms, server_ms = self._translate(closed)
        total_ms = (time.perf_counter() - t0) * 1000

        with self.lock:
            self.spanish_text = text
            self.semantic_busy = False

        timing = {}
        if via == "llm":
            from semantic.remote import last_timing

            timing = last_timing()
        pause = float(pause_ms) if pause_ms is not None else round(cfg.UTTERANCE_PAUSE_SEC * 1000, 1)
        queue_ms = timing.get("queue_ms") if timing.get("queue_ms") is not None else timing.get("queued_ms")
        infer_ms = timing.get("inference_ms") if timing.get("inference_ms") is not None else (
            timing.get("infer_ms") if timing.get("infer_ms") is not None else (
                round(server_ms, 1) if server_ms is not None else None
            )
        )
        transport = timing.get("transport_ms")
        if transport is None and via == "llm":
            from utils.benchmark_logger import transport_ms as net_ms

            transport = net_ms(timing.get("roundtrip_ms") or remote_ms, queue_ms, infer_ms)

        print(
            f"[metrics] Tiempo de Pausa (Heurística de Cierre): {pause:.1f} ms "
            f"— NOT model time (t_pausa_inicio → t_pausa_fin at "
            f"{cfg.UTTERANCE_PAUSE_SEC:.1f}s idle; excluded from RNF-01)",
            flush=True,
        )

        snap = self.snapshot()
        snap.update({
            "closed": True,
            "glosses": closed,
            "spanish": text,
            "via": via,
            "ms": round(total_ms, 1),
            "pause_ms": round(pause, 1),
            "queue_ms": queue_ms,
            "inference_ms": infer_ms,
            "transport_ms": transport,
            "delivery_ms": 0.0,
        })
        from backend.metrics import log_event, record_pipeline_turn, turn_extra

        extra = turn_extra(
            timing,
            via=via,
            n_glosses=len(closed),
            input_chars=len(" ".join(closed)),
            output_chars=len(text or ""),
        )
        log_event(
            "utterance",
            glosses=" ".join(closed),
            spanish=text,
            via=via,
            pause_ms=round(pause, 1),
            queued_ms=queue_ms,
            queue_ms=queue_ms,
            infer_ms=infer_ms,
            inference_ms=infer_ms,
            transport_ms=transport,
            delivery_ms=0.0,
            remote_ms=round(remote_ms, 1),
            server_ms=round(server_ms, 1) if server_ms is not None else None,
            total_ms=round(total_ms, 1),
            n_glosses=len(closed),
            prompt_tokens=timing.get("prompt_tokens"),
            completion_tokens=timing.get("completion_tokens"),
            eval_tokens=timing.get("eval_tokens"),
            sys_tokens=timing.get("sys_tokens"),
            kv_hit=timing.get("kv_hit"),
            kv_mb=timing.get("kv_mb"),
            t_prompt_eval_ms=timing.get("t_prompt_eval_ms"),
            t_eval_ms=timing.get("t_eval_ms"),
            tokens_per_s=timing.get("tokens_per_s"),
            model_used=timing.get("model_used"),
            n_ctx=timing.get("n_ctx"),
            n_tokens=timing.get("n_tokens"),
            queue_depth=timing.get("queue_depth"),
            n_threads=timing.get("n_threads"),
            cloud_rss_mb=timing.get("cloud_rss_mb"),
            cloud_cpu_percent=timing.get("cloud_cpu_percent"),
        )
        record_pipeline_turn(
            "direct",
            pause_ms=pause,
            queue_ms=queue_ms,
            inference_ms=infer_ms,
            transport_ms=transport,
            delivery_ms=0,
            extra=extra,
        )
        srv = f" server={server_ms:.0f}ms" if server_ms is not None else ""
        print(
            f"[backend] /utterance/end response: spanish={text!r} via={via} "
            f"{total_ms:.0f}ms{srv}"
        )
        return snap

    def _translate(self, glosses: list[str]) -> tuple[str, str, float, float | None]:
        # A lone gloss or a fingerspell/digits run skips Cloud Run. A sentence does not.
        fixed = resolve_letter_lookalikes(glosses)
        if fixed != list(glosses):
            print(f"[backend] Letter lookalikes: {glosses!r} => {fixed!r}")
            glosses = fixed
        joined = " ".join(glosses)
        literal = format_literal_utterance(glosses)
        if literal is not None:
            print(f"[backend] Utterance → literal (no semantic): {joined} => {literal}")
            return literal, "literal", 0.0, None
        simple = simple_lexical_spanish(glosses)
        if simple is not None:
            print(f"[backend] Utterance → single gloss (no semantic): {joined} => {simple}")
            return simple, "lexical", 0.0, None
        collapsed = " ".join(collapse_literal_runs(glosses))
        print(f"[backend] Utterance → Cloud Run POST /sign: {collapsed}")
        self._ensure_remote()
        with self._semantic_gate:
            with self.lock:
                fn = self._translate_glosses
        if fn is None:
            print("[backend] Semantic not ready; no sentence.")
            return collapsed, "fallback", 0.0, None
        try:
            from semantic.remote import last_timing

            t0 = time.perf_counter()
            text = fn(collapsed)
            remote_ms = (time.perf_counter() - t0) * 1000
            timing = last_timing()
            if not text:
                print("[backend] Cloud Run /sign returned empty")
                return collapsed, "fallback", remote_ms, timing.get("server_ms")
            return text, "llm", remote_ms, timing.get("server_ms")
        except Exception as e:
            print(f"[backend] Error Cloud Run /sign: {e}")
            with self.lock:
                self.semantic_error = str(e)
                self.semantic_ready = False
            return collapsed, "fallback", 0.0, None

    def capture_config(self) -> dict[str, Any]:
        return {
            "max_frames": cfg.MAX_FRAMES,
            "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
            "inference_cooldown_sec": cfg.INFERENCE_COOLDOWN_SEC,
            "motion_pixel_threshold": cfg.MOTION_PIXEL_THRESHOLD,
            "landmark_motion_threshold": cfg.LANDMARK_MOTION_THRESHOLD,
            "static_hands_frames_to_start": cfg.STATIC_HANDS_FRAMES_TO_START,
            "hands_frames_to_start": cfg.HANDS_FRAMES_TO_START,
            "still_frames_limit": cfg.STILL_FRAMES_LIMIT,
            "capture_buffer_size": cfg.CAPTURE_BUFFER_SIZE,
            "missing_hands_limit": cfg.MISSING_HANDS_LIMIT,
            "min_capture_frames": cfg.MIN_CAPTURE_FRAMES,
            "capture_mode": cfg.CAPTURE_MODE,
            "num_classes": len(self.idx_to_class),
            "utterance_pause_sec": cfg.UTTERANCE_PAUSE_SEC,
            "pose_dim": cfg.POSE_DIM,
            "mode": "signer",
            "left_handed": self.left_handed,
        }
