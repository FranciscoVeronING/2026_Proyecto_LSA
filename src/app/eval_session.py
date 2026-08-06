"""Modo evaluación: recorre el listado de señas y registra el top-3 en un CSV."""

import csv
import os
from datetime import datetime


class EvalSession:
    FIELDNAMES = [
        "timestamp",
        "eval_index",
        "expected_sign",
        "top1",
        "conf1",
        "top2",
        "conf2",
        "top3",
        "conf3",
        "hit_top1",
        "hit_top3",
        "handedness",
        "capture_mode",
    ]

    def __init__(self, sign_list: list[str], csv_path: str, handedness: str):
        self.sign_list = sign_list
        self.csv_path = csv_path
        self.handedness = handedness
        self.index = 0
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    @property
    def finished(self):
        return self.index >= len(self.sign_list)

    @property
    def expected_sign(self):
        if self.finished:
            return None
        return self.sign_list[self.index]

    def _write_row(self, row: dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
        self.index += 1

    def log_prediction(self, top3: list[tuple[str, float]], capture_mode: str):
        if self.finished:
            return

        expected = self.expected_sign
        top_names = [t[0] for t in top3]
        self._write_row(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "eval_index": self.index + 1,
                "expected_sign": expected,
                "top1": top3[0][0] if len(top3) > 0 else "",
                "conf1": f"{top3[0][1]:.4f}" if len(top3) > 0 else "",
                "top2": top3[1][0] if len(top3) > 1 else "",
                "conf2": f"{top3[1][1]:.4f}" if len(top3) > 1 else "",
                "top3": top3[2][0] if len(top3) > 2 else "",
                "conf3": f"{top3[2][1]:.4f}" if len(top3) > 2 else "",
                "hit_top1": int(top3[0][0] == expected) if top3 else 0,
                "hit_top3": int(expected in top_names),
                "handedness": self.handedness,
                "capture_mode": capture_mode,
            }
        )

    def skip_current(self, capture_mode: str):
        if self.finished:
            return
        self._write_row(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "eval_index": self.index + 1,
                "expected_sign": self.expected_sign,
                "top1": "SKIP",
                "conf1": "",
                "top2": "",
                "conf2": "",
                "top3": "",
                "conf3": "",
                "hit_top1": 0,
                "hit_top3": 0,
                "handedness": self.handedness,
                "capture_mode": capture_mode,
            }
        )
