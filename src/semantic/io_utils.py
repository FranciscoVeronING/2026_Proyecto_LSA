"""Log y JSON de corridas."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def log_line(log_fp: TextIO | None, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()


def load_summary(summary_path: Path) -> dict[str, Any]:
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"runs": []}
    return {"runs": []}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
