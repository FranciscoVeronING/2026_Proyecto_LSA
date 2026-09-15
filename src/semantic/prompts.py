"""Prompt de sistema y plantilla de chat."""
from __future__ import annotations

import json
from pathlib import Path

from config import DEFAULT_FEW_SHOT_FILE, DEFAULT_PROMPT_FILE
from gloss_format import formato_canonico


def user_message(spanish: str) -> str:
    return f"Español: {spanish.strip()}"


def build_messages(
    system_prompt: str,
    spanish: str,
    glosses: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message(spanish)},
    ]
    if glosses is not None:
        messages.append({"role": "assistant", "content": glosses})
    return messages


def load_system_prompt(
    prompt_path: Path | None = None,
    few_shot_path: Path | None = None,
    include_few_shots: bool = True,
) -> str:
    prompt_path = prompt_path or DEFAULT_PROMPT_FILE
    few_shot_path = few_shot_path or DEFAULT_FEW_SHOT_FILE
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    if not include_few_shots or not few_shot_path.exists():
        return system_prompt

    payload = json.loads(few_shot_path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    if not examples:
        return system_prompt

    system_prompt += "\n\nEjemplos de glosado:"
    for example in examples:
        spanish = str(example["spanish"]).strip()
        glosses = formato_canonico(example["glosses"])
        system_prompt += f"\n{user_message(spanish)} -> Glosas: {glosses}"
    return system_prompt
