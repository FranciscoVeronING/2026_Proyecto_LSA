"""System prompt and the user/assistant messages used for training and eval."""
from __future__ import annotations

from pathlib import Path

from config import DEFAULT_PROMPT_FILE


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


def load_system_prompt(prompt_path: Path | None = None) -> str:
    prompt_path = prompt_path or DEFAULT_PROMPT_FILE
    return prompt_path.read_text(encoding="utf-8").strip()
