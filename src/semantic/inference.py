"""GGUF generation of one sentence from glosses. Used by train and eval."""
from __future__ import annotations

import time

from glosses import clean_output
from models import GGUF_STOP, MAX_NEW_TOKENS, MAX_SEQ_LENGTH


def gloss_messages(system_prompt: str, glosses: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Glosas: {glosses}"},
    ]


def load_gguf(path):
    from llama_cpp import Llama

    return Llama(
        model_path=str(path),
        n_gpu_layers=-1,
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False,
    )


def translate_gguf(llm, system_prompt: str, glosses: str) -> tuple[str, float]:
    """Return the sentence and the latency in milliseconds."""
    started = time.time()
    response = llm.create_chat_completion(
        messages=gloss_messages(system_prompt, glosses),
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        repeat_penalty=1.05,
        stop=GGUF_STOP,
    )
    latency_ms = (time.time() - started) * 1000
    text = clean_output(response["choices"][0]["message"]["content"])
    return text, latency_ms
