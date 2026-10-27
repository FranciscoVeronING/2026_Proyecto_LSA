"""Generación GGUF de una oración a partir de glosas. La usan el train y el eval."""
from __future__ import annotations

import time

from glosas import limpiar_salida
from modelos import GGUF_STOP, MAX_NEW_TOKENS, MAX_SEQ_LENGTH


def mensajes_glosas(system_prompt: str, glosas: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Glosas: {glosas}"},
    ]


def cargar_gguf(ruta):
    from llama_cpp import Llama

    return Llama(
        model_path=str(ruta),
        n_gpu_layers=-1,
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False,
    )


def traducir_gguf(llm, system_prompt: str, glosas: str) -> tuple[str, float]:
    """Devuelve la oración y la latencia en milisegundos."""
    inicio = time.time()
    respuesta = llm.create_chat_completion(
        messages=mensajes_glosas(system_prompt, glosas),
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        repeat_penalty=1.05,
        stop=GGUF_STOP,
    )
    latencia_ms = (time.time() - inicio) * 1000
    texto = limpiar_salida(respuesta["choices"][0]["message"]["content"])
    return texto, latencia_ms
