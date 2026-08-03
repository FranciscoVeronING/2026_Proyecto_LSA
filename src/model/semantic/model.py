"""
Traductor semántico LSA → español.

- NO importar Unsloth/transformers a nivel de módulo (rompe `python -m src.camera`).
- load_model_and_tokenizer() carga en diferido.
- Por defecto usa PEFT (más estable). Unsloth opcional vía config.USE_UNSLOTH.
"""

from __future__ import annotations

import json

try:
    from src.model.semantic.config import (
        MODEL_PATH,
        MODEL,
        MAX_SEQ_LENGTH,
        SYSTEM_PROMPT_PATH,
        FEW_SHOTS_PATH,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        REPETITION_PENALTY,
        USE_UNSLOTH,
        LOAD_IN_4BIT,
    )
except ImportError:
    from .config import (
        MODEL_PATH,
        MODEL,
        MAX_SEQ_LENGTH,
        SYSTEM_PROMPT_PATH,
        FEW_SHOTS_PATH,
        MAX_NEW_TOKENS,
        TEMPERATURE,
        REPETITION_PENALTY,
        USE_UNSLOTH,
        LOAD_IN_4BIT,
    )

SYSTEM_PROMPT = ""
MODEL = None
TOKENIZER = None
_LOADED = False


def load_prompt():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt_base = f.read().strip()

    try:
        with open(FEW_SHOTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            few_shots = data.get("examples", [])
    except FileNotFoundError:
        few_shots = []

    system_prompt_completo = system_prompt_base
    if few_shots:
        system_prompt_completo += "\n\nEjemplos de traducción:"
        for ex in few_shots:
            glosas_str = " ".join(ex["glosses"])
            system_prompt_completo += f"\nGlosas: {glosas_str} -> Español: {ex['spanish']}"

    return system_prompt_completo


def _load_with_unsloth():
    from transformers import logging
    from unsloth import FastLanguageModel

    logging.set_verbosity_error()
    print(f"[semantic] Loading with Unsloth from: {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_with_peft():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging
    from peft import PeftModel

    logging.set_verbosity_error()
    print(f"[semantic] Loading base={MODEL}")
    print(f"[semantic] Adapter={MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    return model, tokenizer


def load_model_and_tokenizer():
    """Carga diferida. Preferí PEFT si USE_UNSLOTH=False (default)."""
    global SYSTEM_PROMPT, MODEL, TOKENIZER, _LOADED

    if _LOADED and MODEL is not None and TOKENIZER is not None:
        return

    SYSTEM_PROMPT = load_prompt()

    if USE_UNSLOTH:
        try:
            MODEL, TOKENIZER = _load_with_unsloth()
        except Exception as e:
            print(f"[semantic] Unsloth falló ({e}). Probando PEFT...")
            MODEL, TOKENIZER = _load_with_peft()
    else:
        try:
            MODEL, TOKENIZER = _load_with_peft()
        except Exception as e:
            print(f"[semantic] PEFT falló ({e}). Probando Unsloth...")
            MODEL, TOKENIZER = _load_with_unsloth()

    if TOKENIZER.pad_token is None or TOKENIZER.pad_token == TOKENIZER.eos_token:
        TOKENIZER.pad_token = TOKENIZER.unk_token or "<|endoftext|>"

    _LOADED = True
    print("[semantic] Model ready for real-time translation.\n")


def translate_glosses(glosses_input):
    """
    glosses_input: str, ej. "YO LLAMAR POLICIA"
    """
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("LLM no cargada. Llamá load_model_and_tokenizer() primero.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Glosas: {glosses_input}"},
    ]

    model_inputs = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # device del modelo (evita .to("cuda") hardcodeado si cae a CPU)
    try:
        device = next(MODEL.parameters()).device
    except StopIteration:
        device = "cpu"
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    outputs = MODEL.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=TOKENIZER.pad_token_id,
        eos_token_id=TOKENIZER.eos_token_id,
    )

    prompt_len = model_inputs["input_ids"].shape[1]
    return TOKENIZER.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True,
    ).strip()
