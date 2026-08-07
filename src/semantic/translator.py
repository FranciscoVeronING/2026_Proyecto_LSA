"""
Traductor semántico LSA → español.

- NO importar Unsloth/transformers a nivel de módulo: tardan segundos y pueden
  abortar el proceso, y la cámara tiene que arrancar aunque la LLM falle.
- load_model_and_tokenizer() carga en diferido, desde el hilo semántico.
- Por defecto usa PEFT (más estable). Unsloth opcional vía config.USE_UNSLOTH.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from semantic.config import (
    ADAPTER_PATH,
    BASE_MODEL_ID,
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

# Qwen2.5-3B en safetensors ≈ 6 GB repartidos en 2 shards. Menos de 1 GB total
# indica descarga truncada (común en Windows sin symlinks en el cache de HF).
_MIN_BASE_MODEL_BYTES = 1_000_000_000


def _hub_cache_dir(model_id: str) -> Optional[Path]:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    slug = model_id.replace("/", "--")
    return Path(HF_HUB_CACHE) / f"models--{slug}"


def _check_base_model_cache(model_id: str) -> None:
    """Detecta descargas truncadas antes de que safetensors crashee el proceso."""
    cache = _hub_cache_dir(model_id)
    if cache is None or not cache.exists():
        return

    shards = list(cache.rglob("*.safetensors"))
    if not shards:
        return

    total = sum(p.stat().st_size for p in shards)
    if total >= _MIN_BASE_MODEL_BYTES:
        return

    mb = total / 1e6
    print(
        f"\n[!] Cache de {model_id} parece CORRUPTO ({mb:.0f} MB; "
        f"esperado ~6000 MB).\n"
        f"    Esto provoca crash silencioso en 'Loading checkpoint shards'.\n\n"
        f"    Borrá el cache y volvé a descargar:\n"
        f"      Remove-Item -Recurse -Force \"{cache}\"\n"
        f"      python run.py --eval-semantic\n\n"
        f"    Tip Windows: activá Modo Desarrollador para symlinks en el cache HF.\n"
    )
    raise RuntimeError(f"Modelo base incompleto en cache ({mb:.0f} MB)")


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
    print(f"[semantic] Loading with Unsloth from: {ADAPTER_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _print_vram_status() -> None:
    import torch

    if not torch.cuda.is_available():
        print("[semantic] CUDA no disponible -> se cargara en CPU (lento).")
        return
    try:
        free, total = torch.cuda.mem_get_info(0)
        print(f"[semantic] VRAM libre: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
        if free < 3e9:
            print(
                "[!] Poca VRAM libre. Cerrá otros procesos que usen la GPU "
                "(otra instancia de Python, juegos, Chrome con aceleración HW)."
            )
    except Exception:
        pass


def _load_with_peft():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging
    from peft import PeftModel

    logging.set_verbosity_error()
    print(f"[semantic] Loading base={BASE_MODEL_ID}")
    print(f"[semantic] Adapter={ADAPTER_PATH}")
    _print_vram_status()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if LOAD_IN_4BIT:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
                print("[semantic] Modo: 4-bit (BitsAndBytes)")
            except ImportError:
                print(
                    "[semantic] LOAD_IN_4BIT=True pero falta bitsandbytes; "
                    "usando float16. Instala: pip install bitsandbytes"
                )
                load_kwargs["dtype"] = torch.float16
                load_kwargs["device_map"] = "cpu"
                print("[semantic] Modo: float16 (CPU -> GPU)")
        else:
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "cpu"
            print("[semantic] Modo: float16 (CPU -> GPU)")
    else:
        load_kwargs["dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"
        print("[semantic] Modo: CPU float32")

    print("[semantic] Descargando/cargando pesos base (puede tardar 1-3 min)...", flush=True)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
    except Exception as exc:
        if "quantization_config" in load_kwargs:
            print(f"[semantic] Carga 4-bit fallo ({exc}). Reintentando float16...")
            load_kwargs.pop("quantization_config", None)
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "cpu"
            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
        else:
            raise

    if torch.cuda.is_available() and load_kwargs.get("device_map") == "cpu":
        print("[semantic] Pesos en RAM OK. Moviendo a GPU...", flush=True)
        base_model = base_model.to("cuda:0")

    print("[semantic] Pesos base OK. Aplicando adapter LoRA...", flush=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def load_model_and_tokenizer():
    """Carga diferida. Preferí PEFT si USE_UNSLOTH=False (default)."""
    global SYSTEM_PROMPT, MODEL, TOKENIZER, _LOADED

    if _LOADED and MODEL is not None and TOKENIZER is not None:
        return

    SYSTEM_PROMPT = load_prompt()
    _check_base_model_cache(BASE_MODEL_ID)

    try:
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
    except Exception as e:
        print(f"\n[!] No se pudo cargar la LLM: {e}")
        print(
            "[!] Si el proceso murió sin mensaje de error, suele ser falta de VRAM.\n"
            "    Probá: cerrar otras apps GPU, pip install bitsandbytes, o\n"
            "    set LOAD_IN_4BIT = True en src/semantic/config.py"
        )
        raise

    if TOKENIZER.pad_token is None or TOKENIZER.pad_token == TOKENIZER.eos_token:
        TOKENIZER.pad_token = TOKENIZER.unk_token or "<|endoftext|>"

    _LOADED = True
    print("[semantic] Model ready for real-time translation.\n")


def translate_glosses(glosses_input, history_messages=None):
    """
    glosses_input: str, ej. "YO LLAMAR POLICIA"
    history_messages: lista opcional de dicts {role, content} (ventana conversacional)
    """
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("LLM no cargada. Llamá load_model_and_tokenizer() primero.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": f"Glosas: {glosses_input}"})

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
