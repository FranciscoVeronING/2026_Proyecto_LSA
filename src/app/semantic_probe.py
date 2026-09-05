"""
Prueba interactiva del traductor semántico: glosas → español, sin cámara.

Sirve para ver cómo traduce la LLM cuando las glosas son las correctas
(el clasificador no interviene).

    python run.py --probe-semantic
    python run.py --probe-semantic --model qwen2.5-0.5b
    python run.py --probe-semantic YO LLAMAR POLICIA
    python run.py --probe-semantic --file enunciados.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from app.semantic_eval import predict_utterance
from core.conversation_memory import ConversationMemory
from semantic.config import CONVERSATION_HISTORY_SIZE, DEFAULT_MODEL_ID, USE_CONVERSATION_HISTORY
from semantic.models import list_semantic_models

HELP_TEXT = """
Comandos:
  :q / :quit     salir
  :c / :clear    limpiar historial conversacional
  :h / :hist     prender/apagar historial
  :m / :model    listar modelos
  :m ID          cambiar modelo (ej. :m qwen2.5-1.5b)
  :help          esta ayuda

Un enunciado es una línea de glosas separadas por espacio:
  YO LLAMAR POLICIA
  CUANDO VOS CASA
  J U A N
""".strip()


def parse_glosses(text: str) -> List[str]:
    cleaned = (text or "").replace(",", " ").replace(";", " ").replace("|", " ")
    return [tok.upper() for tok in cleaned.split() if tok.strip()]


def load_utterances_from_file(path: Path) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        examples = data.get("examples", data if isinstance(data, list) else [])
        utterances = []
        for ex in examples:
            if isinstance(ex, dict) and "glosses" in ex:
                glosses = ex["glosses"]
                if isinstance(glosses, str):
                    glosses = parse_glosses(glosses)
                utterances.append([str(g).upper() for g in glosses])
            elif isinstance(ex, (list, tuple)):
                utterances.append([str(g).upper() for g in ex])
        return utterances

    utterances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        utterances.append(parse_glosses(line))
    return utterances


def _print_models(active_id: Optional[str]) -> None:
    print("Modelos en src/semantic/outputs/:")
    for item in list_semantic_models():
        mark = "*" if item["id"] == active_id else " "
        avail = "" if item["available"] else "  (sin .gguf)"
        print(f"  [{mark}] {item['id']}{avail}")


def _run_one(
    glosses: Sequence[str],
    translate_glosses,
    memory: ConversationMemory,
    use_history: bool,
) -> None:
    joined = " ".join(glosses)
    history = memory.as_messages() if use_history and memory.turns else None
    t0 = time.perf_counter()
    text, via = predict_utterance(glosses, translate_glosses, history)
    elapsed = time.perf_counter() - t0
    memory.add_signer(text, glosses=joined)
    print(f"  glosas : {joined}")
    print(f"  via    : {via}")
    print(f"  español: {text}")
    print(f"  tiempo : {elapsed:.1f}s | historial: {len(memory.turns)}/{memory.maxlen}")
    print()


def cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probar el traductor semántico con glosas escritas (sin clasificador)."
    )
    parser.add_argument(
        "--probe-semantic",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Id del modelo GGUF (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--file",
        dest="file_path",
        default=None,
        help="Archivo .txt (una línea = un enunciado) o JSON con examples.glosses.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="No inyectar turnos previos en el prompt.",
    )
    parser.add_argument(
        "utterance",
        nargs="*",
        help="Glosas de un solo enunciado. Si se omiten, entra al modo interactivo.",
    )
    args = parser.parse_args(argv)

    from semantic.translator import (
        get_active_model_id,
        load_model_and_tokenizer,
        switch_model,
        translate_glosses,
    )

    model_id = args.model
    use_history = USE_CONVERSATION_HISTORY and not args.no_history
    memory = ConversationMemory(maxlen=CONVERSATION_HISTORY_SIZE)

    print("[*] Probe semántico (sin cámara / sin clasificador)")
    _print_models(model_id)
    print(f"[*] Cargando {model_id}...", flush=True)
    try:
        load_model_and_tokenizer(model_id=model_id)
    except Exception as exc:
        print(f"[!] No se pudo cargar el modelo: {exc}", file=sys.stderr)
        return 1

    active = get_active_model_id() or model_id
    print(f"[*] Listo. Modelo activo: {active}")
    print(f"[*] Historial: {'on' if use_history else 'off'}")
    print()

    if args.file_path:
        path = Path(args.file_path)
        try:
            utterances = load_utterances_from_file(path)
        except Exception as exc:
            print(f"[!] No se pudo leer {path}: {exc}", file=sys.stderr)
            return 1
        if not utterances:
            print(f"[!] {path} no tiene enunciados.")
            return 1
        print(f"[*] {len(utterances)} enunciados desde {path}\n")
        for glosses in utterances:
            if glosses:
                _run_one(glosses, translate_glosses, memory, use_history)
        return 0

    if args.utterance:
        glosses = parse_glosses(" ".join(args.utterance))
        if not glosses:
            print("[!] No hay glosas para traducir.")
            return 1
        _run_one(glosses, translate_glosses, memory, use_history)
        return 0

    print(HELP_TEXT)
    print()
    while True:
        try:
            line = input("glosas> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Listo.")
            return 0

        if not line:
            continue

        if line.startswith(":"):
            cmd, _, rest = line[1:].partition(" ")
            cmd = cmd.lower().strip()
            rest = rest.strip()
            if cmd in {"q", "quit", "exit"}:
                print("[*] Listo.")
                return 0
            if cmd in {"c", "clear"}:
                memory.clear()
                print("[*] Historial limpiado.\n")
                continue
            if cmd in {"h", "hist", "history"}:
                use_history = not use_history
                print(f"[*] Historial: {'on' if use_history else 'off'}\n")
                continue
            if cmd in {"m", "model"}:
                if rest:
                    try:
                        switch_model(rest)
                        active = get_active_model_id() or rest
                        print(f"[*] Modelo activo: {active}\n")
                    except Exception as exc:
                        print(f"[!] No se pudo cambiar a {rest}: {exc}\n")
                else:
                    _print_models(get_active_model_id())
                    print()
                continue
            if cmd in {"help", "?"}:
                print(HELP_TEXT)
                print()
                continue
            print(f"[!] Comando desconocido: :{cmd}  (probá :help)\n")
            continue

        glosses = parse_glosses(line)
        if not glosses:
            continue
        try:
            _run_one(glosses, translate_glosses, memory, use_history)
        except Exception as exc:
            print(f"[!] Error de la LLM: {exc}\n")
