"""Chat: escribís español, el modelo devuelve glosas. Sin historial entre turnos."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DEFAULT_OUTPUT_BASE, MODELS, model_folder_name
from gguf_export import find_q4_gguf
from metrics import load_gguf_engine, predecir_glosas
from prompts import load_system_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hablá con el glosador: escribís español, responde glosas LSA.",
    )
    parser.add_argument(
        "--gguf",
        default=None,
        help="Ruta al .gguf. Si se omite, se busca en outputs/ según --model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="ID del catálogo (ej. qwen2.5-0.5b). Default: el único GGUF disponible.",
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help="Carpeta de outputs (para resolver --model).",
    )
    parser.add_argument(
        "--no-few-shots",
        action="store_true",
        help="System prompt sin ejemplos few-shot.",
    )
    parser.add_argument(
        "texto",
        nargs="*",
        help="Una oración. Si se omite, entra al modo interactivo.",
    )
    return parser.parse_args()


def available_ggufs(output_base: Path) -> list[tuple[dict, Path]]:
    found: list[tuple[dict, Path]] = []
    for model in MODELS:
        gguf_dir = output_base / f"{model_folder_name(model['name'])}_gguf"
        try:
            found.append((model, find_q4_gguf(gguf_dir)))
        except FileNotFoundError:
            continue
    return found


def resolve_gguf(output_base: Path, gguf_arg: str | None, model_id: str | None) -> tuple[str, Path]:
    if gguf_arg:
        path = Path(gguf_arg)
        if not path.exists():
            raise SystemExit(f"No existe el GGUF: {path}")
        return path.stem, path

    found = available_ggufs(output_base)
    if model_id:
        for model, path in found:
            if model["id"] == model_id:
                return model["label"], path
        known = ", ".join(m["id"] for m, _ in found) or "(ninguno en outputs/)"
        raise SystemExit(f"No hay GGUF para '{model_id}'. Disponibles: {known}")

    if not found:
        raise SystemExit(
            "No encontré GGUF en outputs/. Pasá --gguf RUTA o entrená con train_all_models.py."
        )
    if len(found) > 1:
        ids = ", ".join(m["id"] for m, _ in found)
        raise SystemExit(f"Hay varios GGUF ({ids}). Elegí uno con --model ID o --gguf RUTA.")
    model, path = found[0]
    return model["label"], path


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def chat_loop(llm, system_prompt: str) -> None:
    print("Escribí una oración en español. Vacío, :q o Ctrl+C para salir.")
    while True:
        try:
            spanish = input("español> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not spanish or spanish in {":q", ":quit", "salir"}:
            return
        glosas, latency = predecir_glosas(llm, system_prompt, spanish)
        print(f"glosas>  {glosas}")
        print(f"         ({latency:.0f} ms)")


def main() -> int:
    _configure_stdout()
    args = parse_args()
    label, gguf = resolve_gguf(Path(args.output_base), args.gguf, args.model)
    system_prompt = load_system_prompt(include_few_shots=not args.no_few_shots)

    print(f"Cargando {label}")
    print(f"GGUF: {gguf}")
    llm = load_gguf_engine(gguf)

    if args.texto:
        spanish = " ".join(args.texto)
        glosas, latency = predecir_glosas(llm, system_prompt, spanish)
        print(glosas)
        print(f"({latency:.0f} ms)")
        return 0

    chat_loop(llm, system_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
