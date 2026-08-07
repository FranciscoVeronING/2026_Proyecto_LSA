"""
Evaluación offline del traductor semántico: glosas → español.

Corre el pipeline de producción (atajo literal + LLM) sobre un JSON de pares
referencia y reporta métricas agregadas + CSV detallado.

    python run.py --eval-semantic
    python run.py --eval-semantic --eval-semantic-dataset src/semantic/prompts/few_shots_examples.json

IMPORTANTE: evaluar sobre few_shots_examples.json infla las métricas porque
esos mismos pares ya están inyectados en el system prompt. Usar eval_dataset.json
(o un dataset externo) para medir generalización.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from core.repeat_policy import format_literal_utterance
from semantic.config import CONVERSATION_HISTORY_SIZE, FEW_SHOTS_PATH

_SEMANTIC_DIR = Path(__file__).resolve().parents[1] / "semantic"
DEFAULT_DATASET = _SEMANTIC_DIR / "eval_dataset.json"

CSV_FIELDS = [
    "index",
    "glosses",
    "reference",
    "prediction",
    "via",
    "exact_match",
    "token_f1",
    "rouge_l",
    "bleu4",
]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    """Minúsculas, sin acentos ni puntuación periférica, espacios colapsados."""
    text = (text or "").strip().lower()
    text = _strip_accents(text)
    text = re.sub(r"[\"\'`]", "", text)
    text = re.sub(r"[^\w\s¿?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0]
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def token_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = 0
    hyp_counts = {}
    for tok in hyp_tokens:
        hyp_counts[tok] = hyp_counts.get(tok, 0) + 1
        if hyp_counts[tok] <= ref_counts.get(tok, 0):
            overlap += 1

    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _ngram_counts(tokens: Sequence[str], n: int) -> dict:
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def bleu4(reference: str, hypothesis: str) -> float:
    """BLEU-4 simplificado con suavizado add-one (sin dependencias externas)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        hyp_counts = _ngram_counts(hyp_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        if not hyp_counts:
            precisions.append(0.0)
            continue
        clipped = 0
        total = 0
        for gram, count in hyp_counts.items():
            clipped += min(count, ref_counts.get(gram, 0))
            total += count
        precisions.append((clipped + 1) / (total + 1))

    import math

    if not any(p > 0 for p in precisions):
        return 0.0
    geo = math.exp(sum(math.log(max(p, 1e-9)) for p in precisions) / 4)
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)
    return bp * geo


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalize_text(reference) == normalize_text(hypothesis)


def load_dataset(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = data.get("examples", data)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Dataset vacío o inválido: {path}")
    return examples


def _warn_if_few_shots_leakage(dataset_path: Path) -> None:
    try:
        resolved = dataset_path.resolve()
        few_shots = Path(FEW_SHOTS_PATH).resolve()
    except OSError:
        return
    if resolved == few_shots:
        print(
            "[!] ATENCIÓN: estás evaluando sobre few_shots_examples.json.\n"
            "    Esos pares ya están en el system prompt → las métricas van infladas.\n"
            "    Para medir generalización usá src/semantic/eval_dataset.json.\n"
        )


def predict_utterance(
    glosses: Sequence[str],
    translate_glosses,
    history_messages: Optional[List[dict]] = None,
) -> Tuple[str, str]:
    """Devuelve (texto, via) donde via es 'literal' o 'llm'."""
    joined = " ".join(glosses)
    literal = format_literal_utterance(glosses)
    if literal is not None:
        return literal, "literal"

    text = translate_glosses(joined, history_messages=history_messages) or joined
    return text.strip(), "llm"


def _append_turn(history: List[dict], glosses: Sequence[str], text: str) -> None:
    """Acumula un turno como lo haría ConversationMemory.as_messages()."""
    joined = " ".join(glosses)
    history.append({"role": "user", "content": f"Glosas: {joined}"})
    history.append({"role": "assistant", "content": text})
    max_msgs = CONVERSATION_HISTORY_SIZE * 2
    if len(history) > max_msgs:
        del history[:-max_msgs]


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Entry point liviano: no importa OpenCV ni MediaPipe."""
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluacion offline del traductor semantico (glosas → espanol)."
    )
    parser.add_argument(
        "--eval-semantic",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-semantic-dataset",
        default=None,
        help="JSON con pares {glosses, spanish}. Default: src/semantic/eval_dataset.json",
    )
    parser.add_argument(
        "--eval-semantic-output",
        default=None,
        help="Ruta del CSV (default: eval_semantic_<fecha>.csv en la raiz).",
    )
    parser.add_argument(
        "--eval-semantic-history",
        action="store_true",
        help="Acumular historial turno a turno (simula conversacion encadenada).",
    )
    args = parser.parse_args(argv)
    return run_semantic_eval(
        dataset_path=args.eval_semantic_dataset,
        output_path=args.eval_semantic_output,
        use_history=args.eval_semantic_history,
        repo_root=repo_root,
    )


def run_semantic_eval(
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    use_history: bool = False,
    repo_root: Optional[Path] = None,
) -> int:
    dataset = Path(dataset_path) if dataset_path else DEFAULT_DATASET
    if not dataset.is_absolute() and repo_root is not None:
        candidate = repo_root / dataset
        if candidate.exists():
            dataset = candidate
    if not dataset.exists():
        print(f"[!] No se encontró el dataset: {dataset}")
        return 1

    _warn_if_few_shots_leakage(dataset)
    examples = load_dataset(dataset)

    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = repo_root or Path.cwd()
        output_path = str(base / f"eval_semantic_{stamp}.csv")
    csv_path = Path(output_path)

    print(f"[*] Evaluación semántica: {len(examples)} ejemplos")
    print(f"[*] Dataset: {dataset}")
    print(f"[*] Historial conversacional: {'sí' if use_history else 'no (recomendado para eval)'}")
    print("[*] Cargando LLM...", flush=True)

    try:
        from semantic.translator import load_model_and_tokenizer, translate_glosses

        load_model_and_tokenizer()
    except Exception as exc:
        print(f"\n[!] Falló la carga de la LLM: {exc}", file=sys.stderr)
        print(
            "[!] Si el proceso terminó en silencio en 'Loading checkpoint shards', "
            "es casi seguro falta de VRAM.\n"
            "    1. Cerrá otras apps que usen la GPU.\n"
            "    2. pip install bitsandbytes  (activa LOAD_IN_4BIT en config)\n"
            "    3. Probá sin --eval-semantic-history primero\n"
            "    4. Reiniciá Python/conda si quedó VRAM tomada por un crash previo",
            file=sys.stderr,
        )
        return 1

    print("[*] LLM lista. Traduciendo...\n", flush=True)

    rows = []
    totals = {"exact": 0, "token_f1": 0.0, "rouge_l": 0.0, "bleu4": 0.0}
    via_counts = {"literal": 0, "llm": 0}
    history_acc: List[dict] = []

    for idx, ex in enumerate(examples, start=1):
        glosses = ex["glosses"]
        reference = ex["spanish"]
        hist = list(history_acc) if use_history else None
        prediction, via = predict_utterance(glosses, translate_glosses, hist)
        if use_history:
            _append_turn(history_acc, glosses, prediction)
        via_counts[via] += 1

        em = exact_match(reference, prediction)
        tf1 = token_f1(reference, prediction)
        rl = rouge_l(reference, prediction)
        b4 = bleu4(reference, prediction)

        totals["exact"] += int(em)
        totals["token_f1"] += tf1
        totals["rouge_l"] += rl
        totals["bleu4"] += b4

        gloss_str = " ".join(glosses)
        rows.append(
            {
                "index": idx,
                "glosses": gloss_str,
                "reference": reference,
                "prediction": prediction,
                "via": via,
                "exact_match": int(em),
                "token_f1": f"{tf1:.4f}",
                "rouge_l": f"{rl:.4f}",
                "bleu4": f"{b4:.4f}",
            }
        )

        mark = "OK" if em else "  "
        print(f"  [{mark}] {idx:2d}. {gloss_str}")
        if not em:
            print(f"       ref: {reference}")
            print(f"       got: {prediction}  ({via})")

    n = len(examples)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 52)
    print("RESUMEN")
    print("=" * 52)
    print(f"  Ejemplos           : {n}")
    print(f"  Vía literal / LLM  : {via_counts['literal']} / {via_counts['llm']}")
    print(f"  Exact match        : {totals['exact']}/{n} ({100*totals['exact']/n:.1f}%)")
    print(f"  Token F1 (prom.)   : {totals['token_f1']/n:.4f}")
    print(f"  ROUGE-L (prom.)    : {totals['rouge_l']/n:.4f}")
    print(f"  BLEU-4 (prom.)     : {totals['bleu4']/n:.4f}")
    print(f"  CSV                : {csv_path}")
    print("=" * 52)
    print(
        "\nNota: exact match es estricto (ignora mayúsculas/acentos). "
        "Token F1, ROUGE-L y BLEU-4 capturan parcialidad.\n"
        "Para comparar con el paper de entrenamiento, sumá sacrebleu sobre el mismo CSV."
    )
    return 0
