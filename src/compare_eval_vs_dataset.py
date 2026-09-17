"""
Cruza una eval de cámara contra la separabilidad medida en el dataset.

La pregunta: cuando el modelo falla en cámara, ¿el dato ya era ambiguo o el
dato estaba bien y el problema es la captura? Son dos arreglos distintos:

  - Dato ambiguo  → feature engineering / regrabar / colapsar clases.
  - Dato separable pero falla en cámara → domain shift: cómo se seña en vivo
    no coincide con cómo se grabó. Reentrenar con las mismas features no ayuda.

Uso (desde src/, entorno lsa_gpu):

    python compare_eval_vs_dataset.py model/eval_36letras_numeros_20260915_141001.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from analyze_sign_confusions import (
    ALL_FEATURES,
    BASE_FEATURES,
    load_dataset,
    pair_combined_acc,
    pair_feature_table,
    resolve_name,
    verdict,
)


def read_eval(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def model_scores(rows: list[dict]) -> list[dict]:
    per_model = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "conf_ok": [], "conf_bad": []})
    for row in rows:
        bucket = per_model[row["model"]]
        bucket["n"] += 1
        hit1 = int(float(row["hit_top1"]))
        bucket["top1"] += hit1
        bucket["top3"] += int(float(row["hit_top3"]))
        (bucket["conf_ok"] if hit1 else bucket["conf_bad"]).append(float(row["conf1"]))

    out = []
    for model, b in per_model.items():
        out.append(
            {
                "model": model,
                "n": b["n"],
                "acc1": b["top1"] / b["n"],
                "acc3": b["top3"] / b["n"],
                "conf_ok": float(np.mean(b["conf_ok"])) if b["conf_ok"] else float("nan"),
                "conf_bad": float(np.mean(b["conf_bad"])) if b["conf_bad"] else float("nan"),
            }
        )
    out.sort(key=lambda r: r["acc1"], reverse=True)
    return out


def failures(rows: list[dict]) -> list[dict]:
    """Un registro por (seña, atractor) con en cuántos modelos falló así."""
    grouped = defaultdict(lambda: {"models": [], "confs": [], "top3_miss": 0})
    for row in rows:
        if int(float(row["hit_top1"])):
            continue
        key = (row["expected_sign"], row["top1"])
        grouped[key]["models"].append(row["model"])
        grouped[key]["confs"].append(float(row["conf1"]))
        grouped[key]["top3_miss"] += 0 if int(float(row["hit_top3"])) else 1

    out = []
    for (expected, got), b in grouped.items():
        out.append(
            {
                "expected": expected,
                "got": got,
                "n_models": len(b["models"]),
                "conf": float(np.mean(b["confs"])),
                "top3_miss": b["top3_miss"],
                "models": b["models"],
            }
        )
    out.sort(key=lambda r: (-r["n_models"], -r["conf"]))
    return out


def near_misses(rows: list[dict]) -> list[dict]:
    """Aciertos donde el top2 delata al competidor. Confusión latente."""
    grouped = defaultdict(list)
    for row in rows:
        if not int(float(row["hit_top1"])):
            continue
        grouped[(row["expected_sign"], row["top2"])].append(float(row["conf2"]))
    out = [
        {"expected": exp, "rival": rival, "n_models": len(v), "conf2": float(np.mean(v))}
        for (exp, rival), v in grouped.items()
        if len(v) >= 2
    ]
    out.sort(key=lambda r: (-r["n_models"], -r["conf2"]))
    return out


def diagnose(data: dict, expected: str, got: str) -> dict | None:
    """¿Qué decía el dataset sobre este par?"""
    a = resolve_name(expected, data["signs"])
    b = resolve_name(got, data["signs"])
    if not a or not b or a == b:
        return None
    table = pair_feature_table(data, a, b)
    base_best = next(r for r in table if r["feature"] in BASE_FEATURES)
    return {
        "a": a,
        "b": b,
        "best": table[0]["feature"],
        "sep": table[0]["sep"],
        "base_feature": base_best["feature"],
        "base_sep": base_best["sep"],
        "acc_12": pair_combined_acc(data, a, b, BASE_FEATURES),
        "acc_all": pair_combined_acc(data, a, b, ALL_FEATURES),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eval de cámara vs separabilidad del dataset.")
    parser.add_argument("eval_csv", help="CSV de camera.py --eval.")
    parser.add_argument("--max-clips", type=int, default=None, help="Tope de clips por seña.")
    parser.add_argument("--out", default="inspect_out/confusions/EVAL_VS_DATASET.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_eval(args.eval_csv)
    signs_evaluated = sorted({r["expected_sign"] for r in rows}, key=str.casefold)

    print(f"Eval: {len(rows)} filas, {len(signs_evaluated)} señas, "
          f"{len({r['model'] for r in rows})} modelos.")
    print("Cargando dataset para cruzar...")
    data = load_dataset(args.max_clips)

    scores = model_scores(rows)
    fails = failures(rows)
    latent = near_misses(rows)

    report = [
        f"# Eval de cámara vs dataset — {os.path.basename(args.eval_csv)}",
        "",
        f"{len(signs_evaluated)} señas evaluadas, 1 toma por seña por modelo.",
        "",
        "## Modelos",
        "",
        "| modelo | top-1 | top-3 | conf media (acierto) | conf media (fallo) |",
        "|---|---|---|---|---|",
    ]
    for s in scores:
        report.append(
            f"| `{s['model']}` | **{s['acc1']:.1%}** | {s['acc3']:.1%} "
            f"| {s['conf_ok']:.3f} | {s['conf_bad']:.3f} |"
        )

    report += [
        "",
        "## Fallos, y qué decía el dataset",
        "",
        "`sep dataset` = separabilidad del par en los .npy. Si es alta y el modelo",
        "igual falla, el dato no es el problema: es la captura en vivo.",
        "",
        "| esperada | salió | modelos | conf | fuera de top-3 | sep dataset | 1-NN feats | veredicto |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for f in fails:
        d = diagnose(data, f["expected"], f["got"])
        if d is None:
            report.append(
                f"| **{f['expected']}** | {f['got']} | {f['n_models']}/3 | {f['conf']:.3f} "
                f"| {f['top3_miss']} | — | — | atractor fuera del dataset |"
            )
            continue
        if d["sep"] >= 0.8 and d["acc_all"] >= 0.9:
            tag = "DATO OK → problema de captura"
        elif d["sep"] < 0.4:
            tag = "DATO AMBIGUO → irresoluble así"
        else:
            tag = "DATO LIMITE"
        verdicts.append({**f, **d, "tag": tag})
        report.append(
            f"| **{f['expected']}** | {f['got']} | {f['n_models']}/3 | {f['conf']:.3f} "
            f"| {f['top3_miss']} | `{d['best']}` {d['sep']:.2f} | {d['acc_all']:.0%} | {tag} |"
        )

    report += [
        "",
        "## Confusiones latentes (acertó, pero el top-2 avisa)",
        "",
        "Pares donde al menos 2 modelos pusieron al mismo rival segundo.",
        "",
        "| seña | rival en top-2 | modelos | conf2 media | sep dataset |",
        "|---|---|---|---|---|",
    ]
    for m in latent:
        d = diagnose(data, m["expected"], m["rival"])
        sep_txt = f"`{d['best']}` {d['sep']:.2f}" if d else "—"
        report.append(
            f"| {m['expected']} | {m['rival']} | {m['n_models']}/3 "
            f"| {m['conf2']:.4f} | {sep_txt} |"
        )

    groups = defaultdict(list)
    for v in verdicts:
        groups[v["tag"]].append(f"{v['expected']}→{v['got']}")
    report += ["", "## Resumen", ""]
    for tag, items in sorted(groups.items()):
        report.append(f"- **{tag}**: {', '.join(items)}")
    report.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print(f"\nGuardado {os.path.abspath(args.out)}")
    print("\n".join(report[4 : 4 + 8 + len(scores)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
