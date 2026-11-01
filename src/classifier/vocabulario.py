"""Classifier signs grouped like informe appendix A, labels from mapeo_clases.json."""

from __future__ import annotations

import json
from pathlib import Path

from classifier.config import CLASSES_PATH

# Same buckets as informe/capitulos/apendice_a_vocabulario.tex.
# Keys must match mapeo_clases.json exactly (`ñ`, `años`, `esposo a`).
# The mapping stores a space; the table shows the underscore form.
_DISPLAY = {"esposo a": "esposo_a"}
CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Alfabeto manual",
        tuple("ABCDEFGHIJKLMN") + ("ñ",) + tuple("OPQRSTUVWXYZ"),
    ),
    ("Dígitos", tuple("0123456789")),
    ("Interrogativos", ("como", "cuando", "donde", "que", "quien", "cuantos")),
    ("Afirmación y negación", ("si", "no")),
    ("Valoración", ("bien", "mal")),
    ("Pronombres", ("yo", "vos", "el_ella", "nosotros", "ellos")),
    ("Posesivos", ("mio", "tuyo")),
    ("Saludos y cortesía", ("hola", "chau", "gracias")),
    ("Tiempo", ("pasado", "futuro", "dia", "hora", "ahora_hoy", "ayer")),
    (
        "Días de la semana",
        ("lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"),
    ),
    ("Lugares", ("lugar", "casa", "calle", "plaza")),
    ("Identificación", ("nombre", "apellido", "documento", "numero", "años")),
    ("Parentesco", ("familia", "mama", "papa", "hermano_a", "hijo_a", "esposo a")),
    ("Partes del cuerpo", ("brazo", "cara", "ojo")),
    ("Objeto", ("cuchillo",)),
    (
        "Verbos",
        ("tener", "poder", "vivir", "vivir_en", "ver", "llamar", "llevar", "repetir"),
    ),
    ("Auxilio", ("ayudar", "robar", "policia")),
]


def load_class_names(path: Path | None = None) -> list[str]:
    dest = Path(path) if path is not None else Path(CLASSES_PATH)
    data = json.loads(dest.read_text(encoding="utf-8"))
    return [name for name, _ in sorted(data.items(), key=lambda kv: int(kv[1]))]


def vocab_rows(class_names: list[str] | None = None) -> list[dict[str, object]]:
    """Rows: category, count, signs (only classes that exist in the mapping)."""
    names = set(class_names if class_names is not None else load_class_names())
    rows: list[dict[str, object]] = []
    for title, labels in CATEGORIES:
        present = [_DISPLAY.get(lab, lab) for lab in labels if lab in names]
        rows.append({"category": title, "count": len(present), "signs": present})
    return rows


def uncategorized(class_names: list[str] | None = None) -> list[str]:
    names = class_names if class_names is not None else load_class_names()
    known = {lab for _, labels in CATEGORIES for lab in labels}
    return [name for name in names if name not in known]
