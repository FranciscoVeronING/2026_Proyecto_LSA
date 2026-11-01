"""Classifier vocab table matches mapeo_clases.json (informe appendix A)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classifier.vocabulario import (  # noqa: E402
    CATEGORIES,
    load_class_names,
    uncategorized,
    vocab_rows,
)


def test_all_mapped_classes_are_categorized():
    names = load_class_names()
    assert len(names) == 100
    assert uncategorized(names) == []
    rows = vocab_rows(names)
    total = sum(int(row["count"]) for row in rows)
    assert total == 100
    assert rows[0]["category"] == "Alfabeto manual"
    assert rows[0]["count"] == 27
    assert "ñ" in rows[0]["signs"]
    parentesco = next(row for row in rows if row["category"] == "Parentesco")
    assert "esposo_a" in parentesco["signs"]
    ident = next(row for row in rows if row["category"] == "Identificación")
    assert "años" in ident["signs"]


def test_category_labels_are_unique():
    seen: set[str] = set()
    for _, labels in CATEGORIES:
        for lab in labels:
            assert lab not in seen, lab
            seen.add(lab)


if __name__ == "__main__":
    test_all_mapped_classes_are_categorized()
    print("ok test_all_mapped_classes_are_categorized")
    test_category_labels_are_unique()
    print("ok test_category_labels_are_unique")
    print("all vocabulario tests passed")
