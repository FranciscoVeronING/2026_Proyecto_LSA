"""Tests de formato de glosas. Correr desde src/semantic: python test_gloss_format.py"""
from __future__ import annotations

from gloss_format import canonicalize_glosses, formato_canonico, unificar_glosas_dactilologicas


def test_nombre_dactilologico() -> None:
    assert unificar_glosas_dactilologicas(["YO", "NOMBRE", "M", "A", "I", "T", "E"]) == [
        "YO",
        "NOMBRE",
        "Maite",
    ]


def test_dni_compacto() -> None:
    assert formato_canonico(["YO", "DOCUMENTO", "NUMERO", "1", "8", "3", "3", "3", "9", "4", "0"]) == (
        "YO DOCUMENTO NUMERO 18333940"
    )


def test_letra_lexica_x() -> None:
    assert unificar_glosas_dactilologicas(["X"]) == ["X"]


def test_glosas_ya_compactas() -> None:
    assert formato_canonico("FUTURO YO CUCHILLO LLEVAR NO") == "FUTURO YO CUCHILLO LLEVAR NO"


def test_canon_hijo_hombre() -> None:
    assert canonicalize_glosses("Mi hijo.", ["HIJO", "MIO"]) == ["HIJO", "HOMBRE", "MIO"]
    assert canonicalize_glosses("Tengo un hijo.", ["YO", "HIJO", "TENER"]) == [
        "YO",
        "HIJO",
        "HOMBRE",
        "TENER",
    ]
    assert canonicalize_glosses("Mi hija.", ["HIJO", "MUJER", "MIO"]) == ["HIJO", "MUJER", "MIO"]


def test_canon_ellos_y_tacito() -> None:
    assert canonicalize_glosses("Ellos viven acá.", ["ELLOS", "VIVIR"]) == ["ELLOS", "HOMBRE", "VIVIR"]
    assert canonicalize_glosses("Ellas viven acá.", ["ELLOS", "MUJER", "VIVIR"]) == [
        "ELLOS",
        "MUJER",
        "VIVIR",
    ]
    assert canonicalize_glosses("Viven acá.", ["ELLOS", "HOMBRE", "VIVIR"]) == ["ELLOS", "VIVIR"]


def test_canon_posesivo() -> None:
    assert canonicalize_glosses("Mi cuchillo.", ["MIO", "CUCHILLO"]) == ["CUCHILLO", "MIO"]


if __name__ == "__main__":
    test_nombre_dactilologico()
    test_dni_compacto()
    test_letra_lexica_x()
    test_glosas_ya_compactas()
    test_canon_hijo_hombre()
    test_canon_ellos_y_tacito()
    test_canon_posesivo()
    print("ok")
