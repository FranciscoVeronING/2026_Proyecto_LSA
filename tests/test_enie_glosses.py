"""Hearing captions must keep Ñ even when the SLM returns T-pose stems."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from avatar.names import tpose_stem  # noqa: E402
from core.enie import restore_enie_glosses  # noqa: E402


def test_anos_from_spanish_años():
    out = restore_enie_glosses("YO ANOS 24", spanish="tengo veinticuatro años")
    assert out == "YO AÑOS 24", out


def test_manana_already_has_enie():
    out = restore_enie_glosses("MAÑANA PLAZA NOSOTROS VER", spanish="mañana nos vemos en la plaza")
    assert out == "MAÑANA PLAZA NOSOTROS VER", out


def test_manana_folded_from_spanish():
    out = restore_enie_glosses("MANANA YO VOS LLAMAR", spanish="mañana te voy a llamar")
    assert out == "MAÑANA YO VOS LLAMAR", out


def test_pose_inventory_without_spanish():
    assert restore_enie_glosses("YO ANOS 23") == "YO AÑOS 23"
    assert restore_enie_glosses("OTONO FRIO") == "OTOÑO FRIO"
    assert restore_enie_glosses("ESPANA") == "ESPAÑA"


def test_canada_is_not_cañada():
    assert restore_enie_glosses("CANADA") == "CANADA"


def test_nfc_composes_decomposed_enie():
    decomposed = "MA" + "N\u0303" + "ANA"
    out = restore_enie_glosses(decomposed, spanish="mañana")
    assert out == "MAÑANA", repr(out)
    assert "\u0303" not in out


def test_single_n_is_not_enie():
    out = restore_enie_glosses("N K AÑOS R O", spanish="Mi apellido es Nkgro")
    assert out == "N K AÑOS R O", out


def test_proper_names_keep_casing():
    out = restore_enie_glosses("NOMBRE MIO Maite", spanish="Me llamo Maite")
    assert out == "NOMBRE MIO Maite", out


def test_tpose_stem_still_finds_ascii_files():
    assert tpose_stem("AÑOS") == "ANOS"
    assert tpose_stem("MAÑANA") == "MANANA"
    assert tpose_stem("Ñ") == "ENIE"


if __name__ == "__main__":
    tests = [
        test_anos_from_spanish_años,
        test_manana_already_has_enie,
        test_manana_folded_from_spanish,
        test_pose_inventory_without_spanish,
        test_canada_is_not_cañada,
        test_nfc_composes_decomposed_enie,
        test_single_n_is_not_enie,
        test_proper_names_keep_casing,
        test_tpose_stem_still_finds_ascii_files,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all enie caption tests passed")
