"""Restore Ñ on LSA gloss tokens so Meet captions keep Spanish eñe.

The hearing SLM often emits ASCII-folded stems that match T-pose filenames
(ANOS.pose, MANANA.pose). Those stems must stay valid for clip lookup
(``tpose_stem`` still maps Ñ→N), but the subtitle line should show AÑOS.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ_]+")
_SPANISH_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
_ACUTE = str.maketrans("ÁÉÍÓÚÜ", "AEIOUU")

# T-pose stems that dropped Ñ in the filename. Do not include CANADA (country).
_POSE_ENIE = {
    "ANO": "AÑO",
    "ANOS": "AÑOS",
    "MANANA": "MAÑANA",
    "PASADOMANANA": "PASADOMAÑANA",
    "SENA": "SEÑA",
    "SENAS": "SEÑAS",
    "SENAPERSONAL": "SEÑAPERSONAL",
    "OTONO": "OTOÑO",
    "PANUELO": "PAÑUELO",
    "PANUELOPARANARIZ": "PAÑUELOPARANARIZ",
    "PANUELOPARACUELLO": "PAÑUELOPARACUELLO",
    "ESPANA": "ESPAÑA",
    "CUNADO": "CUÑADO",
    "CUNADA": "CUÑADA",
    "NINO": "NIÑO",
    "NINA": "NIÑA",
    "NINOS": "NIÑOS",
    "NINAS": "NIÑAS",
    "SENOR": "SEÑOR",
    "SENORA": "SEÑORA",
    "MONTANA": "MONTAÑA",
}


def _nfc_upper(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").upper()


def _gloss_form(word: str) -> str:
    """Uppercase lexical form: keep Ñ, drop other accents (años → AÑOS)."""
    return _nfc_upper(word).translate(_ACUTE)


def _fold_enie(token: str) -> str:
    return _gloss_form(token).replace("Ñ", "N")


def _maps_from_spanish(spanish: str) -> dict[str, str]:
    """If the source sentence has ñ, the matching folded gloss token gets Ñ back."""
    maps: dict[str, str] = {}
    for word in _SPANISH_WORD.findall(spanish or ""):
        display = _gloss_form(word)
        if "Ñ" not in display:
            continue
        folded = display.replace("Ñ", "N")
        if folded == "N":
            # Lone N is a fingerspell letter; only restore when the Spanish
            # token itself was ñ / eñe.
            if display not in {"Ñ", "EÑE"}:
                continue
        maps[folded] = display
    return maps


def restore_enie_glosses(glosses: str, spanish: str = "") -> str:
    """Rewrite hearing glosses so captions keep Ñ (AÑOS, MAÑANA, …)."""
    text = unicodedata.normalize("NFC", (glosses or "").strip())
    if not text:
        return ""
    from_spanish = _maps_from_spanish(spanish)

    def one(token: str) -> str:
        nfc = unicodedata.normalize("NFC", token)
        folded = _fold_enie(nfc)
        if folded in from_spanish:
            return from_spanish[folded]
        if folded in _POSE_ENIE:
            return _POSE_ENIE[folded]
        return nfc

    parts: list[str] = []
    last = 0
    for match in _TOKEN.finditer(text):
        parts.append(text[last : match.start()])
        parts.append(one(match.group(0)))
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)
