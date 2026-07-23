"""Reordena glosas segun LSA: TIEMPO-LUGAR-SUJETO/OBJETO-VERBO-NEG/AFIR-PREGUNTA."""
import json
from pathlib import Path

SALUDO = {"HOLA", "CHAU"}
TIEMPO = {
    "AYER", "AHORA_HOY", "LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES",
    "SABADO", "DOMINGO", "DIA", "HORA",
}
LUGAR = {"CASA", "CALLE", "PLAZA", "DEPARTAMENTO", "LUGAR", "VIVIR_EN"}
LUGAR_CON_DELETREO = {"CALLE", "CASA", "PLAZA", "DEPARTAMENTO", "LUGAR"}
SUJETO = {"YO", "VOS", "EL_ELLA", "NOSOTROS", "ELLOS"}
VERBO = {
    "TENER", "VER", "LLAMAR", "REPETIR", "VIVIR", "GOLPEAR", "PODER", "SACAR",
    "ROBAR", "PASAR", "LLEVAR", "LASTIMAR",
}
NEG_AFF = {"NO", "SI", "BIEN", "MAL", "TUYO"}
PREGUNTA = {"COMO", "CUANDO", "DONDE", "QUE", "QUIENES", "CUANTOS"}
HEAD_SPELL = {"NOMBRE", "APELLIDO", "DOCUMENTO", "NUMERO"}
OBJETO = {
    "FAMILIA", "MAMA", "PAPA", "HERMANO_A", "HIJO_A", "ESPOSO_A",
    "ARMA", "CUCHILLO", "BRAZO", "CARA", "OJO", "AÑOS", "ANOS", "NUMERO",
}


def is_spelling(gloss: str) -> bool:
    return len(gloss) == 1 and (gloss.isdigit() or gloss.isalpha())


def collect_spelling_group(glosses: list[str], start: int) -> tuple[list[str], int]:
    group = [glosses[start]]
    i = start + 1
    if group[0] == "DOCUMENTO" and i < len(glosses) and glosses[i] == "NUMERO":
        group.append(glosses[i])
        i += 1
    while i < len(glosses) and is_spelling(glosses[i]):
        group.append(glosses[i])
        i += 1
    return group, i


def collect_edad_group(glosses: list[str], start: int) -> tuple[list[str], int]:
    group = [glosses[start]]
    i = start + 1
    while i < len(glosses) and is_spelling(glosses[i]):
        group.append(glosses[i])
        i += 1
    return group, i


def reorder_glosses(glosses: list[str]) -> list[str]:
    """TIEMPO -> LUGAR -> SUJETO/OBJETO -> VERBO -> NEG/AFIR -> PREGUNTA."""
    saludo: list[str] = []
    tiempo: list[str] = []
    lugar: list[str] = []
    sujeto: list[str] = []
    objeto: list[str] = []
    verbo: list[str] = []
    neg_aff: list[str] = []
    pregunta: list[str] = []

    i = 0
    n = len(glosses)
    while i < n:
        g = glosses[i]

        if g == "VIVIR_EN":
            group = [g]
            i += 1
            while i < n and glosses[i] in LUGAR_CON_DELETREO | {"CASA"}:
                group.append(glosses[i])
                i += 1
                while i < n and is_spelling(glosses[i]):
                    group.append(glosses[i])
                    i += 1
            lugar.extend(group)
            continue

        if g in HEAD_SPELL or (g == "DOCUMENTO" and i + 1 < n and glosses[i + 1] == "NUMERO"):
            group, i = collect_spelling_group(glosses, i)
            objeto.extend(group)
            continue

        if g in {"AÑOS", "ANOS"}:
            group, i = collect_edad_group(glosses, i)
            objeto.extend(group)
            continue

        if g in LUGAR_CON_DELETREO:
            group = [g]
            i += 1
            while i < n and is_spelling(glosses[i]):
                group.append(glosses[i])
                i += 1
            lugar.extend(group)
            continue

        if g in SALUDO:
            saludo.append(g)
        elif g in TIEMPO:
            tiempo.append(g)
        elif g in LUGAR:
            lugar.append(g)
        elif g in SUJETO:
            sujeto.append(g)
        elif g in VERBO:
            verbo.append(g)
        elif g in NEG_AFF:
            neg_aff.append(g)
        elif g in PREGUNTA:
            pregunta.append(g)
        elif g in OBJETO:
            objeto.append(g)
        elif is_spelling(g):
            objeto.append(g)
        else:
            objeto.append(g)
        i += 1

    # HOLA/CHAU + BIEN/MAL forman un bloque de saludo al inicio
    greeting_aff = [x for x in neg_aff if x in {"BIEN", "MAL"} and saludo]
    other_aff = [x for x in neg_aff if x not in {"BIEN", "MAL"} or not saludo]
    if greeting_aff and saludo[0] in {"HOLA", "CHAU"}:
        saludo = saludo + greeting_aff
        neg_aff = other_aff

    return saludo + tiempo + lugar + sujeto + objeto + verbo + neg_aff + pregunta


def main() -> None:
    path = Path(__file__).parent / "dataset_glosas1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for item in data["dataset"]:
        original = item["glosses"]
        reordered = reorder_glosses(original)
        if reordered != original:
            changed += 1
            item["glosses"] = reordered
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Procesados {len(data['dataset'])} ejemplos, reordenados {changed}.")


if __name__ == "__main__":
    main()
