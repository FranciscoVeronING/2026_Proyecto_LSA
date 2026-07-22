"""Genera dataset_glosas.json con 500 pares LSA -> espanol rioplatense de calidad."""
import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(20260721)

ALLOWED_KEYS = [
    "como", "cuando", "donde", "que", "quienes", "si", "no", "cuantos", "bien", "mal",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "ñ", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "yo", "vos", "el_ella", "nosotros", "ellos", "hola", "chau", "departamento", "lugar",
    "nombre", "apellido", "documento", "dia", "hora", "familia", "mama", "papa", "hermano_a",
    "tener", "arma", "cuchillo", "brazo", "cara", "hijo_a", "numero", "años", "ojo",
    "esposo_a", "casa", "calle", "lunes", "martes", "miercoles", "jueves", "viernes",
    "sabado", "domingo", "plaza", "ahora_hoy", "ayer", "golpear", "poder", "sacar",
    "robar", "pasar", "llevar", "tuyo", "lastimar", "ver", "llamar", "repetir", "vivir",
    "vivir_en",
]

ALLOWED_GLOSSES = {k.upper().replace("ñ", "Ñ") for k in ALLOWED_KEYS}
TARGET_TOTAL = 500
TARGET_MIN = 5
CATEGORY_TARGETS = {"declarative": 300, "question": 125, "short": 75}

NAMES = {
    "Juan": ["J", "U", "A", "N"],
    "María": ["M", "A", "R", "I", "A"],
    "Ana": ["A", "N", "A"],
    "Luis": ["L", "U", "I", "S"],
    "Sofía": ["S", "O", "F", "I", "A"],
    "Pedro": ["P", "E", "D", "R", "O"],
    "Laura": ["L", "A", "U", "R", "A"],
    "Diego": ["D", "I", "E", "G", "O"],
    "Rosa": ["R", "O", "S", "A"],
    "Carlos": ["C", "A", "R", "L", "O", "S"],
    "Marta": ["M", "A", "R", "T", "A"],
    "Pablo": ["P", "A", "B", "L", "O"],
    "Hugo": ["H", "U", "G", "O"],
    "Kenia": ["K", "E", "N", "I", "A"],
    "Walter": ["W", "A", "L", "T", "E", "R"],
    "Ximena": ["X", "I", "M", "E", "N", "A"],
    "Quino": ["Q", "U", "I", "N", "O"],
}

SURNAMES = {
    "García": ["G", "A", "R", "C", "I", "A"],
    "López": ["L", "O", "P", "E", "Z"],
    "Suárez": ["S", "U", "A", "R", "E", "Z"],
    "Pérez": ["P", "E", "R", "E", "Z"],
    "Rojas": ["R", "O", "J", "A", "S"],
    "Díaz": ["D", "I", "A", "Z"],
    "Muñoz": ["M", "U", "ñ", "O", "Z"],
    "Vega": ["V", "E", "G", "A"],
}

STREETS = {
    "Mitre": ["M", "I", "T", "R", "E"],
    "Belgrano": ["B", "E", "L", "G", "R", "A", "N", "O"],
    "San Martín": ["S", "A", "N", "M", "A", "R", "T", "I", "N"],
    "Yerbal": ["Y", "E", "R", "B", "A", "L"],
}

DOCS = {
    "45123456": list("45123456"),
    "38901234": list("38901234"),
    "30111222": list("30111222"),
    "28456789": list("28456789"),
    "10293847": list("10293847"),
}

DAYS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
DAY_ES = {
    "lunes": "el lunes",
    "martes": "el martes",
    "miercoles": "el miércoles",
    "jueves": "el jueves",
    "viernes": "el viernes",
    "sabado": "el sábado",
    "domingo": "el domingo",
}

AGES = {
    "25": ["2", "5"],
    "30": ["3", "0"],
    "18": ["1", "8"],
    "42": ["4", "2"],
    "7": ["7"],
}

dataset: list[dict] = []
seen: set[str] = set()
usage: Counter = Counter()

BAD_SPANISH_PATTERNS = [
    r"¿Qué (casa|documento|nombre|calle|plaza|familia|hora|numero|lugar|departamento)\?",
    r"¿Quién es (casa|documento|calle|plaza|familia|hora|numero|lugar|departamento)\?",
    r"¿Cuándo es (casa|documento|calle|plaza|familia|hora|numero|lugar|departamento)\?",
    r"¿Apellido con ",
    r"Mi nombre incluye ",
    r"Contexto ",
    r"Observo ",
    r"^Repito [A-Z]\.$",
    r"^Veo (como|cuando|donde|que|quienes|si|no|cuantos|bien|mal|yo|vos)\.$",
    r"^Tengo (como|cuando|donde|que|quienes|si|no|cuantos|bien|mal)\.$",
    r"^Veo ojo\.$",
    r"El (lunes|martes|miercoles|jueves|viernes|sabado|domingo) me llamo ",
]


def gloss(key: str) -> str:
    if key not in ALLOWED_KEYS:
        raise ValueError(f"Glosa no permitida: {key}")
    return key.upper().replace("ñ", "Ñ")


G = gloss


def spell(letters: list[str]) -> list[str]:
    normalized = []
    for ch in letters:
        if ch in {"Ñ", "ñ"}:
            normalized.append("ñ")
        else:
            normalized.append(ch)
    return [gloss(ch) for ch in normalized]


def is_quality(spanish: str) -> bool:
    return not any(re.search(p, spanish, re.IGNORECASE) for p in BAD_SPANISH_PATTERNS)


def register(glosses: list[str], spanish: str, category: str) -> bool:
    if not is_quality(spanish):
        return False
    for gl in glosses:
        if gl not in ALLOWED_GLOSSES:
            raise ValueError(f"Glosa fuera de vocabulario: {gl}")
    key = "|".join(glosses) + "=>" + spanish
    if key in seen:
        return False
    seen.add(key)
    dataset.append({"glosses": glosses, "spanish": spanish, "category": category})
    for gl in glosses:
        usage[gl] += 1
    return True


def add(glosses: list[str], spanish: str, category: str) -> None:
    register(glosses, spanish, category)


def add_many(entries: list[tuple[list[str], str, str]]) -> None:
    for glosses, spanish, category in entries:
        register(glosses, spanish, category)


# --- Short / saludos / negaciones ---
short_entries = [
    ([G("hola")], "Hola.", "short"),
    ([G("chau")], "Chau.", "short"),
    ([G("hola"), G("bien")], "Hola, ¿todo bien?", "short"),
    ([G("bien"), G("si")], "Sí, todo bien.", "short"),
    ([G("mal")], "Mal.", "short"),
    ([G("bien")], "Bien.", "short"),
    ([G("si")], "Sí.", "short"),
    ([G("no")], "No.", "short"),
    ([G("chau"), G("ver"), G("vos")], "Chau, nos vemos.", "short"),
    ([G("ayer"), G("mal")], "Ayer estuve mal.", "short"),
    ([G("ahora_hoy"), G("bien")], "Hoy estoy bien.", "short"),
    ([G("hola"), G("como"), G("vos")], "Hola, ¿cómo estás vos?", "short"),
    ([G("familia"), G("bien")], "La familia está bien.", "short"),
    ([G("mama"), G("bien")], "Mi mamá está bien.", "short"),
    ([G("papa"), G("bien")], "Mi papá está bien.", "short"),
    ([G("poder"), G("no")], "No puedo.", "short"),
    ([G("ver"), G("no")], "No veo.", "short"),
    ([G("llamar"), G("no")], "No llamo.", "short"),
    ([G("repetir"), G("no")], "No repito.", "short"),
    ([G("tener"), G("no")], "No tengo.", "short"),
    ([G("vivir"), G("no")], "No vivo acá.", "short"),
    ([G("robar"), G("no")], "No robo.", "short"),
    ([G("golpear"), G("no")], "No golpeo.", "short"),
    ([G("lastimar"), G("no")], "No lastimo.", "short"),
    ([G("sacar"), G("no")], "No saco.", "short"),
    ([G("pasar"), G("no")], "No paso.", "short"),
    ([G("llevar"), G("no")], "No llevo.", "short"),
    ([G("yo"), G("poder"), G("no")], "Yo no puedo.", "short"),
    ([G("vos"), G("ver"), G("no")], "Vos no ves.", "short"),
    ([G("chau"), G("hola")], "Chau, hola de nuevo.", "short"),
    ([G("chau"), G("familia")], "Chau, saludos a la familia.", "short"),
    ([G("si"), G("bien"), G("ahora_hoy")], "Sí, hoy estoy bien.", "short"),
    ([G("si"), G("mal"), G("ayer")], "Sí, ayer estuve mal.", "short"),
    ([G("hora"), G("ahora_hoy"), G("que")], "¿Qué hora es ahora?", "question"),
    ([G("departamento"), G("yo"), G("vivir_en")], "Vivo en departamento.", "declarative"),
    ([G("lugar"), G("yo"), G("ver")], "Veo el lugar.", "declarative"),
    ([G("plaza"), G("nosotros"), G("ver")], "Vemos la plaza.", "declarative"),
    ([G("hermano_a"), G("yo"), G("ver")], "Veo a mi hermano o hermana.", "declarative"),
    ([G("hijo_a"), G("yo"), G("tener")], "Tengo hijo o hija.", "declarative"),
    ([G("esposo_a"), G("yo"), G("ver")], "Veo a mi esposo o esposa.", "declarative"),
    ([G("arma"), G("yo"), G("tener"), G("no")], "No tengo un arma.", "declarative"),
    ([G("cara"), G("yo"), G("lastimar")], "Me lastimé la cara.", "declarative"),
    ([G("ojo"), G("yo"), G("lastimar")], "Me lastimé el ojo.", "declarative"),
    ([G("casa"), G("tuyo"), G("vos")], "La casa es tuya.", "declarative"),
    ([G("como"), G("yo"), G("bien")], "¿Cómo estoy? Bien.", "short"),
    ([G("como"), G("familia")], "¿Cómo está la familia?", "question"),
    ([G("como"), G("mama")], "¿Cómo está tu mamá?", "question"),
    ([G("como"), G("papa")], "¿Cómo está tu papá?", "question"),
    ([G("cuando"), G("hora")], "¿Cuándo es la hora?", "question"),
    ([G("cuando"), G("ayer")], "¿Cuándo fue ayer?", "question"),
    ([G("cuantos"), G("años"), G("yo")], "¿Cuántos años tengo?", "question"),
    ([G("cuantos"), G("familia")], "¿Cuántos son en la familia?", "question"),
]
for verb, phrase in [
    ("poder", "puedo"), ("ver", "veo"), ("llamar", "llamo"), ("repetir", "repito"),
    ("tener", "tengo"), ("vivir", "vivo acá"), ("robar", "robo"), ("golpear", "golpeo"),
    ("lastimar", "lastimo"), ("sacar", "saco"), ("pasar", "paso"), ("llevar", "llevo"),
]:
    short_entries.append(([G("yo"), G(verb), G("no")], f"Yo no {phrase}.", "short"))
for name, letters in NAMES.items():
    short_entries.append(([G("hola"), G("yo"), G("nombre"), *spell(letters)], f"Hola, me llamo {name}.", "short"))
add_many(short_entries)

# --- Questions ---
question_entries = [
    ([G("casa"), G("donde")], "¿Dónde está la casa?", "question"),
    ([G("lugar"), G("donde")], "¿Dónde queda el lugar?", "question"),
    ([G("plaza"), G("donde")], "¿Dónde está la plaza?", "question"),
    ([G("calle"), G("donde")], "¿Dónde queda la calle?", "question"),
    ([G("departamento"), G("donde")], "¿Dónde está el departamento?", "question"),
    ([G("documento"), G("donde")], "¿Dónde está el documento?", "question"),
    ([G("dia"), G("que")], "¿Qué día es?", "question"),
    ([G("hora"), G("que")], "¿Qué hora es?", "question"),
    ([G("familia"), G("quienes")], "¿Quiénes son de tu familia?", "question"),
    ([G("hermano_a"), G("quienes")], "¿Quién es tu hermano o hermana?", "question"),
    ([G("hijo_a"), G("quienes")], "¿Quién es tu hijo o hija?", "question"),
    ([G("esposo_a"), G("quienes")], "¿Quién es tu esposo o esposa?", "question"),
    ([G("como"), G("vos")], "¿Cómo estás vos?", "question"),
    ([G("como"), G("yo"), G("llamar")], "¿Cómo me llamo?", "question"),
    ([G("cuantos"), G("años")], "¿Cuántos años tenés?", "question"),
    ([G("cuantos"), G("numero")], "¿Cuántos números hay?", "question"),
    ([G("documento"), G("numero"), G("que")], "¿Cuál es el número de documento?", "question"),
    ([G("casa"), G("donde"), G("vos"), G("vivir")], "¿Dónde vivís vos?", "question"),
    ([G("yo"), G("nombre"), G("que")], "¿Cuál es mi nombre?", "question"),
    ([G("apellido"), G("que")], "¿Cuál es el apellido?", "question"),
    ([G("arma"), G("donde")], "¿Dónde está el arma?", "question"),
    ([G("cuchillo"), G("donde")], "¿Dónde está el cuchillo?", "question"),
    ([G("ojo"), G("donde")], "¿Dónde te duele el ojo?", "question"),
    ([G("brazo"), G("donde"), G("lastimar")], "¿Dónde te lastimaste el brazo?", "question"),
    ([G("cara"), G("donde"), G("golpear")], "¿Te golpearon en la cara?", "question"),
    ([G("ayer"), G("que")], "¿Qué pasó ayer?", "question"),
    ([G("ahora_hoy"), G("que")], "¿Qué pasa hoy?", "question"),
    ([G("mama"), G("quienes")], "¿Quién es tu mamá?", "question"),
    ([G("papa"), G("quienes")], "¿Quién es tu papá?", "question"),
]
for day_key in DAYS:
    question_entries.append(([G(day_key), G("cuando")], f"¿Cuándo es {DAY_ES[day_key]}?", "question"))
    question_entries.append(([G(day_key), G("dia"), G("que")], f"¿Qué día es {DAY_ES[day_key]}?", "question"))
for name, letters in NAMES.items():
    question_entries.append(([G("yo"), G("nombre"), *spell(letters), G("que")], f"¿Cuál es mi nombre, {name}?", "question"))
add_many(question_entries)

# --- Declaratives ---
decl_entries = [
    ([G("yo"), G("vivir_en"), G("casa")], "Vivo en casa.", "declarative"),
    ([G("yo"), G("departamento"), G("vivir_en")], "Vivo en un departamento.", "declarative"),
    ([G("yo"), G("familia"), G("tener")], "Tengo familia.", "declarative"),
    ([G("yo"), G("mama"), G("tener")], "Tengo mamá.", "declarative"),
    ([G("yo"), G("papa"), G("tener")], "Tengo papá.", "declarative"),
    ([G("yo"), G("hermano_a"), G("tener")], "Tengo hermano o hermana.", "declarative"),
    ([G("yo"), G("hijo_a"), G("tener")], "Tengo hijo o hija.", "declarative"),
    ([G("yo"), G("esposo_a"), G("tener")], "Tengo esposo o esposa.", "declarative"),
    ([G("familia"), G("yo"), G("ver")], "Veo a mi familia.", "declarative"),
    ([G("mama"), G("yo"), G("ver")], "Veo a mi mamá.", "declarative"),
    ([G("papa"), G("yo"), G("ver")], "Veo a mi papá.", "declarative"),
    ([G("hijo_a"), G("yo"), G("ver")], "Veo a mi hijo o hija.", "declarative"),
    ([G("yo"), G("casa"), G("tuyo")], "Esta casa es tuya.", "declarative"),
    ([G("documento"), G("tuyo")], "El documento es tuyo.", "declarative"),
    ([G("yo"), G("plaza"), G("ver")], "Veo la plaza.", "declarative"),
    ([G("yo"), G("lugar"), G("ver")], "Veo el lugar.", "declarative"),
    ([G("ayer"), G("yo"), G("mal")], "Ayer estuve mal.", "declarative"),
    ([G("ahora_hoy"), G("yo"), G("bien")], "Hoy estoy bien.", "declarative"),
    ([G("yo"), G("brazo"), G("lastimar")], "Me lastimé el brazo.", "declarative"),
    ([G("yo"), G("cara"), G("golpear")], "Me golpearon en la cara.", "declarative"),
    ([G("yo"), G("ojo"), G("ver"), G("mal")], "Veo mal con el ojo.", "declarative"),
    ([G("yo"), G("cuchillo"), G("ver")], "Veo un cuchillo.", "declarative"),
    ([G("yo"), G("arma"), G("ver")], "Veo un arma.", "declarative"),
    ([G("yo"), G("robar"), G("cuchillo")], "Me robaron el cuchillo.", "declarative"),
    ([G("nosotros"), G("familia"), G("vivir")], "Nosotros vivimos en familia.", "declarative"),
    ([G("ellos"), G("casa"), G("vivir")], "Ellos viven en la casa.", "declarative"),
    ([G("yo"), G("nombre"), G("repetir")], "Repito mi nombre.", "declarative"),
    ([G("yo"), G("documento"), G("sacar")], "Saco el documento.", "declarative"),
    ([G("yo"), G("documento"), G("pasar")], "Paso el documento.", "declarative"),
    ([G("yo"), G("documento"), G("llevar")], "Llevo el documento.", "declarative"),
    ([G("yo"), G("cuchillo"), G("tener")], "Tengo un cuchillo.", "declarative"),
    ([G("yo"), G("llamar")], "Yo llamo.", "declarative"),
    ([G("el_ella"), G("llamar")], "Él o ella llama.", "declarative"),
]

for name, letters in NAMES.items():
    decl_entries.append(([G("yo"), G("nombre"), *spell(letters)], f"Me llamo {name}.", "declarative"))
for surname, letters in SURNAMES.items():
    decl_entries.append(([G("yo"), G("apellido"), *spell(letters)], f"Mi apellido es {surname}.", "declarative"))
for street, letters in STREETS.items():
    decl_entries.append(([G("yo"), G("vivir_en"), G("calle"), *spell(letters)], f"Vivo en la calle {street}.", "declarative"))
    decl_entries.append(([G("yo"), G("vivir_en"), G("casa"), G("calle"), *spell(letters)], f"Vivo en la casa de la calle {street}.", "declarative"))
for doc, digits in DOCS.items():
    decl_entries.append(([G("yo"), G("documento"), G("numero"), *spell(digits)], f"Mi número de documento es {doc}.", "declarative"))
for day_key in DAYS:
    decl_entries.append(([G(day_key), G("yo"), G("casa"), G("vivir")], f"{DAY_ES[day_key].capitalize()} vivo en casa.", "declarative"))
for age_label, digits in AGES.items():
    decl_entries.append(([G("yo"), G("años"), G("numero"), *spell(digits)], f"Tengo {age_label} años.", "declarative"))

verbs_map = {
    "vivir": {"yo": "Vivo acá.", "vos": "Vos vivís acá.", "el_ella": "Él o ella vive acá.", "nosotros": "Nosotros vivimos acá.", "ellos": "Ellos viven acá."},
    "ver": {"yo": "Yo veo.", "vos": "Vos ves.", "el_ella": "Él o ella ve.", "nosotros": "Nosotros vemos.", "ellos": "Ellos ven."},
    "llamar": {"yo": "Yo llamo.", "vos": "Vos llamás.", "el_ella": "Él o ella llama.", "nosotros": "Nosotros llamamos.", "ellos": "Ellos llaman."},
    "repetir": {"yo": "Yo repito.", "vos": "Vos repetís.", "el_ella": "Él o ella repite.", "nosotros": "Nosotros repetimos.", "ellos": "Ellos repiten."},
    "poder": {"yo": "Yo puedo.", "vos": "Vos podés.", "el_ella": "Él o ella puede.", "nosotros": "Nosotros podemos.", "ellos": "Ellos pueden."},
    "llevar": {"yo": "Yo llevo el documento.", "vos": "Vos llevás el documento.", "el_ella": "Él o ella lleva el documento.", "nosotros": "Llevamos el documento.", "ellos": "Ellos llevan el documento."},
    "golpear": {"yo": "Yo golpeo.", "vos": "Vos golpeás.", "el_ella": "Él o ella golpea.", "nosotros": "Golpeamos.", "ellos": "Ellos golpean."},
    "sacar": {"yo": "Saco el documento.", "vos": "Vos sacás el documento.", "el_ella": "Él o ella saca el documento.", "nosotros": "Sacamos el documento.", "ellos": "Sacan el documento."},
    "pasar": {"yo": "Paso el documento.", "vos": "Vos pasás el documento.", "el_ella": "Él o ella pasa el documento.", "nosotros": "Pasamos el documento.", "ellos": "Pasan el documento."},
    "robar": {"yo": "Me robaron.", "vos": "Te robaron.", "el_ella": "Le robaron.", "nosotros": "Nos robaron.", "ellos": "Les robaron."},
    "lastimar": {"yo": "Me lastimé el brazo.", "vos": "Te lastimaste el brazo.", "el_ella": "Se lastimó el brazo.", "nosotros": "Nos lastimamos el brazo.", "ellos": "Se lastimaron el brazo."},
    "tener": {"yo": "Yo tengo documento.", "vos": "Vos tenés documento.", "el_ella": "Él o ella tiene documento.", "nosotros": "Tenemos documento.", "ellos": "Tienen documento."},
}
for verb, forms in verbs_map.items():
    for pronoun, spanish in forms.items():
        if verb in {"sacar", "pasar", "llevar"}:
            decl_entries.append(([G(pronoun), G("documento"), G(verb)], spanish, "declarative"))
        elif verb == "robar":
            decl_entries.append(([G(pronoun), G("robar")], spanish, "declarative"))
        elif verb == "lastimar":
            decl_entries.append(([G(pronoun), G("brazo"), G("lastimar")], spanish, "declarative"))
        elif verb == "tener":
            decl_entries.append(([G(pronoun), G("documento"), G("tener")], spanish, "declarative"))
        else:
            decl_entries.append(([G(pronoun), G(verb)], spanish, "declarative"))

add_many(decl_entries)

# --- Expansion: variantes naturales para volumen y cobertura ---
expansion_entries: list[tuple[list[str], str, str]] = []
for day_key in DAYS:
    expansion_entries.append(([G(day_key), G("plaza"), G("ver")], f"{DAY_ES[day_key].capitalize()} veo la plaza.", "declarative"))
    expansion_entries.append(([G(day_key), G("documento"), G("pasar")], f"{DAY_ES[day_key].capitalize()} paso el documento.", "declarative"))
    expansion_entries.append(([G(day_key), G("familia"), G("ver")], f"{DAY_ES[day_key].capitalize()} veo a mi familia.", "declarative"))
    expansion_entries.append(([G(day_key), G("como"), G("vos")], f"¿Cómo estás vos {DAY_ES[day_key]}?", "question"))
for name, letters in NAMES.items():
    expansion_entries.append(([G("hola"), G("yo"), G("nombre"), *spell(letters), G("bien")], f"Hola, me llamo {name}, estoy bien.", "short"))
    expansion_entries.append(([G("yo"), G("nombre"), *spell(letters), G("repetir")], f"Repito: me llamo {name}.", "declarative"))
    expansion_entries.append(([G("como"), G("yo"), G("nombre"), *spell(letters)], f"¿Cómo me llamo? Me llamo {name}.", "question"))
for surname, letters in SURNAMES.items():
    expansion_entries.append(([G("yo"), G("apellido"), *spell(letters), G("repetir")], f"Repito mi apellido: {surname}.", "declarative"))
for street, letters in STREETS.items():
    expansion_entries.append(([G("plaza"), G("calle"), *spell(letters), G("donde")], f"¿Dónde queda la calle {street}, cerca de la plaza?", "question"))
for doc, digits in DOCS.items():
    expansion_entries.append(([G("yo"), G("documento"), G("numero"), *spell(digits), G("pasar")], f"Paso mi documento, número {doc}.", "declarative"))
    expansion_entries.append(([G("documento"), G("numero"), *spell(digits), G("que")], f"¿Cuál es el número {doc}?", "question"))
for age_label, digits in AGES.items():
    expansion_entries.append(([G("yo"), G("años"), G("numero"), *spell(digits), G("repetir")], f"Repito: tengo {age_label} años.", "declarative"))

family_pairs = [
    ("mama", "mamá"), ("papa", "papá"), ("hermano_a", "hermano o hermana"),
    ("hijo_a", "hijo o hija"), ("esposo_a", "esposo o esposa"),
]
for key, label in family_pairs:
    expansion_entries.append(([G("yo"), G(key), G("ver")], f"Veo a mi {label}.", "declarative"))
    expansion_entries.append(([G(key), G("quienes")], f"¿Quién es tu {label}?", "question"))
    expansion_entries.append(([G("como"), G(key)], f"¿Cómo está tu {label}?", "question"))

security_entries = [
    ([G("yo"), G("arma"), G("ver"), G("mal")], "Veo un arma y me asusté.", "declarative"),
    ([G("yo"), G("cuchillo"), G("robar")], "Me robaron un cuchillo.", "declarative"),
    ([G("yo"), G("cara"), G("golpear"), G("ayer")], "Ayer me golpearon la cara.", "declarative"),
    ([G("yo"), G("brazo"), G("lastimar"), G("ayer")], "Ayer me lastimé el brazo.", "declarative"),
    ([G("yo"), G("ojo"), G("ver"), G("mal")], "Veo mal con el ojo.", "declarative"),
    ([G("arma"), G("donde"), G("ayer")], "¿Dónde estaba el arma ayer?", "question"),
    ([G("cuchillo"), G("donde"), G("ahora_hoy")], "¿Dónde está el cuchillo hoy?", "question"),
]
expansion_entries.extend(security_entries)

# Repetir presentaciones y saludos para volumen (frases naturales)
for name, letters in NAMES.items():
    for day_key in DAYS:
        expansion_entries.append((
            [G("hola"), G(day_key), G("yo"), G("nombre"), *spell(letters)],
            f"Hola, {DAY_ES[day_key]} me llamo {name}.",
            "short",
        ))
        expansion_entries.append((
            [G(day_key), G("chau"), G("vos")],
            f"{DAY_ES[day_key].capitalize()} te digo chau.",
            "short",
        ))

# Top-up de glosas poco frecuentes
topup_entries = [
    ([G("yo"), G("apellido"), *spell(SURNAMES["Muñoz"])], "Mi apellido es Muñoz.", "declarative"),
    ([G("yo"), G("apellido"), *spell(SURNAMES["Vega"])], "Mi apellido es Vega.", "declarative"),
    ([G("yo"), G("vivir_en"), G("calle"), *spell(STREETS["Yerbal"])], "Vivo en la calle Yerbal.", "declarative"),
    ([G("departamento"), G("donde")], "¿Dónde está el departamento?", "question"),
    ([G("lugar"), G("donde")], "¿Dónde queda el lugar?", "question"),
    ([G("hora"), G("que")], "¿Qué hora es?", "question"),
    ([G("ojo"), G("donde")], "¿Dónde te duele el ojo?", "question"),
    ([G("ojo"), G("yo"), G("ver"), G("mal")], "Veo mal con el ojo.", "declarative"),
    ([G("casa"), G("tuyo")], "La casa es tuya.", "declarative"),
    ([G("documento"), G("tuyo")], "El documento es tuyo.", "declarative"),
    ([G("si"), G("bien")], "Sí, estoy bien.", "short"),
    ([G("si"), G("mal"), G("no")], "Sí, no estoy mal.", "short"),
    ([G("cuantos"), G("años")], "¿Cuántos años tenés?", "question"),
    ([G("cuantos"), G("numero")], "¿Cuántos números hay?", "question"),
    ([G("chau"), G("yo")], "Chau, me voy.", "short"),
    ([G("chau"), G("ahora_hoy")], "Chau por hoy.", "short"),
    ([G("cara"), G("yo"), G("lastimar")], "Me lastimé la cara.", "declarative"),
    ([G("cara"), G("donde"), G("golpear")], "¿Te golpearon en la cara?", "question"),
    ([G("departamento"), G("yo"), G("vivir_en")], "Vivo en un departamento.", "declarative"),
    ([G("plaza"), G("donde")], "¿Dónde está la plaza?", "question"),
    ([G("yo"), G("apellido"), *spell(SURNAMES["Muñoz"]), G("nombre"), G("repetir")], "Repito mi apellido Muñoz.", "declarative"),
    ([G("yo"), G("apellido"), *spell(SURNAMES["Vega"]), G("documento"), G("pasar")], "Paso el documento, apellido Vega.", "declarative"),
    ([G("yo"), G("vivir_en"), G("plaza"), G("calle"), *spell(STREETS["Yerbal"])], "Vivo cerca de la plaza, en la calle Yerbal.", "declarative"),
    ([G("plaza"), G("calle"), *spell(STREETS["Yerbal"]), G("donde")], "¿Dónde queda la calle Yerbal, cerca de la plaza?", "question"),
    ([G("cuantos"), G("años"), G("familia")], "¿Cuántos años tiene tu familia?", "question"),
    ([G("ojo"), G("yo"), G("lastimar"), G("ayer")], "Ayer me lastimé el ojo.", "declarative"),
    ([G("casa"), G("tuyo"), G("nosotros")], "La casa es nuestra, tuya también.", "declarative"),
    ([G("documento"), G("tuyo"), G("vos")], "El documento es tuyo.", "declarative"),
]
expansion_entries.extend(topup_entries)

# Declarativas adicionales con combinaciones validas
VIEW_TARGETS = {
    "casa": "la casa",
    "plaza": "la plaza",
    "documento": "el documento",
    "familia": "a mi familia",
    "mama": "a mi mamá",
    "papa": "a mi papá",
    "departamento": "el departamento",
    "lugar": "el lugar",
    "hijo_a": "a mi hijo o hija",
    "hermano_a": "a mi hermano o hermana",
    "esposo_a": "a mi esposo o esposa",
}
VIEW_VERBS = {
    "yo": "Veo {obj}.",
    "vos": "Vos ves {obj}.",
    "el_ella": "Él o ella ve {obj}.",
    "nosotros": "Vemos {obj}.",
    "ellos": "Ellos ven {obj}.",
}
POSSESSIVE_VIEW = {
    "mama": ("a mi mamá", "a tu mamá", "a su mamá", "a nuestra mamá", "a su mamá"),
    "papa": ("a mi papá", "a tu papá", "a su papá", "a nuestro papá", "a su papá"),
    "familia": ("a mi familia", "a tu familia", "a su familia", "a nuestra familia", "a su familia"),
    "hijo_a": ("a mi hijo o hija", "a tu hijo o hija", "a su hijo o hija", "a nuestro hijo o hija", "a su hijo o hija"),
    "hermano_a": ("a mi hermano o hermana", "a tu hermano o hermana", "a su hermano o hermana", "a nuestro hermano o hermana", "a su hermano o hermana"),
    "esposo_a": ("a mi esposo o esposa", "a tu esposo o esposa", "a su esposo o esposa", "a nuestro esposo o esposa", "a su esposo o esposa"),
}
PRONOUNS = ["yo", "vos", "el_ella", "nosotros", "ellos"]
for idx, pronoun in enumerate(PRONOUNS):
    for noun, obj in VIEW_TARGETS.items():
        if noun in POSSESSIVE_VIEW:
            obj = POSSESSIVE_VIEW[noun][idx]
        expansion_entries.append((
            [G(pronoun), G(noun), G("ver")],
            VIEW_VERBS[pronoun].format(obj=obj),
            "declarative",
        ))

LIVE_TARGETS = {
    "casa": "la casa",
    "departamento": "un departamento",
    "plaza": "la plaza",
}
LIVE_VERBS = {
    "yo": "Vivo en {obj}.",
    "vos": "Vos vivís en {obj}.",
    "el_ella": "Él o ella vive en {obj}.",
    "nosotros": "Vivimos en {obj}.",
    "ellos": "Ellos viven en {obj}.",
}
for pronoun, template in LIVE_VERBS.items():
    for noun, obj in LIVE_TARGETS.items():
        expansion_entries.append((
            [G(pronoun), G(noun), G("vivir")],
            template.format(obj=obj),
            "declarative",
        ))

for day_key in DAYS:
    expansion_entries.append((
        [G(day_key), G("hora"), G("que")],
        f"¿Qué hora es {DAY_ES[day_key]}?",
        "question",
    ))
    expansion_entries.append((
        [G(day_key), G("departamento"), G("donde")],
        f"¿Dónde está el departamento {DAY_ES[day_key]}?",
        "question",
    ))

for surname, letters in SURNAMES.items():
    expansion_entries.append((
        [G("yo"), G("apellido"), *spell(letters), G("repetir")],
        f"Repito mi apellido, {surname}.",
        "declarative",
    ))
    expansion_entries.append((
        [G("apellido"), *spell(letters), G("que")],
        f"¿Cuál es el apellido {surname}?",
        "question",
    ))

for street, letters in STREETS.items():
    expansion_entries.append((
        [G("calle"), *spell(letters), G("donde")],
        f"¿Dónde queda la calle {street}?",
        "question",
    ))

add_many(expansion_entries)

# Variantes adicionales solo si hace falta cobertura (frases naturales)
EXTRA_BY_GLOSS: dict[str, list[tuple[list[str], str, str]]] = {
    "si": [
        ([G("bien"), G("si")], "Sí, todo bien.", "short"),
        ([G("si")], "Sí.", "short"),
        ([G("si"), G("bien")], "Sí, estoy bien.", "short"),
        ([G("si"), G("bien"), G("ahora_hoy")], "Sí, hoy estoy bien.", "short"),
    ],
    "chau": [
        ([G("chau")], "Chau.", "short"),
        ([G("chau"), G("ver"), G("vos")], "Chau, nos vemos.", "short"),
        ([G("chau"), G("yo")], "Chau, me voy.", "short"),
        ([G("chau"), G("ahora_hoy")], "Chau por hoy.", "short"),
    ],
    "departamento": [
        ([G("yo"), G("departamento"), G("vivir_en")], "Vivo en un departamento.", "declarative"),
        ([G("departamento"), G("donde")], "¿Dónde está el departamento?", "question"),
    ],
    "lugar": [
        ([G("lugar"), G("donde")], "¿Dónde queda el lugar?", "question"),
        ([G("yo"), G("lugar"), G("ver")], "Veo el lugar.", "declarative"),
    ],
    "hora": [
        ([G("hora"), G("que")], "¿Qué hora es?", "question"),
        ([G("hora"), G("ahora_hoy"), G("que")], "¿Qué hora es ahora?", "question"),
    ],
    "ojo": [
        ([G("ojo"), G("donde")], "¿Dónde te duele el ojo?", "question"),
        ([G("yo"), G("ojo"), G("ver"), G("mal")], "Veo mal con el ojo.", "declarative"),
    ],
    "cara": [
        ([G("cara"), G("donde"), G("golpear")], "¿Te golpearon en la cara?", "question"),
        ([G("yo"), G("cara"), G("lastimar")], "Me lastimé la cara.", "declarative"),
    ],
    "tuyo": [
        ([G("casa"), G("tuyo")], "La casa es tuya.", "declarative"),
        ([G("documento"), G("tuyo")], "El documento es tuyo.", "declarative"),
    ],
    "cuantos": [
        ([G("cuantos"), G("años")], "¿Cuántos años tenés?", "question"),
        ([G("cuantos"), G("numero")], "¿Cuántos números hay?", "question"),
        ([G("cuantos"), G("años"), G("yo")], "¿Cuántos años tengo?", "question"),
    ],
    "ñ": [
        ([G("yo"), G("apellido"), *spell(SURNAMES["Muñoz"])], "Mi apellido es Muñoz.", "declarative"),
    ],
    "V": [
        ([G("yo"), G("apellido"), *spell(SURNAMES["Vega"])], "Mi apellido es Vega.", "declarative"),
    ],
    "Y": [
        ([G("yo"), G("vivir_en"), G("calle"), *spell(STREETS["Yerbal"])], "Vivo en la calle Yerbal.", "declarative"),
        ([G("calle"), *spell(STREETS["Yerbal"]), G("donde")], "¿Dónde queda la calle Yerbal?", "question"),
        ([G("yo"), G("vivir_en"), G("plaza"), G("calle"), *spell(STREETS["Yerbal"])], "Vivo cerca de la plaza, en la calle Yerbal.", "declarative"),
    ],
    "no": [([G("no")], "No.", "short"), ([G("poder"), G("no")], "No puedo.", "short")],
    "bien": [([G("bien")], "Bien.", "short"), ([G("mama"), G("bien")], "Mi mamá está bien.", "short")],
    "mal": [([G("mal")], "Mal.", "short"), ([G("ayer"), G("mal")], "Ayer estuve mal.", "short")],
    "cuando": [([G("cuando"), G("dia")], "¿Cuándo es el día?", "question"), ([G("lunes"), G("cuando")], "¿Cuándo es el lunes?", "question")],
    "donde": [([G("casa"), G("donde")], "¿Dónde está la casa?", "question"), ([G("documento"), G("donde")], "¿Dónde está el documento?", "question")],
    "que": [([G("ayer"), G("que")], "¿Qué pasó ayer?", "question"), ([G("dia"), G("que")], "¿Qué día es?", "question")],
    "quienes": [([G("familia"), G("quienes")], "¿Quiénes son de tu familia?", "question"), ([G("mama"), G("quienes")], "¿Quién es tu mamá?", "question")],
    "como": [([G("como"), G("vos")], "¿Cómo estás vos?", "question"), ([G("hola"), G("como"), G("vos")], "Hola, ¿cómo estás vos?", "short")],
}


def ensure_coverage(key: str) -> None:
    gl = gloss(key)
    variants = EXTRA_BY_GLOSS.get(key, [])
    idx = 0
    attempts = 0
    while usage[gl] < TARGET_MIN and attempts < 100:
        if idx < len(variants):
            register(*variants[idx])
            idx += 1
        else:
            break
        attempts += 1
    if usage[gl] < TARGET_MIN:
        print(f"WARNING: cobertura baja para {gl}: {usage[gl]}")


for key in ALLOWED_KEYS:
    if usage[gloss(key)] < TARGET_MIN:
        ensure_coverage(key)


def select_final_dataset(items: list[dict], target: int) -> list[dict]:
    pool = [item for item in items if is_quality(item["spanish"])]
    random.shuffle(pool)

    selected: list[dict] = []
    selected_keys: set[str] = set()
    final_usage: Counter = Counter()

    def entry_key(item: dict) -> str:
        return "|".join(item["glosses"]) + "=>" + item["spanish"]

    def add_item(item: dict) -> bool:
        key = entry_key(item)
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(item)
        final_usage.update(item["glosses"])
        return True

    def missing_glosses() -> list[str]:
        return [gloss(k) for k in ALLOWED_KEYS if final_usage[gloss(k)] < TARGET_MIN]

    remaining = pool[:]

    # 1) Priorizar cobertura minima de cada glosa
    for _ in range(target * 3):
        need = missing_glosses()
        if not need:
            break
        need_set = set(need)
        picked = next((item for item in remaining if entry_key(item) not in selected_keys and need_set.intersection(item["glosses"])), None)
        if picked is None:
            break
        remaining.remove(picked)
        add_item(picked)

    # 2) Balance por categoria
    by_category: dict[str, list[dict]] = {cat: [] for cat in CATEGORY_TARGETS}
    for item in remaining:
        if entry_key(item) not in selected_keys:
            by_category[item["category"]].append(item)
    for cat in by_category:
        random.shuffle(by_category[cat])

    for cat, target_count in CATEGORY_TARGETS.items():
        current = sum(1 for item in selected if item["category"] == cat)
        for item in by_category[cat]:
            if current >= target_count:
                break
            if add_item(item):
                current += 1

    # 3) Completar hasta target
    random.shuffle(remaining)
    for item in remaining:
        if len(selected) >= target:
            break
        if entry_key(item) not in selected_keys:
            add_item(item)

    return selected[:target]


final_dataset = select_final_dataset(dataset, TARGET_TOTAL)
output_dataset = [{"glosses": item["glosses"], "spanish": item["spanish"]} for item in final_dataset]

final_usage = Counter()
for item in output_dataset:
    final_usage.update(item["glosses"])

missing = [gloss(k) for k in ALLOWED_KEYS if final_usage[gloss(k)] < TARGET_MIN]
if missing:
    print(f"WARNING: {len(missing)} glosas con cobertura < {TARGET_MIN}: {missing[:15]}")

out = {"dataset": output_dataset}
path = Path(__file__).parent / "dataset_glosas.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(output_dataset)} examples to {path}")
print("Pool size:", len(dataset))
print(
    "Pool categories:",
    {cat: sum(1 for d in dataset if d.get("category") == cat) for cat in CATEGORY_TARGETS},
)
print("Min coverage:", min(final_usage[gloss(k)] for k in ALLOWED_KEYS))
print("Max coverage:", max(final_usage[gloss(k)] for k in ALLOWED_KEYS))
