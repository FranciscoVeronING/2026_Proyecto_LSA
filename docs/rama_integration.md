# Rama `integration` — bitácora técnica y guía explicativa

Este documento describe **qué hace el sistema**, **cómo está armado por dentro** y **por qué tomamos ciertas decisiones** mientras desarrollamos la rama `integration`.


---

## Glosario rápido (términos que vas a ver seguido)

| Término | Explicación simple | Detalle técnico (si aplica) |
|--------|-------------------|----------------------------|
| **LSA** | Lengua de Señas Argentina. El sistema reconoce *señas* de este lenguaje. | No es español hablado; es otro idioma con gramática propia. |
| **Glosa** | Etiqueta escrita que representa una seña. Ej: `HOLA`, `YO`, `A`. | En lingüística de señas, la glosa es la unidad léxica escrita en MAYÚSCULAS. |
| **Landmark** | Punto del cuerpo que la cámara “ve” (hombro, muñeca, dedo…). | Coordenadas `(x, y, z)` por frame, salida de MediaPipe Holistic. |
| **Clasificador** | El “cerebro” que adivina qué seña estás haciendo. | Red `TinySkeletonClassifier` sobre secuencias de landmarks. |
| **LLM** | Modelo de lenguaje grande: traduce / interpreta las glosas a español natural. | Qwen2.5-3B-Instruct fine-tuneado con PEFT (LoRA adapter). |
| **Buffer / cola** | Memoria temporal donde se van guardando cosas antes de procesarlas. | `UtteranceBuffer` (glosas), `Queue` (trabajo para la LLM). |
| **Worker (hilo de trabajo)** | Proceso en segundo plano que no bloquea la cámara. | `InferenceWorker` (clasificador), `SemanticWorker` (LLM). |
| **PEFT / adapter** | Ajuste liviano sobre un modelo base ya entrenado. | No cargamos 3B parámetros desde cero; usamos LoRA en `adapter_model.safetensors`. |
| **Prompt** | Instrucciones que le damos a la LLM para que se comporte como intérprete LSA. | `sys_prompt.txt` + few-shots en JSON. |
| **Few-shot** | Ejemplos dentro del prompt (“Glosas X → Español Y”). | Mejora consistencia sin reentrenar. |
| **Dedup / anti-rebote** | Evitar que la misma seña se registre muchas veces por error del clasificador. | Regla por igualdad en `RepeatGate`, sin ventana temporal. |
| **Deletreo** | Deletrear con señas de letras: `J U A N` → “Juan”. | Secuencia de glosas de un solo carácter alfabético. |
| **Turno (conversación)** | Una intervención en el diálogo: alguien “dijo” algo. | Objeto `Turn` con rol `signer` o `hearing`. |
| **signer** | Persona sorda que señala (LSA). | Turno con glosas + interpretación en español. |
| **hearing** | Persona oyente | Turno con texto/voz capturado (API lista; UI pendiente). |
| **CUDA / GPU** | Aceleración por placa de video NVIDIA. | Necesaria para inferencia rápida de clasificador y LLM. |
| **Hilo (thread)** | Tarea que corre “en paralelo” sin frenar la pantalla. | `threading.Thread` daemon + `Queue` como buzón de trabajo. |
| **Race condition** | Dos partes del programa tocan el mismo dato al mismo tiempo y se pisan. | Se evita con `Lock`/`RLock`. |

---

## 1. ¿Qué problema resuelve esta rama?

### En palabras simples

Queremos que una persona sorda pueda **señar frente a la cámara** y que el sistema:

1. Reconozca las señas una por una.
2. Las vaya juntando como si fueran “palabras sueltas” en una frase.
3. Cuando hace una pausa, **traduzca / interprete** todo a español argentino.
4. Muestre el texto en pantalla y, opcionalmente, lo **diga en voz alta**.
5. Recuerde un poco de la conversación reciente para entender mejor la siguiente frase.

### En términos técnicos

La rama `integration` une en **tiempo real (RT)**:

1. **Captura:** webcam + MediaPipe Holistic → landmarks normalizados.
2. **Clasificación:** TinySkeleton sobre ventanas de frames → glosa + confianza.
3. **Agregación:** `UtteranceBuffer` + `RepeatGate` → lista de glosas por enunciado.
4. **Semántica:** LLM (PEFT) o reglas literales → oración en español.
5. **Contexto:** `ConversationMemory` → últimos 10 turnos inyectados al chat de la LLM.

---

## 2. Recorrido completo: qué pasa cuando alguien seña

Imaginemos que la persona sorda seña **“YO NOMBRE J U A N”** y hace una pausa.

### Paso a paso (versión humana)

1. **La cámara filma** tu cuerpo y manos.
2. **MediaPipe** dibuja puntos sobre hombros, brazos y dedos.
3. Cada cierto tiempo, el sistema **corta un pedazo de movimiento** (cuando detecta que empezaste y terminaste una seña).
4. Ese pedazo va al **clasificador**, que responde algo como: “esto es la seña `J` con 87% de confianza”.
5. Si la confianza es suficiente y la glosa no es un rebote duplicado, se **agrega al listado** en pantalla: `J`, luego `U`, luego `A`, luego `N`…
6. Si pasan **4 segundos** sin una nueva seña, el sistema entiende: “terminó el enunciado”.
7. Como solo hay letras, **no hace falta la LLM**: arma directamente **“Juan.”** y lo muestra / dice.
8. Ese resultado queda guardado en la **memoria de conversación** por si la próxima frase necesita contexto.

### Paso a paso (versión técnica)

```text
Frame (BGR)
  → MediaPipe Holistic.process()
  → extract_vector() + mirror (zurdo) + LandmarkSmoother (EMA α=0.6)
  → frames_temp_buffer (captura dinámica/estática según CAPTURE_MODE)
  → enqueue → InferenceWorker
  → TinySkeleton → top-3 + confidence
  → UtteranceBuffer.try_add() → RepeatGate.allow()
  → (pausa UTTERANCE_PAUSE_SEC) → maybe_close()
  → SemanticWorker.submit(glosses)
  → format_literal_utterance? → sí: "Juan." / no: translate_glosses(..., history)
  → shared_state["spanish_text"] + pyttsx3 + ConversationMemory.add_signer()
```

---

## 3. Arquitectura del sistema

### Diagrama general

```text
┌─────────────┐   landmarks    ┌──────────────────┐
│  Webcam +   │ ─────────────► │ InferenceWorker  │  ← hilo aparte (no bloquea UI)
│  MediaPipe  │                │ TinySkeleton     │
└─────────────┘                └────────┬─────────┘
                                        │ glosa + confianza
                                        ▼
                               ┌──────────────────┐
                               │ UtteranceBuffer  │  ← “frase en construcción”
                               │ + RepeatGate     │  ← filtra repeticiones
                               └────────┬─────────┘
                                        │ pausa → lista cerrada
                                        ▼
                               ┌──────────────────┐
                               │ SemanticWorker   │  ← hilo aparte (LLM pesada)
                               │ + ConversationMemory
                               └────────┬─────────┘
                        literal │       │ LLM
                   (letras/díg) │       ▼
                                │  translate_glosses(history)
                                ▼
                          pantalla / voz (pyttsx3)
```

### ¿Por qué hay “workers” (hilos separados)?

**Problema:** la LLM tarda segundos en cargar y traducir. Si eso corriera en el mismo hilo que la cámara, el video se congelaría.

**Solución:** dos workers en background:

| Worker | Qué hace | Por qué existe |
|--------|----------|----------------|
| `InferenceWorker` | Corre el clasificador PyTorch | La red neuronal no debe frenar OpenCV |
| `SemanticWorker` | Carga y ejecuta la LLM | Transformers es pesado; además se carga *lazy* (solo cuando arranca el hilo) |
| `VoiceWorker` | Sintetiza voz (pyttsx3) | Hablar bloquea hasta terminar el audio; si viviera en el hilo semántico, la traducción siguiente esperaría al parlante |

Patrón: **productor-consumidor** con `Queue` / `deque` + `Lock` en `shared_state`.

Sobre la voz: `pyttsx3` en Windows usa COM, que exige crear y usar el engine **en el mismo hilo**. Por eso `VoiceWorker` instancia el engine dentro de su propio loop y no lo comparte.

### Módulos del código y su responsabilidad

El proyecto se organiza en cuatro paquetes bajo `src/`, separados por qué tan pesadas son sus dependencias:

```
run.py                    punto de entrada único
src/
├── app/                  todo lo que necesita cámara y pantalla
│   ├── main.py             loop principal y orquestación
│   ├── ui.py               Button, Slider, modal de mano dominante
│   ├── capture.py          WebcamStream, suavizado, vector de landmarks
│   ├── workers.py          InferenceWorker, SemanticWorker, VoiceWorker
│   ├── utterance.py        UtteranceBuffer, normalize_gloss
│   ├── eval_session.py     modo --eval y su CSV
│   └── state.py            shared_state + lock
├── core/                 lógica pura: ni OpenCV, ni torch, ni GPU
│   ├── repeat_policy.py    RepeatGate, format_literal_utterance
│   ├── conversation_memory.py
│   └── landmarks.py        normalización espacial y subsampleo
├── classifier/           clasificador de señas
│   ├── config.py           umbrales, hiperparámetros, GLOSS_NORMALIZER
│   ├── arch.py             TinySkeletonClassifier
│   └── weights/            .pth, mapeo_clases.json, metrics.json
└── semantic/             traductor a español
    ├── config.py           rutas, params de generación, tamaño de contexto
    ├── translator.py       carga PEFT/Unsloth y translate_glosses
    ├── prompts/            sys_prompt.txt, few_shots_examples.json
    └── adapter/            adaptador LoRA (safetensors vía Git LFS)
```

| Paquete | Para el usuario | Para el desarrollador |
|---------|-----------------|----------------------|
| `app/` | Ventana, botones, sliders, teclas | Loop OpenCV, hilos, orquestación |
| `core/` | (invisible) | Reglas de negocio testeables sin cámara ni GPU |
| `classifier/` | Umbrales de sensibilidad / confianza | Config, arquitectura, pesos entrenados |
| `semantic/` | Calidad de la traducción | Prompt, few-shots, carga PEFT |

**El criterio de corte es la dependencia, no el tema.** `core/` importa solo la librería estándar y NumPy, así que se puede ejecutar y testear en milisegundos desde cualquier máquina. `app/` es lo único que abre una cámara. Esa frontera es la que permite escribir tests de `RepeatGate` o `ConversationMemory` sin montar el pipeline entero.

Los artefactos binarios viven en carpetas propias (`classifier/weights/`, `semantic/adapter/`) y no mezclados con el código que los consume: el código cambia todos los días y los pesos casi nunca.

---

## 4. Glosas: de la seña al listado en pantalla

### 4.1 ¿Qué es una glosa en este proyecto?

No es español. Es la **representación escrita** de una unidad de LSA.

Ejemplos:

- Seña de saludo → glosa `HOLA`
- Seña de “yo” → glosa `YO`
- Letra J del abecedario manual → glosa `J`

El clasificador devuelve claves internas (`el_ella`, `vivir_en`…). Antes de mostrar o traducir, **`normalize_gloss`** las pasa a tokens estándar (`EL/ELLA`, `VIVIR-EN`…) usando `GLOSS_NORMALIZER` en config.

### 4.2 `UtteranceBuffer`: la “oración en construcción”

**Analogía:** es como el autocorrector que va juntando palabras hasta que hacés una pausa larga.

- Cada glosa aceptada se appendea a una lista.
- Si pasan **`UTTERANCE_PAUSE_SEC` = 4 segundos** sin glosa nueva → se **cierra** el enunciado y se manda a traducir.
- Mientras tanto, en pantalla ves algo como: `Glosas: YO NOMBRE J U A N`.

**Parámetros importantes** (`src/classifier/config.py`):

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `UTTERANCE_PAUSE_SEC` | 4.0 | Segundos sin actividad de señado para cerrar frase |
| `CONFIDENCE_THRESHOLD` | 0.75 | Confianza mínima del clasificador |
| `INFERENCE_COOLDOWN_SEC` | 1.0 | Espera entre inferencias (anti-spam) |

### 4.3 `RepeatGate`: repeticiones inteligentes

#### El problema que teníamos

El clasificador a veces **rebota**: la misma seña se predice 2–3 veces seguidas en menos de 1 segundo aunque la persona solo hizo la seña una vez.

La solución original era simple: **“si la glosa es igual a la anterior y pasó menos de 1 s, descartala”**.

Eso funcionaba para palabras (`HOLA HOLA` → una sola), pero **rompía casos legítimos**:

- Deletreo: `J U A N` necesita repetir letras distintas, no el problema… pero sí **números repetidos** (`1 1 1`) o **dobles letras** (`A A` en “Ana”).
- Números de documento: `1 1 2 2 3` debe conservarse.

#### La solución: política por tipo de glosa

Archivo: `src/core/repeat_policy.py`.

| Tipo | Cómo se detecta | Regla | Ejemplo |
|------|-----------------|-------|---------|
| **Dígito** | un carácter `0-9` | Siempre aceptar repeticiones | `1 1 1` → `1 1 1` |
| **Letra** | un carácter alfabético (incl. `Ñ`) | Máximo 2 iguales seguidas; la 3ª se descarta | `A A A` → `A A` |
| **Other** | todo lo demás (`HOLA`, `VER`…) | Nunca dos veces seguidas | `HOLA HOLA` → `HOLA` |

```python
# Ejemplo de comportamiento (RepeatGate)
gate = RepeatGate(max_letter_consecutive=2)

gate.allow("A")       # True  — primera A
gate.allow("A")       # True  — segunda A (válida en deletreo)
gate.allow("A")       # False — tercera A (probable rebote o exceso)

gate.allow("1")       # True
gate.allow("1")       # True  — números repetidos OK
gate.allow("1")       # True

gate.allow("hola")    # True
gate.allow("hola")    # False — seña léxica repetida: rebote
gate.allow("ver")     # True
gate.allow("hola")    # True  — no es consecutiva, hay un VER en el medio
```

**Constante:** `LETTER_MAX_CONSECUTIVE = 2`.

#### Por qué la regla de `other` dejó de ser temporal

La versión original descartaba una glosa léxica repetida solo si llegaba **dentro de
`GLOSS_DEDUP_SEC = 1 s`**. En la práctica no descartaba nada, y la razón es aritmética:
`INFERENCE_COOLDOWN_SEC` también vale 1 s, así que el sistema **garantiza** que dos
inferencias consecutivas estén separadas por al menos un segundo. La ventana de dedup era
inalcanzable por construcción: toda glosa repetida llegaba "tarde" y se aceptaba.

Se detectó en uso real: señar `HOLA`, esperar, volver a señar `HOLA`, y ver las dos en el
buffer. Subir el umbral hubiera sido un parche frágil, porque habría que mantenerlo siempre
por encima del cooldown.

La regla nueva es por **igualdad, no por tiempo**: dos señas léxicas idénticas pegadas se
descartan siempre. El fundamento lingüístico es que `HOLA HOLA` no significa nada distinto de
`HOLA` dentro de un mismo enunciado; si aparece, es rebote. Para volver a decir `HOLA` hay que
cerrar el enunciado, y al cerrarse se llama a `reset()`. Las repeticiones **no consecutivas**
(`HOLA VER HOLA`) se conservan intactas.

Esto además simplificó la clase: `RepeatGate` ya no necesita saber la hora. `allow()` perdió
el parámetro `now` y el atributo `last_at`, y quedó como una máquina de estados de tres reglas
sin dependencia temporal, mucho más fácil de testear.

### 4.4 Cuándo se cierra el enunciado

El cierre lo decide `UtteranceBuffer` cuando pasan `UTTERANCE_PAUSE_SEC` (4 s) **sin actividad
de señado**. La sutileza está en qué cuenta como actividad, y son tres cosas:

1. Una glosa aceptada.
2. Una glosa **reconocida pero descartada** por repetida.
3. Las manos moviéndose frente a la cámara, aunque el clasificador todavía no haya resuelto nada.

Los puntos 2 y 3 no estaban antes, y su ausencia se volvió un problema al endurecer la regla
de repeticiones. Si el reloj solo se reiniciara con glosas aceptadas, alguien que repite una
seña dejaría correr la cuenta regresiva mientras sigue señando, y el enunciado podría cerrarse
en medio de la frase.

El punto 3 usa **movimiento de manos**, no mera presencia. Es deliberado: si bastara con tener
las manos en cámara, una persona que apoya las manos en un lugar visible mantendría el
enunciado abierto para siempre. Con movimiento, las manos quietas o fuera de pantalla dejan
correr los 4 segundos y el enunciado cierra, que es el comportamiento esperado.

---

## 5. Traductor semántico (LLM)

### 5.1 ¿Qué hace la LLM aquí?

**En simple:** convierte una secuencia de glosas (que suenan raro en español) en **una oración argentina natural**.

Ejemplo:

```text
Glosas:  YO LLAMAR POLICIA
Español: Llamo a la policía.
```

La LLM también **reordena** (la gramática LSA ≠ gramática española), **agrega artículos** y **resuelve ambigüedades** cuando el clasificador se equivocó entre dos señas parecidas.

### 5.2 ¿Cómo se carga el modelo?

No usamos un modelo gigante entrenado desde cero. Usamos:

1. **Modelo base:** `unsloth/Qwen2.5-3B-Instruct` (~3 mil millones de parámetros, formato chat).
2. **Adapter PEFT (LoRA):** pesos pequeños entrenados para LSA, en `src/semantic/adapter/`.

**PEFT** (*Parameter-Efficient Fine-Tuning*): en lugar de modificar todo el modelo, solo se entrenan/adaptan unas capas livianas. Es más rápido de cargar y ocupa menos disco.

**Flujo de carga** (`load_model_and_tokenizer`):

1. Lee prompt + few-shots.
2. Intenta **PEFT** (default, `USE_UNSLOTH = False`).
3. Si falla, intenta **Unsloth** (requiere GPU NVIDIA y paquete `unsloth`).

**Carga diferida (*lazy load*):** la LLM **no** se importa al arrancar la aplicación. Solo cuando arranca `SemanticWorker`. Así la cámara puede abrir aunque Transformers no esté instalado (modo “solo glosas”).

**Parámetros de generación** (`src/semantic/config.py`):

| Parámetro | Valor | Por qué |
|-----------|-------|---------|
| `MAX_NEW_TOKENS` | 64 | Las respuestas son una oración corta |
| `TEMPERATURE` | 0.1 | Casi determinístico: interpretar, no inventar |
| `REPETITION_PENALTY` | 1.0 | Ver abajo |

La penalización por repetición estaba en `1.2`. Ese parámetro castiga tokens que **ya aparecieron en el contexto**, y el contexto incluye el prompt con las glosas. Para una tarea que es casi transcripción, eso es contraproducente: al traducir `DOCUMENTO 1 2 O 4 5` el modelo queda desincentivado justamente a copiar esos dígitos, y peor todavía con dígitos repetidos (`1 1 2 2`). Se bajó a `1.0` (sin penalización).

### 5.3 Prompt del sistema y few-shots

**Prompt** (`sys_prompt.txt`): le dice a la LLM que es intérprete LSA→español rioplatense, qué no inventar, cómo formatear la respuesta, etc.

**Few-shots** (`few_shots_examples.json`): pares ejemplo:

```json
{
  "glosses": ["YO", "NOMBRE", "J", "U", "A", "N"],
  "spanish": "Me llamo Juan."
}
```

### 5.4 Ambigüedades del clasificador

A veces dos señas **se ven casi iguales** para la cámara. El clasificador elige una u otra casi al azar, y el contexto es lo único que permite decidir. Los pares detectados hasta ahora se agrupan en tres familias según qué tipo de contexto los desempata.

**Grupo 1 — letra contra número.** Se resuelve por el tipo de secuencia.

| Par confuso | Cuándo es letra/nombre | Cuándo es número |
|-------------|------------------------|------------------|
| **O ↔ 0** | Deletreo, nombre, apellido → **O** | Documento, edad, teléfono → **0** |
| **2 ↔ V** | Deletreo → **V** | Secuencia numérica → **2** |
| **AÑOS ↔ G** | Dentro de un deletreo → **G** | Después de dígitos, contexto de edad → **AÑOS** |

`AÑOS ↔ G` entra en este grupo porque el desempate es el mismo: `YO 2 5 G` es claramente una edad, mientras que `NOMBRE AÑOS U I D O` es un deletreo donde la seña léxica no tiene lugar y debe leerse como `G`.

**Grupo 2 — seña léxica contra seña léxica.** Acá no hay tipos que ayuden; desempata el campo semántico de las glosas vecinas.

| Par confuso | Cuándo es una | Cuándo es la otra |
|-------------|---------------|-------------------|
| **CHAU ↔ MARTES** | Junto a glosas de tiempo (`HOY`, `AYER`, `DIA`, `HORA`, otros días) → **MARTES** | Sola, o cerrando un saludo o despedida → **CHAU** |

Ejemplo del corrimiento: `HOY CHAU` no es un saludo, es *"Hoy es martes"*. La presencia de `HOY` es lo único que lo revela.

**Grupo 3 — I ↔ T ↔ OJO.** El más difícil, porque las tres conviven en contexto de letras. La configuración de la mano es prácticamente la misma; lo único que cambia es **dónde se apoya**:

| Seña | Ubicación de la mano |
|------|----------------------|
| **I** | Mejilla |
| **T** | Debajo de la boca |
| **OJO** | Pómulo, con el índice cerca del ojo |

Como el clasificador trabaja sobre landmarks normalizados, esa diferencia de posición es sutil y se pierde con facilidad. Las reglas que le damos a la LLM:

- Dentro de un deletreo → elegir entre **I** y **T** la que forme una palabra o nombre real en español (`S O F T A` → Sofía).
- Suelta o entre glosas léxicas → interpretar **OJO** (`YO I MAL` → “Me duele el ojo”).

La LLM recibe estas reglas en el prompt **y** ejemplos few-shot:

```text
DOCUMENTO 1 2 O 4 5  →  documento 12045
NOMBRE 0 S C A R     →  Oscar
NOMBRE A 2 A         →  Ava
NOMBRE S O F T A     →  Sofía
YO I MAL             →  Me duele el ojo
```

**Importante:** esto no reemplaza al clasificador; es una **capa de interpretación** que usa el contexto de las glosas vecinas. Tiene un techo claro: cuando las dos lecturas son palabras válidas (`MARIA` vs `MARTA`) ningún contexto textual alcanza, y la única salida real es mejorar el clasificador para que capture la posición de la mano.

### 5.5 Deletreo y números: camino rápido sin LLM

**Problema detectado:** si el enunciado era *solo* letras (`J U A N`) o *solo* dígitos (`1 2 3 4`), la LLM a veces respondía **vacío** o algo incoherente.

**Solución:** función `format_literal_utterance(glosses)`:

| Entrada | Salida | ¿Pasa por LLM? |
|---------|--------|----------------|
| `J U A N` | `Juan.` | No |
| `1 2 3 4` | `1234.` | No |
| `YO NOMBRE J U A N` | — | Sí (mixto / léxico) |

Ventajas: **instantáneo**, **predecible**, **funciona aunque la LLM falle**.

#### Cuándo el camino rápido cede ante la LLM

El atajo solo se toma si no hay ambigüedad que el propio tipo de secuencia resuelva.

**O/0 y 2/V no necesitan excepción.** Si la secuencia quedó toda de un tipo, el carácter ya es del tipo correcto (`1 2 0 4 5` son dígitos, el `0` está bien). Y si el clasificador se equivocó de tipo, la secuencia queda **mixta** (`1 2 O 4 5`) y cae en la LLM sola. El caso se cubre sin código extra.

**I/T/OJO sí necesita excepción**, porque las tres lecturas conviven en contexto de letras y el tipo no desempata. Por eso `format_literal_utterance` devuelve `None` (delega en la LLM) cuando:

| Situación | Por qué |
|-----------|---------|
| Una sola glosa de letra (`I`, `T`, `A`) | Puede ser una seña léxica mal clasificada, no un deletreo |
| Deletreo que contiene `I` o `T` (`M A R I A`) | Solo el sentido de la palabra decide entre las dos |

Las secuencias de dígitos conservan el atajo siempre. El conjunto vive en `AMBIGUOUS_LETTERS` dentro de [`src/core/repeat_policy.py`](../src/core/repeat_policy.py) y se puede sobrescribir por parámetro.

El costo es que la mayoría de los deletreos de nombres pasan ahora por la LLM (muchos nombres en español llevan `i` o `t`). Se prioriza la interpretación correcta sobre la latencia.

### 5.6 Dependencias y ejecución

**Entorno recomendado:** conda `lsa_gpu` con PyTorch + CUDA.

Instalación reproducible (`requirements.txt` en la raíz del repo):

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

PyTorch va aparte porque la build con CUDA no vive en PyPI.

**Cómo correr la cámara**, siempre desde la raíz del repo:

```bash
python run.py            # cámara + traducción
python run.py --no-llm   # solo glosas, sin cargar la LLM
python run.py --eval     # recorrido de evaluación, salida CSV
```

Hay un único punto de entrada a propósito. Antes convivían `python -m camera` (parado en `src/`) y `python -m src.camera` (parado en la raíz), y cuál funcionaba dependía del directorio actual. `run.py` agrega `src/` a `sys.path` y delega en `app.main`, así que el comando es siempre el mismo.

Dos decisiones sostienen esto:

1. **Un solo lugar donde se toca `sys.path`.** Es `run.py`. Los módulos importan entre sí con rutas absolutas desde la raíz de fuentes (`from core.repeat_policy import ...`), sin `try/except ImportError` repartidos.
2. **Paths absolutos a los artefactos.** Los `config.py` calculan rutas de pesos, prompts y adapter con `Path(__file__).resolve().parent`. Cuando eran relativas al directorio actual, el proyecto solo arrancaba parado en `src/`.


## 6. Memoria conversacional

### 6.1 ¿Para qué sirve?

**En simple:** si hace un minuto dijiste “Me llamo Juan” y ahora señás solo “DOCUMENTO …”, la LLM puede entender que el número es **tu** documento, no un dato suelto.

**En técnico:** inyectamos los últimos turnos como mensajes `user`/`assistant` en el chat template de Qwen, antes del turno actual.

### 6.2 ¿Por qué no guardamos “toda” la charla?

Los modelos tienen **límite de contexto** (cantidad de texto que “ven” de una vez). Mandar toda una conversación larga:

- Hace más lenta cada traducción.
- Mezcla temas viejos con el turno actual.
- Puede confundir más que ayudar.

Por eso usamos una **ventana deslizante** de los últimos **10 turnos**.

### 6.3 ¿Por qué 10 turnos y no “5 del sordo + 5 del oyente”?

Porque en conversaciones reales **no se alterna perfecto**:

```text
Sordo: Me llamo Juan.
Sordo: Vivo en Córdoba.
Oyente: ¿Documento?
Sordo: (números)
```

Si reserváramos cupos fijos por lado, podríamos **tirar contexto reciente** solo porque un lado habló más veces.

Con “últimos 10 turnos con rol”, en un diálogo equilibrado suele haber ~5 intervenciones de cada uno, pero el sistema se adapta si uno habla más.

### 6.4 Estructura de datos

Archivo: `src/core/conversation_memory.py`

```python
@dataclass
class Turn:
    role: str       # "signer" | "hearing"
    text: str       # texto en español (signer) o mensaje oyente (hearing)
    glosses: str    # opcional, solo signer: "YO NOMBRE J U A N"
    ts: float       # timestamp
```

**Dos niveles de memoria:**

| Nivel | Qué guarda | Límite | Uso |
|-------|------------|--------|-----|
| `window` | Últimos N turnos | 10 | Lo que ve la LLM |
| `session_log` | Toda la sesión | Sin límite | Auditoría, export futuro, UI |

**API pública:**

```python
memory.add_signer("Me llamo Juan.", glosses="YO NOMBRE J U A N")
memory.add_hearing("¿Cuál es tu documento?")   # preparado, UI/STT pendiente
memory.clear()          # corta el contexto de la LLM, conserva session_log (tecla 'c')
memory.reset_session()  # descarta también el log completo
memory.as_messages()    # → lista de mensajes para translate_glosses
```

**Dos formas de olvidar, a propósito.** `clear()` sirve para cuando cambia el interlocutor o el tema y el contexto viejo estorba, pero seguís queriendo el registro de lo conversado. `reset_session()` es empezar de cero. Al principio `clear()` borraba las dos cosas, lo que anulaba el propósito de auditoría del log.

**Thread-safety.** La memoria se escribe desde el hilo semántico y se puede limpiar desde el hilo de UI (tecla `c`). Sin protección, limpiar justo mientras se arma el historial podía cortar con `RuntimeError: deque mutated during iteration`. Por eso todos los accesos van bajo un `RLock` interno.

### 6.5 Cómo se ve el historial dentro de la LLM

```text
system:  (instrucciones del intérprete LSA)

user:    Glosas: YO NOMBRE J U A N
assistant: Me llamo Juan.

user:    Persona oyente: ¿Cuál es tu documento?
assistant: (mensaje del oyente registrado como contexto)

user:    Glosas: DOCUMENTO 1 2 O 4 5 6 7 8    ← turno actual
→ la LLM genera la respuesta
```

Constante: `CONVERSATION_HISTORY_SIZE = 10` en `src/semantic/config.py`.

**Tecla `c`:** corta el contexto (`semantic_worker.clear_conversation()`). El contador de turnos activos se muestra en el panel inferior como `Contexto: N/10`.

### 6.6 Riesgo abierto: el adapter se entrenó sin historial

Este es el punto más delicado de la feature y conviene tenerlo escrito.

El fine-tuning del adapter LoRA se hizo con **pares de un solo turno**:

```json
{
  "glosses": ["AHORA_HOY", "YO", "BIEN"],
  "spanish": "Hoy estoy bien."
}
```

No hay turnos previos en ninguno de los ejemplos de entrenamiento. Inyectar historial conversacional en inferencia lo pone frente a una estructura de prompt que nunca vio, y eso puede degradar la salida en lugar de mejorarla. En jerga: lo saca **fuera de distribución**.

Un matiz importante: con la memoria vacía, `as_messages()` devuelve una lista vacía y el prompt queda **idéntico** al de entrenamiento. O sea, el primer enunciado de cada conversación nunca se ve afectado; la diferencia aparece recién del segundo en adelante.

Por eso el historial es apagable:

```python
# src/semantic/config.py
USE_CONVERSATION_HISTORY = True   # False para volver al comportamiento de un turno
```

**Cómo medirlo.** El dataset de evaluación (~200 secuencias de glosas con su traducción de referencia, medido con BLEU y ROUGE-L) es de un solo turno, así que **no puede detectar esta degradación**: al evaluar ejemplo por ejemplo el historial siempre está vacío. Para medirlo de verdad hacen falta diálogos encadenados.

Camino sugerido, en orden:

1. Armar un set chico de evaluación **multi-turno** (20–30 diálogos de 3–5 intervenciones), con la traducción de referencia de cada turno.
2. Correr BLEU y ROUGE-L con `USE_CONVERSATION_HISTORY` en `True` y en `False`, y comparar.
3. Si el historial baja las métricas, la solución de fondo no es apagarlo sino **agregar ejemplos multi-turno al dataset de entrenamiento** y reentrenar el adapter. El contexto es valioso; lo que falta es que el modelo lo haya visto.

### 6.7 Pendiente (producto)

- Campo de texto o micrófono para el oyente → `add_hearing()`.
- Panel en pantalla mostrando el texto de los últimos turnos (hoy solo se ve el contador).
- Exportar `session_log` a JSON/CSV.
- Revisar el "ack" de los turnos del oyente: hoy insertamos un `assistant: "(mensaje del oyente registrado como contexto)"` solo para mantener la alternancia user/assistant. Funciona, pero le está mostrando al modelo que esa frase es una respuesta posible. Alternativa: incorporar lo dicho por el oyente dentro del mensaje `user` del turno siguiente.

---


## 7. Patrones de diseño utilizados 

| Patrón | Dónde | Beneficio |
|--------|-------|-----------|
| **Worker + Queue** | `InferenceWorker`, `SemanticWorker`, `VoiceWorker` | UI fluida; tareas pesadas en background |
| **Lazy loading** | import LLM solo en hilo semántico | Arranque rápido; degradación graceful |
| **Policy object** | `RepeatGate` | Reglas de repetición testeables y extensibles |
| **Strategy / fallback** | literal vs LLM; PEFT vs Unsloth | Robustez ante fallos |
| **Sliding window** | `ConversationMemory` | Contexto acotado y predecible |
| **Shared state + Lock** | `shared_state` dict, `RLock` en la memoria | Comunicación thread-safe entre workers y UI |
| **Separation of concerns** | `repeat_policy`, `conversation_memory`, `utils` | Cada archivo un dominio claro |
| **Location independence** | paths vía `Path(__file__)` | El programa corre desde cualquier directorio |

---

## 8. Archivos modificados o creados

| Archivo | Descripción |
|---------|-------------|
| `run.py` | **Nuevo.** Punto de entrada único; el único lugar que toca `sys.path` |
| `src/core/repeat_policy.py` | **Nuevo.** Clasificación digit/letter/other, `RepeatGate`, literales |
| `src/core/conversation_memory.py` | **Nuevo.** Turnos signer/hearing, ventana + session log, `RLock` |
| `src/core/landmarks.py` | Ex `src/utils.py`. Normalización espacial y subsampleo |
| `src/app/main.py` | Ex `src/camera.py`, ahora solo argparse y loop principal |
| `src/app/ui.py` | **Nuevo.** `Button`, `Slider`, modal de mano dominante, panel top-3 |
| `src/app/capture.py` | **Nuevo.** `WebcamStream`, `LandmarkSmoother`, vector de landmarks |
| `src/app/workers.py` | **Nuevo.** `InferenceWorker`, `SemanticWorker`, `VoiceWorker` |
| `src/app/utterance.py` | **Nuevo.** `UtteranceBuffer`, `normalize_gloss` |
| `src/app/eval_session.py` | **Nuevo.** Modo `--eval` y escritura del CSV |
| `src/app/state.py` | **Nuevo.** `shared_state` y su lock, aislados de los workers |
| `src/classifier/config.py` | `LETTER_MAX_CONSECUTIVE`, umbrales utterance, paths a `weights/` |
| `src/classifier/arch.py` | Ex `model_arch.py`, sin cambios de lógica |
| `src/semantic/config.py` | Paths absolutos, `CONVERSATION_HISTORY_SIZE`, `REPETITION_PENALTY` |
| `src/semantic/translator.py` | Ex `model.py`. Fix `BASE_MODEL_ID`, historial en `translate_glosses` |
| `src/semantic/prompts/sys_prompt.txt` | Reglas O/0, 2/V, literales, historial |
| `src/semantic/prompts/few_shots_examples.json` | Ejemplos ampliados |
| `src/config.py` | **Eliminado.** Importaba un módulo inexistente; nadie lo usaba |
| `unsloth_compiled_cache/` | **Sacado de git.** 3,2 MB de caché autogenerada, ahora ignorada |
| `requirements.txt` | **Nuevo.** Dependencias con versiones probadas |
| `docs/rama_integration.md` | Este documento |

---

## 9. Escenarios de ejemplo (casos de uso)

### Caso A — Deletreo de nombre

```text
Persona seña: J → U → A → N → (pausa 4s)

Glosas en buffer: J U A N
format_literal_utterance → "Juan."
Pantalla / voz: "Juan."
Memoria: add_signer("Juan.", glosses="J U A N")
LLM: no se invoca
```

### Caso B — Frase lexical con LLM

```text
Persona seña: YO → LLAMAR → POLICIA → (pausa)

Glosas: YO LLAMAR POLICIA
No es solo letras/dígitos → translate_glosses()
Español: "Llamo a la policía."
```

### Caso C — Documento con ambigüedad O/0

```text
Historial: "Me llamo Oscar." (deletreo previo)
Glosas actuales: DOCUMENTO 1 2 O 4 5 6 7 8

LLM interpreta O como 0 por contexto numérico
Español: "Mi documento es 12045678."
```

### Caso D — Rebote vs doble letra

```text
Deletreo "ANA":
  A → OK
  A → OK (segunda A permitida)
  A → descartada (tercera)
  N → OK
  A → OK

Resultado buffer: A A N A  (si la 3ª A era rebote del clasificador, se filtró bien)
```

---

## 10. Guía rápida de operación

### Requisitos

- Webcam funcionando.
- Entorno Python con GPU (recomendado): `lsa_gpu`.
- Pesos del clasificador en `src/classifier/weights/` (`tinyskeleton_best.pth`, `mapeo_clases.json`).
- Adapter LLM en `src/semantic/adapter/` (viene por Git LFS: `git lfs pull`).
- Dependencias instaladas con `pip install -r requirements.txt`.

### Pasos

1. Activar entorno: `conda activate lsa_gpu`
2. Ejecutar `python run.py` desde la raíz del repo.
3. Elegir mano dominante en el modal inicial.
4. Esperar en consola: `[*] Traductor semántico listo.`
5. Señar frente a la cámara; observar glosas acumularse.
6. Pausar ~4 s → aparece español + voz (si VOICE activo).
7. Tecla **`c`** → corta el contexto conversacional (el log de sesión se conserva).
8. Tecla **`q`** → salir.

Teclas útiles: `m` cambia el modo de captura, `n` saltea seña en modo eval.

### Si la LLM no carga

La cámara **no se cae**: sigue reconociendo señas y mostrando glosas. Revisar consola para el error exacto (deps faltantes, red para descargar base model, etc.).

---

## 11. Próximos pasos sugeridos

1. **Input del oyente:** caja de texto o STT (Speech-to-Text) → `semantic_worker.add_hearing(text)`.
2. **UI de historial:** mostrar el texto de los últimos turnos en el canvas OpenCV.
3. **Tests automáticos** de `RepeatGate` y `ConversationMemory`.
4. **Suavizado visual** de landmarks en el overlay.
5. **Export de sesión** (`session_log` → JSON/CSV) para revisión posterior.
6. **Eval multi-turno del historial** (ver 6.6): armar diálogos encadenados y comparar BLEU / ROUGE-L con `USE_CONVERSATION_HISTORY` en `True` y `False`. Si baja, agregar ejemplos multi-turno al entrenamiento del adapter.
7. **Revisar si los few-shots siguen sumando:** con un modelo ya fine-tuneado para esta tarea, los 16 ejemplos del prompt pueden ser redundantes y solo gastar contexto. Medir con y sin ellos sobre el dataset de evaluación.
8. **Reset por tiempo en `RepeatGate`:** hoy una letra repetida por tercera vez se descarta sin importar cuánto tiempo pasó. Si alguien hace `A`, espera 3 segundos y vuelve a hacer `A` a propósito, se pierde.
9. **Atacar I/T/OJO desde el clasificador:** la diferencia entre las tres es *dónde se apoya la mano* (mejilla / debajo de la boca / junto al ojo). Es información posicional que los landmarks tienen, pero que se diluye en la normalización. Una feature explícita de distancia mano–rostro resolvería el caso mejor que cualquier regla de prompt, sobre todo cuando las dos lecturas son palabras válidas (`MARIA` vs `MARTA`).

---

## 12. Resumen ejecutivo (una página)

La rama `integration` conecta **cámara → reconocimiento de señas → lista de glosas → interpretación en español**, con tres mejoras importantes respecto a una versión ingenua:

1. **Repeticiones inteligentes:** permite deletreo y números reales sin dejar pasar rebotes del clasificador.
2. **Desambiguación por contexto:** O/0, 2/V, AÑOS/G, CHAU/MARTES e I/T/OJO se resuelven mirando las glosas vecinas.
3. **Memoria conversacional corta:** la LLM entiende mejor frases sucesivas y referencias a lo ya dicho.

El sistema está diseñado para **no bloquearse** si la LLM falla: la cámara y el clasificador siguen. Los números puros y los deletreos sin ambigüedad funcionan **sin** LLM.

**Lo que queda por validar:** el adapter se entrenó con ejemplos de un solo turno, así que el valor real de la memoria conversacional todavía no está medido. Es la deuda técnica principal de la rama (sección 6.6).

Esta bitácora debe actualizarse cada vez que se agreguen features (voz del oyente, persistencia, etc.) para mantener alineado al equipo técnico y no técnico.
