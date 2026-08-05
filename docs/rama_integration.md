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
| **Dedup / anti-rebote** | Evitar que la misma seña se registre muchas veces por error del clasificador. | Ventana temporal `GLOSS_DEDUP_SEC`. |
| **Deletreo** | Deletrear con señas de letras: `J U A N` → “Juan”. | Secuencia de glosas de un solo carácter alfabético. |
| **Turno (conversación)** | Una intervención en el diálogo: alguien “dijo” algo. | Objeto `Turn` con rol `signer` o `hearing`. |
| **signer** | Persona sorda que señala (LSA). | Turno con glosas + interpretación en español. |
| **hearing** | Persona oyente | Turno con texto/voz capturado (API lista; UI pendiente). |
| **CUDA / GPU** | Aceleración por placa de video NVIDIA. | Necesaria para inferencia rápida de clasificador y LLM. |

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

Patrón: **productor-consumidor** con `Queue` / `deque` + `Lock` en `shared_state`.

### Módulos del código y su responsabilidad

| Archivo | Para el usuario | Para el desarrollador |
|---------|-----------------|----------------------|
| `src/camera.py` | Ventana, botones, sliders, teclas | Loop principal OpenCV, orquestación |
| `src/utils.py` | (invisible) | Normalización espacial, recorte de silencio, subsampleo a 32 frames |
| `src/repeat_policy.py` | Evita duplicados raros; permite deletreo | `RepeatGate`, `format_literal_utterance` |
| `src/conversation_memory.py` | (futuro) contexto de diálogo | `ConversationMemory`, roles signer/hearing |
| `src/model/classifier/*` | Umbrales de sensibilidad / confianza | Config, pesos `.pth`, arquitectura |
| `src/model/semantic/*` | Calidad de la traducción | Prompt, few-shots, carga PEFT |

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

**Parámetros importantes** (`model/classifier/config.py`):

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `UTTERANCE_PAUSE_SEC` | 4.0 | Segundos de silencio para cerrar frase |
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

Archivo: `src/repeat_policy.py`.

| Tipo | Cómo se detecta | Regla | Ejemplo |
|------|-----------------|-------|---------|
| **Dígito** | un carácter `0-9` | Siempre aceptar repeticiones | `1 1 1` → `1 1 1` |
| **Letra** | un carácter alfabético (incl. `Ñ`) | Máximo 2 iguales seguidas; la 3ª se descarta | `A A A` → `A A` |
| **Other** | todo lo demás (`HOLA`, `VER`…) | Dedup temporal 1 s | rebote filtrado |

```python
# Ejemplo de comportamiento (RepeatGate)
gate = RepeatGate(dedup_sec=1.0, max_letter_consecutive=2)

gate.allow("A", 0.0)    # True  — primera A
gate.allow("A", 0.5)    # True  — segunda A (válida en deletreo)
gate.allow("A", 1.0)    # False — tercera A (probable rebote o exceso)

gate.allow("1", 2.0)    # True
gate.allow("1", 2.1)    # True  — números repetidos OK
gate.allow("1", 2.2)    # True

gate.allow("hola", 3.0)   # True
gate.allow("hola", 3.5)   # False — mismo rebote en <1s
```

**Constantes:** `GLOSS_DEDUP_SEC = 1.0`, `LETTER_MAX_CONSECUTIVE = 2`.

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
2. **Adapter PEFT (LoRA):** pesos pequeños entrenados para LSA, en `model/semantic/unsloth_Qwen2.5-3B-Instruct/`.

**PEFT** (*Parameter-Efficient Fine-Tuning*): en lugar de modificar todo el modelo, solo se entrenan/adaptan unas capas livianas. Es más rápido de cargar y ocupa menos disco.

**Flujo de carga** (`load_model_and_tokenizer`):

1. Lee prompt + few-shots.
2. Intenta **PEFT** (default, `USE_UNSLOTH = False`).
3. Si falla, intenta **Unsloth** (requiere GPU NVIDIA y paquete `unsloth`).

**Carga diferida (*lazy load*):** la LLM **no** se importa al abrir `camera.py`. Solo cuando arranca `SemanticWorker`. Así la cámara puede abrir aunque Transformers no esté instalado (modo “solo glosas”).

### 5.3 Prompt del sistema y few-shots

**Prompt** (`sys_prompt.txt`): le dice a la LLM que es intérprete LSA→español rioplatense, qué no inventar, cómo formatear la respuesta, etc.

**Few-shots** (`few_shots_examples.json`): pares ejemplo:

```json
{
  "glosses": ["YO", "NOMBRE", "J", "U", "A", "N"],
  "spanish": "Me llamo Juan."
}
```

### 5.4 Ambigüedades del clasificador (O↔0 y 2↔V)

A veces dos señas **se ven casi iguales** para la cámara. El clasificador elige una u otra al azar.

| Par confuso | Cuándo es letra/nombre | Cuándo es número |
|-------------|------------------------|------------------|
| **O ↔ 0** | Deletreo, nombre, apellido → **O** | Documento, edad, teléfono → **0** |
| **2 ↔ V** | Deletreo → **V** | Secuencia numérica → **2** |

La LLM recibe estas reglas en el prompt **y** ejemplos few-shot:

```text
DOCUMENTO 1 2 O 4 5  →  documento 12045
NOMBRE 0 S C A R     →  Oscar
NOMBRE A 2 A         →  Ava
```

**Importante:** esto no reemplaza al clasificador; es una **capa de interpretación** que usa el contexto de las glosas vecinas.

### 5.5 Deletreo y números: camino rápido sin LLM

**Problema detectado:** si el enunciado era *solo* letras (`J U A N`) o *solo* dígitos (`1 2 3 4`), la LLM a veces respondía **vacío** o algo incoherente.

**Solución:** función `format_literal_utterance(glosses)`:

| Entrada | Salida | ¿Pasa por LLM? |
|---------|--------|----------------|
| `J U A N` | `Juan.` | No |
| `1 2 3 4` | `1234.` | No |
| `YO NOMBRE J U A N` | — | Sí (mixto / léxico) |

Ventajas: **instantáneo**, **predecible**, **funciona aunque la LLM falle**.

### 5.6 Dependencias y ejecución

**Entorno recomendado:** conda `lsa_gpu` con PyTorch + CUDA.

Paquetes extra para la LLM:

```bash
pip install transformers peft accelerate huggingface_hub
```

**Cómo correr la cámara:**

```bash
# Desde la carpeta src/
python -m camera

# Desde la raíz del repositorio
python -m src.camera
```


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

Archivo: `src/conversation_memory.py`

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
memory.clear()                                  # tecla 'c' en cámara
memory.as_messages()                            # → lista para translate_glosses
```

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

Constante: `CONVERSATION_HISTORY_SIZE = 10` en `model/semantic/config.py`.

**Tecla `c`:** limpia la memoria (`semantic_worker.clear_conversation()`).

### 6.6 Pendiente (producto)

- Campo de texto o micrófono para el oyente → `add_hearing()`.
- Panel en pantalla mostrando últimos turnos.
- Exportar `session_log` a JSON/CSV.

---


## 7. Patrones de diseño utilizados 

| Patrón | Dónde | Beneficio |
|--------|-------|-----------|
| **Worker + Queue** | `InferenceWorker`, `SemanticWorker` | UI fluida; tareas pesadas en background |
| **Lazy loading** | import LLM solo en hilo semántico | Arranque rápido; degradación graceful |
| **Policy object** | `RepeatGate` | Reglas de repetición testeables y extensibles |
| **Strategy / fallback** | literal vs LLM; PEFT vs Unsloth | Robustez ante fallos |
| **Sliding window** | `ConversationMemory` | Contexto acotado y predecible |
| **Shared state + Lock** | `shared_state` dict | Comunicación thread-safe entre workers y UI |
| **Separation of concerns** | `repeat_policy`, `conversation_memory`, `utils` | Cada archivo un dominio claro |

---

## 8. Archivos modificados o creados

| Archivo | Descripción |
|---------|-------------|
| `src/repeat_policy.py` | **Nuevo.** Clasificación digit/letter/other, `RepeatGate`, literales |
| `src/conversation_memory.py` | **Nuevo.** Turnos signer/hearing, ventana + session log |
| `src/camera.py` | Orquestación RT, integración memoria, tecla `c`, imports compat |
| `src/model/classifier/config.py` | `LETTER_MAX_CONSECUTIVE`, umbrales utterance |
| `src/model/semantic/config.py` | Paths, `CONVERSATION_HISTORY_SIZE` |
| `src/model/semantic/model.py` | Fix `BASE_MODEL_ID`, historial en `translate_glosses` |
| `src/model/semantic/prompts/sys_prompt.txt` | Reglas O/0, 2/V, literales, historial |
| `src/model/semantic/prompts/few_shots_examples.json` | Ejemplos ampliados |
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
- Pesos del clasificador entrenados (`tinyskeleton_best.pth`, `mapeo_clases.json`).
- Adapter LLM en `model/semantic/unsloth_Qwen2.5-3B-Instruct/`.
- Paquetes: `transformers`, `peft`, `accelerate`, `huggingface_hub`.

### Pasos

1. Activar entorno: `conda activate lsa_gpu`
2. Ir a `src/` y ejecutar: `python -m camera`
3. Elegir mano dominante en el modal inicial.
4. Esperar en consola: `[*] Semantic worker ready (translate_glosses).`
5. Señar frente a la cámara; observar glosas acumularse.
6. Pausar ~4 s → aparece español + voz (si VOICE activo).
7. Tecla **`c`** → nueva conversación (borra memoria).
8. Tecla **`q`** → salir.

### Si la LLM no carga

La cámara **no se cae**: sigue reconociendo señas y mostrando glosas. Revisar consola para el error exacto (deps faltantes, red para descargar base model, etc.).

---

## 11. Próximos pasos sugeridos

1. **Input del oyente:** caja de texto o STT (Speech-to-Text) → `semantic_worker.add_hearing(text)`.
2. **UI de historial:** mostrar últimos turnos en el canvas OpenCV.
3. **Tests automáticos** de `RepeatGate` y `ConversationMemory`.
4. **Suavizado visual** de landmarks en el overlay.
5. **Export de sesión** (`session_log` → JSON/CSV) para revisión posterior.
6. **Documentar environment.yml** con todas las deps LLM para evitar installs manuales.

---

## 12. Resumen ejecutivo (una página)

La rama `integration` conecta **cámara → reconocimiento de señas → lista de glosas → interpretación en español**, con dos mejoras importantes respecto a una versión ingenua:

1. **Repeticiones inteligentes:** permite deletreo y números reales sin dejar pasar rebotes del clasificador.
2. **Memoria conversacional corta:** la LLM entiende mejor frases sucesivas y ambigüedades (O/0, 2/V).

El sistema está diseñado para **no bloquearse** si la LLM falla: la cámara y el clasificador siguen. Deletreos y números puros funcionan **sin** LLM.

Esta bitácora debe actualizarse cada vez que se agreguen features (voz del oyente, persistencia, etc.) para mantener alineado al equipo técnico y no técnico.
