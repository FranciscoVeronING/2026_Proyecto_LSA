# Proyecto LSA 2026 — Ficha Técnica de Entrega

| | |
|---|---|
| **Módulo** | Integrador (rama `integration`) |
| **Iteración** | _(completar)_ |
| **Fecha estimada** | _(completar)_ |
| **Fecha entregado** | _(completar)_ |

---

## Alcance y objetivos de la entrega

Este módulo es el que **une las dos mitades del sistema**: el clasificador de señas, que
venía funcionando por separado, y el traductor semántico basado en una LLM. El objetivo de
esta ronda fue que dejaran de ser dos piezas independientes y pasaran a formar una
aplicación única que interpreta Lengua de Señas Argentina en tiempo real.

**Entrada:** video en vivo de una webcam (640×480 a 30 fps). La persona seña frente a la
cámara de forma continua, sin marcar dónde empieza ni dónde termina cada seña.

**Salida:** una oración en español, mostrada en pantalla y pronunciada en voz alta.

Entre esos dos extremos, el módulo resuelve tres problemas que no existían mientras cada
componente corría aislado:

1. **Segmentar el flujo continuo.** El clasificador espera secuencias recortadas de 32
   frames, pero la cámara entrega video sin cortes. El módulo detecta cuándo empieza y
   termina cada seña, y cuándo termina la *frase* completa.
2. **Decidir qué repeticiones son reales.** Una seña detectada dos veces seguidas puede ser
   un rebote del clasificador o una repetición legítima (el `1 1` de un documento, la doble
   letra de un nombre). La regla no puede ser la misma para todos los casos.
3. **Sostener la fluidez.** La LLM tarda segundos en responder y la síntesis de voz bloquea
   hasta terminar de hablar. Si eso ocurriera en el hilo de la cámara, el video se congelaría.

Alcance funcional entregado: reconocimiento de **91 señas** (abecedario dactilológico,
dígitos 0–9 y vocabulario de contexto policial/civil), traducción a español con contexto
conversacional de los últimos 10 turnos, salida por voz, modo de evaluación del clasificador
(`--eval`, 91 señas → CSV) y **evaluación offline del traductor** (`--eval-semantic`, glosas
→ español con métricas Token F1 / ROUGE-L / BLEU-4).

**Fuera de alcance en esta iteración:** captura de la persona oyente (la estructura de datos
ya contempla sus turnos, pero no hay ni micrófono ni entrada de texto conectados), y el
código de entrenamiento y preprocesamiento, que vive en otra rama.

---

## Especificaciones Técnicas y Decisiones de Diseño

### Arquitectura general

El sistema es una **cadena de cuatro etapas** conectadas por colas, donde cada etapa corre en
su propio hilo para que ninguna frene a la anterior:

```
webcam → MediaPipe Holistic → vector (32, 225) → TinySkeletonClassifier → glosa
                                                                             ↓
                                                                      UtteranceBuffer
                                                                      + RepeatGate
                                                                             ↓ (pausa de 4 s)
                                        pantalla / voz ← Qwen2.5-3B + LoRA ← enunciado
```

**Etapa 1 — Extracción de landmarks.** MediaPipe Holistic devuelve 33 puntos de pose y 21 por
mano. Se normalizan contra un ancla corporal (el punto medio de los hombros) y se escalan por
la distancia inter-hombros, de modo que el vector no dependa de la distancia de la persona a
la cámara ni de su altura. Resultado: 225 valores por frame (33×3 + 21×3 + 21×3).

**Etapa 2 — Clasificación.** `TinySkeletonClassifier`, una red propia de 413.148 parámetros
que combina dos capas Conv1D con un encoder Transformer de 2 capas. En lugar del *mean
pooling* habitual usa **attention pooling**: una capa lineal aprende qué frames de la
secuencia son informativos y pondera en consecuencia. Esto importa porque muchas señas son
estáticas y el gesto útil ocupa solo una fracción de los 32 frames capturados.

**Etapa 3 — Acumulación de glosas.** `UtteranceBuffer` junta las glosas aceptadas hasta que
pasan 4 segundos sin actividad, momento en que cierra el enunciado y lo despacha.

**Etapa 4 — Traducción.** Qwen2.5-3B-Instruct con un adaptador LoRA entrenado sobre pares
`{glosas → español}`, cargado vía PEFT.

### Tecnologías

| Componente | Tecnología | Rol |
|---|---|---|
| Extracción de landmarks | MediaPipe Holistic | Pose + manos, 75 puntos por frame |
| Clasificador | PyTorch 2.6 (Conv1D + Transformer) | 91 clases de señas |
| Traductor | Qwen2.5-3B-Instruct + LoRA (PEFT) | Glosas → español |
| Interfaz | OpenCV | Video, botones, sliders, panel de estado |
| Síntesis de voz | pyttsx3 | Salida hablada |
| Concurrencia | `threading` + `queue` de la stdlib | Cuatro hilos desacoplados |

### Decisiones de diseño

**Política de repeticiones por tipo de glosa.** La decisión de aceptar una seña repetida
depende de qué tipo de seña es, y esa lógica se aisló en una clase propia (`RepeatGate`) para
poder cambiarla sin tocar el resto:

| Tipo | Regla | Razón |
|---|---|---|
| Dígito (`0`–`9`) | Repeticiones ilimitadas | Un documento puede ser `1 1 2 2 3` |
| Letra (`A`–`Z`, `Ñ`) | Máximo 2 consecutivas | Existen dobles letras (`ANNA`), no triples |
| Resto (`HOLA`, `VER`…) | Nunca dos veces seguidas | `HOLA HOLA` no significa nada distinto de `HOLA`; si aparece, es rebote |

La regla del último tipo es por igualdad y no por tiempo. La primera versión usaba una
ventana de 1 segundo, pero resultó inalcanzable: el cooldown entre inferencias también es de
1 segundo, así que el sistema garantiza que dos glosas consecutivas estén separadas por al
menos ese lapso y ninguna caía dentro de la ventana. Las repeticiones no consecutivas
(`HOLA VER HOLA`) se conservan.

**Cierre del enunciado por inactividad.** Los 4 segundos se cuentan desde la última
*actividad de señado*, que incluye tres cosas: una glosa aceptada, una glosa reconocida pero
descartada por repetida, y las manos moviéndose en cámara. Se usa movimiento de manos y no
mera presencia, porque si bastara con tenerlas en pantalla, alguien que apoya las manos en
posición visible mantendría el enunciado abierto indefinidamente.

**Atajo para secuencias literales.** Si un enunciado es puro deletreo o puros dígitos, no
tiene sentido gastar segundos de LLM: `J U A N` se resuelve como `Juan.` directamente. Esto
además elimina un problema observado, en el que la LLM a veces devolvía cadena vacía ante
secuencias de letras sueltas.

**Desambiguación por contexto.** Varios pares de señas son casi idénticos en configuración de
mano y se distinguen solo por la ubicación o por un matiz de altura. Los identificados hasta
ahora son `O`/`0`, `2`/`V`, `AÑOS`/`G`, `CHAU`/`MARTES` y el trío `I`/`T`/`OJO`. Como el
clasificador trabaja frame a frame y no ve la frase completa, la resolución se delega a la LLM
mediante reglas explícitas en el *system prompt* y ejemplos en los *few-shots*. Los tres
primeros pares se desempatan por el tipo de secuencia (dígitos contra deletreo); `CHAU`/`MARTES`
se desempata por campo semántico, según haya o no glosas de tiempo alrededor. En consecuencia, un deletreo que contenga `I` o `T` **no** toma el atajo
literal: se prioriza interpretar bien sobre responder rápido.

**Memoria conversacional.** Una ventana deslizante de los últimos 10 turnos se inyecta como
mensajes `user`/`assistant` antes del turno actual. Se eligieron 10 turnos con rol, en vez de
reservar cupos fijos por interlocutor, porque las conversaciones reales no alternan de forma
pareja y un cupo fijo descartaría contexto reciente solo porque un lado habló más veces.

**Cuatro hilos, un estado compartido.** Cámara, clasificador, traductor y voz corren en hilos
separados que se comunican por colas y escriben en un diccionario compartido protegido por un
lock. Dos detalles concretos motivaron el diseño: `pyttsx3` bloquea hasta terminar el audio,
así que si viviera en el hilo del traductor la frase siguiente esperaría al parlante; y en
Windows usa COM, que exige crear y usar el motor de voz **en el mismo hilo**.

**Carga diferida de la LLM.** El traductor no se importa al arrancar. Si Transformers no está
instalado o la GPU no alcanza, la cámara abre igual y funciona en modo "solo glosas". En Windows,
los pesos se cargan primero en RAM y luego se mueven a GPU para evitar un crash silencioso de
`safetensors` al mapear directo a CUDA.

**Organización del código por dependencia.** El proyecto se estructura en cuatro paquetes
según qué tan pesadas son sus dependencias, no según el tema:

```
run.py              punto de entrada único
src/
├── app/            lo único que necesita cámara y pantalla
├── core/           lógica pura: stdlib + NumPy, testeable en milisegundos
├── classifier/     arquitectura, config y pesos entrenados
└── semantic/       LLM, adaptador LoRA y prompts
```

El criterio permite ejercitar las reglas de negocio (repeticiones, memoria conversacional)
sin montar cámara ni GPU. Los artefactos binarios viven en carpetas separadas del código que
los consume, porque el código cambia a diario y los pesos casi nunca.

---

## Métricas de Rendimiento y Validación

### Clasificador de señas

Medido sobre el conjunto de validación al cierre del entrenamiento
(`src/classifier/weights/metrics.json`, 31/07/2026):

| Métrica | Valor |
|---|---|
| Exactitud top-1 (validación) | **96,48 %** |
| Mejor pérdida de validación | 1,0427 |
| Clases | 91 |
| Muestras de entrenamiento / validación | 3.640 / 910 |
| Muestras por clase | 50 |
| Épocas ejecutadas | 19 (con *early stopping*) |
| Aumentación de datos | Activada (ruido σ≈0,03 + escala 0,85–1,15) |

Tamaño del modelo: 413.148 parámetros, 4,1 MB en disco. Entrada de (32, 225).

### Rendimiento en tiempo real

Medido en este equipo (RTX 3060), promedio de 100 inferencias tras 10 de calentamiento:

| Medición | Valor |
|---|---|
| Latencia del clasificador (GPU) | 1,08 ms |
| Latencia del clasificador (CPU) | 0,78 ms |
| Cooldown entre inferencias | 1,0 s (configurable) |
| Pausa para cerrar enunciado | 4,0 s (configurable) |

La CPU resulta marginalmente más rápida que la GPU porque, con 413 k parámetros, el costo de
lanzar los kernels CUDA supera al del cómputo. La GPU sigue siendo necesaria para la LLM. En
la práctica, el cuello de botella no es el clasificador sino MediaPipe y la generación de la
LLM.

### Validación funcional

Batería ejecutada sobre la lógica de negocio, comparando entrada y salida esperadas:

| Caso | Entrada | Salida | Estado |
|---|---|---|---|
| Dígitos repetidos | `1 1 1 2 2` | `1 1 1 2 2` | OK |
| Letra repetida tres veces | `A A A B` | `A A B` | OK |
| Rebote del clasificador | `HOLA HOLA` (a 2 s) | `HOLA` | OK |
| Repetición no consecutiva | `HOLA VER HOLA` | `HOLA VER HOLA` | OK |
| Descarte que reinicia el reloj | Repetida a los 3 s | No cierra hasta 4 s después | OK |
| Manos activas | Movimiento hasta t=5 s | No cierra hasta t=9 s | OK |
| Deletreo literal | `J U A N` | `Juan.` sin LLM | OK |
| Deletreo ambiguo | `T I N A` | Derivado a la LLM | OK |
| Letra suelta | `A` | Derivado a la LLM | OK |
| Secuencia mixta | `YO 5` | Derivado a la LLM | OK |
| Memoria conversacional | 2 turnos | 4 mensajes de chat | OK |
| `clear()` | — | Ventana en 0, log intacto | OK |

Verificación de integración: cadena completa de imports, resolución de las seis rutas a
artefactos, carga real de los pesos (91 clases) y construcción del prompt con *few-shots*
(5.220 caracteres). Ejecución en vivo con cámara confirmada de forma manual.

### Traductor semántico (evaluación offline)

Modo `--eval-semantic`: corre el pipeline glosas → español sobre un JSON de pares referencia,
sin cámara. Dataset hold-out por defecto: `src/semantic/eval_dataset.json` (20 oraciones que
**no** están en los few-shots del prompt).

```bash
python run.py --eval-semantic
python run.py --eval-semantic --eval-semantic-history   # conversación encadenada
python run.py --eval-semantic --eval-semantic-dataset ruta/dataset.json
```

Métricas: exact match, Token F1, ROUGE-L, BLEU-4. Salida: consola + CSV `eval_semantic_<fecha>.csv`.

**Primera corrida** (RTX 3060, hold-out, sin historial, 06/08/2026):

| Métrica | Valor |
|---|---|
| Exact match | 2/20 (10 %) |
| Token F1 (promedio) | 0,50 |
| ROUGE-L (promedio) | 0,49 |
| BLEU-4 (promedio) | 0,20 |

Interpretación: muchas traducciones son cercanas pero no idénticas a la referencia
(p. ej. “Me siento bien” vs “Estoy bien”). Token F1 y ROUGE-L capturan mejor esa
partialidad que el exact match. Para métricas comparables con el entrenamiento (~200 secuencias),
falta correr el dataset completo de fine-tuning.

**Carga de la LLM en Windows:** se detectó crash silencioso al mapear pesos directo a GPU
(`Loading checkpoint shards: 0%`). Solución: cargar en CPU y mover a GPU después (~15 s extra
en el primer arranque). Los prints de consola evitan caracteres Unicode (`→`, `…`) incompatibles
con `cp1252`.

### Pendiente de medición

| Métrica | Estado |
|---|---|
| BLEU/ROUGE-L sobre dataset completo (~200 secuencias) | Modo `--eval-semantic` listo; falta commitear el JSON de evaluación del entrenamiento |
| Exactitud en vivo (91 señas) | El modo `--eval` genera el CSV, falta ejecutar la corrida completa |
| Latencia de la LLM por enunciado | Sin instrumentar |
| Eval con historial conversacional | Flag `--eval-semantic-history` implementado; falta dataset de diálogos encadenados |

### Hallazgos resueltos durante la iteración

**Discrepancia de cabezas de atención entre entrenamiento e inferencia.** El checkpoint se
entrenó con `num_heads = 4` (registrado en `metrics.json`), pero la configuración instanciaba
el modelo con `NUM_HEADS = 8`. **PyTorch no detectaba el error**: las matrices de
`MultiheadAttention` tienen la misma forma con cualquier cantidad de cabezas, así que
`load_state_dict` cargaba sin protestar y el modelo corría repartiendo las 128 dimensiones en
8 grupos de 16 en lugar de 4 de 32. Se verificó experimentalmente que, sobre la misma entrada,
las dos configuraciones diferían hasta 0,062 en probabilidad; con un umbral de aceptación de
0,75, esa diferencia podía cambiar si una glosa entraba o no al enunciado. **Corregido**:
`NUM_HEADS = 4`, coincidente con el checkpoint, por lo que el 96,48 % reportado ahora sí
describe la configuración que corre en cámara.

**Señas con movimiento listadas como estáticas.** `STATIC_SIGN_CLASSES` incluía `H` y `Z`, que
tienen desplazamiento y por criterio del equipo no corresponden a ese grupo. **Corregido**: la
lista quedó sin `H`, `Z`, `R` ni `J`, consistente con el criterio de que esas cuatro señas
tienen movimiento aunque duren poco.

### Hallazgos abiertos

**1. La cuantización de 4 bits no está activa en la ruta por defecto.** `LOAD_IN_4BIT = True`
solo lo consume el camino de Unsloth, que está desactivado (`USE_UNSLOTH = False`). El camino
PEFT, que es el que efectivamente se usa, carga en `float16`. Conviene alinear la
configuración con el comportamiento real, o bien pasar `quantization_config` a PEFT si se
busca reducir el consumo de VRAM. Es relevante para la redacción del informe: describir el
módulo como "cuantizado en 4 bits" no sería exacto hoy.

**2. El adaptador LoRA se entrenó sin historial conversacional.** El dataset de fine-tuning
son pares de un solo turno. Inyectar turnos previos aleja el prompt de esa distribución. Con
la memoria vacía el prompt es idéntico al de entrenamiento, así que el primer enunciado de
cada conversación nunca se ve afectado y la diferencia aparece recién del segundo en adelante.
Por eso el historial es apagable (`USE_CONVERSATION_HISTORY`). El dataset de evaluación
actual, al ser de un solo turno, **no puede detectar esta degradación**: haría falta armar
diálogos encadenados.

**3. Pares confundibles sin cobertura de ejemplos en ambas direcciones.** Para `CHAU ↔ MARTES`
hay ejemplos que enseñan a leer `CHAU` como `MARTES` en contexto temporal, pero no la
dirección inversa, porque con el vocabulario disponible no se encontró una frase natural donde
un `MARTES` detectado deba interpretarse como despedida. Queda pendiente sumarla si aparece el
caso en uso real.
