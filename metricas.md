# Métricas de evaluación ILSA

Catálogo de todo lo que el sistema registra en una sesión de evaluación.
Una **seña** no es lo mismo que un **turno de traducción**:

| Unidad | Qué es | Archivo |
|--------|--------|---------|
| Seña | Un recorte de manos que entra a TinySkeleton (`POST /sign`) | `metrics_signs.csv` |
| Turno | Una oración traducida por el GGUF en Cloud Run (Directo o Inverso) | `metrics_breakdown.csv` |
| Evento | Cualquier hito (seña, turno, STT, FPS, espera) | `live-AAAAMMDD.jsonl` |

Carpeta de salida: `%LOCALAPPDATA%\ILSA\metrics\`.

Relojes: Python usa `time.perf_counter()`; el navegador usa `performance.now()` o `Date.now()` cuando hay que cruzar contextos (offscreen ↔ Meet).

---

## Qué ejecutar para tomar las métricas

No hay un script aparte de “empezar a medir”. El muestreo arranca con ILSA
(`start_sampler`) y cada seña / turno / STT / entrega en Meet **escribe sola**
en esa carpeta. La evaluación es una sesión real: motor + extensión + Meet.

### 1. Motor ILSA

En la raíz del repo, con el entorno que ya usás para el proyecto:

```bat
LSABackend.bat
```

o:

```bat
python run_backend.py
```

(`ILSA.exe` también vale: el sampler se enciende igual.)

En la ventana: esperá el splash (traductor + T-pose) → **Sordo** o **Oyente**
→ si es Sordo, zurdo/diestro → **Iniciar**. Dejá ILSA abierto durante toda la
prueba. No hace falta `--headless` para esta evaluación.

### 2. Extensión Chrome

`chrome://extensions` → modo desarrollador → **Cargar descomprimida** →
carpeta `extension/`. Si recargaste código, recargá la extensión **y** la
pestaña de Meet.

En el popup de la extensión: modo de salida (subtítulos / voz / ambos) según
lo que quieras medir. `Espera_voz_ms` solo se llena si la voz en off está
habilitada y el micrófono de Meet no está muteado.

### 3. Google Meet

Abrí una llamada, dejá que el HUD de ILSA se conecte (`En marcha` en la
ventana). Entonces:

| Modo | Qué hacer para generar filas |
|------|------------------------------|
| **Sordo (Directo)** | Señas a cámara. Cada seña → fila en `metrics_signs.csv`. Al bajar las manos ~4 s → un turno en `metrics_breakdown.csv` + espera de subtítulos/voz |
| **Oyente (Inverso)** | Hablar al mic de Meet o escribir en el chat. Cada oración → fila Inverso (STT, GGUF, T-pose) |

RAM/CPU de ILSA y del navegador se muestrean todo el tiempo que el motor está
encendido (hace falta Chrome o Edge abierto para `Browser_*`).

### 4. Dónde están los archivos

Al terminar la sesión (o en cualquier momento), abrí:

```
%LOCALAPPDATA%\ILSA\metrics\
```

| Archivo | Se crea cuando… |
|---------|-----------------|
| `metrics_signs.csv` | Primera seña aceptada (`POST /sign`) |
| `metrics_breakdown.csv` y `.json` | Primer turno GGUF (cierre de oración Directo o `POST /hearing`) |
| `live-AAAAMMDD.jsonl` | Cualquier evento (seña, STT, FPS, espera) |

Si la carpeta está vacía: ILSA no llegó a Iniciar, Meet no habló con
`127.0.0.1:8765`, o no hubo ninguna seña/oración.

### Qué no alcanza para esta evaluación

| Comando | Para qué sirve | Qué **no** cubre |
|---------|----------------|------------------|
| `python tests/test_performance_pipeline.py` | Tests locales del CSV, sin Meet | No es una medición real |
| `python scripts/bench_latency.py --channel signer --n 20` | RNF-01 contra Cloud Run (glosas de dataset) | No hay cámara, Holistic, pausa real, HUD, TTS ni RAM del navegador. Entrega Directo = 0. Escribe `bench-*.jsonl`, no el CSV de sesión |

`bench_latency.py` es un banco del **traductor en la nube**, no de la
evaluación de escritorio. Para el catálogo de este documento hay que correr
el flujo 1–3.

---

## Canales

| `Canal` | Modo de producto | Entrada | Salida |
|---------|------------------|---------|--------|
| **Directo** | Sordo (LSA → español) | Señas en cámara | Subtítulos en Meet y, si está habilitada, voz en off |
| **Inverso** | Oyente (español → LSA) | Voz (Vosk) o chat de Meet | Glosas + avatar T-pose en el canvas |

---

## Pipeline de un turno (cinco tramos)

```
  [manos quietas]     [espera HTTP]     [tokens GGUF]     [red sola]      [pantalla / voz]
   Tramo 1               Tramo 2           Tramo 3          Tramo 4           Tramo 5
    Pausa                 Cola           Inferencia       Transporte         Entrega
```

**RNF-01** mira solo el tramo 3: el GGUF cuantizado 4-bit debe terminar en **≤ 1500 ms** (techo 2000 ms en banda ancha). La pausa, la cola, la red y la entrega se miden, pero **no entran** a ese presupuesto.

TinySkeleton (clasificador de señas) **no** es el GGUF. Su tiempo es `classifier_ms`, en el CSV de señas.

---

## `metrics_breakdown.csv` — un turno GGUF

### Identificación

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `Timestamp` | Momento UTC del turno | Reloj de ILSA al cerrar el turno |
| `Canal` | Directo o Inverso | Modo de la sesión |
| `Via` | Camino de traducción | `llm` = Cloud Run; `literal` / `lexical` = una sola glosa sin GGUF; `fallback` = respuesta de reserva |
| `Modelo` | Archivo GGUF usado | Lo informa Cloud Run (`model_used`) |
| `N_glosses` | Glosas de entrada (Directo) o de salida (Inverso) | Conteo de tokens |
| `Input_chars` / `Output_chars` | Longitud del texto de entrada y salida | `len()` del string |

### Latencia del pipeline

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `Pausa_ms` | Heurística de cierre: manos fuera de cuadro hasta `POST /utterance/end`. **No es tiempo de modelo.** | Extensión: `t_pausa_inicio` al bajar las manos; `pause_ms` al cumplir ~4 s (`UTTERANCE_PAUSE_SEC`). En Inverso vale **0** (la oración ya viene cerrada por STT/chat) |
| `Cola_ms` | Espera en Cloud Run antes de que llama.cpp tome el turno | `t_inicio_inferencia − t_llegada_servidor`. Ambos canales comparten un mutex |
| `Inferencia_ms` | Tiempo de generación GGUF. **Esta es RNF-01.** | `t_fin_inferencia − t_inicio_inferencia` en Cloud Run |
| `Transporte_ms` | Red solamente | RTT cliente − cola − inferencia |
| `Entrega_ms` | Presentación en cliente | Directo: pintar subtítulo / arrancar TTS (Meet reporta `delivery_ms`). Inverso: descarga T-pose + compose en ILSA, más el tiempo de aplicar el clip en Meet |
| `Latencia_Total_ms` | Suma **sin** pausa | `Cola + Inferencia + Transporte + Entrega` |
| `Cumple_RNF01` | ¿La inferencia GGUF ≤ 2000 ms? | `true` / `false` sobre `Inferencia_ms` |

### Espera percibida (incluye pausa)

Reloj desde **fin de seña** (manos abajo) hasta que el otro ve o escucha el resultado. Incluye la pausa de ~4 s.

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `Espera_subtitulos_ms` | Fin de seña → subtítulo visible en Meet | Reloj de pared (`Date.now()`) desde `t_pausa_inicio` hasta `LSA_CAPTION`. Si Meet aún no reportó, se estima como `Pausa + Cola + Inferencia + Transporte + Entrega` |
| `Espera_voz_ms` | Fin de seña → voz en off sonando en la llamada | Mismo origen; se cierra cuando el WAV hace `source.start()` en el mix de Meet. Vacía si el modo es solo subtítulos, el micrófono está muteado o el TTS falla |

En Inverso no hay “fin de seña”: `Espera_subtitulos_ms` queda como suma de tramos (pausa 0) y `Espera_voz_ms` no aplica.

### Recursos locales (PC del usuario)

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `RAM_mb` | RAM del **proceso Python de ILSA** | `psutil` sobre `os.getpid()` (working set / RSS), muestreo ~1 s |
| `CPU_percent` | CPU de **ese mismo proceso** | `psutil.cpu_percent` desde la muestra anterior. 0–100+ si usa más de un núcleo |
| `Browser_RAM_mb` | RAM del **navegador** | Suma del working set de todos los `chrome.exe` o `msedge.exe` (familia agrupada, como el Administrador de tareas). Si hay Chrome y Edge, se toma la de más RAM |
| `Browser_CPU_percent` | CPU de esa familia | Suma del CPU de esos subprocesos. Puede **pasar de 100 %** |
| `Browser_name` | `chrome` o `msedge` | Nombre de la familia elegida |
| `Browser_n_proc` | Cuántos subprocesos se sumaron | Conteo al muestrear |
| `Js_heap_mb` | Heap V8 del tab/extensión | `performance.memory.usedJSHeapSize`. **No** es la RAM del proceso Chrome |

`RAM_mb` y `Browser_RAM_mb` son procesos distintos. El heap JS es una tercera cifra, más chica.

### GGUF / cache KV (Cloud Run)

Estas columnas se llenan cuando Cloud Run responde con stats de llama.cpp. Si el servicio aún no exporta esos campos, quedan vacías.

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `Prompt_tokens` | Tokens del prompt de este turno | llama.cpp |
| `Completion_tokens` | Tokens generados | llama.cpp |
| `Eval_tokens` | Tokens evaluados en el forward | llama.cpp (`n_eval`) |
| `Sys_tokens` | Tokens del system prompt anclado en KV | llama.cpp / motor |
| `Kv_hit` | ¿Se reutilizó la cache KV (rewind, sin reprocesar el system prompt)? | Motor semántico |
| `Kv_mb` | Estimación de memoria de la cache KV | `n_layer × n_embd × n_tokens` (orden de magnitud) |
| `T_prompt_eval_ms` | Tiempo de evaluar el prompt | `llama_get_timings` |
| `T_eval_ms` | Tiempo de generar tokens | `llama_get_timings` |
| `Tokens_per_s` | Velocidad de generación | tokens / `T_eval_ms` |
| `N_ctx` | Contexto configurado | Motor |
| `N_tokens` | Tokens ocupados en KV en ese momento | Motor |
| `Queue_depth` | Pedidos esperando el mutex de inferencia | Contador `_WAITERS` en Cloud Run |
| `N_threads` | Hilos de llama.cpp | Motor |
| `Cloud_RAM_mb` | RAM del **proceso GGUF en Cloud Run** | RSS del contenedor, no de ILSA local |
| `Cloud_CPU_percent` | CPU de ese proceso en la nube | `psutil` en Cloud Run |

TinySkeleton no tiene cache KV entre señas.

### Inverso: STT, T-pose, Ñ

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `Stt_ms` | Vosk transcribiendo el WAV | Reloj de pared de `POST /stt` |
| `Tpose_fetch_ms` | Bajar clips `.pose` (red + disco) | Suma de GET / lecturas de cache |
| `Tpose_compose_ms` | Armar el clip compuesto | Compositor local |
| `Tpose_gets` | Pedidos de pose (incluye cache) | Contador |
| `Tpose_cache_hits` | Poses servidas desde disco, sin GET | Contador |
| `Tpose_missing_n` | Glosas sin clip | Largo de `missing` |
| `Enie_restored` | Cuántas Ñ se restauraron en glosas (p. ej. `ANOS` → `AÑOS`) | `restore_enie_report` |

---

## `metrics_signs.csv` — una seña (TinySkeleton)

No es un turno GGUF. Cada fila es un `POST /sign`.

| Columna | Qué es | Cómo se obtiene |
|---------|--------|-----------------|
| `Timestamp` | UTC de la seña | ILSA |
| `accepted` | ¿Pasó cooldown y largo mínimo? | `true` si se infirió |
| `added` | ¿Entró al buffer de la oración? | `false` si umbral, `DESCONOCIDO` o repetición |
| `reason` | Por qué no se agregó | `confidence`, `repeat`, `cooldown`, `too_short`, `bad_tensor` |
| `gloss` | Primera predicción | TinySkeleton top-1 |
| `conf1` `conf2` `conf3` | Confianzas top-3 | Softmax del clasificador |
| `conf_margin` | Separación top-1 vs top-2 | `conf1 − conf2` |
| `classifier_ms` | Forward del modelo 16×225 en CPU | Reloj alrededor de la inferencia TinySkeleton |
| `prepare_ms` | EMA + armar la matriz 16×225 | Reloj antes del forward |
| `total_ms` | Todo el `POST /sign` en ILSA | Reloj del handler |
| `capture_ms` | Duración del recorte en la extensión | De abrir buffer a `flushSign` |
| `holistic_ms` | Un tick de MediaPipe Holistic | Offscreen, último frame |
| `jpeg_ms` | JPEG recibido → landmarks de ese frame | Offscreen (`lastJpegMs`) |
| `time_to_record_ms` | Desde que hay mano hasta que arranca el recorte | `captureStartedAt − handsSince` |
| `n_frames` | Frames que ILSA recibió | Largo del JSON |
| `n_raw` | Frames crudos en el buffer | Antes de subsamplear a 16 |
| `n_sent` | Frames enviados (≤ 16) | Tras subsampleo uniforme |
| `payload_kb` | Tamaño del POST | `body.length / 1024` en la extensión |
| `left_handed` | ¿Modo zurdo? | Sesión ILSA |
| `RAM_mb` / `CPU_percent` | ILSA en ese instante | Igual que en el CSV de turnos |
| `Browser_*` | Navegador en ese instante | Igual que en el CSV de turnos |

Rechazos por cooldown / too_short van al JSONL; no siempre a este CSV.

---

## `live-AAAAMMDD.jsonl` — log completo

Una línea JSON por evento. Incluye lo de los CSV más campos que no tienen columna propia.

| `kind` | Cuándo | Extra típico |
|--------|--------|----------------|
| `sign` | Cada `POST /sign` | Igual que `metrics_signs.csv` |
| `utterance` | Cierre Directo (`/utterance/end`) | Pausa, cola, inferencia, KV, espera |
| `hearing` | `POST /hearing` | STT, T-pose, Ñ, KV |
| `stt` | `POST /stt` | `stt_ms`, tamaño del WAV |
| `client` | `POST /metrics/client` | FPS, heap JS, `delivery_ms`, esperas de subtítulo/voz |

FPS (no están en los CSV de turnos/señas; sí en JSONL):

| Campo | Qué es | Cómo se obtiene |
|---------------|-----------------|
| `fps` | `canvas.captureStream(24)` hacia Meet | Extensión, hook de cámara |
| `holistic_fps` | MediaPipe en el sandbox offscreen | Extensión |
| `js_heap_total_mb` / `js_heap_limit_mb` | Heap V8 total y tope | `performance.memory` |

`/health` resume percentiles (p50/p95 de clasificador, inferencia, espera, etc.).

---

## Qué no se mide

| Hueco | Por qué |
|-------|---------|
| RAM de **un solo tab** de Meet | Chrome no lo expone a la extensión; se suma **toda** la familia Chrome/Edge |
| WER / calidad semántica vs referencia | No hay etiquetas en vivo; eso es `--eval-semantic` |
| CPU/RAM de Chrome **antes** de abrir el navegador | Columnas `Browser_*` vacías |
| KV / tokens GGUF | Vacío hasta que Cloud Run se despliegue con el export de stats |
| Duración total del audio TTS | Se marca el **arranque** de la voz, no el final |

---

## Cómo leer una fila Directo (ejemplo)

Seña `HOLA` aceptada → ~4 s de manos abajo → Cloud Run traduce → subtítulo y (si aplica) voz.

1. `metrics_signs.csv`: `classifier_ms` / `prepare_ms` / `conf1` de esa seña.
2. `metrics_breakdown.csv`, canal Directo:
   - `Pausa_ms` ≈ 4000
   - `Inferencia_ms` = GGUF (RNF-01)
   - `Latencia_Total_ms` = cola + GGUF + red + entrega (**sin** los 4 s)
   - `Espera_subtitulos_ms` ≈ 4000 + esa latencia (lo que espera quien mira Meet)
   - `Espera_voz_ms` un poco mayor, cuando arranca el audio
   - `RAM_mb` = ILSA; `Browser_RAM_mb` = Chrome/Edge

---

## Archivos de código

| Archivo | Rol |
|---------|-----|
| `src/utils/process_stats.py` | Muestreo ILSA vs navegador |
| `src/utils/benchmark_logger.py` | CSV / JSON de turnos y señas |
| `src/backend/metrics.py` | JSONL + resumen `/health` |
| `src/backend/signer_session.py` | Señas + cierre Directo |
| `src/backend/hearing_session.py` | Turno Inverso |
| `extension/lib/capture.js` | Pausa y fin de seña |
| `extension/content/meet.js` | Espera de subtítulos y de voz |
| `extension/offscreen.js` | Holistic, JPEG, `sign_end_ms` |
