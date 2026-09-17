# Proyecto LSA 2026
## Ficha Técnica de Entrega

# Refinamiento: feature engineering

| | |
|---|---|
| **Módulo** | Clasificador de señas — feature engineering y diagnóstico de captura |
| **Iteración** | Septiembre 2026 |
| **Fecha estimada** | 15/09/2026 |
| **Fecha entregado** | 17/09/2026 |
| **Autores** | Francisco Veron, Maite Nigro |
| **Rama** | `scratch-mediapipe-v2` |
| **Entorno** | Conda `lsa_gpu` · Python 3.11 · MediaPipe 0.10.21 · PyTorch (CUDA) |

---

## Alcance y objetivos de la entrega

Esta iteración **no cambia la arquitectura del clasificador ni el tensor que consume**. Es una ronda de *refinamiento*: construir instrumentación para responder, con evidencia numérica, dos preguntas que hasta ahora se contestaban por intuición.

1. **¿Qué señas se van a confundir, y por qué?** Antes se detectaban a mano, mirando fallos sueltos de cámara. Ahora hay un análisis cuantitativo sobre el dataset completo que rankea pares confundibles y, para cada uno, identifica qué medida geométrica los separaría.
2. **Cuando el modelo falla en vivo pero acierta sobre el dataset, ¿dónde está la diferencia?** Antes esto era un callejón sin salida («en la webapp anda peor»). Ahora hay una referencia dorada que aísla inferencia de extracción de landmarks, y trazabilidad de las condiciones de captura en cada evaluación.

Como consecuencia de ese diagnóstico se corrigió un artefacto de captura (aspect ratio 4:3 vs 16:9 de entrenamiento) y se reentrenó el baseline de **97 clases / 16 frames**. La eval de cámara del 17/09 (padding activo, 1 toma por seña) cerró el experimento: **`I → ojo` desapareció**; el top-1 crudo sobre 97 señas es **77,3 %** (top-3 **94,8 %**); las 36 letras/números quedan en **88,9 %**. Detalle: [eval_97senias_20260917.md](eval_97senias_20260917.md).

### Qué recibe como entrada

| Entrada | Detalle |
|---|---|
| Dataset de video | 6.748 MP4 en `dataset/`, todos 1920×1080 (16:9), ~69 clips por seña |
| Dataset de landmarks | 6.680 clips `.npy` de forma `(16, 225)` en `dataset_landmarks_16frames/` |
| Checkpoints previos | `2026_08_29_model_opt` (10 frames), `2026_08_26_model_opt` (12 frames), `2026_08_18_model_no_opt` (16 frames), 94 clases |
| Sesiones de cámara | CSV de `camera.py --eval` más los tensores de entrada de cada toma |

### Qué genera como salida

| Salida | Detalle |
|---|---|
| Reporte de confusiones | Ranking de pares, separabilidad por feature y un PNG comparativo por par |
| Mapa por seña | Un directorio por seña con snapshot de pose, series temporales y métricas en JSON |
| Diagnóstico de evaluación | Cruce entre fallos de cámara y separabilidad del dataset: ambigüedad real vs problema de captura |
| Referencia dorada | Predicciones reproducibles sobre `.npy`, sin cámara, para testear paridad entre implementaciones |
| Corrección de captura | Letterboxing a 16:9 en `camera.py`, con test de validación |
| Checkpoint nuevo | TinySkeleton baseline 16 frames, **97 clases**, val top-1 98,28 % |
| Eval de cámara 17/09 | `eval_97senias_20260917_160148.csv`: 77,3 % top-1 / 94,8 % top-3, padding 16:9 |

El artefacto conceptual central es un **descriptor de 26 features interpretables** que se calcula al vuelo sobre el vector normalizado de 225 dimensiones, sin modificar los `.npy` ni el pipeline de producción.

---

## Especificaciones técnicas y decisiones de diseño

### Contexto: por qué feature engineering y no más capacidad de red

El clasificador (TinySkeleton: Conv1D + Transformer con attention pooling) recibe por frame **225 números**: 33 puntos de pose × 3 + 21 × 2 manos × 3. Cada frame se centra en el punto medio de los hombros y se divide por la distancia inter-hombros, lo que da invarianza a persona y distancia a cámara.

En iteraciones previas se buscó arquitectura con Optuna. El search **mejoraba la pérdida de validación offline y empeoraba el top-1 de webcam**: una configuración llegó a 99 % offline y rindió peor en cámara, además mal calibrada. Con ~60 videos por clase, el cuello de botella no es capacidad del modelo sino **señal**.

De ahí la decisión de esta ronda: hacer *explícita* una relación que el modelo podría aprender en teoría, pero que con poco dato y 225 dimensiones diluidas no aprende de forma estable. El feature engineering cumple dos roles — **diagnóstico** (medir si la señal existe en el dato antes de gastar un reentrenamiento) y, potencialmente, **entrada al modelo** (solo si el diagnóstico lo justifica). En esta entrega el descriptor **no se cableó al entrenamiento**: se usó para explicar confusiones y para decidir *qué* reentrenar (97 clases, V nueva, 16 frames), no *con qué features extra*.

### El descriptor de 26 features

Implementado en `src/hand_face_features.py`. Tres familias, con unidades y propósitos distintos.

**Familia 1 — Geometría mano–rostro (6 features).** Normalizadas en anchos de hombro, igual que el clasificador.

`dy_index_mouth`, `dy_index_eye`, `dy_index_nose`, `dist_index_mouth`, `dist_index_eye`, `dy_wrist_nose`

Los `dy` son **con signo**. En MediaPipe el eje Y crece hacia abajo, así que `dy_index_mouth < 0` significa índice por encima de la boca. La boca se toma como punto medio de las comisuras (landmarks 9 y 10 de pose) y el ojo como **el más cercano al índice**, sin asumir lado de la cara. Si falta la pose o las dos manos, la función devuelve ceros en lugar de calcular distancia al origen, que se leería erróneamente como «mano en el ancla de hombros».

Esta familia ataca la distinción por **lugar de apoyo**: en LSA, `I`, `T` y `ojo` comparten configuración de mano y difieren en dónde se apoya (mejilla / debajo de la boca / cerca del ojo).

**Familia 2 — Movimiento de muñeca (6 features).** Desvío estándar y rango de la trayectoria de muñeca en X, Y y Z a lo largo del clip.

`std_wrist_x/y/z`, `range_wrist_x/y/z`

Ataca la familia **letra estática vs seña dinámica**, donde el rasgo distintivo es el movimiento y no la forma: `L`/`lunes` (agite lateral, eje X), `G`/`años` (vertical, eje Y), `F`/`donde` y `V`/`viernes` (profundidad, eje Z).

**Familia 3 — Configuración de dedos (14 features).** Agregada al descubrir que las 12 anteriores eran ciegas a la forma de mano.

| Subfamilia | Cómo se calcula | Qué captura |
|---|---|---|
| `ext_*` (5) | punta → muñeca, sobre tamaño de mano | dedo estirado vs recogido |
| `curl_*` (5) | punta → nudillo, sobre largo estirado del dedo | 1 = recto, ~0,5 = cerrado |
| `spread_*` (4) | punta → punta vecina, sobre tamaño de mano | dedos juntos vs separados |

Decisión clave: estas 14 se normalizan por **muñeca → nudillo del medio**, no por ancho de hombros. Eso las vuelve invariantes a escala *y a posición del brazo*, que es exactamente lo que se necesita para comparar handshapes independientemente de dónde esté la mano.

### Decisión de diseño: el descriptor no se guarda en los `.npy`

Tres funciones del pipeline asumen que **todo el vector son triples (x, y, z)**:

- `mirror_landmarks_for_left_handed` hace `reshape(-1, 3)` y flip del eje X.
- `augment_batch_3d` rota, escala y agrega ruido sobre `view(..., -1, 3)`.
- el trim de movimiento trata todo lo que sigue a la pose como manos.

Si se concatenaran 26 escalares al `.npy` guardando `(T, 251)`, el reshape mezclaría distancias con coordenadas, la rotación 3D **giraría distancias como si fueran puntos**, y el recorte del gesto se contaminaría. Por eso el cálculo es **al vuelo** sobre el `(225,)` ya normalizado, y el día que se integren al modelo debe hacerse *después* del augmentation espacial. Los `.npy` no cambian y no hace falta re-preprocesar para diagnosticar.

Tampoco se activa Face Mesh: sumaría 468 × 3 = 1.404 dimensiones, que con ~60 clips por clase invita a memorizar. Y no se usa Z como señal principal, porque el depth de MediaPipe es el eje más ruidoso.

### Metodología de análisis de confusiones

`src/analyze_sign_confusions.py` trabaja en dos niveles independientes.

**Nivel A — espacio crudo.** Clasificación 1-NN *leave-one-out* sobre el `(16, 225)` aplanado, o sea el mismo espacio que consume TinySkeleton. Responde «¿qué pares están geométricamente pegados?» sin depender de ningún checkpoint. El número absoluto es optimista, porque clips de la misma sesión de grabación se parecen entre sí; lo informativo es **qué pares aparecen cruzados**.

**Nivel B — separabilidad por feature.** Para cada par confundido y cada una de las 26 features, se calcula el AUC de Mann-Whitney entre las dos distribuciones y se reporta

```
sep = |AUC − 0,5| × 2
```

Escala directa e interpretable: **1,0** las distribuciones no se tocan, **0,0** son idénticas. Umbrales del reporte: ≥ 0,80 separable, ≥ 0,60 con margen, ≥ 0,30 parcial, < 0,30 no separable.

La combinación de ambos niveles es lo que da la respuesta accionable: el nivel A dice *dónde* está el problema y el nivel B dice *con qué medida* se resolvería.

### Instrumentación de la evaluación en vivo

Tres piezas para cerrar la brecha entre «anda bien acá» y «anda peor en la webapp».

**`src/predict_npy.py` — referencia dorada.** El pipeline tiene dos mitades sospechosas:

```
landmarks (MediaPipe) → normalización + trim + subsampleo → modelo
```

Este script fija la segunda mitad corriendo los modelos sobre un `.npy` ya extraído, sin cámara. Si la webapp alimentada con el **mismo** `.npy` da el mismo top-3, la divergencia está enteramente en la extracción de landmarks (versión de MediaPipe, resolución, fps, espejado). Si da distinto, está en la carga del modelo o el armado del tensor. Convierte una pregunta difusa en un test binario.

**Persistencia de tensores de toma.** `camera.py --eval` ahora guarda el tensor exacto que se le pasó al modelo en cada toma, como `.npy` en un directorio `<csv>_takes/`. Esto permite reanalizar una sesión de cámara sin volver a grabarla, que es lo que hizo posible el hallazgo de aspect ratio.

**Trazabilidad de condiciones.** El CSV de evaluación registra `frame_size` y la versión de MediaPipe en cada fila. Sin esto, dos evaluaciones corridas en condiciones distintas son indistinguibles a posteriori.

**`src/compare_eval_vs_dataset.py`.** Cruza los fallos de una sesión de cámara contra la separabilidad del dataset para el par involucrado, distinguiendo dos diagnósticos que se veían iguales: **ambigüedad real** (el dataset tampoco separa ese par) contra **problema de captura** (el dataset separa perfecto y el modelo acierta sobre él, así que la ejecución en vivo se salió de la distribución de entrenamiento).

### Corrección del aspect ratio de captura

El hallazgo más consecuente de la iteración, y la justificación de por qué la instrumentación valía la pena.

**El problema.** MediaPipe normaliza x por el ancho de la imagen e y por el alto. Para un mismo tamaño físico, el cociente entre una medida vertical y una horizontal es exactamente W/H. Los videos de entrenamiento son 1920×1080 (W/H = 1,778) y la cámara de desarrollo (DroidCam, versión libre) entrega 640×480 (1,333) sin importar lo que se le pida. Como el ancla es el punto medio de hombros por eje y la escala es la distancia entre hombros —casi horizontal—, el efecto neto es que **toda medida vertical le llega al modelo a 0,75× de la escala que vio entrenando**, mientras las horizontales quedan intactas. Es una distorsión anisotrópica sistemática del 25 % en un eje.

**La solución.** Rellenar el frame con bandas negras hasta 853×480 **antes** de pasarlo a MediaPipe. Eso restaura W/H = 1,778 sin recortar campo de visión ni escalar la imagen, y es no-op si algún día la cámara entrega 16:9 nativo. Implementada en `letterbox_to_aspect` y aplicada dentro de `WebcamStream`. Se desactiva con `--no-pad-aspect` para reproducir evaluaciones anteriores.

**Efecto secundario que hubo que tapar.** El detector de movimiento compara `countNonZero` contra un umbral en píxeles absolutos. Al cambiar la resolución de captura, el trigger de auto-captura habría cambiado de sensibilidad. Ahora el cálculo se hace sobre un gris redimensionado a 640×480 fijo, de forma que `MOTION_PIXEL_THRESHOLD` conserva su calibración.

### Reentrenamiento (97 clases)

Con el diagnóstico cerrado se reentrenó el **baseline** (no Optuna), alineado a la recomendación de agosto: 16 frames, 128 dim, 4 heads, 2 capas, dropout 0,40, data augmentation. Vocabulario de **97 clases** (incluye `ñ`, `pasado`, `futuro`, `mio`) y la **V regrabada** el 08/09/2026. `viernes` no se regrabó: la eval del 17/09 confirmó el swap bidireccional.

### Componentes entregados

| Archivo | Rol |
|---|---|
| `src/hand_face_features.py` | Las 26 features, trayectoria de muñeca, puntos 2D para graficar |
| `src/inspect_sign_features.py` | CLI de inspección por seña, grupo, dataset completo o CSV de eval |
| `src/analyze_sign_confusions.py` | Análisis de confusiones: 1-NN LOO + AUC por feature |
| `src/compare_eval_vs_dataset.py` | Cruce entre fallos de cámara y separabilidad del dataset |
| `src/predict_npy.py` | Referencia dorada de inferencia sobre `.npy` |
| `src/test_aspect_correction.py` | Test offline de la corrección de aspect ratio |
| `src/camera.py` | Padding a 16:9, persistencia de tomas, `--signs`, `--cam-index`, metadatos en CSV |
| `src/config.py` | 97 clases, 16 frames, `dataset_landmarks_16frames` |
| `src/model/metrics.json` | Checkpoint del 15/09/2026 (97c / 16f) |
| `docs/inspeccion_features_mano_rostro.md` | Marco teórico y manual de uso del inspector |
| `docs/eval_97senias_20260917.md` | Análisis de la eval de cámara del 17/09 |

Tecnologías: MediaPipe Holistic 0.10.21, NumPy, SciPy (Mann-Whitney), Matplotlib, PyTorch, OpenCV (captura y letterboxing).

---

## Métricas de rendimiento y validación

### Análisis de confusiones sobre el dataset completo

Corrida sobre **6.680 clips en 97 señas**. Precisión 1-NN leave-one-out en el espacio crudo: **88,4 %**.

**Hallazgo principal: las confusiones duras no son las que se creía.** El foco previo estaba en `I` / `T` / `ojo`. Los pares realmente pegados son el **alfabeto y los números**, donde la mano está en el mismo lugar y casi no se mueve:

| Seña | clips | acc 1-NN | atractor principal |
|---|---|---|---|
| **D** | 73 | 46,6 % | B |
| **3** | 77 | 55,8 % | 2 |
| **U** | 63 | 57,1 % | K |
| **O** | 56 | 58,9 % | B |
| **K** | 66 | 59,1 % | L |
| **B** | 67 | 62,7 % | D |

Sobre esos pares, las 12 features originales no tenían nada que medir: la geometría mano–rostro y el movimiento de muñeca son ciegos a la forma de la mano. Esto es lo que motivó la tercera familia de features.

**Impacto de las features de configuración de dedos.** La ganancia en separabilidad es grande y específica:

| Par | sep con 12 features | sep con dedos | feature que lo resuelve |
|---|---|---|---|
| `2 ↔ 5` | 0,21 | **1,00** | `spread_middle_ring` |
| `1 ↔ 2` | 0,22 | **1,00** | `ext_middle` |
| `M ↔ N` | 0,25 | **1,00** | `curl_pinky` |
| `6 ↔ A` | 0,27 | **0,99** | `curl_thumb` |
| `C ↔ K` | 0,31 | **1,00** | `spread_middle_ring` |
| `D ↔ O` | 0,30 | **0,97** | `curl_index` |
| `K ↔ L` | 0,35 | **1,00** | `spread_thumb_index` |
| `Martes ↔ Chau` | 0,23 | **0,96** | `ext_index` |

**Control negativo — la métrica no dice «1,00» siempre.** `0 ↔ O` queda en **0,53** y `Años ↔ G` en **0,59**. Son casos donde la ambigüedad es real: `0` y `O` comparten forma y movimiento, y ninguna feature del descriptor puede separarlos porque la información no está en el dato. Que estos casos no mejoren es la evidencia de que las ganancias anteriores no son un artefacto de la metodología.

### Validación de la corrección de aspect ratio

Se testeó **offline**, aplicando la corrección derivada analíticamente (multiplicar y por 4/3) sobre los tensores de 36 tomas ya grabadas. Determinista, sin ruido de ejecución, mismas tomas en las tres filas:

| Variante | top-1 (modelo 08-29, 36 señas) |
|---|---|
| Original (captura 4:3) | 66,7 % |
| **Corrección en y (×1,333)** | **88,9 %** |
| Corrección en y y z | 75,0 % |

**+22,2 puntos.** El caso emblemático: `I` se predecía como `ojo` con confianza 1,000 y pasó a `I` con confianza 1,000. Se corrigieron además `D`, `K`, `L`, `N`, `P`, `T` y `Z` — ocho de diez cambios fueron hacia la respuesta correcta.

De los 12 fallos originales quedaron **4: E, G, O y V**. Es un resultado coherente con el análisis de dataset: `O` es la ambigüedad real `0 ↔ O` (sep 0,53) y `V` correspondía a la grabación vieja (ya reemplazada en este reentrenamiento). Removido el artefacto de captura, lo que queda son los problemas ya identificados como reales.

**Validación cruzada con MediaPipe real.** Se recortaron frames de video de entrenamiento a 4:3 y se compararon los landmarks contra la versión con padding, midiendo el ratio por eje sobre 472 muestras:

| Eje | Ratio medido | Ratio predicho |
|---|---|---|
| x | 1,008 | 1,000 |
| y | 1,321 | 1,333 |
| z | 1,033 | 1,000 |

La derivación analítica se sostiene empíricamente. Que `z` no cambie explica por qué la variante que también corrige `z` rinde peor: MediaPipe expresa `z` en la misma escala que `x`, así que el padding no debe alterarlo.

### Test de paridad

La variante «original» del test offline dio 24/36 = **66,7 %**, idéntico al 66,7 % que registró el CSV de la sesión en vivo para ese modelo. `predict_npy.py` reproduce `camera.py` exactamente: **carga de modelo y armado de tensor son idénticos entre ambos caminos**. Cualquier divergencia de la webapp queda localizada en la extracción de landmarks, no en la inferencia.

Sobre clips del dataset (I, Ojo, C, D, O, R, Y), los tres checkpoints de agosto aciertan con confianza ≈ 1,0. El modelo y el dato están sanos; los fallos de cámara eran de captura.

### Hallazgo metodológico: varianza entre corridas

Dos evaluaciones del mismo set de 36 letras y números, misma resolución, mismos modelos, sin ningún cambio en el pipeline:

| Modelo | Corrida A | Corrida B | Δ |
|---|---|---|---|
| `2026_08_18_model_no_opt` | 80,6 % | 69,4 % | −11,1 |
| `2026_08_26_model_opt` | 88,9 % | 75,0 % | −13,9 |
| `2026_08_29_model_opt` | 86,1 % | 66,7 % | −19,4 |

Entre 5 y 11 señas cambiaron de resultado por modelo. **Con una sola toma por seña, la evaluación de cámara no puede detectar efectos menores a ±14 puntos.** Esto invalida el uso del 88,9 % histórico como línea de base para comparar el modelo reentrenado, y es la razón por la cual la corrección de aspect ratio se validó offline y no con una segunda sesión de cámara: el efecto real (+22 puntos) es apenas mayor que el ruido del método de medición.

### Checkpoint reentrenado (15/09/2026) y eval de cámara (17/09/2026)

| | |
|---|---|
| Arquitectura | TinySkeleton baseline: 128 dim / 4 heads / 2 capas / dropout 0,40 |
| Frames | 16 |
| Clases | **97** |
| Train / val | 4.656 / 1.164 samples (60 por clase) |
| Data augmentation | sí (virtual ×25) |
| Épocas | 35 |
| **Val top-1** | **98,28 %** |
| **Val loss** | **0,062** |
| Captura | `640x480→pad853x480` · MediaPipe 0.10.21 · 1 toma |
| **Webcam top-1 / top-3** | **77,3 % / 94,8 %** (75/97 y 92/97) |
| Webcam 36 letras/números | **88,9 %** top-1, **100 %** top-3 |
| CSV | `src/eval_97senias_20260917_160148.csv` |

La métrica offline no cierra la entrega: val 98 % vs webcam 77 % es la misma brecha de siempre. Lo que sí cierra el experimento de aspect ratio es **`I`**: el 15/09 los tres modelos la mandaban a `ojo` con confianza ≈ 1,0; el 17/09, con padding, sale `I` (0,89). `T`, `E` y `G` también aciertan.

**Fallos que el dataset ya explicaba (no son sorpresa):**

| Par | Qué pasó | Lectura |
|---|---|---|
| `O → 0` | confianza 0,996 | homógrafo; sep 0,53; colapsarlo sube el global a 78,4 % |
| `V ↔ viernes` | bidireccional | `viernes` no se regrabó; sep de movimiento 0,28 |
| `años → G`, `donde → F`, `lunes → L` | la dinámica se lee como letra | el corpus *sí* las separa; en vivo faltó el agite |
| `ayer/cuando → pasado` | atractor nuevo | `pasado` entra al vocabulario y se come señas temporales |

De 22 fallos, el cruce con el dataset marca **17 como problema de captura/ejecución**, 2 ambiguos (`V`/`viernes`) y 3 límite (`O/0`, `años/G`, `ojo/vivir`). `ojo` fue un casi acierto (0,53 vs 0,47 a `vivir`), no el atractor viejo sobre `I`.

Sobre las 94 señas comunes con la eval del 31/08, el baseline 16f de agosto tenía 86,2 % y esta corrida 77,7 %. No es comparable uno a uno: cambió el aspect, el vocabulario y `V`. El número que sí es atribuible es el de `I`.

### Estado del dataset

| Seña | Videos | Fechas | Estado |
|---|---|---|---|
| `V` | 65 | todos 08/09/2026 | Regrabada (hold estático). En cámara se confunde con `viernes` |
| `viernes` | 77 | hasta 17/08/2026 | **Sin regrabar** — confirmado en eval: swap bidireccional |

---

## Bitácora de horas

Estimación a partir de las ventanas de trabajo evidenciada por artefactos y sesiones (no es un registro medido con reloj). Ajustar si hay horas propias no cubiertas acá.

| Fecha | Sesión | Artefactos | Horas (est.) |
|---|---|---|---|
| 09/09/2026 | Inspector de features por seña y modo dataset completo | `inspect_sign_features.py` | 4 |
| 15/09/2026 | Features de configuración de dedos; análisis de confusiones sobre 6.680 clips; marco teórico | `hand_face_features.py`, `analyze_sign_confusions.py`, `inspeccion_features_mano_rostro.md` | 5 |
| 15/09/2026 | Evaluación de cámara (36 letras/números); cruce eval vs dataset; referencia dorada | `compare_eval_vs_dataset.py`, `predict_npy.py`, CSV de eval | 3 |
| 15/09/2026 | Diagnóstico de aspect ratio; letterboxing; test offline y validación con MediaPipe; paridad | `camera.py`, `test_aspect_correction.py` | 4 |
| 15/09/2026 | Reentrenamiento 97 clases / 16 frames (35 épocas) | `metrics.json`, checkpoint `.pth` | 3 |
| 17/09/2026 | Eval de cámara 97 señas con padding 16:9; cruce vs dataset | `eval_97senias_20260917_160148.csv` | 2 |
| **Total** | | | **~21 h** |

---

## Trabajo pendiente para la próxima iteración

1. **Regrabar `viernes`** (2 golpes al mentón en Z) y reentrenar. Es el único par que el dataset marca como irresoluble con el dato actual; la eval lo confirmó en ambos sentidos.
2. **Protocolo de ejecución en eval** para las dinámicas (`lunes`, `donde`, `años`): si se sostienen como la letra, el modelo predice la letra. El corpus ya las separa.
3. **Cerrar la paridad con la webapp** con `predict_npy.py` y anotar MediaPipe del navegador (Python: 0.10.21). El lado Python ya tiene línea de base con padding.
4. **Varias tomas por seña** si se quiere un porcentaje global comparable; con una toma el ±14 pp sigue valiendo. Priorizar re-tomas de `ojo`, `cara`, `T` y `B` (casi aciertos / aciertos frágiles).
5. **No cablear las 26 features solo por I/T/ojo** — eso lo resolvió el padding. Sí tiene sentido para alfabeto/números (`D`, handshapes), *después* de estabilizar captura y `viernes`.
6. **Producto:** colapsar `O≡0` en la métrica que se informa (78,4 %). No colapsar L/lunes, G/años, F/donde, V/viernes.
