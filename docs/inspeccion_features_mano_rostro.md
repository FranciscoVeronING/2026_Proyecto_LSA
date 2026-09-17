# Inspección de features mano–rostro

> **Rama:** `scratch-mediapipe-v2`  
> **Estado:** diagnóstico. **No** cambia el clasificador ni el tensor de entrada.  
> **Código:** `src/hand_face_features.py` (cálculo) y `src/inspect_sign_features.py` (CLI + gráficos).  
> **Para qué:** ver *antes de reentrenar* si I / T / OJO (y atractores como `quien`) ya se separan con geometría explícita.

Este documento tiene dos partes: **marco teórico** (por qué existe esto) y **manual** (cómo corre y cómo probarlo).

---

## 1. Qué es este agregado (y qué no es)

El clasificador actual recibe, por cada frame, **225 números**: 33 puntos de pose × 3 + 21 × 2 manos × 3. MediaPipe Holistic **sí** estima puntos de cara en la pose (nariz, ojos, boca), pero van mezclados con hombros, codos y torso, todos normalizados contra el ancho de hombros. La red tiene que *descubir sola* que “índice cerca del ojo” es OJO y “índice bajo la boca” es T.

Este agregado **no mete esas 6 features en el modelo**. Calcula, *por fuera*, seis cantidades interpretables (altura y distancia del índice respecto de boca / ojo / nariz) y las dibuja. La pregunta que responde es:

> ¿El dato ya separa las señas que la cámara confunde? Si sí, vale la pena concatenar esas features al tensor y reentrenar. Si no, el problema es de grabación o de normalización, no de arquitectura.

Todavía **no** está cableado a `train.py` ni a `camera.py`. Eso es el paso siguiente, solo si los gráficos lo justifican.

| Pieza | Rol |
|-------|-----|
| `hand_face_features.py` | De un vector `(225,)` saca 6 escalares. Reutilizable el día que se entrenen. |
| `inspect_sign_features.py` | Elige señas/videos, calcula esas 6, imprime una tabla y guarda PNG. |

---

## 2. Marco teórico

### 2.1 De píxeles a esqueleto

El sistema no clasifica RGB. Extrae un **esqueleto** con MediaPipe Holistic y trabaja sobre coordenadas. Motivos ya documentados en el entregable del clasificador:

- Un frame de video son millones de píxeles; un frame de landmarks son 225 números.
- Fondo, ropa e iluminación importan menos (aunque MediaPipe **sí** se degrada con mala luz: +21 pp de top-1 entre evals de agosto).
- Es el enfoque estándar en *skeleton-based sign language recognition*.

La pose de Holistic incluye una cara **gruesa** (no es Face Mesh de 468 puntos):

| Índice pose | Punto |
|-------------|--------|
| 0 | nariz |
| 2 / 5 | ojo izquierdo / derecho |
| 9 / 10 | comisuras de la boca |
| 11 / 12 | hombros (ancla y escala) |

En cada mano, el landmark **8** es la punta del índice y el **0** es la muñeca.

### 2.2 Normalización espacial: qué gana y qué pierde

Cada frame se centra en el **punto medio de los hombros** y se divide por la **distancia inter-hombros**. Dos personas a distinta distancia de la cámara quedan en unidades comparables (“fracciones de ancho de hombro”).

El costo: un desplazamiento chico en la cara (mejilla vs boca vs ojo) se vuelve un matiz dentro de 225 coords. Eso es exactamente la diferencia entre **I**, **T** y **OJO** en LSA: misma configuración de mano, distinto **lugar de apoyo**.

```text
I     → mejilla
T     → debajo de la boca
OJO   → índice cerca del ojo
```

En julio `ojo` se comía I y T. En agosto, con buena luz, a veces se separan y a veces I cae en el atractor **`quien`** (también una seña cerca de la cara). El módulo semántico puede desempatar con contexto (`YO I MAL` → “ojo”), pero **no** cuando las dos lecturas son palabras válidas (`MARIA` vs `MARTA`). Ahí solo sirve que el clasificador vea la posición.

### 2.3 Feature engineering, no más capacidad de red

En este proyecto ya se buscó arquitectura (Optuna v2: frames, `hidden_dim`, heads, layers). El search **gana val loss offline y pierde webcam top-1**. Agrandar la red (10 frames / 3 capas) llegó a 99 % offline y empeoró cámara, además mal calibrada.

La lección: con ~60 videos/clase el cuello no es “falta un Transformer más grande”. Es **señal**. Feature engineering consiste en **hacer explícita** una relación que el modelo podría aprender en teoría, pero que con poco dato y 225 dims diluidas no aprende de forma estable.

Formalmente, si \(x \in \mathbb{R}^{225}\) es el frame normalizado, se define un mapa

\[
\phi(x) = \bigl(\Delta y_{\text{índice,boca}},\;
\Delta y_{\text{índice,ojo}},\;
\Delta y_{\text{índice,nariz}},\;
d_{\text{XY}}(\text{índice},\text{boca}),\;
d_{\text{XY}}(\text{índice},\text{ojo}),\;
\Delta y_{\text{muñeca,nariz}}\bigr)
\in \mathbb{R}^{6}.
\]

\(\Delta y\) es **con signo** (no el valor absoluto). En MediaPipe el eje **Y crece hacia abajo** (como en una imagen). Entonces:

- \(\Delta y_{\text{índice,boca}} < 0\) ⇒ el índice está **más arriba** que la boca (típico de I / OJO).
- \(\Delta y_{\text{índice,boca}} \approx 0\) o \(> 0\) ⇒ a la altura o debajo de la boca (típico de T).
- \(d_{\text{XY}}\) chica al ojo ⇒ candidato a OJO.

No se usa Z como señal principal: el depth de MediaPipe es el eje más ruidoso. Tampoco se prende Face Mesh (`USE_FACE = True` sumaría 468×3 = **1404** dims). Con 60 clips/clase eso invita a memorizar, no a generalizar.

### 2.4 Por qué no guardar \(\phi(x)\) dentro del `.npy`

Tres funciones del pipeline asumen que **todo el vector son triples \((x,y,z)\)**:

- `mirror_landmarks_for_left_handed` hace `reshape(-1, 3)` y flip de X.
- `augment_batch_3d` rota, escala y mete ruido sobre `view(..., -1, 3)`.
- el trim de movimiento trata “todo lo que viene después de la pose” como manos.

Si se concatenan 6 distancias al `.npy` y se guarda `(T, 231)`:

1. el reshape falla o mezcla distancias con coordenadas;
2. la rotación 3D **gira distancias como si fueran puntos**;
3. el recorte del gesto se contamina.

Por eso el cálculo es **al vuelo**, sobre el `(225,)` ya normalizado (y, el día que se entrenen, **después** del augmentation espacial). Los `.npy` no cambian. No hace falta `preprocessing.py --force` para inspeccionar.

### 2.5 Un valor por video, no 16 frames sueltos

Para histogramas y scatter se resume cada clip en **un** vector de 6: el frame visible con menor \(d_{\text{XY}}(\text{índice},\text{ojo})\). Idea: el instante en que la mano está más cerca del ojo es el más informativo para este trío (coherente con el attention pooling del TinySkeleton, que ya pondera frames).

Si en ese frame las nubes de I / T / OJO se separan, la geometría **existe en el dato**. Recién ahí tiene sentido concatenar \(\phi(x)\) al input de la red y reentrenar el baseline 16f / 128 / 4 heads.

### 2.6 Dos familias que no son I/T/OJO

Hay que separar dos cosas que el protocolo de eval de agosto agrupó como “homógrafos”:

**A — Misma forma, sin contraste de movimiento:** `O` ≡ `0`. El clasificador no tiene (ni debe tener) una pista visual usable. Se siguen colapsando en eval.

**B — Letra estática vs seña dinámica** (misma o casi la misma configuración de mano; el rasgo distintivo es el movimiento):

| Estática | Dinámica | Qué se mueve |
|----------|----------|----------------|
| **L** | **lunes** | Agitar de lado a lado (eje **X**) |
| **G** | **años** | Mano de arriba abajo (eje **Y**) |
| **F** | **donde** | Leve diferencia de mano + brazo en profundidad, ~3 toques al **pecho** (eje **Z**) |
| **V** | **viernes** | V se sostiene; viernes son **2 golpes al mentón** en profundidad (eje **Z**), mano en configuración de V |

\(\phi\) mano–rostro **no** es la herramienta para esta familia: el frame “más cerca del ojo” se ve parecido en L y lunes. Lo que hay que mirar es la **trayectoria** (desvío/rango de la muñeca a lo largo del clip). El inspector ya imprime esa tabla y `histograms_motion.png`.

Si esas nubes de movimiento se separan, el modelo *puede* aprender el par — siempre que el trim y el subsampleo a 16 frames no aplasten el agite (dos golpes de viernes o tres de donde caben justos en 16 frames). Colapsar L≡lunes en eval fue un parche de producto, no una verdad lingüística.

`donde` y `viernes` comparten eje Z; se distinguen por **dónde** pegan (pecho vs mentón) y por la mano (F vs V). Ahí sí puede ayudar un poco \(\phi\) (altura de la mano: pecho más abajo que mentón) además del movimiento.

**Otras señas** (`repetir`, `ellos`, `vos`): tampoco las resuelve \(\phi\). `--group persist` es el control negativo.

**Luz:** MediaPipe inestable mueve más el top-1 que cualquier feature.

### 2.7 Recaptura de V

V se grabó mal y se va a rehacer. Protocolo alineado con L vs lunes:

- **V:** entrar a la pose (índice y mayor en V) → **sostener 1–2 s sin golpear el mentón** → retirar. Está en `STATIC_SIGN_CLASSES`.
- **viernes:** mano en V, **2 golpes al mentón en el eje Z** (profundidad). No es un hold. No va en esa lista.
- Después de reemplazar los MP4: `python preprocessing.py --force` **solo si** ya había `.npy` de V; si el inspector lee MP4, alcanza con los videos nuevos.
- Inspeccionar: `python inspect_sign_features.py --signs V viernes --max-clips 20 --no-show` y mirar sobre todo `std_wrist_z` / `range_wrist_z`. V baja; viernes alta. Puede verse algo de Y (subir al mentón); el rasgo distintivo que describimos es Z.
- En eval, **no** tratar V≡viernes como O≡0 hasta ver si el modelo las separa. Si V vieja estaba mal, cualquier métrica histórica de esa clase queda sucia.

---

## 3. Cómo funciona el programa, paso a paso

```text
señas elegidas ──► buscar clips (.npy o MP4)
                         │
                         ▼
              vector (T, 225)  [mismo pipeline que train/cámara]
                         │
                         ▼
              φ por frame → (T, 6)
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     tabla consola   histograma     scatter
                     series         snapshot 2D
```

### 3.1 Elegir señas

Origen, en este orden (se pueden combinar):

| Flag | Efecto |
|------|--------|
| `--group face` | I, T, ojo, quien, papa |
| `--group persist` | I, ellos, papa, repetir, tener, vos |
| `--group same_form` | O, 0 (misma forma, sin movimiento distintivo) |
| `--group static_dyn` | L/lunes, G/años, F/donde, V/viernes |
| `--group homo` | Alias de `static_dyn` |
| `--signs I T ojo` | Lista explícita |
| `--from-eval CSV` | Señas con `hit_top1 = 0` en un eval de cámara |
| (nada) | Default = grupo `face` |

Con `--from-eval`, por defecto también entra el **atractor** (el `top1` erróneo). Si I fue a `quien`, el gráfico muestra las dos. `--no-attractors` lo apaga. En CSVs apareadas (varias redes en el mismo take) hay que pasar `--model 2026_08_18_model_no_opt`.

Las carpetas del dataset suelen llamarse `Ojo` y las clases del modelo `ojo`. El script compara **sin importar mayúsculas**.

### 3.2 De dónde salen los clips

1. Si existe `dataset_landmarks_32frames/<clase>/*.npy` → los usa (rápido).
2. Si no → lee `dataset/<clase>/*.mp4` y corre **el mismo** `process_video_to_landmarks` que el preprocesado (MediaPipe → interpolar ceros → trim → subsampleo a `--frames`, default 16).

Los MP4 con `copia` en el nombre se ignoran. Sin `.npy` y sin `--max-clips` / `--clips`, el tope automático es **12 videos por seña** (extraer 60 con MediaPipe tarda mucho).

`--source npy|video|auto` fuerza el origen. `--clips 0 3 7` usa índices de `--list-clips`. `--clip-files I-10.npy` (o `I-10.mp4`) elige por nombre.

### 3.3 Cálculo de las 6 features (un frame)

Sobre el vector ya normalizado:

1. Partir en pose (33×3), mano izquierda (21×3), mano derecha (21×3).
2. Mano activa: derecha si no es toda ceros (después del espejado zurdo la dominante queda a la derecha); si no, izquierda.
3. Punta = landmark 8; si falta, muñeca.
4. Boca = punto medio de comisuras 9 y 10. Ojo = el **más cercano** al índice (no asume lado de la cara).
5. Si falta pose o las dos manos → **seis ceros**. No se calcula distancia al origen: eso parecería “mano en el ancla de hombros”.

Unidades: las mismas que el clasificador (ancho de hombros ≈ 1).

### 3.4 Salidas

En consola, dos tablas: (1) las 6 features mano–rostro en el frame más cerca del ojo; (2) std/rango de la muñeca a lo largo del clip (estática vs dinámica).

En `src/inspect_out/` (gitignored):

| Archivo | Qué muestra |
|---------|-------------|
| `histograms.png` | Un histograma por feature mano–rostro, color = seña. ¿Se pisan las nubes? |
| `histograms_motion.png` | Desvío/rango de la muñeca en X/Y/Z. Es el plot de L vs lunes, G vs años, F vs donde, V vs viernes |
| `scatter_mouth_eye.png` | Eje X: `dy_index_mouth`. Eje Y: `dist_index_eye`. Es el plot que más importa para I/T/OJO. |
| `series_dy_index_mouth.png` | La feature a lo largo de los frames (¿un instante o todo el gesto?). `--series` cambia cuál. |
| `snapshots_2d.png` | Dibujo 2D: hombros, ojos, boca, estrella = índice. Y invertido como en imagen. |

`--no-show` solo guarda PNG (útil en SSH o para no abrir ventanas). `--out otra_carpeta` cambia el destino.

---

## 4. Cómo probarlo

Todo desde `src/`, entorno **`lsa_gpu`**. El Python de Windows no trae MediaPipe; sin ese env el modo MP4 falla.

```powershell
conda activate lsa_gpu
cd src
python check_env.py
```

### 4.1 Humo (segundos, sin MediaPipe pesado)

```powershell
python inspect_sign_features.py --list
python inspect_sign_features.py --list-clips I
python inspect_sign_features.py --list-clips ojo
```

Esperado:

- `--list` muestra `.npy` (si existen) y/o MP4 en `dataset/` (`I`, `Ojo`, `T`, `Quien`, …).
- `--list-clips I` numera archivos. Esos números son los de `--clips`.

Si las dos raíces están vacías, no hay corpus local: el inspector no puede inventar videos.

### 4.2 Corrida corta (recomendada la primera vez)

Sin `.npy` cada video pasa por MediaPipe (~5–10 s por clip en CPU).

```powershell
python inspect_sign_features.py --signs I T ojo --max-clips 2 --no-show
```

Esperado en consola: 6 clips, `mano%` cerca de 100, tres filas de medias **distintas**. En una prueba de 2 clips por seña se vio aproximadamente:

| Seña | `dy_index_mouth` | `dist_index_eye` | Lectura |
|------|------------------|------------------|---------|
| T | ~0,03 | ~0,42 | Índice a la altura de la boca, lejos del ojo |
| I | ~−0,19 | ~0,21 | Más arriba, distancia media |
| ojo | ~−0,35 | ~0,08 | Más arriba, pegado al ojo |

Eso **no** es prueba científica (n=2). Sirve para verificar que el código corre y que la geometría no está invertida.

Archivos: `src/inspect_out/*.png`.

### 4.3 Corrida que sí informa una decisión

```powershell
python inspect_sign_features.py --group face --max-clips 20 --no-show
```

Después, las que el modelo no detecta bien en cámara (no solo I/T/OJO):

```powershell
python inspect_sign_features.py --group persist --max-clips 12 --no-show
python inspect_sign_features.py --group static_dyn --max-clips 15 --no-show
```

Desde un CSV de `camera.py --eval` (eval apareada del 31/08):

```powershell
python inspect_sign_features.py --from-eval eval_94senias_20260831_151057.csv --model 2026_08_18_model_no_opt --max-clips 15 --no-show
```

Un video puntual:

```powershell
python inspect_sign_features.py --signs I --clips 0 3 7 --snapshots 3
```

(sin `--no-show` abre las figuras en pantalla).

### 4.4 Si ya existen `.npy`

Más rápido; se puede no poner `--max-clips` y usar todos los clips de esas clases.

```powershell
python inspect_sign_features.py --source npy --group face --no-show
```

### 4.5 Dataset completo (todas las señas)

Con `preprocessing.py` ya corrido, inspecciona **97 señas** leyendo solo `.npy` (~2 min). Por defecto **1 clip por seña** (suficiente para un mapa global; subir a 5–10 en pares confusos).

```powershell
python inspect_sign_features.py --dataset --clips-per-sign 1 --no-show
```

Salida en `src/inspect_out/dataset/`:

| Ruta | Contenido |
|------|-----------|
| `summary.csv` | Tabla con las 6 features mano–rostro + movimiento por seña |
| `by_sign/<seña>/metrics.json` | Valores numéricos del clip elegido |
| `by_sign/<seña>/snapshot.png` | Pose 2D en el frame más cerca del ojo |
| `by_sign/<seña>/series_hand_face.png` | Las 6 features frame a frame |
| `by_sign/<seña>/series_wrist.png` | Trayectoria xyz de la muñeca |
| `overview/overview_std_wrist_*.png` | Barras de movimiento de **todas** las señas |
| `groups/face/`, `static_dyn/`, … | Mismos plots que `--group`, pero en una sola corrida |

Para más variabilidad en I/T/ojo o pares estática/dinámica:

```powershell
python inspect_sign_features.py --dataset --clips-per-sign 10 --no-show --out inspect_out/dataset_10clips
```

### 4.6 Cómo leer el resultado (criterio de “sí / no”)

| Qué ves | Qué implica | Qué hacer |
|---------|-------------|-----------|
| I / T / ojo en nubes distintas en el scatter, poco solape en `dy_index_mouth` | El dato ya tiene la señal | Cablear las 6 al train/cámara y reentrenar el **baseline 16f / 128 / 4 heads** |
| I y T juntos, ojo aparte | Altura no alcanza para I↔T | Revisar grabaciones; no reentrenar todavía |
| `quien` encima de I | Esperable; \(\phi\) igual puede bajar el atractor | Incluir `quien` en el retrain y medir esa celda |
| Estática y dinámica juntas en `std_wrist_*` | El recorte/subsampleo aplastó el movimiento, o los videos de la “estática” también se agitaron | Revisar captura (hold vs gesto); no colapsar el par en eval todavía |
| Estática con std bajo y dinámica con std alto en el eje esperado | El dato distingue el par | Features temporales (std/rango de muñeca) o cuidar que 16 frames no aliasen el agite |
| Todo superpuesto | Normalización o recorte mataron la señal | No concatenar features a ciegas |
| `repetir` / `ellos` no se separan (grupo persist) | Correcto: no es el problema de \(\phi\) | No usar ese grupo como éxito de este experimento |

No uses el 98 % de validación offline para decidir. La métrica de este diagnóstico es **separación visual + tabla**, y la del paso siguiente será `camera.py --eval` apareado, misma luz.

---

## 4bis. Análisis de confusiones sobre todo el dataset

`inspect_sign_features.py` describe **una** seña a la vez. Para responder *"¿qué pares se van a confundir?"* hay un script aparte que corre sobre los 6680 clips:

```powershell
python analyze_sign_confusions.py --top-pairs 35
```

Salida en `src/inspect_out/confusions/`: `REPORTE.md`, `worst_signs.png`, `top_confusions.png`, `confusion_heatmap.png`, `per_sign.csv`, `pairs.csv` y un PNG por par en `pairs/`.

### Los dos niveles

| Nivel | Qué mide | Responde |
|-------|----------|----------|
| **A — espacio crudo** | 1-NN leave-one-out sobre `(16, 225)` aplanado, o sea lo que ve TinySkeleton | ¿Qué pares están geométricamente pegados? |
| **B — features** | AUC de Mann-Whitney por feature entre las dos clases | ¿Cuál feature los separaría? |

`sep = |AUC − 0.5| × 2`. Es una escala directa: **1.0** las distribuciones no se tocan, **0.0** son idénticas. Umbrales usados en el reporte: ≥0.80 SEPARABLE, ≥0.60 con margen, ≥0.30 parcial, <0.30 NO SEPARABLE.

El 1-NN da **88.4 %** global. Ese número es optimista (clips de la misma sesión se parecen entre sí); lo que importa es **qué pares** aparecen cruzados.

### Hallazgo: las 12 features son ciegas a la forma de mano

Las confusiones más duras del dataset **no** son I/T/OJO. Son el **alfabeto y los números**: `D↔B` (29 clips), `4↔5`, `2↔3`, `C↔L`, `M↔N`, `1↔2`, `U↔K`. En todas, la mano está en el mismo lugar y casi no se mueve: \(\phi\) mano–rostro y `std_wrist_*` no tienen nada que medir. En `pairs/d__vs__b.png` las 12 barras están en rojo y las trayectorias de muñeca se superponen.

Por eso se agregó `handshape_features_frame` / `clip_handshape_row` en `hand_face_features.py`: 14 valores del **interior de la mano**, invariantes a escala y posición (se normalizan por `muñeca → nudillo del medio`, no por ancho de hombros).

| Familia | Cómo se calcula | Qué captura |
|---------|-----------------|-------------|
| `ext_*` (5) | punta → muñeca / tamaño de mano | dedo estirado vs recogido |
| `curl_*` (5) | punta → nudillo / largo estirado del dedo | 1 = recto, ~0.5 = cerrado |
| `spread_*` (4) | punta → punta vecina / tamaño de mano | dedos juntos vs separados |

El salto es grande en los pares de letras:

| Par | sep con las 12 | sep con dedos | feature que lo resuelve |
|-----|----------------|---------------|--------------------------|
| `M ↔ N` | 0.25 | **1.00** | `curl_pinky` |
| `1 ↔ 2` | 0.22 | **1.00** | `ext_middle` |
| `2 ↔ 5` | 0.21 | **1.00** | `spread_middle_ring` |
| `K ↔ L` | 0.35 | **1.00** | `spread_thumb_index` |
| `D ↔ O` | 0.30 | **0.97** | `curl_index` |
| `Martes ↔ Chau` | 0.23 | **0.96** | `ext_index` |

Contraejemplos que validan que la métrica no dice "1.00" siempre: `0 ↔ O` queda en **0.53** (misma forma, mismo movimiento) y `Años ↔ G` en **0.59**.

Estas 14 features **tampoco** se guardan en el `.npy`, por el mismo motivo del §2.4.

---

## 5. Relación con el clasificador (hoy y después)

Hoy el flujo de producto **no cambia**:

```text
webcam → MediaPipe → (T, 225) → TinySkeleton → glosa
```

El inspector es un **banco de pruebas** paralelo. El día que se integre:

1. `append_hand_face_features` **después** de `augment_batch_3d` en train y **después** de `extract_normalized_vector` + mirror en cámara.
2. `input_dim = 225 + 6`. Los `.pth` viejos no cargan: reentrenar de cero el baseline 16f.
3. Medir con eval apareada contra el checkpoint actual, no con val acc.

Hasta que los plots del §4.3 no muestren separación, ese cableado no está justificado **para I/T/ojo**. La eval de cámara del 17/09 (padding 16:9, 97 clases) cambió esa cuenta: **`I` acierta** y el atractor `ojo` desapareció. Ese trío lo resolvió el aspect ratio, no hace falta concatenar \(\phi\) solo por ellos. Las 14 features de handshape siguen justificadas para alfabeto y números (`D→0` en vivo es captura; en el dataset `curl_index` las separa).

Eval 17/09: [docs/eval_97senias_20260917.md](eval_97senias_20260917.md).

---

## 6. Archivos tocados

| Ruta | Qué |
|------|-----|
| `src/hand_face_features.py` | \(\phi\), movimiento de muñeca, configuración de dedos, puntos 2D |
| `src/inspect_sign_features.py` | CLI por seña / grupo / dataset, CSV de eval, gráficos |
| `src/analyze_sign_confusions.py` | Confusiones sobre los 6680 clips (1-NN + AUC por feature) |
| `src/inspect_out/` | PNG y reportes generados (gitignored) |
| `.gitignore` | `src/inspect_out/` |

No se modifican `train.py` para este diagnóstico. `camera.py` y `config.py` sí se tocaron después (letterbox 16:9, 97 clases). El inspector usa `--frames 16` por default, alineado al baseline integrado.
