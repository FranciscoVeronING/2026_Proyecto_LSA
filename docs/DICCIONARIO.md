# Diccionario de movimiento (laboratorio)

Este archivo es el **laboratorio** que fabrica clips `.pose` a partir del
Señario CAS (PDF → QR → YouTube → Holistic). **No** es el runtime de Meet.

En vivo, ILSA pide esos clips a Cloud Run. Cómo se reproducen en la llamada:
[AVATAR.md](AVATAR.md). Cómo encaja en el producto: [ARQUITECTURA.md](ARQUITECTURA.md).

El §17 (“cómo se uniría a Meet”) quedó como diseño de aquella rama. La unión
ya está implementada.

---

# Rama `avatar`: del Señario CAS a un esqueleto que encadena glosas

Apunte técnico-pedagógico de la rama `avatar` del proyecto *Intérprete de Lengua de Señas Argentina* (2026). Está escrito como una clase: primero la analogía, después el contrato del código.

**Rama / commit de referencia:** `avatar` · `2a06b1ac` (16 oct 2026).  
**Script central vigente:** `extraction/render_pose_pair.py`.  
**Python:** 3.11 (MediaPipe Holistic clásico + pose-format).  
**Diccionario usable:** ≈ 518 señas del Señario + 37 del `dataset/` ≈ 555 archivos `.pose`.

Este informe cubre dos caminos que se recorrieron en la misma rama:

1. **El que quedó:** un diccionario de clips `.pose` y un montajista que los empalma en video de esqueleto.
2. **El que se descartó:** un avatar 3D VRM (Three.js + retarget de landmarks a huesos Mixamo). El código vive en el worktree `2026_Proyecto_LSA-avatar-letras`, no en este workspace.

---

## Cómo leer esto

Hay tres capas. Si no programás visión por computadora, leé el texto corrido, las analogías y los recuadros *Aula abierta*. Si vas a tocar el repo, leé *Aula técnica*, las tablas de flags y los casos que rompieron el compositor (CASA, LLEVAR, CUÁNTOS).

Una sola frase para llevarse: **esta rama no genera lengua de señas; monta un diccionario medido.** El oficio está en no mentirle al interpolador (ceros, duplicados, manos idle), no pisar el negativo (`.pose` crudo) y no confundir velocidad de reproducción con resampleo.

---

## Índice

0. [Qué cubre y qué no](#0-qué-cubre-esta-rama-y-qué-no)
1. [El problema](#1-el-problema-dicho-dos-veces)
2. [Vocabulario](#2-vocabulario-de-la-materia)
3. [Mapa del laboratorio](#3-mapa-del-laboratorio)
4. [Anatomía de un fotograma](#4-anatomía-de-un-fotograma)
5. [Del PDF al catálogo de QR](#5-paso-1--del-pdf-al-catálogo-de-qr)
6. [Extraer el movimiento crudo](#6-paso-2--extraer-el-movimiento-pose-crudo)
7. [Recortar el mosaico](#7-recortar-el-mosaico-antes-de-medir)
8. [Dónde empieza y termina la seña](#8-dónde-empieza-y-termina-la-seña)
9. [Un mismo cuerpo en un mismo cuadro](#9-un-mismo-cuerpo-en-un-mismo-cuadro)
10. [Completar el diccionario desde el dataset](#10-completar-el-diccionario-desde-el-dataset-de-laboratorio)
11. [El compositor](#11-el-compositor-cómo-se-arma-un-enunciado)
12. [Casos que enseñan](#12-casos-que-enseñan-más-que-la-teoría)
13. [Knobs](#13-knobs-qué-tocar-según-lo-que-ves)
14. [Recetas de consola](#14-recetas-de-consola)
15. [Qué hay en git](#15-qué-hay-en-git-y-por-qué-los-pose-no)
16. [**El intento VRM (descartado)**](#16-el-intento-vrm-descartado)
17. [Cómo se uniría esto a Meet](#17-cómo-se-uniría-esto-a-meet-diseño-no-código-de-esta-rama)
18. [Decisiones de diseño](#18-decisiones-de-diseño)
19. [Limitaciones](#19-limitaciones-honestas)
20. [Minitest](#20-minitest)
21. [Referencias](#21-referencias)

---

## 0. Qué cubre esta rama y qué no

| Cubre (rama `avatar`, este workspace) | No cubre (vive en otra parte) |
|---|---|
| Diccionario animable: glosa → clip de esqueleto → video | Traductor neuronal texto → seña (difusión, SMPL-X) |
| Continuidad intra-seña (EMA, relleno) e inter-seña (puente) | Avatar VRM en producción (se **probó** y se **descartó**; ver §16) |
| Extracción Holistic cruda a `.pose`, todos los frames | Clasificador TinySkeleton / cámara en vivo |
| 518 señas del Señario + 37 tomadas del `dataset/` | Unificación de glosas con el módulo semántico |
| Explicación de Meet 50/50 más adelante | La extensión de Chrome ya interceptando `getUserMedia` |

**Mapa mental.** El Señario es un archivo de negativos. Cada QR es la etiqueta del sobre. YouTube revela el negativo (el mp4). Holistic clava alfileres en cada fotograma. El `.pose` es el sobre con esas marcas, a tamaño real. `render_pose_pair.py` es el montajista: recorta cola y cabeza, fabrica un fundido de 4 a 8 fotogramas, pega el cartel `HOLA` / `puente` / `MAMA`.

---

## 1. El problema, dicho dos veces

### 1.1 En español llano

La Confederación Argentina de Sordos publicó el *Señario de términos y expresiones básicas en la LSA*: dos PDF (2019 y 2022) con tarjetas. Cada una tiene foto, glosa y un QR que abre el video de esa seña. El objetivo de producto, más adelante, es que en Google Meet la cámara se parta **50/50 a igual altura**: persona que habla a un lado, intérprete-esqueleto al otro. El esqueleto espera quieto hasta que llega una lista de glosas.

Esta rama no mete todavía el esqueleto en Meet. Hace el trabajo sin el cual Meet no tendría qué reproducir: un **diccionario de movimiento** y un **empalmador**.

> **Aula abierta.** Una glosa no es una oración. *MAMÁ* es el nombre de una seña. Cuando el compositor escribe `MAMA` en el video (sin acento) no es un error lingüístico: el tipo Hershey de OpenCV no dibuja Á. El archivo en disco sí puede llamarse `MAMÁ.pose`.

### 1.2 Contrato técnico

Entrada: PDF vectorial → URL `youtu.be` → mp4. Intermedio: `.pose` de pose-format, Holistic sobre **todos** los fotogramas, coordenadas en **píxeles de imagen**, sin ancla de hombros. Visible: mp4 `PoseVisualizer` + OpenCV `mp4v`. Python **3.11**: en 3.13 MediaPipe 1.0 ya no expone `mp.solutions.holistic` y pose-format 0.15 lo necesita.

> **Aula técnica.** El pipeline viejo (`qr_video_pipeline.py`) guardaba tensores `(32, 1629)` normalizados por hombros y persistía ceros cuando Holistic no veía a nadie. 214 de 506 `.npy` eran nulos. Eso se abandonó para el avatar. El clasificador puede seguir subsampleando; el diccionario animable no.

---

## 2. Vocabulario de la materia

| Término | Qué es, sin jerga | Detalle de implementación |
|---|---|---|
| LSA | Lengua de Señas Argentina. Gramática propia. | Este diccionario es léxico aislado, no discurso espontáneo. |
| CAS / Señario | Diccionario pedagógico en PDF con foto, glosa y QR. | Vol. 1 ~86 pp.; vol. 2 ~205 pp. QR vectoriales, no JPEG. |
| Glosa | Etiqueta de una seña (`HOLA`, `CASA`). | Se resuelve a un stem en `out/poses/`. |
| Landmark | Punto estimado sobre una articulación o la cara. | Holistic: 33 pose + 468 cara + 21+21 manos. |
| Holistic | Un solo modelo que ve cuerpo, cara y dos manos. | `mp.solutions.holistic`, complejidad 1, `smooth_landmarks`. |
| `.pose` / pose-format | Archivo SLP: secuencia de puntos + fps + visor. | Cuerpo `(T, 1, 576, 3)` si incluye world; el visor usa 543. |
| Confianza | 0–1: qué tan seguro está el modelo de ese punto. | El visor **no dibuja** si confianza = 0. “Esconder” es poner 0. |
| Núcleo / preparación / retracción | El medio porta el significado; los bordes son llegar y volver. | El trim corta bordes. El puente **no** mezcla núcleos. |
| Coarticulación | El final de una seña ya anticipa la siguiente. | El puente (4–8 frames, smoothstep) es un sucedáneo barato. |
| EMA | Cada pose se mezcla con la anterior para que no tiemble. | `p_t = α x_t + (1-α) p_{t-1}`, default α = 0.50. |
| Puente | Fotogramas **nuevos** entre el último de A y el primero de B. | Lerp cartesiano, no SLERP de huesos. |
| Park / idle | Si una mano no está en la seña, se cuelga al costado. | `IDLE_HAND_HIT=0.10`, `DUPLICATE_HAND_FRAC=0.07`. |
| Mosaico | Video del diccionario con panel grande + cámaras chicas. | `crop_mosaic.py` recorta el panel izquierdo. |
| Hábil / no hábil | Mano dominante (naranja) vs. la otra (amarilla). | Naranja = anatómica derecha = `RIGHT_HAND`. |
| VRM | Formato de avatar 3D para la web (huesos Mixamo + malla). | Se probó y se descartó (§16). |
| Retarget | Pasar puntos de una persona a rotaciones de *otro* esqueleto. | El cuello de botella del camino 3D. |
| IK / CCD | Calcular rotaciones para que un extremo llegue a un punto. | CCD estiraba la malla; se abandonó para dactilología. |

---

## 3. Mapa del laboratorio

```
PDF Señario (extraction/*.pdf, gitignored)
    │  extract_qr_catalog.py     ZXing sobre página raster ×3
    ▼
out/qr_catalog.json              glosa + URL YouTube  (SÍ se commitea)
    │  extract_catalog_poses.py  yt-dlp + Holistic 3.11
    ├──────────────────────────► out/videos/*.mp4     (gitignored)
    │                            crop_mosaic.py       recorte 2×2 → panel frontal
    ▼                            extract de nuevo si recortaste
out/poses/<GLOSA>.pose           crudo, todos los frames, píxeles (gitignored, ~672 MB)
    │  trim_sign_windows.py      ráfagas de movimiento de manos
    ▼
out/sign_windows.json            start_sec / end_sec por seña  (SÍ se commitea)
    │  extract_dataset_poses.py  completa glosas que el Señario no tiene
    ▼
render_pose_pair.py              load → fps → trim → idle → fill → park/snap → EMA → canvas → puente → mp4
```

Camino 3D (descartado, worktree `avatar-letras`):

```
.pose crudo → export_pose_json.py → clips JSON
                                    → avatar_viewer (Three.js + three-vrm + retarget.js)
                                    → character.vrm
```

| Archivo | Rol | ¿En git? |
|---|---|---|
| `extraction/extract_qr_catalog.py` | PDF → JSON de QR | sí |
| `extraction/sign_names.py` | Nombres canónicos (`F G` → `F`); rename auto desactivado | sí |
| `extraction/extract_catalog_poses.py` | URL/mp4 → `.pose` válido (no persiste ceros) | sí |
| `extraction/crop_mosaic.py` | Detecta franja negra y recorta el panel grande | sí |
| `extraction/trim_sign_windows.py` | Marca inicio/fin útiles; no pisa el `.pose` | sí |
| `extraction/canonicalize_landmarks.py` | Mide afinidad hombros→canvas 720×960; **no pisa** crudos | sí |
| `extraction/extract_dataset_poses.py` | Agrega señas del `dataset/` que faltan | sí |
| `extraction/render_pose_pair.py` | Compositor: N señas → un mp4 con puentes | sí |
| `extraction/out/qr_catalog.json` | Índice chico (~134 KB) | sí |
| `extraction/out/sign_windows.json` | Índice chico (~188 KB) | sí |
| `.gitignore` | Ignora mp4, PDF, `out/poses/`, `out/videos/` | sí |
| `extraction/out/poses/*.pose` | El diccionario de movimiento (~672 MB) | **no** (copiar a parte) |
| `extraction/export_pose_json.py` | Export a JSON para el visor VRM | leftover; no es el camino vigente |
| `extraction/qr_video_pipeline.py` | Extractor viejo 32 frames / hombros | sí, no usarlo para avatar |

---

## 4. Anatomía de un fotograma

### 4.1 Aula abierta: alfileres sobre una foto

MediaPipe Holistic clava alfileres numerados: 33 en el cuerpo, 21 en cada mano, 468 en la cara. Un video es una pila de esas fotos. Un `.pose` es esa pila, más el fps, más un número 0–1 de confianza.

Si el modelo no ve la mano izquierda no deja un hueco honesto. A menudo escribe **(0, 0)** o **copia la mano que sí ve**. Para un clasificador es ruido. Para un montajista es un desastre: el puente interpola hacia la esquina, o deja un fantasma trabado. Casi todo el compositor existe para pelear contra eso.

### 4.2 Layout en memoria

Índices **0-based**, eje de puntos `P` en `body.data` con forma `(T, 1, P, 3)`:

```
[  0 ..  32]  POSE_LANDMARKS         33  cuerpo en píxeles de imagen
[ 33 .. 500]  FACE_LANDMARKS         468 malla facial
[501 .. 521]  LEFT_HAND_LANDMARKS     21  mano ANATÓMICA izquierda
[522 .. 542]  RIGHT_HAND_LANDMARKS    21  mano ANATÓMICA derecha
[543 .. 575]  POSE_WORLD_LANDMARKS    33  cuerpo en METROS (se tira al dibujar)

T = fotogramas    1 = una persona    3 = (x, y, z)
Ejemplo HOLA del Señario: (108, 1, 576, 3) a ~29.97 fps
```

| Índice | Qué es | Para qué lo usa el compositor |
|---|---|---|
| 11 / 12 | Hombro izq. / der. | Ancho de hombros, ancla del canvas, escala del park |
| 13 / 14 | Codo izq. / der. | Se reescribe al parquear o al snap |
| 15 / 16 | Muñeca del CUERPO izq. / der. | Distancia del puente; a menudo no coincide con la mano |
| 17–22 | Meñique / índice / pulgar del CUERPO | Se **apagan** al dibujar: pisan las 21 de cada mano |
| 0–10 | Nariz, ojos, orejas, boca del CUERPO | Se **apagan** al dibujar: pisan el facemesh |
| 23 / 24 | Cadera izq. / der. | Dirección “hacia abajo” cuando se estaciona un brazo |
| 25–32 | Rodillas, tobillos, talones, puntas | Confianza 0: plano cintura para arriba |
| 501+0 | Muñeca de LEFT_HAND | La verdad de la mano izquierda |
| 522+0 | Muñeca de RIGHT_HAND | Idem derecha. Color naranja = hábil |

### 4.3 Izquierda, derecha, pantalla y naranja

Convención **cerrada**:

- Mano **hábil = naranja = anatómica derecha = `RIGHT_HAND_LANDMARKS`**.
- Mano no hábil = amarilla = anatómica izquierda.
- En un plano frontal, la mano derecha de la persona aparece a la **izquierda de la imagen**. La mano a la derecha del esqueleto en el video es la **izquierda anatómica**.
- Al dibujar (OpenCV / PoseVisualizer): **+X a la derecha del cuadro**, **+Y hacia abajo**, **Z** profundidad de MediaPipe. El Z de cada mano es relativo a su muñeca y ~10× más chico que XY: **no es 3D métrico**. `POSE_WORLD` sí está en metros; por eso el visor lo descarta.

> **Aula técnica — profundidad.** Cuando se habló de un rig 3D, la convención de escena era cielo = +Y y +Z hacia cámara. Eso **no** es lo que usa el compositor 2D: acá Y crece hacia abajo. No mezclar los dos sistemas. Ver §16.

### 4.4 Ejemplo: un puente que se va a la esquina

Último frame de HOLA: muñeca derecha en `(420, 310)`. Primer frame de MAMÁ: Holistic no vio las manos → `(0, 0)`. Un lerp de 6 frames fabrica la mano **volando al rincón**. La solución no es “arreglar el interpolador”: es **no alimentarlo con ceros**.

---

## 5. Paso 1 — del PDF al catálogo de QR

Los QR del Señario **no están pegados como foto**. Son miles de cuadraditos vectoriales. Extraer las imágenes del PDF no trae el código. Hay que rasterizar la página y leerla con un motor de códigos de barras.

`extract_qr_catalog.py`:

1. Busca `extraction/*.pdf` (vol1 = el más chico / “libro”; vol2 = el que tiene “2”).
2. PyMuPDF rasteriza cada página ×3.
3. ZXing detecta todos los QR. OpenCV sobre la página entera **no** los leía: son demasiado chicos.
4. Asocia cada QR al texto más cercano **debajo** (±120 pt en X).
5. Tira ruido (“Señario II”, “cómo escanear”, números de página).
6. Clasifica `kind`: `sign` / `playlist` / `note`.
7. Escribe `extraction/out/qr_catalog.json`.

Resultado: **548 códigos**, **519 URLs** distintas, 518 con nombre usable.

> **Aula abierta — glosas pegadas.** En una grilla densa, el texto de abajo del QR de *BUENAS TARDES* puede incluir *CHAU* de la tarjeta vecina. `sign_names.py` corta letras pegadas (`F G` → `F`). El rename **automático masivo está desactivado**: los nombres se curaron a mano.

`sign_names.py` conserva la **ñ**. `vol1_HOLA` → `HOLA`, `vol1_ABECEDARIO A` → `A`. `fold_loose` iguala `Pañuelo (x)` y `Pañuelo(x)`. Correr el script **no renombra**; `--revert` vuelve a `vol1_` / `vol2_`.

---

## 6. Paso 2 — extraer el movimiento (`.pose` crudo)

### 6.1 Por qué no se normaliza al guardar

El negativo se archiva a resolución nativa. Si guardás “fracción del ancho de hombros”, el visor cree que 0.5 son 0.5 **píxeles** y el cuerpo se aplasta en el origen. La normalización **sí** sirve al clasificador. Al diccionario animable le sirve **después**, en el canvas.

### 6.2 `extract_catalog_poses.py`

Por cada fila `kind=sign`: descarga mp4 con yt-dlp (`best[ext=mp4]`, sin FFmpeg), Holistic complejidad 1 + `smooth_landmarks`, **un worker** (Windows se bankea con varios). Valida: ≥ 5 frames, media |coord| > 1, ≥ 20 % de pose con confianza > 0.3. Si falla, **borra** el `.pose`. Resume si ya es válido.

Lote de 519 URLs: **515 ok**, **3 skipped** (HOLA, MAMÁ, *Meses*), **1 failed**: *Leche* (YouTube privado). HOLA: media |coord| ≈ 365, conf 0.93. Ninguna de las 518 es nula.

```text
py -3.11 extraction/extract_catalog_poses.py --only HOLA,MAMÁ
py -3.11 extraction/extract_catalog_poses.py --mosaic-report extraction/out/mosaic_crop_report.json
```

> **Aula técnica — Z de las manos.** El z de cada mano es relativo a la muñeca y ~10× más chico que XY. Quien diseñe un rig 3D no puede tratar `LEFT_HAND[..., 2]` como metros. Eso, más que cualquier otra cosa, hundió el matching VRM (§16).

---

## 7. Recortar el mosaico antes de medir

Muchos videos del Señario son un mosaico: recuadro grande de frente y tomas chicas a la derecha, separadas por una franja negra. Holistic sobre el cuadro entero se distrae.

`crop_mosaic.py` promedia 7 fotogramas en gris por columna, busca en la mitad derecha la corrida oscura más larga (umbral 8, mínimo 6 px) y se queda con lo de la izquierda. Ejemplo, letra A: recorte `866×1037`. Corrida: 226/226 mosaicos recortados. Después hay que **re-extraer** el `.pose` (`--mosaic-report`): los landmarks viejos están en coordenadas del cuadro completo.

```text
py -3.11 extraction/crop_mosaic.py --all --in-place
```

---

## 8. Dónde empieza y termina la seña

Un video del diccionario incluye espera, seña, espera. Empalmar con las esperas duerme al esqueleto. Recortar de más se come un contacto (el techo de CASA).

`trim_sign_windows.py` **no parte** el video. Toma desde la primera ráfaga de movimiento de manos hasta la última y **conserva pausas internas**. Trabaja sobre el `.pose`; escribe `sign_windows.json`; **no pisa** el archivo.

| Constante | Valor | Significado |
|---|---|---|
| `PAUSE_SECONDS` | 0.20 s | Cuánto “quieto” cierra una ráfaga |
| `PRE_ROLL` / `POST_ROLL` | 0.15 / 0.20 s | Margen antes y después del núcleo |
| `MIN_KEEP_SECONDS` | 0.40 s | Nunca dejar una ventana más corta |
| `WRIST_MIN_PX` / `WRIST_PEAK_FRAC` | 18 px / 0.15 | Muñecas lejos del reposo |

Corrida sobre 517 `.pose`: mediana conservada **73,7 %**, 487 señas con más de una ráfaga, 0 fallbacks. Letra A: frames 12–78 de 102.

El renderer recorta de tres maneras (`--trim-mode`): `motion` lee el JSON; `fraction` (default, `0.25` a cada lado); `none`. El `.pose` crudo **siempre** conserva la espera.

---

## 9. Un mismo cuerpo en un mismo cuadro

Conviven mosaicos ~860×1030 (persona grande) y planos 1920×1080 (persona chica). Sin reencuadre, el esqueleto cambia de tamaño en el puente.

| Herramienta | Canvas | ¿Pisa el `.pose`? | Cuándo |
|---|---|---|---|
| `canonicalize_landmarks.py` | 720×960, hombros en (360, 400), ancho 230 | **No.** Solo mide. | Offline, visor/export 3:4 |
| `place_on_canvas` del renderer | 720×780, hombros al 55 % de altura, ancho 250 | Solo en memoria | Default `--center` |

Traslación **por frame** (el torso no camina fuera); escala **una por seña** (mediana del ancho de hombros).

```
x' = (x - mid_x) * scale + canvas_width/2
y' = (y - mid_y) * scale + canvas_height * 0.55
z' = (z - mid_z) * scale
```

---

## 10. Completar el diccionario desde el dataset de laboratorio

El Señario no tiene todas las glosas del clasificador. `dataset/` es otra toma, otra persona, otro encuadre. Se usó **solo** para glosas que no existen en el catálogo: mejor un Mio de laboratorio que un YO haciéndose pasar por MIO.

`extract_dataset_poses.py` pliega nombres. Dígitos `0`–`9` → `CERO`–`NUEVE`. `Ahora_Hoy` → `Hoy _ Ahora`. `Papa` → `PAPÁ`. Inventario: 100 carpetas, 63 cubiertas, **37 faltantes**. Default: clip `*-1.mp4`, Holistic completo.

Las 37: Ayudar, Años, Bien, Brazo, Calle, Cara, Casa, Como, Cuando, Cuantos, Cuchillo, Documento, Donde, El_Ella, Ellos, Llamar, Llevar, Lugar, Mal, Mio, No, Nosotros, numero, Ojo, Plaza, Poder, Que, Quien, Repetir, Robar, Si, Tener, Tuyo, Ver, Vivir, Vivir_En, vos.

`--retry Brazo,Mio --tries 7` puntúa el tercio central:

```
score = 3 * hand_hit + pose_hit + min(hand_motion, 40) / 80
```

Eso no reemplaza mirar. Clips fijados a mano: Brazo-1, Casa-33, Documento-48, Donde-29, Llevar-48, Lugar-57, Mal-36, Mio-23, Plaza-61, Quien-25, VivirEn-39. Cuantos-1: en este dataset es a una mano; no había toma a dos.

---

## 11. El compositor: cómo se arma un enunciado

`render_pose_pair.py` no “mezcla señas”. Limpia cada clip y **fabrica fotogramas nuevos entre ellos**. Si algo se ve mal *dentro* de CASA, no es el puente. Si la 2.ª mano queda congelada *entre* LLEVAR y LUGAR, casi nunca es la duración del puente: esa mano llegó sucia al empalme.

### 11.1 Orden de `prepare_sign`

| # | Paso | Default | Si lo salteás |
|---|---|---|---|
| 1 | `load_pose` (tira world) | siempre | El visor dibuja metros como píxeles |
| 2 | `interpolate` a `--fps` | 30 | 24 fps y 30 fps no alinean el puente |
| 3 | Trim fraction / motion / none | fraction 0.25 | Preparación y retracción se comen el ritmo |
| 4 | `detect_idle_hands` sobre el clip crudo | hit 0.10, dup 0.07 | Fantasma interpolado, mano trabada |
| 5 | `fill_missing_landmarks`, huecos ≤ 8 | conf 0.30 | Manos a la esquina, o trabadas si rellenás de más |
| 6 | Park del idle, o snap del activo | park on | Muñeca del cuerpo ≠ muñeca de la mano |
| 7 | EMA | α = 0.50 | Temblor (α→1) o gesto de goma (α→0.3) |
| 8 | Canvas 720×780, sin piernas | on | Mosaico y 1080p de distinto tamaño |

Recién después, `compose_sequence` concatena A + puente + B.

### 11.2 Relleno: la regla de los 8 frames

Faltante = confianza < 0.30 **o** `(x,y)` a menos de 8 px del origen.

- Hueco **corto** (≤ 8) con vecinos: lerp (parpadeo de Holistic).
- Hueco **largo**: hold del vecino, **no** interpolar de punta a punta.
- Si el punto no existió nunca: `fill` no lo inventa; park/snap se ocupan.

> **Aula abierta.** Restaurar una cinta: 5 fotogramas quemados → fundido. Medio rollo quemado → congelás el último bueno. El bug viejo interpolaba medio rollo: una mano que apareció 3 frames quedaba deslizándose toda la seña. LLEVAR, LUGAR, MAL, MIO.

### 11.3 Detectar la mano que no debería estar

1. **Sparse.** `frame_hit < 0.10` → idle. El umbral era 0.22 y estacionaba CASA (contacto con el techo, hit ≈ 0.16).
2. **Duplicada.** Distancia mediana de centroides < `0.07 * ancho_de_hombros` → Holistic copió la hábil. 0.18 enganchaba BRAZO (mano sobre el antebrazo no es un duplicado).
3. **Nunca las dos.** Se salva la de más presencia.

**Park:** reescribe codo, muñeca y 21 puntos a un brazo colgando hacia la cadera. No se apaga (el visor la dejaría invisible y el puente la traería de `(0,0)`).

**Snap:** Holistic estima cuerpo y manos en ramas distintas. La muñeca 15/16 no es el punto 0 de la mano. Se copia la muñeca de 21 puntos al cuerpo y el codo queda entre hombro y muñeca.

### 11.4 EMA y puente

```
out[0] = x[0]
out[t] = alpha * x[t] + (1 - alpha) * out[t-1]
```

Puente: último frame de A y primero de B, ya en canvas. Distancia = promedio de muñecas 15 y 16.

```
dist <= 40 px   →  4 frames   (~0.13 s a 30 fps)
dist >= 180 px  →  8 frames   (~0.27 s)
--bridge-fixed N  ignora las muñecas
```

**Smoothstep** `3t² − 2t³`: derivada 0 en los extremos. Lerp cartesiano de todos los puntos. No hay pose neutra. No hay SLERP: los “huesos” se pueden acortar un poco; a 4–8 frames no se nota como en un rig 3D.

### 11.5 Qué se esconde al dibujar

PoseVisualizer pinta todo con confianza > 0. Cara-pose (0–10) y puntas-pose (17–22) duplican facemesh y manos de 21. `hide_pose_overlays` pone esas confianzas a 0 **en una copia al escribir**. El `.pose` no se pisa.

### 11.6 `--fps` no es `--speed`

| Flag | Qué cambia | Qué no | Ejemplo |
|---|---|---|---|
| `--fps 30` | Muestras por segundo (resample) | Duración en segundos | 2 s siguen siendo 2 s |
| `--speed 0.87` | fps del `VideoWriter` = `body.fps * speed` | Cantidad de frames | 15 % más lento: 30 → 26.1 |

Videos de control (`prueba-senias-seniario.mp4`, `APELLIDO_Mio_N_I_G_R_O.mp4`) a **speed 0.87**. El default del script es 1.0.

### 11.7 Nombres y dactilología

`resolve_sign("F")` busca un archivo **de abecedario**, no cualquier glosa con F. `--spell FRANCISCO` → F,R,A,N,C,I,S,C,O; **CH** y **Ñ** son una seña. Los priors de letras L, A, B, C, F, D se aceptaron y **no se tocan** salvo pedido.

```text
py -3.11 extraction/render_pose_pair.py --signs HOLA,MAMÁ,SALUDAR
py -3.11 extraction/render_pose_pair.py --signs HOLA --spell FRANCISCO
py -3.11 extraction/render_pose_pair.py --signs APELLIDO,Mio --spell NIGRO --speed 0.87
py -3.11 extraction/render_pose_pair.py --list-signs letras
```

---

## 12. Casos que enseñan más que la teoría

**HOLA → MAMÁ.** Prueba fundacional. Sin relleno, las manos volaban a la esquina: MAMÁ arrancaba en `(0,0)`. Ahí nació `fill_missing_landmarks`.

**APELLIDO, MIO, N, I, G, R, O.** MIO no está en el Señario; se extrajo del dataset. Usar YO era mentir la seña. ~13.6 s a speed 0.87.

**Las 37 nuevas.** `prueba-senias-seniario.mp4` (~62 s). La hipótesis “son los parámetros del puente” se descartó: el puente solo mira dos frames. Si la 2.ª mano ya venía trabada *dentro* del clip, el puente la entrega trabada.

**LLEVAR / LUGAR / MAL / MIO.** Holistic copiaba la hábil sobre LEFT_HAND (centroides ~0–29 px). El fill sostenía el fantasma. Cura: duplicado 0.07 × hombros + park. No se tocó BRIDGE_MIN/MAX.

**CASA.** La mano no hábil contacta el techo y se ve poco. `IDLE_HAND_HIT=0.22` la estacionaba (hit ≈ 0.16). Bajó a 0.10 y se cambió a Casa-33.

**BRAZO y CUÁNTOS.** 0.18 de duplicado trataba “mano sobre el antebrazo” como copia. 0.07 exige nubes casi en el mismo lugar. CUÁNTOS a una mano es fiel al video de este dataset.

---

## 13. Knobs: qué tocar según lo que ves

Todos viven en `PARÁMETROS EDITABLES` de `render_pose_pair.py` y tienen flag CLI.

| Si ves… | Tocá | Hacia dónde | No toques |
|---|---|---|---|
| Transición seca | `--bridge-min/max` o `--bridge-fixed` | Subir | EMA |
| Pausa larga entre señas | puente, o `--trim` más agresivo | Bajar puente | Park |
| Manos a la esquina | `--fill-missing` on | Bajar `FILL_MAX_GAP` si se “estira” | Canvas |
| 2.ª mano congelada en el aire | park, `IDLE_HAND_HIT`, `DUPLICATE_HAND_FRAC` | Park on; dup más chico si come manos reales | Duración del puente |
| Falta una mano que sí trabaja (CASA) | `IDLE_HAND_HIT` | Bajar; o cambiar de clip | Hide pose |
| Temblor | `--ema` | 0.40–0.50 | Speed |
| Gesto atrasado | `--ema` | 0.7–1.0 | FPS |
| Mosaico y 1080p de distinto tamaño | `--center`, `--shoulder-px` | 250 px | Canonicalize sobre el archivo |
| Mancha en cara o palma | `--hide-pose-face` / `--hide-pose-hands` | Dejarlos on | El `.pose` |
| Va rápido de más | `--speed 0.87` | Bajar speed, **no** fps | `--fps` no alarga |

---

## 14. Recetas de consola

Siempre `py -3.11`.

```text
py -3.11 extraction/extract_qr_catalog.py
py -3.11 extraction/extract_catalog_poses.py --only HOLA,MAMÁ
py -3.11 extraction/extract_catalog_poses.py
py -3.11 extraction/crop_mosaic.py --all --in-place
py -3.11 extraction/extract_catalog_poses.py --mosaic-report extraction/out/mosaic_crop_report.json
py -3.11 extraction/trim_sign_windows.py
py -3.11 extraction/extract_dataset_poses.py --dry-run
py -3.11 extraction/extract_dataset_poses.py
py -3.11 extraction/extract_dataset_poses.py --retry Brazo,Casa,Cuantos,Mio --tries 7

py -3.11 extraction/render_pose_pair.py
py -3.11 extraction/render_pose_pair.py --signs HOLA,MAMÁ,SALUDAR
py -3.11 extraction/render_pose_pair.py --signs HOLA --spell FRANCISCO
py -3.11 extraction/render_pose_pair.py --signs APELLIDO,Mio --spell NIGRO --speed 0.87
py -3.11 extraction/render_pose_pair.py --list-signs letras
py -3.11 extraction/render_pose_pair.py --bridge-fixed 12 --easing linear
py -3.11 extraction/render_pose_pair.py --trim 0.10 --ema 0.85 --no-park-idle
```

El mp4 sale en `extraction/out/<GLOSAS>.mp4`. La consola imprime frames, trim, ema, rellenos, park L/R, y por puente la distancia de muñecas.

---

## 15. Qué hay en git (y por qué los `.pose` no)

Commit `2a06b1ac` en `origin/avatar`: código del pipeline, `.gitignore`, índices `qr_catalog.json` + `sign_windows.json`. Los `.pose` pesan ≈ 672 MB; se **copian a parte**.

Quedaron fuera a propósito: `export_pose_json.py` (resto VRM), `render_skeleton_video.py` (prototipo `.npy`), reportes, `web/`, `src/`. El worktree `2026_Proyecto_LSA-avatar-letras` conserva el experimento VRM; este workspace se revirtió a videos de `.pose` **sin borrar** ese trabajo.

Código heredado que no hay que reactivar: `qr_video_pipeline.py` + `preprocessing.py` + `config.py` (32 frames, ancla de hombros = clasificador).

---

## 16. El intento VRM (descartado)

Este capítulo es el diario de un camino que se recorrió de verdad y se cerró. No es un “no lo hicimos”. Es: lo hicimos, medimos, y no era el oficio que teníamos.

### 16.1 Aula abierta: qué se quería

Un `.pose` es un esqueleto de alfileres. Un **VRM** (Virtual Reality Model) es un muñeco con piel, ropa y huesos, el formato que usan los VTubers en la web. La idea era simple de enunciar y difícil de ejecutar:

> Tomar los alfileres de Holistic y hacer que el muñeco ponga la muñeca, el codo y cada falange donde estaba la persona del video.

Analogía de sastre. Tenés el patrón de una persona (el `.pose`) y un maniquí de otra talla, con otra longitud de brazos y otra palma (el VRM Mixamo). El retarget es hacerle la ropa al maniquí. Si el patrón dice “el índice toca la mejilla” y el maniquí tiene el brazo más corto, o estirás la manga (se ve de goma) o el índice se queda en el aire. **Eso, en dactilología, es una letra distinta.**

Se construyó un visor local:

```text
py -3.11 extraction/avatar_viewer/serve.py
# http://127.0.0.1:8765/extraction/avatar_viewer/
```

Cargaba `character.vrm`, un clip JSON exportado del `.pose`, y un overlay cian con los landmarks encima de la malla para comparar. El motor está en `extraction/avatar_viewer/app.js` + `retarget.js` (worktree `avatar-letras`). Se llegó a letras aceptadas (A, B, C, L, F, D) **a fuerza de priors por clip**, no porque el matching genérico funcionara.

### 16.2 Por qué se descartó (las dos razones de fondo)

**1. No teníamos los conocimientos suficientes de animación 3D para hacerlo bien.**  
Retarget, IK, rest poses Mixamo, quaterniones en la esfera (el camino más corto vs. dar toda la vuelta), twist de antebrazo vs. muñeca, cascos biomecánicos de mano, blendshapes: es un oficio. Lo que se hizo fue ir parcheando síntoma a síntoma (la palma da la vuelta 360°, un cuaternión sale NaN, la manga se rompe, el pulgar atraviesa el medio) con decenas de sliders. Eso no escala a 555 señas ni a Meet.

**2. El matching entre VRM y `.pose` no era bueno.**  
No era un bug de un archivo. Es un desajuste de representación:

| El `.pose` entrega | El VRM necesita |
|---|---|
| Puntos `(x, y)` en píxeles de imagen + un `z` relativo | Rotaciones de huesos (quaterniones) en un rig Mixamo |
| Proporciones de *esa* persona, *ese* encuadre | Longitudes de hueso *fijas* del muñeco |
| 21 puntos de mano, Z ~10× más chico que XY | ~20 DoF por mano, sin estirar malla |
| Confianzas que a veces son 0 aunque la mano se ve | Una pose siempre legal (no NaN, no 360°) |

Si respetás las longitudes del VRM, las yemas no caen en los puntos naranjas del overlay. Si estirás para que caigan (IK/CCD), la mano deja de ser una mano. **Dactilología es exactamente esa diferencia de un centímetro.**

Este workspace se revirtió al compositor 2D. El visor 3D no se destruyó: quedó en el otro worktree.

### 16.3 El pipeline 3D que se armó

1. `export_pose_json.py` leía un `.pose`, aplicaba `canonicalize_pose` (canvas 720×960, sin pisar el crudo), recortaba por `sign_windows` y escribía un JSON `{pose, face, left, right, world}` por frame, XY normalizado 0–1.
2. El visor (`Three.js` + `@pixiv/three-vrm`) cargaba el VRM 1.0 y cada frame:
   - reconstruía un overlay 3D (silueta de frente: XY de imagen en el plano de la muñeca, no `POSE_WORLD`);
   - orientaba cada hueso para que su hijo apuntara al joint del overlay (**retarget por segmento**, sin estirar);
   - medía rizo y abertura de cada falange sobre los 21 puntos;
   - si el clip era una letra con prior (A, B, C, …) y la mano estaba “en hold”, forzaba esa forma.
3. Un HUD de ~50 parámetros (`tune.json` / `localStorage`) regulaba ganancias de curl, spread, Hz de seguimiento, giro de palma, etc.

Convenciones que se cerraron ahí y siguen valiendo para el esqueleto 2D: naranja = derecha hábil; profundidad = Z; cielo = +Y y +Z hacia cámara **en el visor 3D** (distinto del +Y hacia abajo de OpenCV).

### 16.4 Librerías que se usaron y no funcionaron (para *este* problema)

Acá “no funcionaron” no significa que estén rotas. Significa: **no resuelven landmarks Holistic crudos → dactilología LSA sobre un VRM genérico.**

#### Kalidokit (`@yeemachine/kalidokit` v1.1.5)

Hecha para VTubers: webcam en vivo → MediaPipe → ángulos de Euler para un VRM. Tres solvers:

| Solver | Qué calcula | Para qué sirve | Para LSA |
|---|---|---|---|
| `Kalidokit.Face` | Cabeza (yaw/pitch/roll), apertura de ojos, pupilas, cejas, visemas de boca A/E/I/O/U | Un personaje que habla y parpadea | Gestualidad facial de streamer, no rasgos no-manuales de LSA (cejas de interrogación, etc.) y **cero dactilología** |
| `Kalidokit.Pose` | Hombros, codos, caderas, spine a partir de 33 puntos, asumiendo T-pose Mixamo y persona de pie en el cuadro | Brazos que “siguen” a groso modo | No conoce mosaico vs 1080p, ni que el Z de Holistic no es métrico |
| `Kalidokit.Hand` | 21 puntos → rizo grueso de dedos y un wrist | Wave, thumbs-up, mano abierta | A vs S vs E vs O vs L **colapsan**: el rizo es un solo ángulo por cadena, sin oposición de pulgar ni palma de canto |

Se **vendoreó** (`avatar_viewer/vendor/kalidokit.umd.js`) y se cargó en el HTML. En el runtime **no se llaman los solvers**. El nombre `applyKalidokitFingers` en `app.js` es un fósil: esa función mide rizo/abertura **propia** sobre vectores Three.js. Kalidokit quedó como la librería que “debía” hacer el matching y no lo hizo.

> **Aula abierta.** Kalidokit es un traductor simultáneo para un noticiero: entiende “estoy hablando” y “miré a la cámara”. No es un intérprete de LSA: no distingue un puño con pulgar al costado (A) de un puño con pulgar adelante (S).

#### CCD (Inverse Kinematics clásica, en el propio visor)

CCD = Cyclic Coordinate Descent. Desde la yema hacia el hombro, gira cada hueso un poco para acercar el extremo al punto objetivo. Es el IK de los tutoriales.

En el visor se usó primero para que la mano *llegara* al overlay. Efecto: el brazo se volvía elástico, las falanges se estiraban, la dactilología se iba. El flag `useSegmentRetarget` (default on) lo reemplazó: cada hueso **apunta** al joint correspondiente **sin cambiar de largo**. Entonces las yemas ya no coinciden con los puntos naranjas. Elegís entre “mano de goma que toca el overlay” o “mano rígida que no llega”. Ninguno es una letra.

#### `POSE_WORLD` como si fueran huesos Mixamo

Holistic exporta 33 puntos en metros (`POSE_WORLD_LANDMARKS`). La tentación es tratarlos como posiciones de un esqueleto 3D. En la práctica las manos se quedaban en el pecho aunque el video las tuviera en la cara: el world de pose no arrastra las 21 juntas (esas no tienen world métrico). El visor terminó usando **brazos desde 2D** (`useImageArms`): la silueta de frente, no el world. Es admitir que el `.pose` no es un mocap 3D.

#### `@pixiv/three-vrm` (LookAt, BlendShapeProxy, humanoid)

Esta sí **funcionó como cargador**. Lee un VRM 1.0, expone huesos Mixamo (`leftUpperArm`, `rightIndexProximal`, …) y blendshapes. Lo que **no** hace es el matching: no hay `vrm.applyHolistic(landmarks)`. El humanoid asume T-pose, +Y a lo largo del hueso, flexión de codo +X a la izquierda y −X a la derecha. Cada asunción incumplida (un VRM mal exportado, un quaternion que da la vuelta larga) se veía como manga rota o palma que gira 360° (letras F y D).

Three.js + GLTFLoader + OrbitControls son el escenario, no el actor. Sin un retarget correcto, el escenario está vacío.

### 16.5 Herramientas buenas en gestualidad facial, malas en dactilología

Hay una familia entera de software excelente para **caras** y mediocre para **configuraciones de mano**. Se barajaron y no cierran el producto LSA.

| Herramienta | Qué hace bien | Por qué no alcanza |
|---|---|---|
| **Kalidokit.Face** | Cabeza, ojos, cejas, visemas A/E/I/O/U de habla | Visemas de *español hablado*, no no-manuales de LSA. Dedos: no existen en este solver. |
| **Blendshapes VRM / ARKit (52)** | Sonrisa, parpadeo, ceño, visemas; un slider 0–1 por gesto | La dactilología no es un slider: es 20 DoF por mano. ARKit no tiene “letra T de LSA”. |
| **FLAME** (cara de SMPL-X) | Malla facial de investigación, expresiones densas | Vive pegado a SMPL-X; la mano es **MANO**, otro modelo, otro fitting. Cara 10/10, matching Holistic→MANO = otro doctorado. |
| **FaceMesh (468) en el overlay** | Se *ve* la cara del señante, cejas y boca incluidas | Pintar 468 puntos ≠ animar la cara del VRM. Transferir FaceMesh a blendshapes pierde la geometría fina y no mueve un dedo. |
| **pose-to-video** (Pix2Pix / ControlNet sobre `.pose`) | Una persona “de verdad” encima del esqueleto: cara y ropa convincentes | Apariencia fija, GPU, mp4 offline. No es un personaje intercambiable. Las manos las inventa la red, no el diccionario. No entra en Chrome. |
| **Live2D** | Títere 2D; a recuadro de Meet las caras 2D se leen bien | Los dedos 2D también se leen mejor que un VRM chico, pero sigue haciendo falta un rig de mano que no teníamos. Se aplazó como piel, no se usó. |

La lección es la misma en todas: **el cuello de botella de LSA no es la cara, son las manos.** Un sistema pensado para VTubers o para *talking heads* te resuelve el 20 % que menos duele.

### 16.6 Otras piezas que se evaluaron y no se adoptaron

| Pieza | Qué es | Veredicto |
|---|---|---|
| **Open3D** | Visor de nubes y mallas | Dibuja puntos. No es un personaje. No corre en la extensión. |
| **SMPL-X / SMPLify-X / HaMeR / aitviewer** | Cuerpo paramétrico con MANO + FLAME; fitting 3D | Mejor modelo de cuerpo+manos+cara que existe para research. GPU, licencias no comerciales (MPI-IS), no entra en Chrome. **No teníamos el oficio de fitting.** |
| **SignAvatars** (ECCV 2024) | Dataset + SMPL-X de señas (ASL/WLASL/…, no LSA) | No es drop-in. Licencia CC BY-NC-SA. Se robó *una* idea: el casco flexión×abducción de Spurr 2020 (`bioHands` en el HUD). El dataset no. |
| **VRoid / pieles anime / puppet geométrico** | Skins rápidas sobre el mismo rig | El usuario las **rechazó** explícitamente. Aceptación: dactilología + brazos/manos/dedos, avatar de frente, no muñeco geométrico. |
| **Un `tune.json` para 500 señas** | Los ~50 sliders que hicieron pasar A/B/C/F/D/L | No es un sistema. Es un estudio de animación casero. No hay forma honesta de tunear 555 glosas así. |

### 16.7 Lo que sí se aprendió (y se quedó en el camino 2D)

El intento 3D no fue inútil. Varias convenciones y diagnósticos pasaron al compositor vigente:

- Hábil = naranja = derecha. No elegir lado por frame según altura de muñeca.
- El Z de las manos de Holistic **no se usa como metros**.
- `POSE_WORLD` se descarta al dibujar.
- Letter priors L/A/B/C/F/D se aceptaron y no se tocan.
- Si el overlay (esqueleto) y la malla no coinciden, **se cree al overlay**: el `.pose` es la medición; el muñeco era una interpretación.

> **Aula técnica — ejemplo F/D.** La palma de F pide un roll de ~180° en el antebrazo. El camino más corto en la esfera de quaterniones a veces iba al otro lado, o se pasaba de π y daba otra vuelta (el “360 wrap”). Si algún componente salía no-finito, el hueso desaparecía (NaN). Parches: unwrap del ángulo, hold de 4 frames ante un salto, tope de twist en muñeca bajo y el resto en el antebrazo (`rollForearm`). Eso es exactamente el tipo de oficio 3D que no se tenía de entrada, y que no queríamos seguir acumulando seña a seña.

### 16.8 Dónde quedó el código

| Qué | Dónde |
|---|---|
| Visor VRM, `retarget.js`, `character.vrm`, Kalidokit vendoreada | Worktree `C:\Users\franc\Documents\GitHub\2026_Proyecto_LSA-avatar-letras`, rama `avatar-letras` |
| Contexto y constraints del visor | `AVATAR_CONTEXT.md` en ese worktree |
| `export_pose_json.py` | Todavía en este repo; leftover, no forma parte del pipeline de video |
| Este workspace | Solo `.pose` → mp4 de esqueleto |

No hace falta “terminar el VRM” para Meet. El esqueleto de `PoseVisualizer` ya es el personaje que se va a componer en el canvas (§17).

---

## 17. Cómo se uniría esto a Meet (diseño, no código de esta rama)

Pedido: mitad izquierda = cámara, mitad derecha = este mismo esqueleto, **misma altura**. Idle hasta que llega la lista de glosas.

Meet muestra lo que el navegador entrega como `getUserMedia`. Una extensión MV3 puede envolver esa API, dibujar un `canvas` (persona | esqueleto) y publicar `canvas.captureStream()`. Quien está en la llamada ve el canvas.

No conviene generar un mp4 por frase en Python: la latencia mata el turno. El lerp ya es barato. Estado idle = brazos parkeados. Cuando llega `[APELLIDO, MIO, N, I, G, R, O]` se corre el mismo prepare+puente, se reproduce, se vuelve a idle. El modo oyente **no** necesita Holistic WASM (eso es del modo señante).

> **Aula abierta.** El canvas de Meet es un noticiero: a la izquierda el estudio, a la derecha el intérprete. El esqueleto no se genera hablando. Se **dispara** cuando el módulo de glosas le pasa una lista, como un teleprompter.

---

## 18. Decisiones de diseño

| Opción | Decisión | Por qué |
|---|---|---|
| pose-format + `.pose` crudo | **Adoptada** | Encaja con Holistic, no subsamplea, visor listo |
| `.npy` (32, 1629) normalizado | Descartada para avatar | Ceros persistentes; 32 frames; coords no dibujables |
| Normalizar al extraer | Descartada en el archivo | Es un paso de *uso*, no de archivo |
| Open3D | Descartada como motor | Dibuja puntos; no es un personaje |
| SMPL-X / HaMeR | Aplazada (research) | GPU, licencias, no entra en Chrome |
| Kalidokit + VRM + Three.js | **Probada y descartada** | Matching `.pose`↔huesos insuficiente; oficio 3D que no teníamos; dactilología ilegible sin priors por letra (§16) |
| Live2D / títere 2D | Aplazada como piel | Mejor lectura a recuadro chico; sigue faltando un rig de mano |
| pose-to-video / ControlNet | Descartada para Meet | GPU, mp4, apariencia fija |
| WebM por seña como paquete | Descartada | 500 clips pesan de más |
| Difusión texto→moción | Fuera de alcance | No usa el diccionario CAS |
| Re-extraer *todo* desde `dataset/` | No | Otro signer; solo glosas que el Señario no tiene |
| Rename masivo de glosas | Prohibido salvo pedido | Nombres curados a mano |
| Lerp cartesiano vs SLERP | Lerp en esta rama | No hay huesos. El timing 4–8 se reutilizaría en un rig |

---

## 19. Limitaciones honestas

- *Leche* (vol. 2) no se extrae: YouTube privado.
- Algunas `label` del JSON siguen pegadas (`BUENAS TARDES CHAU`).
- Holistic duplica manos y pierde contactos. Park/fill mitigan; no hay modelo de “intención de la seña”.
- El facemesh de 468 es denso para un recuadro de Meet.
- El puente no es coarticulación lingüística.
- Z de manos no es métrico.
- Python 3.11 es un requisito.
- Esta rama no unifica vocabularios con el clasificador.
- Los `.pose` no están en GitHub.
- El camino VRM demostró que un matching genérico Holistic→Mixamo **no** alcanza para dactilología sin un animador 3D y/o un modelo de mano (MANO) con fitting, que no era el alcance.

---

## 20. Minitest

| Pregunta | Respuesta corta | Cap. |
|---|---|---|
| ¿Por qué no 32 frames como el clasificador? | El avatar necesita la curva temporal densa. | 1, 6 |
| ¿Por qué el `.pose` no está normalizado? | El visor espera píxeles. La ancla es de *uso*. | 6, 9 |
| ¿Qué es un puente? | Fotogramas nuevos entre el último de A y el primero de B. | 11.4 |
| ¿`--fps` hace más lento el video? | No. `--speed` baja el fps del archivo. | 11.6 |
| La 2.ª mano queda trabada. ¿Subo el puente? | No. Park/duplicado/fill. | 12, 13 |
| ¿Por qué CASA perdía una mano? | Umbral idle 0.22; bajó a 0.10 + clip 33. | 12 |
| ¿Naranja es izquierda de la pantalla? | Naranja es la derecha ANATÓMICA. En frontal cae a la izquierda de la imagen. | 4.3 |
| ¿Puedo commitear `out/poses/`? | No. ~672 MB. | 15 |
| ¿Python 3.13? | No para Holistic clásico. 3.11. | 6 |
| ¿Esto ya se ve en Meet? | No. El canvas 50/50 es diseño. | 17 |
| ¿Por qué no un muñeco VRM? | Matching `.pose`↔huesos malo; no había oficio 3D suficiente; Kalidokit/CCD/world no resolvían dactilología. | **16** |
| ¿Kalidokit no sirve para nada? | Sirve para caras de VTuber. No distingue letras LSA. Ni siquiera se terminó usando el solver: se reescribieron los dedos. | 16.4 |
| ¿SMPL-X no era “mejor”? | Como modelo, sí (MANO+FLAME). Como producto para Meet y para este equipo, no: GPU, licencias, fitting. | 16.6 |

---

## 21. Referencias

- Confederación Argentina de Sordos. *Señario de términos y expresiones básicas en la Lengua de Señas Argentina*, vols. 1 (2019) y 2 (2022).
- Moryossef, A., Müller, M. y Fahrni, R. *pose-format*. <https://github.com/sign-language-processing/pose>
- MediaPipe Holistic (Google). Landmarks de pose (33), manos (21×2) y FaceMesh (468).
- Liddell, S. K. y Johnson, R. E. (1989). *American Sign Language: The phonological base* — hold / movement / hold.
- Casiez, G., Roussel, N. y Vogel, D. *1€ Filter*, CHI 2012.
- Wei, F. et al. *A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars*, ECCV 2024.
- Yu, Z. et al. *SignAvatars*, ECCV 2024. arXiv:2310.20436 — referencia de MANO/SMPL-X; no es drop-in para LSA.
- Spurr, A. et al. (2020). Weakly supervised 3D hand pose — casco flexión×abducción que se copió en `bioHands`.
- Kalidokit, yeemachine. <https://github.com/yeemachine/kalidokit> — solvers Face/Pose/Hand para VRM; insuficientes para dactilología.
- `@pixiv/three-vrm` — cargador VRM 1.0 para Three.js.
- Pavlakos, G. et al. SMPL-X. <https://smpl-x.is.tue.mpg.de/>
- pose-to-video (sign-language-processing). Persona sobre esqueleto con Pix2Pix/ControlNet; offline, GPU.
- Runtime de Meet (este repo): [AVATAR.md](AVATAR.md), [EXTENSION.md](EXTENSION.md).
- Apuntes VRM / `Avatar_Senario_CAS.md` viven en otros worktrees, no acá.
- Commit de referencia: `2a06b1ac` en <https://github.com/FranciscoVeronING/2026_Proyecto_LSA> rama `avatar`.

---

## Cierre de clase

Si te llevás una sola frase: **esta rama no genera lengua de señas; monta un diccionario medido.**

Se midió cada glosa, se aprendió a no mentirle al interpolador, y se comprobó —con un visor 3D de verdad, no con una opinión— que ponerle piel VRM a esos puntos pide un oficio de animación y un modelo de mano que no eran el alcance. El personaje que sí se puede llevar a Meet es el mismo esqueleto que ya se ve en los mp4. Lo que falta es un canvas y una cola de glosas, no otro extractor ni otro retarget.
