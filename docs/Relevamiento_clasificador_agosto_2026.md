# Relevamiento — clasificador TinySkeleton (agosto 2026)

> **Rama:** `scratch-mediapipe-v2`  
> **Entrenamiento baseline (sin Optuna):** 18/08/2026 — 16 frames, 128/4h/2L (`src/model/metrics.json` de esa fecha)  
> **Entrenamiento Optuna v2:** 27/08/2026 — 12 frames, 256/2h/2L (`src/model/metrics.json`, `optuna_best_v2.json`)  
> **Eval en vivo baseline (luz ambiente):** `src/eval_91senias_20260818_202845.csv` (18/08)  
> **Eval en vivo Optuna (mejor luz):** `src/eval_94senias_20260828_202929_opt.csv` (28/08 20:29)  
> **Eval en vivo baseline (mejor luz):** `src/eval_94senias_20260828_222229_no_opt.csv` (28/08 22:22; mismo modelo 16f/128/4, no Optuna)  
> **Bitácora de la rama:** `docs/cambios_scratch-mediapipe-v2.md`  
> **Autores:** Francisco Veron, Maite Nigro

---

## 1. Qué cambió en esta iteración

Se reentrenó la red desde cero sobre un corpus más grande, con **16 frames** por secuencia y **más data augmentation**. El objetivo fue atacar la brecha entre validación offline (~96–97%) y cámara (~61–71% top-1 en julio).

| Ítem | Iteraciones A–D (julio) | Esta iteración (agosto) |
|------|-------------------------|-------------------------|
| Frames | 16 (A) o 32 (B/C/D) | **16** |
| Videos por clase | 50 (mínimo y tope) | **60** (mínimo y tope) |
| Clases en el modelo | 91 | **94** (entran `chau`, `tener`, `años`) |
| Train / val | 3.640 / 910 | **4.512 / 1.128** (80/20 × 60 × 94) |
| `VIRTUAL_MULTIPLIER` | 10 | **25** |
| Augmentation | ruido gaussiano + escala 3D | las dos anteriores **+ 5 tipos nuevos** |

El preprocesado sigue el mismo pipeline que la cámara: interpolar frames perdidos de MediaPipe → recortar silencio (`trim`) → subsampleo uniforme a 16 frames. Pose + manos, **sin cara** (225 features por frame).

---

## 2. Cómo se entrenó

Arquitectura **TinySkeletonClassifier**: Conv1D + Transformer con attention pooling.

| Hiperparámetro | Valor |
|----------------|-------|
| `HIDDEN_DIM` | 128 |
| `NUM_HEADS` | 4 |
| `NUM_LAYERS` | 2 |
| `DROPOUT_RATE` | 0,4 |
| `BATCH_SIZE` | 16 |
| `LR` | 3,77×10⁻⁵ (AdamW) |
| `WEIGHT_DECAY` | 3,17×10⁻⁴ |
| `LABEL_SMOOTHING` | 0,0015 |
| `EPOCHS` / `PATIENCE` | 200 / 15 (early stopping) |
| Épocas reales | **29** |

La validación **no** usa augmentation ni repetición virtual.

### 2.1 Más videos

Todas las señas del `dataset` llegan al piso de 60 MP4. `SAMPLES_PER_CLASS = 60` descarta clases por debajo y recorta las que tienen de más (shuffle con semilla 42). El split sigue siendo **48 train / 12 val por clase**.

Con el multiplicador virtual, cada época ve 48 × 25 = **1.200 vistas por clase** (no son 1.200 archivos: son el mismo `.npy` con transforms distintas).

### 2.2 Data augmentation

Se aplica **solo en train**, on-the-fly. No se guardan copias extra en disco.

**Ya existía**

| Transformación | Parámetro | Qué simula |
|----------------|-----------|------------|
| Escala 3D isotrópica | 0,85–1,15 | distancia a cámara / tamaño |
| Ruido gaussiano | std ≈ 0,030 | temblor de MediaPipe |
| Repetición virtual | ×25 | más vistas por época |

**Se agregó**

| Transformación | Parámetro | Qué simula |
|----------------|-----------|------------|
| Rotación 3D | yaw ±15°, pitch ±8°, roll ±8° | ángulo de cámara / persona de costado |
| Recorte temporal | hasta 15% por extremo, luego resample a 16 | seña no centrada en el clip |
| Time warp | velocidad 0,80–1,25× | señante más rápido o más lento |
| Frame dropout | 0–3 frames a cero + interpolación | pérdidas de MediaPipe |
| Ruido distinto pose / manos | pose 0,012 / manos ≈ 0,030 | tracker más inestable en manos |

No se usa flip horizontal (en LSA la mano dominante importa). El espejado zurdo sigue siendo **normalización** en el preprocesado (`_zurdo` / `_left`), no augmentation.

El crop, el warp y el dropout corren al cargar cada video (`LabeledSkeletonDataset`). Rotación, escala y ruido corren sobre el batch en GPU.

---

## 3. Resultados offline

Fuente: `src/model/metrics.json` (18/08/2026, 13:32 UTC).

| Métrica | Valor |
|---------|-------|
| Clases | 94 |
| Exactitud top-1 (validación) | **98,40%** |
| Mejor val loss | **0,095** |
| Épocas hasta early stopping | 29 |

Comparado con julio, la val loss baja un orden de magnitud respecto de A/B/C (~1,04–1,10) y queda en la zona de Optuna D (0,087), pero ahora con 16 frames, 94 clases y más dato. La validación offline **sigue sin medir** el dominio de la webcam.

---

## 4. Resultados en cámara

Fuente: `src/eval_91senias_20260818_202845.csv`.  
Sesión de 9,3 min, mano derecha, captura `auto`. El archivo se llama “91 señas” pero tiene **94 filas** (incluye las tres clases nuevas).

### 4.1 Números crudos del CSV

| Métrica | Valor |
|---------|-------|
| Top-1 | 63/94 (**67,0%**) |
| Top-3 | 84/94 (**89,4%**) |
| Fuera del top-3 | 10 |
| Confianza media (aciertos / errores) | 0,94 / 0,74 |

Por familia:

| Familia | n | Top-1 | Top-3 |
|---------|---|-------|-------|
| Dígitos | 10 | 50,0% | 60,0% |
| Letras (A–Z + ñ) | 27 | 77,8% | 96,3% |
| Léxico | 57 | 64,9% | 91,2% |

Las **letras** suben fuerte respecto de la iteración A de 16 frames (51,9% / 77,8%). El desplome aparente de **dígitos** (50% / 60%) se explica en gran parte por takes mal ejecutados (ver §4.2): de `1 2 4 5 6` el evaluador declara error de ejecución.

### 4.2 Ajustes de la evaluación

Cinco correcciones de protocolo, acordadas al relevar las sesiones:

1. **El evaluador ejecutó mal 8 señas:** `1`, `2`, `4`, `5`, `6`, `hermano_a`, `martes`, `nosotros`. Esos takes no se atribuyen al modelo: la etiqueta esperada no coincidía con lo que realmente se señó.
2. **`O` y `0` son la misma forma.** En LSA no hay distinción visual usable para el clasificador. Si una aparece como predicción de la otra (aunque sea en top-2 o top-3), se cuenta **siempre como top-1**.
3. **`G` y `años` son la misma forma.** Misma regla que O ≡ 0.
4. **`L` y `lunes` son la misma forma**, con movimiento de más: `L` es la pose estática y `lunes` es esa pose en movimiento. Misma regla.
5. **`F` y `donde` son la misma forma**, el mismo patrón que L/lunes: `F` estática, `donde` en movimiento. Misma regla.

El 18/08, `L`/`lunes` y `F`/`donde` ya eran top-1 las cuatro filas, así que esos dos pares **no suman** en esta sesión. Sí mueven las evals del 28/08.

Qué hizo el CSV en esas 8 filas:

| Esperada | Top-1 del modelo | ¿Top-3? | Efecto al descontar el take |
|----------|------------------|---------|-----------------------------|
| `1` | `miercoles` | no | era error (fuera de ranking) |
| `2` | `2` (acierto) | sí | ya era top-1; no suma ni resta |
| `4` | `5` | sí (`4` en top-3) | era error top-1 |
| `5` | `R` | no | era error (fuera de ranking) |
| `6` | `5` | no | era error (fuera de ranking) |
| `hermano_a` | `numero` | sí | era error top-1 |
| `martes` | `0` | sí | era error top-1 |
| `nosotros` | `ahora_hoy` | sí | era error top-1 |

Siete de las ocho eran fallos top-1; `2` el modelo la acertó igual. Tres (`1`, `5`, `6`) ni entraban al top-3.

En el CSV, `0` ya era top-1 correcto. `O` salió top-1 = `0` y top-2 = `O` (hit_top1 = 0, hit_top3 = 1). Con la regla O ≡ 0, esa fila pasa a top-1. `O` no está entre las 8 mal ejecutadas.

`G` ya era top-1 correcto (`años` en top-3). `años` salió top-1 = `G` y top-2 = `años` (hit_top1 = 0, hit_top3 = 1). Con G ≡ años, esa fila pasa a top-1.

`L` y `lunes` ya eran top-1 las dos (`L` tenía `lunes` en top-2; `lunes` tenía `L` en top-3). L ≡ lunes **no suma** en esta sesión.

`F` y `donde` ya eran top-1 las dos (`F` tenía `donde` en top-2; `donde` tenía `F` en top-2). F ≡ donde **no suma** en esta sesión.

Hay dos formas de leer el ajuste:

**A — Beneficio de la duda (contar los 7 fallos de ejecución como acierto, más equivalencias)**

| Criterio | Top-1 | Top-3 |
|----------|-------|-------|
| Crudo (CSV, 94 señas) | 63/94 (67,0%) | 84/94 (89,4%) |
| O ≡ 0 | 64/94 (68,1%) | 84/94 (89,4%) |
| O ≡ 0 + G ≡ años | 65/94 (69,1%) | 84/94 (89,4%) |
| + L ≡ lunes + F ≡ donde | 65/94 (69,1%) | 84/94 (89,4%) |
| Equivalencias + 7 takes mal señados | **72/94 (76,6%)** | **87/94 (92,6%)** |

**B — Sacar esas 8 filas del denominador (solo takes bien ejecutados, n = 86, más equivalencias)**

| Criterio | Top-1 | Top-3 |
|----------|-------|-------|
| 86 takes válidos, crudo | 62/86 (72,1%) | 79/86 (91,9%) |
| 86 takes válidos, O ≡ 0 | 63/86 (73,3%) | 79/86 (91,9%) |
| 86 takes válidos, los cuatro pares | **64/86 (74,4%)** | **79/86 (91,9%)** |

La lectura B es la más limpia para comparar con julio: no se premia al modelo por señas que no se hicieron. En dígitos bien ejecutados quedan `0 3 7 8 9` → 4/5 top-1 (`3` falló, fue a `V`).

### 4.3 Comparación con julio (señas en común)

Sobre las 91 clases que ya se evaluaban en A–C, **sin** los ajustes del §4.2:

| Iter. | Frames | Top-1 | Top-3 |
|-------|--------|-------|-------|
| A (30/07) | 16 | 61,5% | 83,5% |
| B (31/07) | 32 | **71,4%** | 87,9% |
| C (31/07) | 32 | 67,0% | **90,1%** |
| D Optuna (31/07) | 32 | 66,0% | 84,6% |
| **Esta (18/08), crudo** | **16** | **68,1%** | **89,0%** |

Con los cuatro pares (O ≡ 0, G ≡ años, L ≡ lunes, F ≡ donde) y sin las 8 señas mal ejecutadas, el top-1 de takes válidos queda en **74,4%** (64/86), por encima del mejor de julio (B, 71,4% sobre 91). B usaba 32 frames; esta red usa 16, más videos y más augmentation. La comparación no es perfecta (julio no descontó takes mal señados), pero deja de verse como un empate flojo.

### 4.4 Errores que no son equivalencia ni “take mal señado”

Después de sacar `1 2 4 5 6 hermano_a martes nosotros` y de tratar los cuatro pares homógrafos, siguen patrones viejos:

- Atractor **`quien`**: se come `I`, `T`, `papa`, `vivir`, `vos`.
- Atractor **`0`**: además de `O` (equivalente), se come `chau`, `nombre`, `si`.
- `familia` → `calle` y `repetir` → `documento`: otra vez fuera del top-3.
- `3` → `V` (el dígito bien ejecutado que sí falla).

---

## 5. Lectura

El retrain de agosto **sí movió el vivo** respecto de la única corrida previa a 16 frames (A: 61,5% → 67,0% crudo / 69,1% con los cuatro pares; **74,4%** en takes válidos). Offline quedó muy alto (98,4%), como suele pasar cuando se sube el dato y el multiplicador virtual.

El golpe a dígitos en el CSV crudo no se puede leer como fallo del modelo: cinco de diez dígitos los ejecutó mal el evaluador. En los cinco takes válidos el top-1 es 80%, salvo `3`.

Lo que más falta, ya sobre takes bien señados:

1. **Fronteras `quien` / `0`** — desambiguación por contexto (top-3 + módulo semántico) más que por el clasificador solo.
2. **`familia` / `repetir`** — otra vez fuera del top-3; frontera que el modelo no tiene.
3. **Calibración** — el modelo está seguro cuando se equivoca; un umbral más alto no alcanza.

La eval de una seña por clase, con un solo señante, sigue siendo una muestra chica. Los ajustes del §4.2 (takes mal ejecutados y los pares O ≡ 0, G ≡ años, L ≡ lunes, F ≡ donde) hay que anotarlos en las próximas corridas de `camera.py --eval`.

---

## 6. Modelo Optuna v2 (27/08)

Study `tinyskeleton_lsa_v2`, 40 trials, K-fold 3. Los `.npy` tenían **T = 16**, así que el candidato 24 frames **no se exploró**. Optuna eligió **12 frames**.

| Hiperparámetro | Baseline 18/08 (manual) | Optuna v2 (ganador) |
|----------------|-------------------------|---------------------|
| `MAX_FRAMES` | 16 | **12** |
| `HIDDEN_DIM` | 128 | **256** |
| `NUM_HEADS` | 4 | **2** |
| `NUM_LAYERS` | 2 | 2 |
| `DROPOUT_RATE` | 0,40 | **0,57** |
| `BATCH_SIZE` | 16 | **32** |
| `LR` | 3,77×10⁻⁵ | 3,24×10⁻⁵ |
| `WEIGHT_DECAY` | 3,17×10⁻⁴ | **1,45×10⁻²** |
| `LABEL_SMOOTHING` | 0,0015 | 0,00025 |
| `AUG_NOISE_STD` | 0,030 | 0,027 |
| Val acc offline | 98,40% | 97,78% |
| Val loss | 0,095 | **0,080** |
| Épocas | 29 | 22 |

Offline el Optuna **no gana en accuracy** (baja 0,6 pp) y sí en val loss. Como en julio (trial D), el offline no anticipa el vivo.

---

## 7. Eval webcam Optuna (28/08)

Fuente: `src/eval_94senias_20260828_202929_opt.csv`.  
94 señas, mano derecha, captura `auto`, ~10,6 min (20:29–20:40).

**Condición de captura:** mejor iluminación que el 18/08. Los landmarks no vibraban. MediaPipe rinde más con buena luz; **parte del salto no se puede atribuir solo a Optuna**. Esa duda se cierra en el §8–9: el mismo baseline del 18/08, re-evaluado a esta luz, queda por encima del Optuna.

### 7.1 Números crudos y equivalencias

| Métrica | Crudo | O ≡ 0 | + G ≡ años | + L ≡ lunes | + F ≡ donde |
|---------|-------|-------|------------|-------------|-------------|
| Top-1 | 76/94 (**80,9%**) | 78/94 (83,0%) | 79/94 (84,0%) | 80/94 (85,1%) | **81/94 (86,2%)** |
| Top-3 | 91/94 (**96,8%**) | 91/94 (96,8%) | **92/94 (97,9%)** | 92/94 (97,9%) | 92/94 (97,9%) |
| Fuera del top-3 | 3 | 3 | **2** | 2 | 2 |
| Confianza media aciertos / errores | 0,96 / 0,68 | — | — | — | — |

`0` salió top-1 = `hola` pero `0` y `O` están en top-2/top-3 → con el protocolo cuenta top-1.  
`O` salió top-1 = `0`, top-2 = `O` → también top-1.  
`años` salió top-1 = `G` (confianza 0,993) y no figuraba como `años` en el ranking → con G ≡ años cuenta top-1 y deja de estar fuera del top-3. `G` ya era top-1 (`años` en top-2).  
`L` salió top-1 = `lunes` (0,995), top-2 = `L` → con L ≡ lunes cuenta top-1. `lunes` ya era top-1 (`L` en top-3).  
`F` salió top-1 = `donde` (0,784), top-2 = `F` → con F ≡ donde cuenta top-1. `donde` ya era top-1 (`F` en top-2).

Fuera del top-3 (con equivalencias): `D` → `hola`, `repetir` → `llevar`.

Por familia (crudo / los cuatro pares):

| Familia | n | Top-1 crudo | Top-1 equivalencias | Top-3 equivalencias |
|---------|---|-------------|---------------------|---------------------|
| Dígitos | 10 | 9/10 (90,0%) | **10/10 (100%)** | 10/10 |
| Letras | 27 | 19/27 (70,4%) | **22/27 (81,5%)** | 26/27 |
| Léxico | 57 | 48/57 (84,2%) | **49/57 (86,0%)** | **56/57 (98,2%)** |

Calibración: por encima de 0,90 acierta **98,6%** (71/72). Error muy seguro que **no** es equivalencia: `nosotros` → `P` (0,983). `años` → `G`, `L` → `lunes` y `F` → `donde` dejan de contar como error.

---

## 8. Eval webcam baseline con mejor luz (28/08 22:22)

Fuente: `src/eval_94senias_20260828_222229_no_opt.csv`.  
Modelo **baseline 16f / 128 dim / 4 heads** (sin Optuna). Misma persona, 94 señas, mano derecha, captura `auto`, ~7,7 min (22:22–22:30).

**Condición de captura:** la misma mejor iluminación que la eval Optuna de las 20:29. Sirve para separar luz vs modelo.

### 8.1 Números crudos y equivalencias

| Métrica | Crudo | O ≡ 0 | + G ≡ años | + L ≡ lunes | + F ≡ donde |
|---------|-------|-------|------------|-------------|-------------|
| Top-1 | 81/94 (**86,2%**) | 82/94 (87,2%) | 83/94 (88,3%) | 84/94 (89,4%) | **85/94 (90,4%)** |
| Top-3 | 90/94 (**95,7%**) | 90/94 (95,7%) | 90/94 (95,7%) | 90/94 (95,7%) | 90/94 (95,7%) |
| Fuera del top-3 | **4** | 4 | 4 | 4 | 4 |
| Confianza media aciertos / errores | 0,96 / 0,77 | — | — | — | — |

`0` ya era top-1. `O` → `0` (top-2 = `O`) → top-1 con el protocolo.  
`G` ya era top-1. `años` → `G` (top-2 = `años`) → top-1 con G ≡ años.  
`L` ya era top-1 (`lunes` en top-2). `lunes` → `L` (0,977, top-2 = `lunes`) → top-1 con L ≡ lunes.  
`donde` ya era top-1 (`F` en top-2). `F` → `donde` (1,000, top-2 = `F`) → top-1 con F ≡ donde.

Fuera del top-3: `martes` → `chau`, `nombre` → `0`, `repetir` → `documento`, `vos` → `K`.

Por familia:

| Familia | n | Top-1 crudo | Top-1 equivalencias | Top-3 |
|---------|---|-------------|---------------------|-------|
| Dígitos | 10 | **10/10 (100%)** | 10/10 (100%) | 10/10 |
| Letras | 27 | 24/27 (88,9%) | **26/27 (96,3%)** | **27/27 (100%)** |
| Léxico | 57 | 47/57 (82,5%) | **49/57 (86,0%)** | 53/57 (93,0%) |

Calibración: por encima de 0,90 acierta **96,2%** (75/78). Errores muy seguros que no son equivalencia: `repetir` → `documento` (0,997), `ellos` → `ahora_hoy` (0,993), `nosotros` → `ahora_hoy` (0,986). `lunes` → `L` y `F` → `donde` dejan de contar como error.

Errores top-1 restantes (con equivalencias): `D` (top-3), `ellos`, `mal`, `martes`, `miercoles`, `nombre`, `nosotros`, `repetir`, `vos`.

---

## 9. Comparativa a tres bandas

Tres evals, misma persona, 94 clases, `camera.py --eval`, protocolo O ≡ 0 + G ≡ años + L ≡ lunes + F ≡ donde.

| Eval | Modelo | Luz | CSV |
|------|--------|-----|-----|
| 18/08 20:28 | baseline 16f 128/4/2 | ambiente (landmarks inestables) | `eval_91senias_20260818_202845.csv` |
| 28/08 20:29 | Optuna 12f 256/2/2 | **buena** | `eval_94senias_20260828_202929_opt.csv` |
| 28/08 22:22 | baseline 16f 128/4/2 | **buena** | `eval_94senias_20260828_222229_no_opt.csv` |

El cruce **18/08 vs Optuna 20:29** mezclaba modelo y luz (era el único par disponible). El cruce **18/08 vs baseline 22:22** aísla la luz (mismo modelo). El cruce **baseline 22:22 vs Optuna 20:29** aísla el modelo (misma luz).

### 9.1 Tabla resumen

| Criterio | 18/08 baseline, luz mala | 28/08 baseline, luz buena | 28/08 Optuna, luz buena |
|----------|--------------------------|---------------------------|-------------------------|
| Top-1 crudo | 63/94 (67,0%) | **81/94 (86,2%)** | 76/94 (80,9%) |
| Top-1 equivalencias | 65/94 (69,1%) | **85/94 (90,4%)** | 81/94 (86,2%) |
| Top-3 equivalencias | 84/94 (89,4%) | 90/94 (95,7%) | **92/94 (97,9%)** |
| Fuera del top-3 (equiv.) | 10 | 4 | **2** |
| Dígitos top-1 (equiv.) | 50,0% | **100%** | **100%** |
| Letras top-1 (equiv.) | 81,5% | **96,3%** | 81,5% |
| Léxico top-1 (equiv.) | 66,7% | **86,0%** | **86,0%** |

Deltas (equivalencias):

| | Δ top-1 | Δ top-3 | Qué cambia |
|--|---------|---------|------------|
| Luz (18/08 → 22:22, mismo baseline) | **+21,3 pp** | +6,3 pp | solo captura / ejecución |
| Modelo (22:22 → Optuna, misma luz) | **−4,3 pp** | **+2,2 pp** | 16f/128/4 → 12f/256/2 |
| Conjunto (18/08 → Optuna) | +17,0 pp | +8,5 pp | las dos cosas a la vez |

La mayor parte del salto que se le atribuyó a Optuna era **luz** (y, en el 18/08, takes de dígitos mal ejecutados). A igual iluminación el baseline **gana en top-1**; Optuna solo gana en top-3. El léxico empata (86,0%).

### 9.2 Luz, mismo modelo (18/08 vs 22:22)

Pareo top-1 con equivalencias:

| | n | Señas |
|--|---|--------|
| Acierto en ambas | **63** | — |
| Error en ambas | **7** | `D`, `ellos`, `martes`, `nombre`, `nosotros`, `repetir`, `vos` |
| Ganó el 22:22 | **22** | `1 3 4 5 6 I N P T calle chau dia familia hermano_a hijo_a jueves llevar numero papa si vivir vivir_en` |
| Perdió el 22:22 | **2** | `mal`, `miercoles` |

- Dígitos 50% → **100%**. Varias de las 22 ganadas eran takes mal señados el 18/08 (`1 4 5 6 hermano_a`); con buena luz y (probable) mejor ejecución dejan de ser un agujero.
- Letras 81,5% → **96,3%**, top-3 de letras **100%**. El atractor `quien` suelta `I T papa vivir`. `F`→`donde` cuenta acierto (letra estática vs seña en movimiento).
- Léxico 66,7% → **86,0%**. `familia` y `vivir` entran a top-1 (el 18/08 estaban fuera del top-3). `lunes`→`L` cuenta acierto.
- Los 7 que fallan en ambas son frontera de modelo, no de luz: `D`, `ellos`, `nosotros`, `repetir`, `vos`, `nombre`, `martes`.
- Las 2 pérdidas (`mal`, `miercoles`) caben en varianza de un take.

### 9.3 Modelo, misma luz (baseline 22:22 vs Optuna 20:29)

Pareo top-1 con equivalencias:

| | n | Señas |
|--|---|--------|
| Acierto en ambas | **78** | — |
| Error en ambas | **6** | `D`, `ellos`, `nombre`, `nosotros`, `repetir`, `vos` |
| Ganó Optuna | **3** | `mal`, `martes`, `miercoles` |
| Perdió Optuna | **7** | `A N V Y brazo cuando ojo` |

- **Letras:** baseline 96,3% vs Optuna 81,5% (**−14,8 pp**). Optuna rompe `A N V Y`; el baseline las acierta. `L`/`lunes` y `F`/`donde` dejan de ser “errores”: son el mismo patrón (pose estática vs seña en movimiento) en las dos redes.
- **Léxico:** empate **86,0%**. Optuna gana `mal martes miercoles`; pierde `brazo cuando ojo`.
- **Top-3:** Optuna 97,9% (fuera: `D`, `repetir`) vs baseline 95,7% (fuera: `martes nombre repetir vos`). El módulo semántico tiene ~2 señas más de colchón con Optuna.
- `repetir` queda fuera del top-3 en **las tres** evals. `D` falla top-1 en las tres (el 22:22 al menos lo mete al top-3).
- Los cuatro pares homógrafos se disparan en las tres fechas: no dependen de la luz ni de Optuna.

### 9.4 Cómo leerlo

1. **La luz mueve ~21 pp** en el mismo checkpoint. Landmarks estables + dígitos bien ejecutados explican el salto 67% → 86% crudo.
2. **Optuna no es un upgrade de top-1.** A igual luz pierde 4,3 pp, casi todo en letras. Coherente con el offline (98,4% → 97,8%) y con julio D (mejor val loss, peor webcam).
3. **Optuna sí cubre un poco más el top-3** (97,9% vs 95,7%). Útil si el producto apuesta al ranking + módulo semántico, no al top-1 crudo.
4. El número honesto con **buena luz** es: baseline **86,2% / 90,4%** top-1 (crudo / equivalencias) y **95,7%** top-3; Optuna 80,9% / 86,2% y 97,9% top-3.

---

## 10. Lectura actualizada

Con buena iluminación, el prototipo baseline (16 frames, 128/4/2) queda en **~86–90% top-1 y ~96% top-3**. Optuna v2, a la misma luz, queda ~4 pp más abajo en top-1 y ~2 pp más arriba en top-3.

Pendiente, ya no es “falta de dato genérico” ni “hay que atribuir 14 pp a Optuna”:

1. **Fronteras que no resuelve ninguna de las dos redes** — `D`, `ellos`/`nosotros`/`ahora_hoy`, `repetir`, `vos`, `nombre`.
2. **Pares homógrafos** — O ≡ 0, G ≡ años, y letra estática / seña en movimiento (L ≡ lunes, F ≡ donde). Tratarlos como una sola clase en eval y, si hace falta, en el mapeo.
3. **Calibración** — umbral 0,75 no salva `repetir`→`documento` a 0,997 ni `nosotros`→`ahora_hoy`/`P`.
4. **Qué checkpoint usar en el prototipo** — 16f/128/4 si importa deletreo y top-1; Optuna 12f/256/2 si importa cubrir top-3 para el módulo semántico.
