# Relevamiento — clasificador TinySkeleton (agosto 2026)

> **Rama:** `scratch-mediapipe-v2`  
> **Entrenamiento:** 18/08/2026 (`src/model/metrics.json`)  
> **Eval en vivo:** `src/eval_91senias_20260818_202845.csv` (18/08/2026, 20:28–20:38)  
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

Dos correcciones de protocolo, acordadas al relevar la sesión:

1. **El evaluador ejecutó mal 8 señas:** `1`, `2`, `4`, `5`, `6`, `hermano_a`, `martes`, `nosotros`. Esos takes no se atribuyen al modelo: la etiqueta esperada no coincidía con lo que realmente se señó.
2. **`O` y `0` son la misma forma.** En LSA no hay distinción visual usable para el clasificador. Si una aparece como predicción de la otra (aunque sea en top-2 o top-3), se cuenta **siempre como top-1**.

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

Hay dos formas de leer el ajuste:

**A — Beneficio de la duda (contar los 7 fallos de ejecución como acierto, más O ≡ 0)**

| Criterio | Top-1 | Top-3 |
|----------|-------|-------|
| Crudo (CSV, 94 señas) | 63/94 (67,0%) | 84/94 (89,4%) |
| O ≡ 0 | 64/94 (68,1%) | 84/94 (89,4%) |
| O ≡ 0 + 7 takes mal señados | **71/94 (75,5%)** | **87/94 (92,6%)** |

**B — Sacar esas 8 filas del denominador (solo takes bien ejecutados, n = 86, más O ≡ 0)**

| Criterio | Top-1 | Top-3 |
|----------|-------|-------|
| 86 takes válidos, crudo | 62/86 (72,1%) | 79/86 (91,9%) |
| 86 takes válidos, O ≡ 0 | **63/86 (73,3%)** | **79/86 (91,9%)** |

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

Con O ≡ 0 y sin las 8 señas mal ejecutadas, el top-1 de takes válidos queda en **73,3%** (63/86), por encima del mejor de julio (B, 71,4% sobre 91). B usaba 32 frames; esta red usa 16, más videos y más augmentation. La comparación no es perfecta (julio no descontó takes mal señados), pero deja de verse como un empate flojo.

### 4.4 Errores que no son O/0 ni “take mal señado”

Después de sacar `1 2 4 5 6 hermano_a martes nosotros` y de tratar `O` como `0`, siguen patrones viejos:

- Atractor **`quien`**: se come `I`, `T`, `papa`, `vivir`, `vos`.
- Atractor **`0`**: además de `O` (equivalente), se come `chau`, `nombre`, `si`.
- `familia` → `calle` y `repetir` → `documento`: otra vez fuera del top-3.
- `3` → `V` (el dígito bien ejecutado que sí falla).

---

## 5. Lectura

El retrain de agosto **sí movió el vivo** respecto de la única corrida previa a 16 frames (A: 61,5% → 68,1% crudo; **73,3%** en takes válidos con O ≡ 0). Offline quedó muy alto (98,4%), como suele pasar cuando se sube el dato y el multiplicador virtual.

El golpe a dígitos en el CSV crudo no se puede leer como fallo del modelo: cinco de diez dígitos los ejecutó mal el evaluador. En los cinco takes válidos el top-1 es 80%, salvo `3`.

Lo que más falta, ya sobre takes bien señados:

1. **Fronteras `quien` / `0`** — desambiguación por contexto (top-3 + módulo semántico) más que por el clasificador solo.
2. **`familia` / `repetir`** — otra vez fuera del top-3; frontera que el modelo no tiene.
3. **Calibración** — el modelo está seguro cuando se equivoca; un umbral más alto no alcanza.

La eval de una seña por clase, con un solo señante, sigue siendo una muestra chica. Los ajustes del §4.2 (takes mal ejecutados y O ≡ 0) hay que anotarlos en las próximas corridas de `camera.py --eval`.
