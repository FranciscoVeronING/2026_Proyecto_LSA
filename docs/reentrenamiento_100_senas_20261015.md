# Reentrenamiento — 100 señas (15/10/2026)

> **Rama:** `scratch-mediapipe-v3`  
> **Para qué:** cerrar el vocabulario en 100 señas y reentrenar TinySkeleton con los landmarks de MediaPipe Holistic web.  
> **Checkpoint:** `src/model_new_2026_10_15/tinyskeleton_best.pth`  
> **Métricas:** `src/model_new_2026_10_15/metrics.json` (15/10/2026, 17:04 ART)

---

## Qué se agregó

El vocabulario pasó de **97 a 100** clases. Las tres señas nuevas son `gracias`, `ayudar` y `policia`.

Los landmarks del resto no se regeneraron. Solo se extrajeron esas tres, con el mismo pipeline de esta rama: `@mediapipe/holistic@0.5.1675471629` (JS/WASM, WebGL en GPU), pose + manos, sin cara, vector de **225-D**, subsampleo a **16 frames**.

```text
cd src
python preprocessing.py --signs Gracias Ayudar Policia
```

Salida en `dataset_landmarks_16frames/`, una carpeta por seña. 229 clips guardados, 0 fallidos. Todos los `.npy` quedaron con forma `(16, 225)`.

| Seña | MP4 en `dataset/` | `.npy` nuevos |
|---|---:|---:|
| `Gracias` | 76 | 76 |
| `Ayudar` | 80 | 80 |
| `Policia` | 73 | 73 |

`NUM_CLASSES = 100` y las tres entradas quedaron en `SIGN_CLASSES` de `src/config.py`. Al armar el split, las 100 carpetas tenían al menos 60 clips: ninguna clase se descartó.

---

## Entrenamiento

Corrida del 15/10/2026 en `lsa_gpu`, CUDA, NVIDIA GeForce RTX 3060. `train.py` usa como máximo 60 clips por clase (split 80/20, semilla 42), aunque las señas nuevas tengan más videos.

| | |
|---|---|
| Clases | 100 / 100 |
| Train / val | 4.800 / 1.200 |
| Augmentation | sí, solo train, ×25 |
| `MAX_FRAMES` | 16 |
| `HIDDEN_DIM` | 128 |
| `NUM_HEADS` | 4 |
| `NUM_LAYERS` | 2 |
| `DROPOUT_RATE` | 0.4 |
| Batch / épocas máx. / patience | 32 / 200 / 15 |
| Épocas corridas | 34 (early stopping) |

Augmentation de la corrida: escala `(0.85, 1.15)`, rotación yaw ±15° / pitch ±8° / roll ±8°, time warp `(0.80, 1.25)`, crop temporal 15 %, dropout de hasta 3 frames, ruido de pose 0.012 y de manos 0.027.

---

## Resultado

El checkpoint guardado es el de **menor val loss**, época **19**.

| | Train | Validación |
|---|---:|---:|
| Época 1 | loss 2.894 · acc 40.16 % | loss 1.108 · acc 87.58 % |
| **Época 19 (modelo guardado)** | **loss 0.030 · acc 99.34 %** | **loss 0.0528 · acc 99.08 %** |
| Época 34 (última) | loss 0.017 · acc 99.70 % | loss 0.062 · acc 99.25 % |

Reevaluación del checkpoint sobre los 1.200 clips de validación:

| Métrica | Valor |
|---|---|
| Val loss | **0.0528** |
| Accuracy top-1 | **99.08 %** |
| Clases | 100 |
| Épocas | 34 |

La época 34 llegó a 99.25 % de accuracy en validación, con val loss 0.062. No reemplazó al checkpoint: el criterio de guardado es la val loss.

En la misma carpeta quedaron `mapeo_clases.json`, `curva_tinyskeleton.png` y `matriz_confusion_tinyskeleton.png`.
