# Cambios en `scratch-mediapipe-v2`

> **Rama:** nace de `main` + merge de `scratch-mediapipe`.  
> **Para qué:** iterar el clasificador TinySkeleton (más dato, más augmentation, 16 frames, Optuna v2).  
> **Docs relacionadas:** `docs/Relevamiento_clasificador_agosto_2026.md` (entrenamiento 18/08 y eval webcam).

---

## 1. Origen de la rama

- Checkout de `main`, branch nueva `scratch-mediapipe-v2`.
- Merge de todo `scratch-mediapipe`.
- Conflictos en `src/camera.py` y `src/config.py`: se quedó la versión de **scratch-mediapipe** (la red a iterar).
- Informe TFG actualizado: `docs/Informe.tex` + `docs/referencias.bib` (versión agosto).

---

## 2. Dataset

| Antes (julio, A–D) | Ahora |
|--------------------|--------|
| 50 videos/clase (`SAMPLES_PER_CLASS`) | **60** |
| 91 clases en el modelo | **94** (`chau`, `tener`, `años`) |
| 3.640 train / 910 val | **4.512 / 1.128** |

Los MP4 viven en `dataset/`. El train lee `.npy` en `dataset_landmarks_32frames` (o la ruta de `DATASET_NPY_DIR`). Si los videos nuevos no pasaron por `preprocessing.py`, el modelo no los ve.

Preprocesado alineado con cámara: interpolar ceros → trim de silencio → subsampleo a `MAX_FRAMES`.

```text
cd src
conda activate lsa_gpu
python preprocessing.py
```

`--force` solo si hay que reprocesar `.npy` que ya existen.

---

## 3. Data augmentation

`VIRTUAL_MULTIPLIER = 25` (antes 10). Solo train; validación sin aug.

**Ya estaba:** escala 3D (0,85–1,15) + ruido gaussiano (std ≈ 0,030).

**Se agregó:**

| Tipo | Parámetros | Qué simula |
|------|------------|------------|
| Rotación 3D | yaw ±15°, pitch ±8°, roll ±8° | ángulo de cámara |
| Recorte temporal | hasta 15% por extremo | seña no centrada |
| Time warp | velocidad 0,80–1,25× | señante más rápido/lento |
| Frame dropout | 0–3 frames + interpolación | MediaPipe pierde frames |
| Ruido pose vs manos | pose 0,012 / manos ≈ 0,030 | tracker más inestable en manos |

Sin flip horizontal (mano dominante en LSA). El mirror zurdo es normalización de preprocesado, no aug.

Implementación: crop/warp/dropout en `LabeledSkeletonDataset`; rotación/escala/ruido en `augment_batch_3d` (`src/train.py`). Knobs en `src/config.py`.

---

## 4. Retrain 18/08/2026 (16 frames)

Arquitectura: Conv1D + Transformer, attention pooling.  
`HIDDEN_DIM=128`, `NUM_HEADS=4`, `NUM_LAYERS=2`, `DROPOUT=0,4`.

Offline (`src/model/metrics.json`): **98,40%** val top-1, val loss **0,095**, 29 épocas, 94 clases.

Eval webcam: `src/eval_91senias_20260818_202845.csv` (94 filas).

| Criterio | Top-1 | Top-3 |
|----------|-------|-------|
| CSV crudo | 63/94 (67,0%) | 84/94 (89,4%) |
| O ≡ 0 | 64/94 (68,1%) | 84/94 (89,4%) |
| Takes válidos n=86 + O ≡ 0 | **63/86 (73,3%)** | **79/86 (91,9%)** |

Protocolo de eval:

- `O` y `0` son la misma forma: si una aparece en el top-3 de la otra, cuenta como **top-1**.
- Takes mal ejecutados por el evaluador (no se atribuyen al modelo): `1`, `2`, `4`, `5`, `6`, `hermano_a`, `martes`, `nosotros`. El `2` igual salió acierto.

Frente a julio, en 91 señas comunes el crudo queda 68,1% / 89,0% (A 16f era 61,5% / 83,5%; B 32f 71,4% / 87,9%).

Detalle: `docs/Relevamiento_clasificador_agosto_2026.md`.

---

## 5. Optuna v2 (max_frames + hiperparámetros)

`src/tune_optuna.py` ahora busca **a la vez**:

| Se busca | Rango |
|----------|--------|
| **`max_frames`** | **8, 12, 16, 24** |
| `hidden_dim` | 64 / 128 / 256 |
| `num_heads` | 2 / 4 / 8 (debe dividir a hidden_dim) |
| `num_layers` | 1–3 |
| `dropout_rate` | 0,2–0,6 |
| `batch_size` | 16 / 32 |
| `lr` | 1e-5–1e-3 (log) |
| `weight_decay` | 1e-4–1e-1 (log) |
| `label_smoothing` | 0–0,15 |
| `aug_noise_std` | 0,005–0,03 |

**Fijo (no se busca):** augmentation ON, `VIRTUAL_MULTIPLIER=25`, rotación/crop/warp/dropout/ruido de pose.  
`use_data_augmentation=False` se sacó del search: con este dataset apagar aug desperdicia trials.

Cada trial **subsamplea** los `.npy` a `max_frames`. Hace falta `T >= 24` en disco (carpeta `dataset_landmarks_32frames`). Si los `.npy` tienen T=16, Optuna omite el 24 y avisa.

Study nuevo (no mezcla con Optuna D de julio):

- Nombre: `tinyskeleton_lsa_v2`
- DB: `src/optuna_study_v2.db`
- K-fold 3, 40 trials, pruner median
- Resume si se corta: `load_if_exists=True`
- Al terminar escribe `src/model/optuna_best_v2.json`

`train_one_run` respeta `hyperparams["max_frames"]` en train y val.

**Cuidado:** Optuna minimiza **val loss offline**. En julio el trial D ganó offline y perdió en webcam. Después del search hay que `train.py` + `camera.py --eval` con el `max_frames` ganador (cámara y train tienen que usar el mismo número).

### Comandos

Desde la raíz del repo, GPU recomendida:

```text
conda activate lsa_gpu
cd src
python check_env.py
```

Prueba corta (2 trials, para ver que levanta):

```text
python tune_optuna.py --n-trials 2
```

Corrida completa (va a tardar mucho: 40 trials × 3 folds × early stopping, ×25 virtual):

```text
python tune_optuna.py
```

equivalente a `python tune_optuna.py --n-trials 40`.

Si se corta, el mismo comando **continúa** el study (no borra `optuna_study_v2.db`).

Dashboard opcional (otra terminal, también desde `src/`):

```text
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_study_v2.db
```

Cuando termine:

1. Abrir `src/model/optuna_best_v2.json`.
2. Copiar `max_frames`, arquitectura, `lr`, etc. a `src/config.py`.
3. `python train.py` (split 80/20 completo, no un fold).
4. `python camera.py --eval` y aplicar el protocolo O ≡ 0 / takes mal señados.

Para dejar la PC sola en Windows, una sesión que no muera al cerrar la terminal:

```text
conda activate lsa_gpu
cd src
python -u tune_optuna.py --n-trials 40 *> optuna_v2.log
```

O con `Start-Process` / `tmux` si hay WSL. El `-u` evita que el log se bufferice.

---

## 6. Archivos tocados (resumen)

| Archivo | Qué |
|---------|-----|
| `src/config.py` | 60 videos/clase, aug nueva, ×25, `MAX_FRAMES=16` |
| `src/train.py` | 5 augs + `max_frames` por hyperparams |
| `src/tune_optuna.py` | study v2, busca max_frames 8/12/16/24 |
| `docs/Informe.tex` / `referencias.bib` | informe agosto |
| `docs/Relevamiento_clasificador_agosto_2026.md` | métricas 18/08 |
| `src/video_divider.py` | recorte de gestos en video (script aparte) |
