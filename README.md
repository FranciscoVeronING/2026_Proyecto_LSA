# 2026_Proyecto_LSA — `scratch-mediapipe-v2`

Prototipo de reconocimiento de **Lengua de Señas Argentina (LSA)** para el TFG (UNMDP). Esta rama itera el clasificador **TinySkeleton**: MediaPipe Holistic (pose + manos, sin cara) + Transformer chico, 94 glosas, inferencia en webcam.

Autores: Francisco Veron, Maite Nigro.

> Rama de trabajo del clasificador. Nace de `main` + merge de `scratch-mediapipe`. No es el informe TFG completo: eso está en `docs/Informe.tex`.

---

## Qué hay acá (y qué no)

**Sí**

- Entrenar y evaluar el reconocedor de señas aisladas (`src/train.py`, `src/camera.py --eval`).
- Dos checkpoints de agosto 2026: baseline 16 frames y Optuna v2 (12 frames).
- Pipeline de landmarks alineado entre preprocesado y cámara.

**No** (viven en otras ramas / carpetas)

- App web, TTS, módulo semántico / LLM.
- El corpus de video (`dataset/`, `dataset_landmarks_*`) no va al git.

---

## Estado — agosto 2026

Con **buena luz**, el checkpoint que conviene para el prototipo es el **baseline 16 frames** (128 dim / 4 heads / 2 capas), no Optuna.

| | Webcam top-1 crudo | Top-1 con pares homógrafos | Top-3 |
|--|--------------------|----------------------------|-------|
| Baseline 16f, luz buena | **86,2%** | **90,4%** | 95,7% |
| Optuna 12f, luz buena | 80,9% | 86,2% | **97,9%** |
| Baseline 16f, luz mala (18/08) | 67,0% | 69,1% | 89,4% |

A igual iluminación Optuna **pierde ~4 pp** en top-1 (sobre todo letras) y gana ~2 pp en top-3. El salto 67% → 86% del 18/08 al 28/08 fue **luz**, no el search. Offline no anticipa el vivo (98,4% vs 97,8% val).

Números, pareos y protocolo: [docs/Relevamiento_clasificador_agosto_2026.md](docs/Relevamiento_clasificador_agosto_2026.md).

---

## Modelos

`camera.py` carga **siempre** `src/model/tinyskeleton_best.pth` y arma la red con `src/config.py`. Arquitectura y `MAX_FRAMES` tienen que coincidir con ese `.pth`.

| | Baseline (recomendado) | Optuna v2 |
|--|------------------------|-----------|
| Carpeta | `src/model/model_no_opt/` | `src/model/model_opt/` |
| Pesos | `tinyskeleton_best.pth` | `tinyskeleton_best_optuna.pth` |
| `MAX_FRAMES` | **16** | **12** |
| `HIDDEN_DIM` / heads / layers | 128 / **4** / 2 | **256** / **2** / 2 |
| Dropout | 0,40 | 0,57 |
| Eval webcam | `eval_94senias_20260828_222229_no_opt.csv` | `eval_94senias_20260828_202929_opt.csv` |

Para probar el baseline:

```text
copy src\model\model_no_opt\tinyskeleton_best.pth src\model\tinyskeleton_best.pth
```

En `config.py`: `MAX_FRAMES = 16`, `HIDDEN_DIM = 128`, `NUM_HEADS = 4`, `NUM_LAYERS = 2`, `DROPOUT_RATE = 0.4`.

Para Optuna: copiar `model_opt/tinyskeleton_best_optuna.pth` → `tinyskeleton_best.pth` y poner 12 / 256 / 2 / 0,57. Si no matchea, `load_state_dict` falla.

El 18/08 (luz ambiente) está en `src/model/2026_08_18_model_no_opt/`.

---

## Entorno

Python **3.11** + `mediapipe==0.10.21` + `protobuf>=4.25.3,<5`. No subir MediaPipe a 0.10.30+: sacaron `mp.solutions`.

Guía: [docs/entorno.md](docs/entorno.md).

```powershell
conda env create -f environment.yml
conda activate lsa_gpu
cd src
python check_env.py
```

---

## Cómo correr

Todo desde `src/`, con `lsa_gpu` activo.

```powershell
python camera.py              # inferencia en vivo
python camera.py --eval       # recorre las 94 señas y escribe CSV
```

`--eval-output ruta.csv` si querés el nombre del archivo.

**Luz:** MediaPipe necesita iluminación decente (landmarks sin jitter). Comparar evals con distinta luz no atribuye el delta al modelo.

**Entrenar de nuevo** (hace falta el corpus local):

```powershell
python preprocessing.py          # MP4 → .npy (omití --force si ya existen)
python train.py                  # escribe tinyskeleton_best.pth + metrics.json
```

Optuna (lento: 40 trials × 3 folds). El ganador **no** se usa en webcam sin reentrenar el split 80/20 y sin `--eval`:

```powershell
python tune_optuna.py            # o --n-trials 2 para una prueba
```

Bitácora de hiperparámetros y comandos: [docs/cambios_scratch-mediapipe-v2.md](docs/cambios_scratch-mediapipe-v2.md).

---

## Protocolo al leer un `--eval`

El CSV crudo **no** es el número que se informa. En LSA hay pares que el clasificador no puede (ni debe) separar:

| Par | Por qué |
|-----|---------|
| `O` ≡ `0` | misma forma |
| `G` ≡ `años` | misma forma |
| `L` ≡ `lunes` | `L` estática, `lunes` esa pose en movimiento |
| `F` ≡ `donde` | `F` estática, `donde` esa pose en movimiento |

Si uno aparece en el top-3 del otro, cuenta **siempre como top-1**.

Takes mal señados no se le cargan al modelo. En la sesión del 18/08 fueron `1 2 4 5 6 hermano_a martes nosotros`.

---

## Dataset y arquitectura (esta iteración)

| | Julio (A–D) | Esta rama |
|--|-------------|-----------|
| Videos / clase | 50 | **60** |
| Clases | 91 | **94** (`chau`, `tener`, `años`) |
| Augmentation | escala + ruido | eso + rotación, crop temporal, time warp, dropout de frames, ruido pose/manos |
| `VIRTUAL_MULTIPLIER` | 10 | **25** |

225 features por frame (33 pose × 3 + 21×2 manos × 3). Sin cara. Sin flip horizontal (en LSA la mano dominante importa).

---

## Documentación

| Doc | Para qué |
|-----|----------|
| [docs/Relevamiento_clasificador_agosto_2026.md](docs/Relevamiento_clasificador_agosto_2026.md) | Entrenamiento, las tres evals webcam, luz vs modelo |
| [docs/cambios_scratch-mediapipe-v2.md](docs/cambios_scratch-mediapipe-v2.md) | Qué cambió en código y cómo relanzar Optuna |
| [docs/entorno.md](docs/entorno.md) | Dependencias y protobuf / MediaPipe |
| [docs/documentacion.md](docs/documentacion.md) | Pipeline técnico (más viejo; julio) |
| [docs/Entregable_Semana_Clasificador.md](docs/Entregable_Semana_Clasificador.md) | Iteraciones A–D de julio |
| [docs/Informe.tex](docs/Informe.tex) | Informe TFG |

---

## Scripts en `src/`

| Script | Rol |
|--------|-----|
| `config.py` | Clases, frames, hiperparámetros, umbral de cámara |
| `preprocessing.py` | Videos → `.npy` |
| `train.py` | Entrenamiento 80/20 |
| `tune_optuna.py` | Búsqueda v2 (`max_frames` 8/12/16/24 + resto) |
| `camera.py` | Webcam y `--eval` |
| `model_arch.py` | TinySkeletonClassifier |
| `video_divider.py` | Recorte de gestos en video (dato, no inferencia) |
| `check_env.py` | Verifica MediaPipe / protobuf / torch |
