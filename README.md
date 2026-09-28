# 2026_Proyecto_LSA — `scratch-mediapipe-v2`

Prototipo de reconocimiento de **Lengua de Señas Argentina (LSA)** para el TFG (UNMDP). Esta rama itera el clasificador **TinySkeleton**: MediaPipe Holistic (pose + manos, sin cara) + Transformer chico, 97 glosas, inferencia en webcam con letterbox 16:9.

Autores: Francisco Veron, Maite Nigro.

> Rama de trabajo del clasificador. Nace de `main` + merge de `scratch-mediapipe`. No es el informe TFG completo: eso está en `docs/Informe.tex`.

Los landmarks se extraen con **`@mediapipe/holistic@0.5.1675471629`** (JS/WASM, GPU/WebGL), no con el paquete Python `mediapipe`. Hay que regenerar los `.npy` (`preprocessing.py --force`) y usar el checkpoint `src/model/tinyskeleton_best.pth`.

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

## Estado — septiembre 2026

Checkpoint actual: **baseline 16 frames / 128 dim / 4 heads / 2 capas**, **97 clases**, padding a 16:9 en cámara. Eval del 17/09 (`eval_97senias_20260917_160148.csv`, 1 toma, `640x480→pad853x480`, MediaPipe 0.10.21):

| | Webcam top-1 | Webcam top-3 | Val offline |
|--|----------------|--------------|-------------|
| 97 señas, 17/09, padding 16:9 | **77,3 %** | **94,8 %** | 98,3 % |
| 36 letras/números (mismo CSV) | **88,9 %** | **100 %** | — |
| 97 con colapso O≡0 | 78,4 % | — | — |

**Lo que cambió respecto de agosto:** `I → ojo` desapareció (era distorsión de aspect 4:3). `V ↔ viernes` no: `viernes` no se regrabó. Detalle: [docs/eval_97senias_20260917.md](docs/eval_97senias_20260917.md).

Números de agosto (4:3, 94 clases, otra luz/sesión) — no son la línea de base de este checkpoint:

| | Webcam top-1 crudo | Top-1 con pares homógrafos | Top-3 |
|--|--------------------|----------------------------|-------|
| Baseline 16f, luz buena (28/08) | **86,2%** | **90,4%** | 95,7% |
| Optuna 12f, luz buena | 80,9% | 86,2% | **97,9%** |
| 10f / 3 capas (31/08) | 77,7% | 80,9% | 87,2% |
| Baseline 16f, luz mala (18/08) | 67,0% | 69,1% | 89,4% |

A igual iluminación Optuna **perdía ~4 pp** en top-1 (sobre todo letras) y ganaba ~2 pp en top-3. Por eso se reentrenó el **baseline**, no Optuna. El salto 67% → 86% del 18/08 al 28/08 fue **luz**, no el search.

**Integrar este baseline 16f / 128 / 4 heads / 97c, con letterbox 16:9.** Ficha: [docs/ficha_tecnica_refinamiento_feature_engineering.md](docs/ficha_tecnica_refinamiento_feature_engineering.md). Agosto: [docs/Relevamiento_clasificador_agosto_2026.md](docs/Relevamiento_clasificador_agosto_2026.md) (§13).

---

## Modelos

`camera.py` descubre los `.pth` en `src/model/` y, en `--eval`, permite elegir varios. La arquitectura tiene que coincidir con el checkpoint (`config.py` hoy: 16 frames, 128 dim, 4 heads, 2 capas, 97 clases).

Checkpoint actual (Holistic web): `src/model/tinyskeleton_best.pth` — baseline 16f / 128 / 4H / 2L, val **99,14%**. El `.pth` de MediaPipe Python (`tinyskeleton_best_optuna_v2.pth`) y la eval del 17/09 quedan como referencia histórica.

Checkpoint de septiembre con MediaPipe Python 0.10.21 (eval del 17/09): `src/model/tinyskeleton_best_optuna_v2.pth` — el nombre dice Optuna; la arquitectura es **baseline** 16f / 128 / 4H / 2L.

Checkpoints de agosto, para comparar:

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

Python **3.11** + Node **18+** + Chrome/Edge con WebGL GPU. Guía: [docs/entorno.md](docs/entorno.md).

```powershell
conda env create -f environment.yml
conda activate lsa_gpu
cd src\holistic_web
npm install
npx playwright install chromium
cd ..
python check_env.py
```

---

## Cómo correr

Todo desde `src/`, con `lsa_gpu` activo.

```powershell
python camera.py              # inferencia en vivo (padding a 16:9 por default)
python camera.py --eval       # recorre las 97 señas y escribe CSV
```

`--eval-output ruta.csv` si querés el nombre del archivo.

**Luz:** MediaPipe necesita iluminación decente (landmarks sin jitter). Comparar evals con distinta luz no atribuye el delta al modelo.

**Entrenar de nuevo** (hace falta el corpus local). Si los `.npy` salieron de MediaPipe Python, `--force`:

```powershell
python preprocessing.py --force  # MP4 → .npy (Holistic web, GPU)
python train.py                  # escribe tinyskeleton_best.pth + metrics.json
```

Optuna (lento: 40 trials × 3 folds). El ganador **no** se usa en webcam sin reentrenar el split 80/20 y sin `--eval`:

```powershell
python tune_optuna.py            # o --n-trials 2 para una prueba
```

Bitácora de hiperparámetros y comandos: [docs/cambios_scratch-mediapipe-v2.md](docs/cambios_scratch-mediapipe-v2.md).

---

## Protocolo al leer un `--eval`

El CSV crudo **no** es el número que se informa. Hay dos familias distintas:

| Par | Qué son | En eval hoy |
|-----|---------|-------------|
| `O` / `0` | Misma forma, sin contraste de movimiento | Si una aparece en el top-3 de la otra, cuenta como top-1 |
| `L` / `lunes`, `G` / `años`, `F` / `donde` | Letra **estática** vs seña **dinámica** (agite en X / Y / Z) | Agosto las colapsó igual que O/0; el objetivo es **separarlas**, no tratarlas como un solo signo |
| `V` / `viernes` | V estática (hold); viernes = mano en V, 2 golpes al mentón en Z | **No colapsar.** Eval 17/09: swap bidireccional; regrabar `viernes` |

Takes mal señados no se le cargan al modelo. En la sesión del 18/08 fueron `1 2 4 5 6 hermano_a martes nosotros`.

---

## Dataset y arquitectura (esta iteración)

| | Julio (A–D) | Esta rama |
|--|-------------|-----------|
| Videos / clase | 50 | **60** |
| Clases | 91 | **97** (`ñ`, `pasado`, `futuro`, `mio`; `chau`/`tener`/`años` ya venían) |
| Augmentation | escala + ruido | eso + rotación, crop temporal, time warp, dropout de frames, ruido pose/manos |
| `VIRTUAL_MULTIPLIER` | 10 | **25** |

225 features por frame (33 pose × 3 + 21×2 manos × 3). Sin cara. Sin flip horizontal (en LSA la mano dominante importa).

---

## Documentación

| Doc | Para qué |
|-----|----------|
| [docs/ficha_tecnica_refinamiento_feature_engineering.md](docs/ficha_tecnica_refinamiento_feature_engineering.md) | Entrega septiembre: features, aspect ratio, eval 97 |
| [docs/eval_97senias_20260917.md](docs/eval_97senias_20260917.md) | Análisis de la eval de cámara del 17/09 |
| [docs/inspeccion_features_mano_rostro.md](docs/inspeccion_features_mano_rostro.md) | Diagnóstico I/T/OJO y confusiones de dataset |
| [docs/Relevamiento_clasificador_agosto_2026.md](docs/Relevamiento_clasificador_agosto_2026.md) | Entrenamiento, las tres evals webcam de agosto, luz vs modelo |
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
| `camera.py` | Webcam, `--eval`, letterbox 16:9 |
| `model_arch.py` | TinySkeletonClassifier |
| `video_divider.py` | Recorte de gestos en video (dato, no inferencia) |
| `check_env.py` | Verifica MediaPipe / protobuf / torch |
| `hand_face_features.py` | 26 features interpretables (diagnóstico; aún no van al modelo) |
| `inspect_sign_features.py` | Histogramas y plots de esas features sobre .npy o MP4 |
| `analyze_sign_confusions.py` | Confusiones 1-NN + AUC sobre el dataset |
| `compare_eval_vs_dataset.py` | Cruza fallos de cámara con separabilidad del dato |
| `predict_npy.py` | Referencia dorada de inferencia sin cámara |
