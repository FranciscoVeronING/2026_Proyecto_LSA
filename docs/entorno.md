# Entorno — clasificador LSA

Landmarks: **MediaPipe Holistic JS/WASM** (`@mediapipe/holistic@0.5.1675471629`, GPU/WebGL), la misma build que la extensión. **No** se usa el paquete Python `mediapipe`.

## Resumen rápido

| Componente | Versión recomendada | Motivo |
|------------|---------------------|--------|
| **Python** | **3.11.x** | PyTorch CUDA y el código del repo |
| **Node.js** | **18+** | Runtime de `@mediapipe/holistic` + Playwright |
| **Chrome / Edge** | con WebGL GPU | Holistic WASM; SwiftShader/CPU no vale |
| **numpy** | **>= 1.26** | Vectores 225-D |
| **PyTorch** | 2.x + CUDA 12.1 (opcional) | Entrenamiento e inferencia del clasificador |

Los `.npy` de MediaPipe Python **no** sirven: hay que `python preprocessing.py --force` y reentrenar.

---

## Opción A — Conda (recomendada, entorno `lsa_gpu`)

Desde la raíz del repo:

```powershell
conda env create -f environment.yml
conda activate lsa_gpu
cd src\holistic_web
npm install
npx playwright install chromium
cd ..
python check_env.py
```

Si WebGL no engancha la placa: `$env:HOLISTIC_WEB_HEADED=1`.

Si `lsa_gpu` todavía tiene `mediapipe` de pip, se puede desinstalar (`pip uninstall mediapipe`); ya no hace falta.

---

## Opción B — venv + pip

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# GPU NVIDIA (Windows, CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd src\holistic_web
npm install
npx playwright install chromium
cd ..
python check_env.py
```

---

## Verificación

```powershell
cd src
python check_env.py
```

Debe imprimir `Entorno OK` y la línea `mediapipe web: @mediapipe/holistic@0.5.1675471629` con vendor/renderer GPU.

---

## Dependencias opcionales

```powershell
pip install optuna   # tune_optuna.py
```

El módulo semántico (`modulo-semantico`) tiene requirements propios — no mezclar con el env del clasificador.
