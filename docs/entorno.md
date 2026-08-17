# Entorno Python — clasificador LSA

Guía para montar un entorno **actual pero estable**, sin los conflictos típicos de MediaPipe + protobuf.

## Resumen rápido

| Componente | Versión recomendada | Motivo |
|------------|---------------------|--------|
| **Python** | **3.11.x** | Soportado por MediaPipe 0.10.21, PyTorch CUDA y el código del repo |
| **mediapipe** | **== 0.10.21** | Última versión con `mp.solutions.holistic` (usado en `preprocessing.py` y `camera.py`) |
| **protobuf** | **>= 4.25.3, < 5** | MediaPipe 0.10.21 no funciona con protobuf 5+ (`GetMessageClass` missing) |
| **numpy** | **>= 1.26, < 2** | Requisito de MediaPipe 0.10.21 |
| **PyTorch** | 2.x + CUDA 12.1 (opcional) | Entrenamiento e inferencia en GPU |

### ⚠️ No actualizar MediaPipe a 0.10.30+

A partir de **0.10.30** Google **eliminó** `mediapipe.solutions`. Este proyecto usa:

```python
mp.solutions.holistic.Holistic(...)
```

Migrar a MediaPipe Tasks implica reescribir extracción de landmarks (fase posterior).

---

## Opción A — Conda (recomendada, entorno `lsa_gpu`)

Desde la raíz del repo:

```powershell
conda env create -f environment.yml
conda activate lsa_gpu
cd src
python check_env.py
```

Si ya tenés `lsa_gpu` roto (protobuf 5+, mediapipe nuevo):

```powershell
conda deactivate
conda env remove -n lsa_gpu
conda env create -f environment.yml
```

---

## Opción B — venv + pip

```powershell
cd C:\Users\franc\Documents\GitHub\2026_Proyecto_LSA
py -3.11 -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# GPU NVIDIA (Windows, CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd src
python check_env.py
```

---

## Arreglo rápido si ya tenés el entorno (`GetMessageClass`)

El error:

```text
module 'google.protobuf.message_factory' has no attribute 'GetMessageClass'
```

significa que tenés **protobuf 5 o 6**, incompatible con MediaPipe 0.10.21.

```powershell
conda activate lsa_gpu
pip install "protobuf>=4.25.3,<5" mediapipe==0.10.21 "numpy>=1.26,<2"
cd src
python check_env.py
python preprocessing.py --force
```

---

## Verificación

```powershell
cd src
python check_env.py
```

Debe imprimir `Entorno OK` y `mediapipe.solutions.holistic: OK`.

---

## ¿Por qué no Python 3.12 o 3.13?

- **3.12:** funciona con mediapipe 0.10.21 (hay wheels). Opción válida si 3.11 no está disponible.
- **3.13:** wheels de mediapipe 0.10.21 pueden faltar o ser inestables en Windows.
- **3.9:** funciona pero es el entorno donde apareció el error `int | None` (ya corregido en el código con `Optional`).

**Recomendación del equipo:** quedarse en **3.11** hasta migrar a MediaPipe Tasks.

---

## Dependencias opcionales

```powershell
pip install optuna   # tune_optuna.py
```

El módulo semántico (`modulo-semantico`) tiene requirements propios (LangChain, Unsloth, etc.) — no mezclar con el env del clasificador salvo que sepas resolver conflictos de protobuf.
