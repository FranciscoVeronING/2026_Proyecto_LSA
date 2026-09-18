# Entrenamiento y evaluación — módulo semántico LSA

Guía para configurar el entorno, entrenar los LLMs y compararlos.

---

## Requisitos de hardware

| Componente | Mínimo recomendado |
|------------|-------------------|
| GPU | NVIDIA con CUDA (6–8 GB VRAM para 0.5B–1.7B en 4-bit; ~8–12 GB para 3B) |
| RAM | 16 GB |
| Disco | ~10 GB libres (modelos base + adaptadores + GGUF) |

Unsloth **requiere GPU**. Sin CUDA el entrenamiento no arranca.

---

## 1. Crear entorno

```powershell
conda create -n lsa-train python=3.11 -y
conda activate lsa-train
```

Python **3.11** es la opción más estable con Unsloth.

```powershell
cd src\semantic
```

---

## 2. Instalar dependencias

> **Importante (Windows):** si instalás `unsloth` con pip normal, puede reemplazar `torch+cu126` por torch CPU. Seguí los 4 pasos en orden.

### Paso A — PyTorch con CUDA

Reemplazá `cu126` por tu versión de CUDA si es distinta (`cu124`, `cu121`, etc.):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verificá la GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Debe mostrar algo como `2.13.0+cu126 True`. Si dice `+cpu False`, **no sigas**.

### Paso B — Dependencias del proyecto (sin unsloth)

```powershell
pip install -r requirements_train.txt
```

### Paso C — Unsloth sin reinstalar torch

```powershell
pip install unsloth unsloth_zoo --no-deps
```

### Paso D — Verificar CUDA y Unsloth

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

---

## 3. Archivos

| Archivo | Propósito |
|---------|-----------|
| [`train_all_models.py`](../src/semantic/train_all_models.py) | Entrena los modelos, exporta GGUF y evalúa el split de test |
| [`eval_compare_models.py`](../src/semantic/eval_compare_models.py) | Compara todos los checkpoints sobre `dataset_trial.json` |
| [`dataset_glosas.json`](../src/semantic/dataset_glosas.json) | Pares glosa → español para entrenamiento |
| [`dataset_trial.json`](../src/semantic/dataset_trial.json) | Set de prueba fijo, con varias traducciones válidas |
| [`prompts/sys_prompt.txt`](../src/semantic/prompts/sys_prompt.txt) | System prompt |
| [`requirements_train.txt`](../src/semantic/requirements_train.txt) | Paquetes pip |

### Modelos del batch

| ID | Modelo Hugging Face |
|----|---------------------|
| `qwen2.5-0.5b` | `unsloth/Qwen2.5-0.5B-Instruct` |
| `qwen2.5-1.5b` | `unsloth/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-3b` | `unsloth/Qwen2.5-3B-Instruct` |
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B-Instruct` |
| `smollm2-1.7b` | `unsloth/SmolLM2-1.7B-Instruct` |

---

## 4. Entrenar

```powershell
conda activate lsa-train
cd src\semantic
python train_all_models.py --continue-on-error
```

Opciones:

```powershell
# Solo algunos modelos
python train_all_models.py --models qwen2.5-0.5b llama-3.2-1b --continue-on-error

# Épocas (default: 5)
python train_all_models.py --epochs 5 --continue-on-error

# Dataset
python train_all_models.py --dataset .\dataset_glosas.json
```

Cada modelo queda en `outputs/` con adaptador LoRA, binario GGUF (`*_gguf/`) y `metrics.json`. El resumen acumulado está en `outputs/training_summary.json`.

---

## 5. Evaluar y comparar (`dataset_trial`)

Usa los GGUF entrenados (o el checkpoint HF si no hay GGUF):

```powershell
conda activate lsa-train
cd src\semantic
python eval_compare_models.py
```

Reportes:

```
src/semantic/outputs/eval_trial/
  comparativa_trial_latest.md
  comparativa_trial_latest.json
```

Métricas: accuracy estricta y normalizada, BLEU, ROUGE-L, METEOR, latencia, desglose por categoría.

---

## 6. Parámetros de entrenamiento (actuales)

| Parámetro | Valor |
|-----------|-------|
| LoRA rank (`r`) | 16 |
| Cuantización | 4-bit (train) / GGUF Q4_K_M (eval) |
| Batch size | 4 × 4 acumulación |
| Learning rate | 5e-4 |
| Épocas | 5 |
| Max seq length | 2048 |

El split interno de `train_all_models.py` es **85/15** con `seed=3407` (sobre `dataset_glosas.json`). La comparativa de `eval_compare_models.py` usa **todo** `dataset_trial.json`.

---

## 7. Problemas frecuentes

### `Unsloth cannot find any torch accelerator`

PyTorch quedó en CPU. Reparación:

```powershell
conda activate lsa-train
cd src\semantic
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install unsloth unsloth_zoo --no-deps
```

**No** instales unsloth sin `--no-deps`. Orden correcto: torch CUDA → `requirements_train.txt` → unsloth `--no-deps`.

### `No se encontró binario GGUF`

Entrená primero con `train_all_models.py`. La evaluación busca `outputs/<modelo>_gguf/*.gguf`.
