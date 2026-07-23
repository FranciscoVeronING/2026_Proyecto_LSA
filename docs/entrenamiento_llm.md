# Entrenamiento LoRA — Módulo semántico LSA

Guía para configurar el entorno y ejecutar los scripts de fine-tuning en `src/semantic/`.

---

## Requisitos de hardware

| Componente | Mínimo recomendado |
|------------|-------------------|
| GPU | NVIDIA con soporte CUDA (6–8 GB VRAM para modelos 0.5B–1.7B en 4-bit) |
| RAM | 16 GB |
| Disco | ~10 GB libres (modelos base + adaptadores LoRA) |

Unsloth **requiere GPU**. Sin CUDA el entrenamiento no arranca.

---

## 1. Crear entorno en Anaconda

Usá un **entorno aislado** para no mezclar dependencias con el resto del sistema.

```powershell
# Abrir Anaconda Prompt o terminal con conda disponible
conda create -n lsa-train python=3.11 -y
conda activate lsa-train
```

**Python 3.11** es la opción más estable con Unsloth. Evitá 3.13 en producción hasta verificar compatibilidad.

Navegá al módulo semántico:

```powershell
cd C:\Users\franc\Documents\GitHub\2026_Proyecto_LSA\src\semantic
```

---

## 2. Instalar dependencias

### Paso A — PyTorch con CUDA (instalar primero)

Reemplazá `cu126` por tu versión de CUDA si es distinta (`cu124`, `cu121`, etc.):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verificá que detecte la GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Debe mostrar algo como `2.10.0+cu126 True`. Si dice `+cpu` o `False`, no instales Unsloth todavía: corregí CUDA/PyTorch primero.

### Paso B — Resto de dependencias del proyecto

```powershell
pip install -r requirements_train.txt
```

### Paso C — Verificar Unsloth

```powershell
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

---

## 3. Archivos del entrenamiento

| Archivo | Propósito |
|---------|-----------|
| [`train_llm.py`](../src/semantic/train_llm.py) | Entrena **un solo modelo**. Editás `MODEL_NAME` en el archivo. |
| [`train_runner.py`](../src/semantic/train_runner.py) | Lógica interna reutilizable (no ejecutar directo). |
| [`train_all_models.py`](../src/semantic/train_all_models.py) | Entrena **los 5 modelos en secuencia**. |
| [`dataset_glosas.json`](../src/semantic/dataset_glosas.json) | Dataset de pares glosa → español. |
| [`prompts/sys_prompt.txt`](../src/semantic/prompts/sys_prompt.txt) | System prompt base. |
| [`prompts/few_shot_examples.json`](../src/semantic/prompts/few_shot_examples.json) | Ejemplos few-shot inyectados al prompt. |
| [`requirements_train.txt`](../src/semantic/requirements_train.txt) | Lista de paquetes pip. |

### Modelos que prueba el batch (`train_all_models.py`)

| ID | Modelo Hugging Face |
|----|---------------------|
| `qwen2.5-0.5b` | `unsloth/Qwen2.5-0.5B-Instruct` |
| `qwen2.5-1.5b` | `unsloth/Qwen2.5-1.5B-Instruct` |
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B-Instruct` |
| `phi-3-mini-4k` | `unsloth/Phi-3-mini-4k-instruct` |
| `smollm2-1.7b` | `unsloth/SmolLM2-1.7B-Instruct` |

---

## 4. Ejecutar entrenamiento de un solo modelo

1. Activar el entorno:

   ```powershell
   conda activate lsa-train
   cd src\semantic
   ```

2. Editar el modelo en [`train_llm.py`](../src/semantic/train_llm.py):

   ```python
   MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
   ```

3. Ejecutar:

   ```powershell
   python train_llm.py
   ```

### Salida (`train_llm.py`)

```
src/semantic/outputs/unsloth_Qwen2.5-0.5B-Instruct/
  metrics.json          # BLEU, ROUGE-L, accuracy, ejemplos evaluados
  adapter_config.json   # config LoRA (y pesos del adaptador)
  ...
```

El script guarda el adaptador y métricas en `./outputs/{nombre_modelo}/`.

---

## 5. Ejecutar los 5 modelos en secuencia

Para comparar todos los candidatos sin tocar código:

```powershell
conda activate lsa-train
cd src\semantic
python train_all_models.py --continue-on-error
```

Si un modelo falla, `--continue-on-error` sigue con el siguiente.

### Opciones útiles

```powershell
# Solo algunos modelos
python train_all_models.py --models qwen2.5-0.5b phi-3-mini-4k --continue-on-error

# Más pasos de entrenamiento (default: 60)
python train_all_models.py --max-steps 120 --continue-on-error

# Dataset alternativo
python train_all_models.py --dataset .\dataset_glosas.json
```

### Salida (`train_all_models.py`)

Cada modelo en su carpeta:

```
src/semantic/outputs/
  unsloth_Qwen2.5-0.5B-Instruct/
    lora_adapter/           # pesos LoRA
    checkpoints/            # checkpoints de entrenamiento
    metrics.json            # métricas de evaluación
    training_history.json   # loss por step
    loss_plot.png           # gráfico de loss
  unsloth_Qwen2.5-1.5B-Instruct/
    ...
  training_summary.json     # resumen acumulado de todas las corridas
  training_batch_YYYYMMDD_HHMMSS.json

src/semantic/logs/
  train_all_YYYYMMDD_HHMMSS.log
```

---

## 6. Dejar corriendo en segundo plano (Windows)

Con el entorno activado:

```powershell
cd src\semantic
Start-Process python -ArgumentList "train_all_models.py --continue-on-error" -RedirectStandardOutput "logs\train_background.log" -RedirectStandardError "logs\train_background_err.log" -NoNewWindow
```

Seguí el progreso:

```powershell
Get-Content logs\train_all_*.log -Wait -Tail 20
```

---

## 7. Métricas generadas

| Métrica | Descripción |
|---------|-------------|
| `accuracy_exact_match_percent` | % de frases idénticas al gold en test |
| `bleu_score` | BLEU × 100 (calidad de traducción) |
| `rouge_l_score` | ROUGE-L × 100 (solapamiento con referencia) |
| `train_loss_final` | Loss final del entrenamiento (solo batch) |
| `ejemplos_evaluados` | Lista glosa / esperado / predicho |

El split train/test es **80/20** con `seed=3407`.

---

## 8. Parámetros de entrenamiento (actuales)

| Parámetro | Valor |
|-----------|-------|
| LoRA rank (`r`) | 16 |
| Cuantización | 4-bit |
| Batch size | 2 × 4 acumulación |
| Learning rate | 2e-4 |
| Max steps | 60 (configurable en batch) |
| Max seq length | 512 |

---

## 9. Problemas frecuentes

### `ModuleNotFoundError: No module named 'datasets'`

El entorno no está activado o faltan dependencias:

```powershell
conda activate lsa-train
pip install -r requirements_train.txt
```

### `Unsloth cannot find any torch accelerator? You need a GPU.`

PyTorch quedó en versión CPU. Reinstalá con CUDA **antes** de unsloth:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### `ValueError: Target modules not found`

Algunas arquitecturas (Phi-3, SmolLM2) no usan los mismos nombres de capas. `train_runner.py` reintenta con `target_modules="all-linear"`. En `train_llm.py` manual podés cambiar a:

```python
target_modules="all-linear"
```

### Unsloth baja la versión de PyTorch

Instalá siempre **torch+cu126 primero**, luego `pip install -r requirements_train.txt`. Si unsloth lo reemplaza, repetí el paso A.

---

## 10. Resumen rápido

```powershell
conda create -n lsa-train python=3.11 -y
conda activate lsa-train
cd src\semantic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_train.txt

# Un modelo
python train_llm.py

# Los cinco
python train_all_models.py --continue-on-error
```
