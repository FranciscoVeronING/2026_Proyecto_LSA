# Entrenamiento — español → glosas LSA

LoRA con Unsloth, export GGUF `Q4_K_M`, métricas sobre el binario cuantizado. Hace falta GPU NVIDIA.

## Entorno

```powershell
conda create -n lsa-train python=3.11 -y
conda activate lsa-train
cd src\semantic

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_train.txt
pip install unsloth unsloth_zoo --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements_train.txt` no incluye Unsloth a propósito. También hace falta `llama-cpp-python` con CUDA para evaluar el GGUF.

Si el repo está en `Mis Archivos`, Unsloth puede fallar al escribir el `.gguf`:

```powershell
$env:LSA_GGUF_TEMP = "D:\temp_gguf"
```

## Batch

```powershell
python train_all_models.py --continue-on-error
python train_all_models.py --models qwen2.5-0.5b llama-3.2-1b --continue-on-error
```

IDs: `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `llama-3.2-1b`, `smollm2-1.7b`.

Salida en `outputs/` (adaptador + carpeta `*_gguf`) y log en `logs/`. Split 85/15, seed 3407.

```powershell
python eval_compare_models.py
python chat.py --model qwen2.5-0.5b
```

| Parámetro | Valor |
|-----------|-------|
| LoRA r / alpha | 16 / 32 |
| Épocas | 5 |
| LR | 5e-4 (3e-4 en 3B) |
| Loss | solo la respuesta |

Si Unsloth no ve GPU, torch quedó en CPU: reinstalar `torch+cu126` y `unsloth --no-deps`.
