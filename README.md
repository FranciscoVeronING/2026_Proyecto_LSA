# Rama semántica — glosas LSA a español

Fine-tuning de LLMs chicos para traducir glosas de Lengua de Señas Argentina a una oración en español rioplatense. El entrenamiento exporta GGUF Q4_K_M y mide ese binario. La comparativa fija corre sobre `dataset_trial.json`.

Los scripts se ejecutan desde `src/semantic` (los módulos se importan como archivos hermanos).

## Archivos

| Ruta | Rol |
|------|-----|
| `entrenar.py` | LoRA 4-bit, export GGUF y evaluación del holdout 15% |
| `evaluar_trial.py` | Comparativa de los checkpoints ya entrenados sobre `dataset_trial.json` |
| `modelos.py` | Catálogo, rutas y topes de generación |
| `glosas.py` | Dactilología, números y normalización del accuracy |
| `inferencia.py` | Carga y traducción GGUF |
| `metricas.py` | BLEU, ROUGE-L, METEOR, BERTScore, COMET y BLEURT |
| `dataset_glosas.json` | Pares glosa → español del entrenamiento |
| `dataset_trial.json` | Set fijo, con varias traducciones válidas por ítem |
| `prompts/sys_prompt.txt` | System prompt de train y eval |
| `requirements.txt` | Dependencias pip, sin Unsloth ni el paquete COMET |
| `outputs/eval_trial/` | Comparativas guardadas, incluidas las anteriores a esta reestructura |
| `outputs/training_summary.json` | Historial de corridas de entrenamiento |

Pesos, GGUF y logs quedan en `outputs/` y `logs/`. Git no los versiona, salvo el resumen de entrenamiento y `outputs/eval_trial/`.

## Entorno

GPU NVIDIA con CUDA. Unsloth no entrena en CPU. Para 0.5B–1.7B en 4-bit alcanzan unos 8 GB de VRAM; el 3B pide cerca de 12 GB. RAM 16 GB y unos 10 GB de disco para bases, adaptadores y GGUF.

```powershell
conda create -n lsa_semantic python=3.10 -y
conda activate lsa_semantic
cd src\semantic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Tiene que imprimir una build `+cu128` y `True`. Si dice `+cpu` o `False`, no sigas.

```powershell
pip install -r requirements.txt
pip install unsloth unsloth_zoo --no-deps
pip install unbabel-comet==2.2.7 --no-deps
python -c "import torch; from unsloth import FastLanguageModel; print(torch.__version__, torch.cuda.is_available())"
```

`unsloth` sin `--no-deps` puede reemplazar el torch de CUDA. `unbabel-comet` sin `--no-deps` baja `transformers` a la serie 4 y rompe Unsloth. El código de COMET y BLEURT incluye los ajustes para seguir usando Transformers 5.

La primera evaluación descarga `bert-base-multilingual-cased`, `Unbabel/wmt22-comet-da` (XLM-RoBERTa large) y `lucadiliello/BLEURT-20`.

## Entrenar

```powershell
python entrenar.py --continue-on-error
python entrenar.py --models qwen2.5-0.5b llama-3.2-1b --continue-on-error
python entrenar.py --epochs 5
python entrenar.py --dataset .\dataset_glosas.json
```

Cada modelo hace LoRA (`r=16`, `alpha=32`, dropout 0, módulos de atención y MLP). Cuantización de entrenamiento: 4-bit. Contexto de carga: 2048. El texto de entrenamiento se trunca a 512. Batch 4 y acumulación 4 (el 3B usa 2 y 8). Learning rate 5e-4, o 3e-4 en el 3B. 5 épocas, warmup 10, AdamW 8-bit, cosine, semilla 3407. El split interno es 85/15 con semilla 3407.

Antes de armar el prompt, las letras sueltas se compactan (`M a i t e` → `Maite`, como máximo dos iguales seguidas) y los dígitos sueltos se concatenan sin truncar. El system prompt va en el mensaje `system`, las glosas en `Glosas: ...` y el español en `assistant`. Solo se entrena la respuesta.

Al terminar se fusiona el adaptador y se exporta GGUF Q4_K_M. La evaluación del holdout corre sobre ese binario, no sobre el modelo en 4-bit. Salidas:

```
outputs/unsloth_<Modelo>/          adaptador y metrics.json
outputs/unsloth_<Modelo>_gguf/     *.Q4_K_M.gguf
outputs/training_summary.json      una entrada por corrida exitosa
outputs/entrenamiento_<fecha>.json resumen del batch
logs/entrenar_<fecha>.log
```

## Evaluar `dataset_trial`

Usa el GGUF Q4_K_M si existe. Si no, carga el checkpoint de Hugging Face de la carpeta del modelo (ese camino no incluye el adaptador LoRA suelto: hace falta `config.json` de un modelo completo).

```powershell
python evaluar_trial.py
python evaluar_trial.py --models qwen2.5-1.5b
python evaluar_trial.py --stop-on-error
```

Generación, igual que en el holdout del entrenamiento: temperatura 0, 24 tokens nuevos, penalización de repetición 1.05, corte en salto de línea y en los tokens de fin de Qwen y Llama. Las glosas del trial pasan por la misma unificación dactilológica.

Cada corrida escribe un par fechado y pisa los `comparativa_trial_latest.*`. Los reportes anteriores se conservan en `outputs/eval_trial/`.

## Métricas

Todas en escala 0–100. Con varias referencias válidas, ROUGE-L, METEOR, BERTScore, COMET y BLEURT usan la mejor de esa oración. BLEU es el BLEU de corpus con la lista de referencias.

| Métrica | Qué mide |
|---------|----------|
| Accuracy estricta | La predicción es idéntica a alguna referencia |
| Accuracy normalizada | Igual, en minúsculas y sin `¿?¡!.,;"` |
| BLEU | N-gramas de superficie |
| ROUGE-L | Subsecuencia común más larga |
| METEOR | Unigramas con stemming y sinónimos |
| BERTScore | F1 de `bert-base-multilingual-cased`, idioma `es` |
| COMET | `Unbabel/wmt22-comet-da`; la fuente es el texto de glosas |
| BLEURT | BLEURT-20, par referencia/predicción con su SentencePiece |
| Latencia | Milisegundos de la generación GGUF o HF, sin contar las métricas |

El ranking de la comparativa ordena por accuracy normalizada y después por BLEU, ROUGE-L, METEOR, BERTScore, COMET, BLEURT y, al final, menor latencia.

## Catálogo

| ID | Modelo |
|----|--------|
| `qwen2.5-0.5b` | `unsloth/Qwen2.5-0.5B-Instruct` |
| `qwen2.5-1.5b` | `unsloth/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-3b` | `unsloth/Qwen2.5-3B-Instruct` |
| `llama-3.2-1b` | `unsloth/Llama-3.2-1B-Instruct` |
| `smollm2-1.7b` | `unsloth/SmolLM2-1.7B-Instruct` |

Para sumar un modelo, una entrada en `MODELOS` (`modelos.py`) alcanza: id, nombre de Hugging Face, etiqueta, plantilla de chat de Unsloth y los marcadores de instrucción y respuesta. Train y eval leen esa lista.

## Fallos frecuentes

**`Unsloth cannot find any torch accelerator`.** Torch quedó en CPU. Desinstalar `torch` y `torchvision`, reinstalar el índice `cu128` y volver a instalar Unsloth con `--no-deps`.

**`No hay GGUF ni checkpoint HF`.** Todavía no se entrenó ese id, o el GGUF no está en `outputs/<carpeta>_gguf/`.

**COMET o BLEURT no importan.** Revisar que `unbabel-comet` se haya instalado con `--no-deps` y que `transformers` siga en 5.x.
