# Semántico oyente — español → glosas LSA

Esta rama entrena y evalúa modelos que convierten una oración en español rioplatense en una secuencia de glosas de Lengua de Señas Argentina.

```
Me llamo Maite.  →  YO NOMBRE Maite
```

El entrenamiento es LoRA 4-bit con Unsloth. Al terminar se fusiona el adaptador, se exporta un GGUF `Q4_K_M` y ese binario es lo que se evalúa. Hace falta una GPU NVIDIA.

## Archivos

Todo el código está en `src/semantic/`.

| Archivo | Rol |
|---------|-----|
| `train.py` | Entrena los modelos en serie |
| `train_model.py` | Un modelo: LoRA, export GGUF y evaluación del split |
| `eval_trial.py` | Compara los GGUF ya exportados sobre `dataset_trial.json` |
| `config.py` | Rutas, hiperparámetros y la lista de modelos |
| `dataset.py` | Carga el dataset de entrenamiento y el trial |
| `prompts.py` | System prompt y mensajes user/assistant |
| `prompts/sys_prompt.txt` | Instrucciones de glosado que ve el modelo |
| `gloss.py` | Une tokens, limpia la salida y normaliza para comparar |
| `gguf.py` | Exporta el GGUF, lo encuentra y genera una glosa |
| `metrics.py` | Accuracy, BLEU, ROUGE-L y METEOR, y arma el informe |
| `metrics_neural.py` | BERTScore, COMET y BLEURT |
| `io_utils.py` | Log con timestamp y JSON de las corridas |
| `dataset_glosas.json` | Pares de entrenamiento |
| `dataset_trial.json` | Paráfrasis de evaluación, fuera del split |
| `requirements_train.txt` | Dependencias. Unsloth no está incluido |

`metrics.py` no se llama `evaluate` porque ese nombre lo usa la librería de Hugging Face.

## Entorno

```powershell
conda create -n lsa-train python=3.11 -y
conda activate lsa-train
cd src\semantic

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_train.txt
pip install unsloth unsloth_zoo --no-deps
pip install unbabel-comet==2.2.7 --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`pip install unsloth` sin `--no-deps` puede reemplazar el torch con CUDA por uno de CPU. `unbabel-comet` fija `transformers<5`; con `--no-deps` no pisa torch ni transformers.

`llama-cpp-python` no está en el requirements: hace falta un wheel con CUDA de la máquina para evaluar el GGUF. Si Unsloth no ve la GPU, reinstalar `torch+cu126` y después `unsloth unsloth_zoo --no-deps`.

Si el repo está en una ruta con espacios (`Mis Archivos`), Unsloth puede fallar al escribir el `.gguf`. El export usa `D:\temp_gguf`, o la carpeta de `LSA_GGUF_TEMP`:

```powershell
$env:LSA_GGUF_TEMP = "D:\temp_gguf"
```

## Entrenar

```powershell
cd src\semantic
python train.py --continue-on-error
python train.py --models qwen2.5-0.5b llama-3.2-1b --continue-on-error
python train.py --max-steps 20
```

IDs: `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen2.5-3b`, `llama-3.2-1b`, `smollm2-1.7b`. Sin `--models` entrena los cinco, en ese orden. `--continue-on-error` sigue con el próximo modelo si uno falla. `--max-steps` mayor que 0 reemplaza las épocas.

Cada ejemplo incluye el system prompt y queda en unos 1600 tokens. En una GPU de 16 GB o menos el batch es 1 con acumulación 16 (el lote efectivo sigue siendo 16). En el modelo de 3B el checkpointing descarga activaciones a RAM. Con más de 16 GB el batch es 4 (2 en el 3B).

| Parámetro | Valor |
|-----------|-------|
| Cuantización de carga | 4-bit |
| LoRA r / alpha / dropout | 16 / 32 / 0 |
| Módulos | q, k, v, o, gate, up, down |
| Épocas | 5 |
| Learning rate | 5e-4; 3e-4 en el 3B |
| Optimizador | adamw 8-bit |
| Scheduler | cosine, 10 pasos de warmup |
| Largo máximo | 2048. Con 512 el prompt se come la glosa |
| Loss | solo la respuesta del asistente |
| Split | 85/15, seed 3407 |
| Generación | temperature 0, máximo 48 tokens nuevos |

Salida, que no se versiona:

- `outputs/<modelo>/` adaptador y `metrics.json` del split
- `outputs/<modelo>_gguf/` binario `Q4_K_M`
- `outputs/training_summary.json` acumulado de todas las corridas
- `outputs/training_batch_<fecha>.json` una corrida
- `logs/train_all_<fecha>.log`

Esos registros de evaluaciones anteriores se conservan. Una corrida nueva agrega una entrada al summary y no borra los JSON ni los logs viejos.

## Evaluar el trial

`dataset_trial.json` no entra en el split. Cada ítem tiene una categoría, una lista de glosas y varias paráfrasis en español que comparten esa glosa.

```powershell
python eval_trial.py
python eval_trial.py --models qwen2.5-0.5b
```

Lee el GGUF de `outputs/`. Si un modelo no tiene binario, lo saltea y lo anota en el reporte. Escribe en `outputs/eval_trial/`:

- `comparativa_trial_<fecha>.json` y `.md`, más el `.log`
- `comparativa_trial_latest.json` y `.md`, copia de la última corrida

Las comparativas fechadas anteriores no se pisan.

## Métricas

Sobre el texto generado por el GGUF, no sobre el modelo en entrenamiento:

- Accuracy estricta: la predicción es idéntica a alguna referencia.
- Accuracy normalizada: igual, pero en minúsculas y sin puntuación.
- BLEU, ROUGE-L y METEOR. Con varias referencias se toma el máximo de ROUGE-L y METEOR.
- BERTScore F1 (`lang=es`), COMET (`Unbabel/wmt22-comet-da`) y BLEURT-20. También usan el máximo por oración. COMET recibe la oración en español como fuente y las glosas como hipótesis y referencia. El promedio se informa en escala 0–100, igual que BLEU.

La primera evaluación descarga BERTScore, COMET y BLEURT.

## Datos y glosas

`dataset_glosas.json` es una lista `dataset` de `{ "spanish", "glosses" }`. `glosses` es una lista de tokens. El cargador los une con espacios y no compacta deletreos ni cifras: un nombre propio es una palabra (`Maite`) y un DNI es un solo token (`18333940`).

El criterio de glosado está en `prompts/sys_prompt.txt`. En resumen: glosas léxicas en mayúsculas, se conserva la Ñ, el orden es tiempo → lugar → sujeto → objeto → posesivo → número → verbo → pregunta o negación, el género explícito se marca con `O` o `A` después de la seña base, y no se traducen artículos ni preposiciones. La respuesta del modelo es solo la secuencia de glosas.
