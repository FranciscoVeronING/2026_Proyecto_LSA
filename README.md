# Proyecto LSA — módulo semántico

Fine-tuning de LLMs para traducir glosas de Lengua de Señas Argentina (LSA) a español rioplatense.

## Qué hay en el repo

| Ruta | Uso |
|------|-----|
| `src/semantic/train_all_models.py` | Entrenar los modelos (LoRA + export GGUF) |
| `src/semantic/eval_compare_models.py` | Evaluar y comparar sobre `dataset_trial.json` |
| `src/semantic/dataset_glosas.json` | Dataset de entrenamiento |
| `src/semantic/dataset_trial.json` | Set de prueba fijo (multi-referencia) |
| `src/semantic/prompts/sys_prompt.txt` | System prompt |
| `src/semantic/requirements_train.txt` | Dependencias pip |
| `docs/entrenamiento_llm.md` | Guía de entorno, train y eval |

Los pesos entrenados quedan en `src/semantic/outputs/` (gitignored).

## Inicio rápido

Ver la guía completa en [`docs/entrenamiento_llm.md`](docs/entrenamiento_llm.md).
