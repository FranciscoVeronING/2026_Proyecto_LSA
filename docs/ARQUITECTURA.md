# Arquitectura

Español → LLM glosador → glosas LSA (para mostrar o para un avatar).

| Archivo | Rol |
|---------|-----|
| `gloss_format.py` | Deletreo/DNI compactos; orden y género |
| `dataset.py` | Carga y split |
| `prompts.py` | System prompt y plantilla de chat |
| `train.py` | LoRA + GGUF + eval de un modelo |
| `metrics.py` | llama.cpp y BLEU/ROUGE/METEOR |
| `gguf_export.py` | Q4_K_M (ruta corta en Windows) |
| `train_all_models.py` | Batch |
| `eval_compare_models.py` | Trial |
| `chat.py` | Terminal |

El corpus es el mismo que el del semántico glosas→español, con el par invertido y una sola secuencia de glosas por oración.
