# Semántico para oyente

Oración en español → secuencia de glosas LSA. Rama de entrenamiento; no incluye cámara ni extensión.

```
Me llamo Maite.  →  YO NOMBRE Maite
```

| Archivo | Qué hace |
|---------|----------|
| `src/semantic/train_all_models.py` | Entrena los 5 modelos, exporta GGUF, mide |
| `src/semantic/dataset_glosas.json` | Pares español / glosas |
| `src/semantic/chat.py` | Probar un GGUF en la terminal |
| `src/semantic/eval_compare_models.py` | Comparativa sobre `dataset_trial.json` |
| `docs/entrenamiento.md` | Entorno conda y Unsloth |

```powershell
conda activate lsa-train
cd src\semantic
python train_all_models.py --continue-on-error
python chat.py --model llama-3.2-1b
```
