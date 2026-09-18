# Semántico para oyente

Oración en español → secuencia de glosas LSA.

```
Me llamo Maite.  →  YO NOMBRE Maite
```

| Archivo | Qué hace |
|---------|----------|
| `src/semantic/train_all_models.py` | Entrena, exporta GGUF, mide el split |
| `src/semantic/eval_compare_models.py` | Comparativa sobre `dataset_trial.json` |
| `src/semantic/dataset_glosas.json` | Pares de entrenamiento |
| `docs/entrenamiento.md` | Entorno conda y Unsloth |

```powershell
conda activate lsa-train
cd src\semantic
python train_all_models.py --continue-on-error
python eval_compare_models.py
```
