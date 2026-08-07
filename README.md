# 2026_Proyecto_LSA

Interpretación de Lengua de Señas Argentina (LSA) en tiempo real: la cámara detecta
señas, las acumula como glosas y una LLM las convierte en español hablado y escrito.

## Cómo correr

```bash
conda activate lsa_gpu
pip install -r requirements.txt
python run.py
```

| Comando | Qué hace |
|---------|----------|
| `python run.py` | Cámara + traducción + voz |
| `python run.py --no-llm` | Solo glosas, sin cargar la LLM (arranque rápido) |
| `python run.py --eval` | Recorre todas las señas y guarda un CSV de aciertos del clasificador |
| `python run.py --eval-semantic` | Evalúa la traducción glosas→español (20 ejemplos hold-out, métricas + CSV) |

Durante la ejecución: `q` sale, `m` cambia el modo de captura, `c` corta el
contexto conversacional y `n` saltea una seña en modo evaluación.

PyTorch con CUDA se instala aparte porque esa build no está en PyPI:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Estructura

```
run.py                punto de entrada único
src/
├── app/              cámara, UI de OpenCV y workers en hilos
├── core/             lógica pura: sin cámara, sin GPU, testeable sola
├── classifier/       TinySkeletonClassifier + pesos entrenados
└── semantic/         Qwen2.5-3B con adaptador LoRA + prompts
docs/                 decisiones técnicas y arquitectura
```

Los pesos del clasificador y el adaptador LoRA están versionados con Git LFS.
Después de clonar, correr `git lfs pull`.

## Documentación

[`docs/rama_integration.md`](docs/rama_integration.md) explica el pipeline completo,
las decisiones de diseño y los problemas abiertos, con vocabulario técnico y no técnico.
