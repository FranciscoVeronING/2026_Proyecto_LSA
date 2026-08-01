# 2026_Proyecto_LSA

Prototipo de traducción LSA ↔ español (Trabajo Final UNMDP).

## Rama activa del clasificador

`scratch-mediapipe` — MediaPipe Holistic + Tiny Transformer, inferencia en `src/camera.py`.

## Entorno Python

**Recomendado:** Python **3.11** + `mediapipe==0.10.21` + `protobuf>=4.25.3,<5`

Guía completa: [docs/entorno.md](docs/entorno.md)

```powershell
conda env create -f environment.yml
conda activate lsa_gpu
cd src
python check_env.py
python preprocessing.py --force
python train.py
python camera.py
```

## Documentación

- [docs/entorno.md](docs/entorno.md) — dependencias y troubleshooting
- [docs/documentacion.md](docs/documentacion.md) — pipeline técnico
- [docs/Entregable_Semana_Clasificador.md](docs/Entregable_Semana_Clasificador.md) — informe de avance
