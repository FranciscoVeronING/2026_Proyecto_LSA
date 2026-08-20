"""
Búsqueda de hiperparámetros con Optuna + validación cruzada (K-fold).

Incluye max_frames {8, 12, 16, 24}: cada trial subsamplea los .npy a esa
longitud. Los .npy tienen que tener T >= 24 (p. ej. dataset_landmarks_32frames).

Uso (desde src/, entorno conda lsa_gpu):
    python tune_optuna.py
    python tune_optuna.py --n-trials 2          # prueba corta
    python tune_optuna.py --n-trials 40         # corrida completa (default)

Dashboard:
    pip install optuna-dashboard
    optuna-dashboard sqlite:///optuna_study_v2.db
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold

import config as cfg
from train import build_file_lists, filter_valid_classes, train_one_run


STUDY_NAME = "tinyskeleton_lsa_v2"
STUDY_DB = "sqlite:///optuna_study_v2.db"
N_SPLITS = 3
N_TRIALS_DEFAULT = 40
MAX_FRAMES_CANDIDATES = (8, 12, 16, 24)
BEST_PARAMS_PATH = os.path.join(cfg.MODEL_SAVE_DIR, "optuna_best_v2.json")


def peek_npy_temporal_length() -> int:
    """Longitud T de un .npy de train. Sirve para no pedir más frames de los que hay."""
    clases = filter_valid_classes()
    if not clases:
        raise RuntimeError("No hay clases con suficientes videos para Optuna.")
    _, train_archivos, _, _, _ = build_file_lists(clases)
    if not train_archivos:
        raise RuntimeError(f"No se encontraron .npy en {cfg.DATASET_NPY_DIR}.")
    seq = np.load(train_archivos[0])
    if seq.ndim != 2:
        raise RuntimeError(f"Forma inesperada de {train_archivos[0]}: {seq.shape}")
    return int(seq.shape[0])


def suggest_hyperparams(trial: optuna.Trial, frame_choices: tuple) -> dict:
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])

    if hidden_dim % num_heads != 0:
        raise optuna.TrialPruned("hidden_dim debe ser divisible por num_heads")

    return {
        "max_frames": trial.suggest_categorical("max_frames", list(frame_choices)),
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "use_data_augmentation": True,
        "virtual_multiplier": cfg.VIRTUAL_MULTIPLIER,
        "aug_noise_std": trial.suggest_float("aug_noise_std", 0.005, 0.03),
        "aug_pose_noise_std": cfg.AUG_POSE_NOISE_STD,
        "aug_scale_range": cfg.AUG_SCALE_RANGE,
        "aug_rot_yaw_deg": cfg.AUG_ROT_YAW_DEG,
        "aug_rot_pitch_deg": cfg.AUG_ROT_PITCH_DEG,
        "aug_rot_roll_deg": cfg.AUG_ROT_ROLL_DEG,
        "aug_time_warp_range": cfg.AUG_TIME_WARP_RANGE,
        "aug_temporal_crop_frac": cfg.AUG_TEMPORAL_CROP_FRAC,
        "aug_frame_dropout_max": cfg.AUG_FRAME_DROPOUT_MAX,
    }


def make_objective(frame_choices: tuple):
    def objective(trial: optuna.Trial) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hyperparams = suggest_hyperparams(trial, frame_choices)

        clases_validas = filter_valid_classes()
        _, train_archivos, train_etiquetas, _, _ = build_file_lists(clases_validas)
        num_classes = len(clases_validas)

        indices = np.arange(len(train_archivos))
        kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold_losses = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            fold_train_files = [train_archivos[i] for i in train_idx]
            fold_train_labels = [train_etiquetas[i] for i in train_idx]
            fold_val_files = [train_archivos[i] for i in val_idx]
            fold_val_labels = [train_etiquetas[i] for i in val_idx]

            best_loss, _, _ = train_one_run(
                fold_train_files,
                fold_train_labels,
                fold_val_files,
                fold_val_labels,
                num_classes=num_classes,
                hyperparams=hyperparams,
                device=device,
                max_epochs=cfg.EPOCHS,
                patience=cfg.PATIENCE,
                model_save_path=None,
                trial=trial,
                verbose=False,
            )
            fold_losses.append(best_loss)
            trial.set_user_attr(f"fold_{fold_idx}_loss", best_loss)

        trial.set_user_attr("max_frames", hyperparams["max_frames"])
        return float(np.mean(fold_losses))

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna K-fold para TinySkeleton (rama v2).")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT, help="Trials a correr en esta sesión.")
    parser.add_argument(
        "--study-db",
        default=STUDY_DB,
        help="Storage SQLite (default: sqlite:///optuna_study_v2.db).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npy_t = peek_npy_temporal_length()
    frame_choices = tuple(f for f in MAX_FRAMES_CANDIDATES if f <= npy_t)
    skipped = [f for f in MAX_FRAMES_CANDIDATES if f > npy_t]

    print(f"[*] Dispositivo: {device}")
    print(f"[*] Study: {STUDY_NAME} | DB: {args.study_db}")
    print(f"[*] K-fold: {N_SPLITS} | Trials esta sesión: {args.n_trials}")
    print(f"[*] .npy en {cfg.DATASET_NPY_DIR} con T={npy_t}")
    print(f"[*] max_frames a explorar: {frame_choices}")
    if skipped:
        print(
            f"[!] Se omiten {skipped}: los .npy tienen T={npy_t}. "
            "Para incluir 24 hace falta T>=24 (dataset_landmarks_32frames)."
        )
    if not frame_choices:
        raise RuntimeError("Ningún candidato de max_frames entra en la longitud de los .npy.")
    print(f"[*] Augmentation fija ON, VIRTUAL_MULTIPLIER={cfg.VIRTUAL_MULTIPLIER}")
    print(f"[*] Clases con >= {cfg.SAMPLES_PER_CLASS} videos (misma regla que train.py)\n")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=args.study_db,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
    )

    study.optimize(make_objective(frame_choices), n_trials=args.n_trials, show_progress_bar=True)

    print("\n=== MEJORES HIPERPARÁMETROS ===")
    print(f"Trials en el study: {len(study.trials)}")
    print(f"Val loss promedio (K-fold): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_name": STUDY_NAME,
        "best_value": float(study.best_value),
        "n_trials_in_study": len(study.trials),
        "npy_temporal_length": npy_t,
        "max_frames_candidates": list(frame_choices),
        "best_params": study.best_params,
        "fixed": {
            "use_data_augmentation": True,
            "virtual_multiplier": cfg.VIRTUAL_MULTIPLIER,
            "samples_per_class": cfg.SAMPLES_PER_CLASS,
        },
    }
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH) or ".", exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Best params guardados en {BEST_PARAMS_PATH}")
    print("[*] Copiá max_frames y el resto a config.py y ejecutá train.py para el modelo final.")
    print("[*] El modelo final se entrena con train+val (split 80/20 completo), no un fold.")
    print("[*] Después: camera.py --eval. Optuna minimiza val loss, no el vivo.")


if __name__ == "__main__":
    main()
