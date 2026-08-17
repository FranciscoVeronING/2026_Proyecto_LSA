"""
Búsqueda de hiperparámetros con Optuna + validación cruzada (K-fold).

Uso (desde src/, con el entorno conda activo):
    pip install optuna
    python tune_optuna.py

Dashboard (opcional):
    pip install optuna-dashboard
    optuna-dashboard sqlite:///optuna_study.db
"""

import os

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold

import config as cfg
from train import build_file_lists, filter_valid_classes, train_one_run


STUDY_DB = "sqlite:///optuna_study.db"
N_SPLITS = 3
N_TRIALS = 40


def suggest_hyperparams(trial: optuna.Trial) -> dict:
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])

    if hidden_dim % num_heads != 0:
        raise optuna.TrialPruned("hidden_dim debe ser divisible por num_heads")

    use_aug = trial.suggest_categorical("use_data_augmentation", [False, True])

    return {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.2, 0.6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "use_data_augmentation": use_aug,
        "virtual_multiplier": cfg.VIRTUAL_MULTIPLIER,
        "aug_noise_std": trial.suggest_float("aug_noise_std", 0.005, 0.03),
        "aug_scale_range": cfg.AUG_SCALE_RANGE,
    }


def objective(trial: optuna.Trial) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = suggest_hyperparams(trial)

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

    return float(np.mean(fold_losses))


def main():
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")
    print(f"[*] K-fold: {N_SPLITS} | Trials: {N_TRIALS}")
    print("[*] Solo se usan clases con >= 50 videos (misma regla que train.py)\n")

    study = optuna.create_study(
        study_name="tinyskeleton_lsa",
        direction="minimize",
        storage=STUDY_DB,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8),
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n=== MEJORES HIPERPARÁMETROS ===")
    print(f"Val loss promedio (K-fold): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n[*] Copiá estos valores a config.py y ejecutá train.py para el modelo final.")
    print("[*] El modelo final debe entrenarse con train+val (split completo), no solo un fold.")


if __name__ == "__main__":
    main()
