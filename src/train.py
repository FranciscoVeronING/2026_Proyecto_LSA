import os
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config as cfg
from model_arch import TinySkeletonClassifier
from utils import normalize_sequence_to_frames


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f"   [*] Nuevo mejor modelo guardado (Loss: {val_loss:.4f})")
        else:
            self.counter += 1
            print(f"   [!] Sin mejoras. Contador EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class LabeledSkeletonDataset(Dataset):
    def __init__(self, archivos: list, etiquetas: list, max_frames: int, multiplier: int = 1):
        self.archivos = archivos
        self.etiquetas = etiquetas
        self.max_frames = max_frames
        self.multiplier = multiplier
        self.real_length = len(self.archivos)

    def __len__(self) -> int:
        return self.real_length * self.multiplier

    def __getitem__(self, idx: int):
        real_idx = idx % self.real_length
        secuencia = np.load(self.archivos[real_idx])
        secuencia = normalize_sequence_to_frames(secuencia, self.max_frames)
        return (
            torch.tensor(secuencia, dtype=torch.float32),
            torch.tensor(self.etiquetas[real_idx], dtype=torch.long),
        )


def augment_batch_3d(batch_data: torch.Tensor, noise_std: float, scale_range: tuple) -> torch.Tensor:
    b_size, seq_len, features = batch_data.shape
    device = batch_data.device

    x_3d = batch_data.view(b_size, seq_len, -1, 3)
    scales = torch.empty(b_size, 1, 1, 1, device=device).uniform_(*scale_range)
    x_augmented = x_3d * scales
    noise = torch.randn_like(x_augmented, device=device) * noise_std
    x_augmented = x_augmented + noise

    return x_augmented.view(b_size, seq_len, features)


def filter_valid_classes():
    """Descarta clases con menos de SAMPLES_PER_CLASS videos."""
    clases_validas = []
    for clase in cfg.SIGN_CLASSES:
        rutas = glob.glob(os.path.join(cfg.DATASET_NPY_DIR, clase, "*.npy"))
        if len(rutas) >= cfg.SAMPLES_PER_CLASS:
            clases_validas.append(clase)
        else:
            print(
                f"  [!] DESCARTADA: '{clase}'. Tiene {len(rutas)} videos "
                f"(Se requieren {cfg.SAMPLES_PER_CLASS})."
            )
    return clases_validas


def build_file_lists(clases_validas):
    """Split 80/20 por clase, máximo SAMPLES_PER_CLASS por clase."""
    class_to_idx = {clase: idx for idx, clase in enumerate(clases_validas)}
    train_archivos, train_etiquetas = [], []
    test_archivos, test_etiquetas = [], []

    for clase in clases_validas:
        rutas = glob.glob(os.path.join(cfg.DATASET_NPY_DIR, clase, "*.npy"))
        np.random.seed(42)
        np.random.shuffle(rutas)
        rutas = rutas[: cfg.SAMPLES_PER_CLASS]

        split_idx = int(cfg.SAMPLES_PER_CLASS * 0.8)
        rutas_train = rutas[:split_idx]
        rutas_test = rutas[split_idx:]

        train_archivos.extend(rutas_train)
        train_etiquetas.extend([class_to_idx[clase]] * len(rutas_train))
        test_archivos.extend(rutas_test)
        test_etiquetas.extend([class_to_idx[clase]] * len(rutas_test))

    return class_to_idx, train_archivos, train_etiquetas, test_archivos, test_etiquetas


def train_one_run(
    train_archivos,
    train_etiquetas,
    val_archivos,
    val_etiquetas,
    num_classes,
    hyperparams,
    device,
    max_epochs=None,
    patience=None,
    model_save_path=None,
    trial=None,
    verbose=True,
):
    """
    Entrena un modelo y devuelve (best_val_loss, train_loss_history, val_loss_history).
    Usado por train.py y tune_optuna.py.
    """
    max_epochs = max_epochs or cfg.EPOCHS
    patience = patience or cfg.PATIENCE

    multiplier = hyperparams.get("virtual_multiplier", cfg.VIRTUAL_MULTIPLIER)
    if not hyperparams.get("use_data_augmentation", cfg.USE_DATA_AUGMENTATION):
        multiplier = 1

    train_dataset = LabeledSkeletonDataset(
        train_archivos, train_etiquetas, max_frames=cfg.MAX_FRAMES, multiplier=multiplier
    )
    val_dataset = LabeledSkeletonDataset(
        val_archivos, val_etiquetas, max_frames=cfg.MAX_FRAMES, multiplier=1
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    model = TinySkeletonClassifier(
        cfg.FRAME_FEATURES_DIM,
        hyperparams["hidden_dim"],
        num_heads=hyperparams["num_heads"],
        num_layers=hyperparams["num_layers"],
        num_classes=num_classes,
        dropout_rate=hyperparams["dropout_rate"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyperparams["label_smoothing"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=patience)
    save_path = model_save_path or os.path.join(cfg.MODEL_SAVE_DIR, "_optuna_tmp.pth")

    train_loss_history, val_loss_history = [], []

    for epoch in range(max_epochs):
        model.train()
        train_loss, correctos_train, muestras_train = 0.0, 0, 0
        bucle_lotes = tqdm(
            train_loader, desc=f"Época {epoch+1}/{max_epochs}", leave=False, disable=not verbose
        )

        for batch_data, batch_labels in bucle_lotes:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            if hyperparams.get("use_data_augmentation", cfg.USE_DATA_AUGMENTATION):
                batch_data = augment_batch_3d(
                    batch_data,
                    noise_std=hyperparams.get("aug_noise_std", cfg.AUG_NOISE_STD),
                    scale_range=hyperparams.get("aug_scale_range", cfg.AUG_SCALE_RANGE),
                )

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicciones = torch.argmax(logits, dim=1)
            correctos_train += (predicciones == batch_labels).sum().item()
            muestras_train += batch_labels.size(0)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        acc_train = (correctos_train / max(muestras_train, 1)) * 100
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss, correctos_val, muestras_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                logits = model(batch_data)
                loss = loss_fn(logits, batch_labels)
                val_loss += loss.item()
                predicciones = torch.argmax(logits, dim=1)
                correctos_val += (predicciones == batch_labels).sum().item()
                muestras_val += batch_labels.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        acc_val = (correctos_val / max(muestras_val, 1)) * 100
        val_loss_history.append(avg_val_loss)

        if verbose:
            print(
                f"Época {epoch+1} | Train Loss: {avg_train_loss:.4f} (Acc: {acc_train:.2f}%) "
                f"| Val Loss: {avg_val_loss:.4f} (Acc: {acc_val:.2f}%)"
            )

        scheduler.step(avg_val_loss)

        if trial is not None:
            import optuna

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        early_stopping(avg_val_loss, model, save_path)
        if early_stopping.early_stop:
            if verbose:
                print("[!] Detención temprana activada.")
            break

    if model_save_path and os.path.exists(save_path):
        os.replace(save_path, model_save_path)

    return early_stopping.best_loss, train_loss_history, val_loss_history


if __name__ == "__main__":
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")
    print(f"[*] MAX_FRAMES (train/inferencia): {cfg.MAX_FRAMES}")

    print("\n--- ANALIZANDO DATASET Y FILTRANDO CLASES ---")
    clases_validas = filter_valid_classes()
    num_classes_real = len(clases_validas)
    class_to_idx, train_archivos, train_etiquetas, test_archivos, test_etiquetas = build_file_lists(
        clases_validas
    )

    ruta_mapeo = os.path.join(cfg.MODEL_SAVE_DIR, "mapeo_clases.json")
    with open(ruta_mapeo, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=4)
    print(
        f"\n[*] Clases válidas finales: {num_classes_real}/{len(cfg.SIGN_CLASSES)}. "
        f"Mapeo guardado en {ruta_mapeo}."
    )

    multiplicador_real = cfg.VIRTUAL_MULTIPLIER if cfg.USE_DATA_AUGMENTATION else 1
    print(f"[*] Videos en Entrenamiento: {len(train_archivos)} (Virtualmente x{multiplicador_real})")
    print(f"[*] Videos en Validación: {len(test_archivos)}\n")

    hyperparams = {
        "hidden_dim": cfg.HIDDEN_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_layers": cfg.NUM_LAYERS,
        "dropout_rate": cfg.DROPOUT_RATE,
        "batch_size": cfg.BATCH_SIZE,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.1,
        "use_data_augmentation": cfg.USE_DATA_AUGMENTATION,
        "virtual_multiplier": cfg.VIRTUAL_MULTIPLIER,
        "aug_noise_std": cfg.AUG_NOISE_STD,
        "aug_scale_range": cfg.AUG_SCALE_RANGE,
    }

    ruta_mejor_modelo = os.path.join(cfg.MODEL_SAVE_DIR, "tinyskeleton_best.pth")
    print("--- INICIANDO ENTRENAMIENTO ---")

    best_loss, train_loss_history, val_loss_history = train_one_run(
        train_archivos,
        train_etiquetas,
        test_archivos,
        test_etiquetas,
        num_classes=num_classes_real,
        hyperparams=hyperparams,
        device=device,
        max_epochs=cfg.EPOCHS,
        patience=cfg.PATIENCE,
        model_save_path=ruta_mejor_modelo,
        verbose=True,
    )
    print(f"\n[*] Mejor val_loss: {best_loss:.4f}")

    print("\n--- EVALUANDO EL MEJOR MODELO ---")
    model = TinySkeletonClassifier(
        cfg.FRAME_FEATURES_DIM,
        cfg.HIDDEN_DIM,
        num_heads=cfg.NUM_HEADS,
        num_layers=cfg.NUM_LAYERS,
        num_classes=num_classes_real,
        dropout_rate=cfg.DROPOUT_RATE,
    ).to(device)
    model.load_state_dict(torch.load(ruta_mejor_modelo, weights_only=True))
    model.eval()

    test_dataset = LabeledSkeletonDataset(test_archivos, test_etiquetas, max_frames=cfg.MAX_FRAMES)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            preds = torch.argmax(model(batch_data), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Curva de Aprendizaje - TinyTransformer")
    plt.legend()
    plt.grid(True)
    plt.savefig("curva_tinyskeleton.png")
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_validas)
    fig, ax = plt.subplots(figsize=(15, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    plt.tight_layout()
    plt.savefig("matriz_confusion_tinyskeleton.png")
    plt.close()
    print("[*] Fin del proceso.")
