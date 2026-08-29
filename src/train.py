import os
import glob
import json
import math
from datetime import datetime, timezone

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
from utils import interpolate_zero_frames, normalize_sequence_to_frames


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


def _rng():
    return np.random.RandomState(int(torch.randint(0, 2**31 - 1, (1,)).item()))


def resample_to_len(seq: np.ndarray, target: int) -> np.ndarray:
    """Interpolación lineal en el eje temporal hasta `target` frames."""
    frames_actuales, features = seq.shape
    if frames_actuales == target:
        return seq.astype(np.float32)
    if frames_actuales == 1:
        return np.repeat(seq.astype(np.float32), target, axis=0)
    src = np.linspace(0, frames_actuales - 1, target, dtype=np.float32)
    t0 = np.floor(src).astype(np.int64)
    t1 = np.minimum(t0 + 1, frames_actuales - 1)
    w = (src - t0).astype(np.float32)[:, None]
    return ((1.0 - w) * seq[t0] + w * seq[t1]).astype(np.float32)


def time_warp_sequence(seq: np.ndarray, speed_range: tuple, rng: np.random.RandomState) -> np.ndarray:
    """speed > 1 = seña más rápida (menos frames + pad); < 1 = más lenta (crop)."""
    lo, hi = speed_range
    speed = float(rng.uniform(lo, hi))
    frames_actuales = seq.shape[0]
    new_t = max(2, int(round(frames_actuales / speed)))
    warped = resample_to_len(seq, new_t)
    if new_t == frames_actuales:
        return warped
    if new_t > frames_actuales:
        start = int(rng.randint(0, new_t - frames_actuales + 1))
        return warped[start : start + frames_actuales]
    pad = frames_actuales - new_t
    left = int(rng.randint(0, pad + 1))
    out = np.empty((frames_actuales, seq.shape[1]), dtype=np.float32)
    out[:left] = warped[0]
    out[left : left + new_t] = warped
    out[left + new_t :] = warped[-1]
    return out


def temporal_crop_sequence(seq: np.ndarray, max_frac: float, rng: np.random.RandomState) -> np.ndarray:
    frames_actuales = seq.shape[0]
    max_drop = int(frames_actuales * max_frac)
    if max_drop < 1:
        return seq
    drop_start = int(rng.randint(0, max_drop + 1))
    drop_end = int(rng.randint(0, max_drop + 1))
    keep = frames_actuales - drop_start - drop_end
    if keep < 2:
        return seq
    end = frames_actuales - drop_end
    return seq[drop_start:end]


def frame_dropout_sequence(seq: np.ndarray, max_drop: int, rng: np.random.RandomState) -> np.ndarray:
    frames_actuales = seq.shape[0]
    if max_drop < 1 or frames_actuales < 4:
        return seq
    n_drop = int(rng.randint(0, max_drop + 1))
    if n_drop == 0:
        return seq
    n_drop = min(n_drop, frames_actuales - 2)
    idx = rng.choice(frames_actuales, size=n_drop, replace=False)
    out = seq.copy()
    out[idx] = 0.0
    filled = interpolate_zero_frames([out[i] for i in range(frames_actuales)])
    return np.stack(filled, axis=0).astype(np.float32)


def apply_temporal_augmentation(seq: np.ndarray, max_frames: int, params: dict) -> np.ndarray:
    rng = _rng()
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] < 2:
        return normalize_sequence_to_frames(seq, max_frames)

    warp_range = params.get("aug_time_warp_range", cfg.AUG_TIME_WARP_RANGE)
    crop_frac = params.get("aug_temporal_crop_frac", cfg.AUG_TEMPORAL_CROP_FRAC)
    drop_max = params.get("aug_frame_dropout_max", cfg.AUG_FRAME_DROPOUT_MAX)

    seq = time_warp_sequence(seq, warp_range, rng)
    seq = temporal_crop_sequence(seq, crop_frac, rng)
    seq = resample_to_len(seq, max_frames)
    seq = frame_dropout_sequence(seq, drop_max, rng)
    return seq


class LabeledSkeletonDataset(Dataset):
    def __init__(
        self,
        archivos: list,
        etiquetas: list,
        max_frames: int,
        multiplier: int = 1,
        augment: bool = False,
        aug_params: dict = None,
    ):
        self.archivos = archivos
        self.etiquetas = etiquetas
        self.max_frames = max_frames
        self.multiplier = multiplier
        self.augment = augment
        self.aug_params = aug_params or {}
        self.real_length = len(self.archivos)

    def __len__(self) -> int:
        return self.real_length * self.multiplier

    def __getitem__(self, idx: int):
        real_idx = idx % self.real_length
        secuencia = np.load(self.archivos[real_idx])
        if self.augment:
            secuencia = apply_temporal_augmentation(secuencia, self.max_frames, self.aug_params)
        else:
            secuencia = normalize_sequence_to_frames(secuencia, self.max_frames)
        return (
            torch.tensor(secuencia, dtype=torch.float32),
            torch.tensor(self.etiquetas[real_idx], dtype=torch.long),
        )


def _euler_rotation_matrices(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch). Ángulos (B,) en radianes → (B, 3, 3)."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    rx = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cp, -sp), dim=-1),
            torch.stack((zeros, sp, cp), dim=-1),
        ),
        dim=-2,
    )
    ry = torch.stack(
        (
            torch.stack((cy, zeros, sy), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((-sy, zeros, cy), dim=-1),
        ),
        dim=-2,
    )
    rz = torch.stack(
        (
            torch.stack((cr, -sr, zeros), dim=-1),
            torch.stack((sr, cr, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )
    return rz @ ry @ rx


def augment_batch_3d(batch_data: torch.Tensor, hyperparams: dict) -> torch.Tensor:
    b_size, seq_len, features = batch_data.shape
    device = batch_data.device
    x_3d = batch_data.view(b_size, seq_len, -1, 3)
    n_joints = x_3d.size(2)

    yaw_deg = hyperparams.get("aug_rot_yaw_deg", cfg.AUG_ROT_YAW_DEG)
    pitch_deg = hyperparams.get("aug_rot_pitch_deg", cfg.AUG_ROT_PITCH_DEG)
    roll_deg = hyperparams.get("aug_rot_roll_deg", cfg.AUG_ROT_ROLL_DEG)
    if yaw_deg or pitch_deg or roll_deg:
        deg_to_rad = math.pi / 180.0
        yaw = torch.empty(b_size, device=device).uniform_(-yaw_deg, yaw_deg) * deg_to_rad
        pitch = torch.empty(b_size, device=device).uniform_(-pitch_deg, pitch_deg) * deg_to_rad
        roll = torch.empty(b_size, device=device).uniform_(-roll_deg, roll_deg) * deg_to_rad
        rotation = _euler_rotation_matrices(yaw, pitch, roll)
        x_3d = torch.matmul(x_3d, rotation.transpose(-1, -2).unsqueeze(1))

    scale_range = hyperparams.get("aug_scale_range", cfg.AUG_SCALE_RANGE)
    scales = torch.empty(b_size, 1, 1, 1, device=device).uniform_(*scale_range)
    x_augmented = x_3d * scales

    pose_std = hyperparams.get("aug_pose_noise_std", cfg.AUG_POSE_NOISE_STD)
    hand_std = hyperparams.get(
        "aug_hand_noise_std",
        hyperparams.get("aug_noise_std", cfg.AUG_HAND_NOISE_STD),
    )
    n_pose = cfg.POSE_DIM // 3
    n_pose = min(n_pose, n_joints)
    stds = x_augmented.new_empty(1, 1, n_joints, 1)
    stds[..., :n_pose, :] = pose_std
    stds[..., n_pose:, :] = hand_std
    x_augmented = x_augmented + torch.randn_like(x_augmented) * stds

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
    max_frames = int(hyperparams.get("max_frames", cfg.MAX_FRAMES))

    multiplier = hyperparams.get("virtual_multiplier", cfg.VIRTUAL_MULTIPLIER)
    if not hyperparams.get("use_data_augmentation", cfg.USE_DATA_AUGMENTATION):
        multiplier = 1

    use_aug = hyperparams.get("use_data_augmentation", cfg.USE_DATA_AUGMENTATION)
    train_dataset = LabeledSkeletonDataset(
        train_archivos,
        train_etiquetas,
        max_frames=max_frames,
        multiplier=multiplier,
        augment=use_aug,
        aug_params=hyperparams,
    )
    val_dataset = LabeledSkeletonDataset(
        val_archivos, val_etiquetas, max_frames=max_frames, multiplier=1, augment=False
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
                batch_data = augment_batch_3d(batch_data, hyperparams)

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
    print(f"[*] Videos en Validación: {len(test_archivos)}")
    if cfg.USE_DATA_AUGMENTATION:
        print(
            "[*] Augmentation: rotación 3D "
            f"(yaw±{cfg.AUG_ROT_YAW_DEG:.0f}° pitch±{cfg.AUG_ROT_PITCH_DEG:.0f}° "
            f"roll±{cfg.AUG_ROT_ROLL_DEG:.0f}°), crop temporal {cfg.AUG_TEMPORAL_CROP_FRAC:.0%}, "
            f"time warp {cfg.AUG_TIME_WARP_RANGE}, dropout ≤{cfg.AUG_FRAME_DROPOUT_MAX} frames, "
            f"ruido pose {cfg.AUG_POSE_NOISE_STD} / manos {cfg.AUG_HAND_NOISE_STD}"
        )
    print()

    hyperparams = {
        "hidden_dim": cfg.HIDDEN_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_layers": cfg.NUM_LAYERS,
        "dropout_rate": cfg.DROPOUT_RATE,
        "batch_size": cfg.BATCH_SIZE,
        "lr": cfg.LR,
        "weight_decay": cfg.WEIGHT_DECAY,
        "label_smoothing": cfg.LABEL_SMOOTHING,
        "max_frames": cfg.MAX_FRAMES,
        "use_data_augmentation": cfg.USE_DATA_AUGMENTATION,
        "virtual_multiplier": cfg.VIRTUAL_MULTIPLIER,
        "aug_noise_std": cfg.AUG_NOISE_STD,
        "aug_hand_noise_std": cfg.AUG_HAND_NOISE_STD,
        "aug_pose_noise_std": cfg.AUG_POSE_NOISE_STD,
        "aug_scale_range": cfg.AUG_SCALE_RANGE,
        "aug_rot_yaw_deg": cfg.AUG_ROT_YAW_DEG,
        "aug_rot_pitch_deg": cfg.AUG_ROT_PITCH_DEG,
        "aug_rot_roll_deg": cfg.AUG_ROT_ROLL_DEG,
        "aug_time_warp_range": cfg.AUG_TIME_WARP_RANGE,
        "aug_temporal_crop_frac": cfg.AUG_TEMPORAL_CROP_FRAC,
        "aug_frame_dropout_max": cfg.AUG_FRAME_DROPOUT_MAX,
    }

    ruta_mejor_modelo = os.path.join(cfg.MODEL_SAVE_DIR, "tinyskeleton_best_optuna_v2.pth")
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
    plt.savefig("curva_tinyskeleton_optuna_v2.png")
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)) * 100)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": float(best_loss),
        "val_accuracy_top1_pct": round(val_acc, 2),
        "num_classes": num_classes_real,
        "num_train_samples": len(train_archivos),
        "num_val_samples": len(test_archivos),
        "max_frames": cfg.MAX_FRAMES,
        "hidden_dim": cfg.HIDDEN_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_layers": cfg.NUM_LAYERS,
        "dropout_rate": cfg.DROPOUT_RATE,
        "use_data_augmentation": cfg.USE_DATA_AUGMENTATION,
        "virtual_multiplier": cfg.VIRTUAL_MULTIPLIER,
        "aug_scale_range": list(cfg.AUG_SCALE_RANGE),
        "aug_pose_noise_std": cfg.AUG_POSE_NOISE_STD,
        "aug_hand_noise_std": cfg.AUG_HAND_NOISE_STD,
        "aug_rot_yaw_deg": cfg.AUG_ROT_YAW_DEG,
        "aug_rot_pitch_deg": cfg.AUG_ROT_PITCH_DEG,
        "aug_rot_roll_deg": cfg.AUG_ROT_ROLL_DEG,
        "aug_time_warp_range": list(cfg.AUG_TIME_WARP_RANGE),
        "aug_temporal_crop_frac": cfg.AUG_TEMPORAL_CROP_FRAC,
        "aug_frame_dropout_max": cfg.AUG_FRAME_DROPOUT_MAX,
        "samples_per_class": cfg.SAMPLES_PER_CLASS,
        "epochs_ran": len(train_loss_history),
        "classes": clases_validas,
    }
    metrics_path = os.path.join(cfg.MODEL_SAVE_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[*] Métricas guardadas en {metrics_path} (val_acc top-1: {val_acc:.2f}%)")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases_validas)
    fig, ax = plt.subplots(figsize=(15, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    plt.tight_layout()
    plt.savefig("matriz_confusion_tinyskeleton_optuna_v2.png")
    plt.close()
    print("[*] Fin del proceso.")
