import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json

import config as cfg

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
        frames_actuales, features = secuencia.shape
        
        if frames_actuales < self.max_frames:
            padding = np.zeros((self.max_frames - frames_actuales, features))
            secuencia = np.vstack((secuencia, padding))
        else:
            secuencia = secuencia[:self.max_frames, :]
            
        return torch.tensor(secuencia, dtype=torch.float32), torch.tensor(self.etiquetas[real_idx], dtype=torch.long)
    

def augment_batch_3d(batch_data: torch.Tensor, noise_std: float, scale_range: tuple) -> torch.Tensor:
    b_size, seq_len, features = batch_data.shape
    device = batch_data.device
    
    x_3d = batch_data.view(b_size, seq_len, -1, 3)
    scales = torch.empty(b_size, 1, 1, 1, device=device).uniform_(*scale_range)
    x_augmented = x_3d * scales
    noise = torch.randn_like(x_augmented, device=device) * noise_std
    x_augmented = x_augmented + noise
    
    return x_augmented.view(b_size, seq_len, features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model) 
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TinySkeletonClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_classes: int, dropout_rate: float):
        super().__init__()
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout1d(p=0.2), 
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_dropout = nn.Dropout(p=dropout_rate)
        self.classification_head = nn.Linear(hidden_dim, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        x = self.conv_extractor(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x_pooled = x.mean(dim=1)
        x_dropped = self.classifier_dropout(x_pooled)
        return self.classification_head(x_dropped)


if __name__ == "__main__":
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")
    
    print("\n--- ANALIZANDO DATASET Y FILTRANDO CLASES ---")
    clases_validas = []
    
    for clase in cfg.SIGN_CLASSES:
        rutas = glob.glob(os.path.join(cfg.DATASET_NPY_DIR, clase, "*.npy"))
        if len(rutas) >= cfg.SAMPLES_PER_CLASS:
            clases_validas.append(clase)
        else:
            print(f"  [!] DESCARTADA: '{clase}'. Tiene {len(rutas)} videos (Se requieren {cfg.SAMPLES_PER_CLASS}).")
            
    NUM_CLASSES_REAL = len(clases_validas)
    class_to_idx = {clase: idx for idx, clase in enumerate(clases_validas)}
    
    ruta_mapeo = os.path.join(cfg.MODEL_SAVE_DIR, "mapeo_clases.json")
    with open(ruta_mapeo, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=4)
    print(f"\n[*] Clases válidas finales: {NUM_CLASSES_REAL}/{len(cfg.SIGN_CLASSES)}. Mapeo guardado en {ruta_mapeo}.")

    train_archivos, train_etiquetas = [], []
    test_archivos, test_etiquetas = [], []
    
    for clase in clases_validas:
        rutas = glob.glob(os.path.join(cfg.DATASET_NPY_DIR, clase, "*.npy"))
        
        np.random.seed(42) 
        np.random.shuffle(rutas)
        rutas = rutas[:cfg.SAMPLES_PER_CLASS] 
        
        split_idx = int(cfg.SAMPLES_PER_CLASS * 0.8) 
        rutas_train = rutas[:split_idx]
        rutas_test = rutas[split_idx:]
        
        train_archivos.extend(rutas_train)
        train_etiquetas.extend([class_to_idx[clase]] * len(rutas_train))
        
        test_archivos.extend(rutas_test)
        test_etiquetas.extend([class_to_idx[clase]] * len(rutas_test))

    multiplicador_real = cfg.VIRTUAL_MULTIPLIER if getattr(cfg, 'USE_DATA_AUGMENTATION', False) else 1
    
    print(f"[*] Videos en Entrenamiento: {len(train_archivos)} (Virtualmente multiplicados x{multiplicador_real})")
    print(f"[*] Videos en Validación: {len(test_archivos)}\n")

    train_dataset = LabeledSkeletonDataset(train_archivos, train_etiquetas, max_frames=cfg.MAX_FRAMES, multiplier=multiplicador_real)
    test_dataset = LabeledSkeletonDataset(test_archivos, test_etiquetas, max_frames=cfg.MAX_FRAMES, multiplier=1)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    model = TinySkeletonClassifier(
        cfg.FRAME_FEATURES_DIM, 
        cfg.HIDDEN_DIM, 
        num_heads=cfg.NUM_HEADS, 
        num_layers=cfg.NUM_LAYERS, 
        num_classes=NUM_CLASSES_REAL,
        dropout_rate=cfg.DROPOUT_RATE
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=cfg.PATIENCE)
    ruta_mejor_modelo = os.path.join(cfg.MODEL_SAVE_DIR, "tinyskeleton_best.pth")
    
    print("--- INICIANDO FINE-TUNING ---")
    train_loss_history, val_loss_history = [], []
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss, correctos_train, muestras_train = 0.0, 0, 0
        bucle_lotes = tqdm(train_loader, desc=f"Época {epoch+1}/{cfg.EPOCHS}", leave=False)
        
        for batch_data, batch_labels in bucle_lotes:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            if getattr(cfg, 'USE_DATA_AUGMENTATION', False):
                batch_data = augment_batch_3d(batch_data, noise_std=cfg.AUG_NOISE_STD, scale_range=cfg.AUG_SCALE_RANGE)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicciones = torch.argmax(logits, dim=1)
            correctos_train += (predicciones == batch_labels).sum().item()
            muestras_train += batch_labels.size(0)
            
        avg_train_loss = train_loss / len(train_loader)
        acc_train = (correctos_train / muestras_train) * 100
        train_loss_history.append(avg_train_loss)
        
        model.eval()
        val_loss, correctos_val, muestras_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                logits = model(batch_data)
                loss = loss_fn(logits, batch_labels)
                
                val_loss += loss.item()
                predicciones = torch.argmax(logits, dim=1)
                correctos_val += (predicciones == batch_labels).sum().item()
                muestras_val += batch_labels.size(0)
                
        avg_val_loss = val_loss / len(test_loader)
        acc_val = (correctos_val / muestras_val) * 100
        val_loss_history.append(avg_val_loss)
        
        print(f"Época {epoch+1} | Train Loss: {avg_train_loss:.4f} (Acc: {acc_train:.2f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {acc_val:.2f}%)")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, ruta_mejor_modelo)
        
        if early_stopping.early_stop:
            print("\n[!] Detención Temprana activada.")
            break
            
    print("\n--- EVALUANDO EL MEJOR MODELO ---")
    model.load_state_dict(torch.load(ruta_mejor_modelo, weights_only=True))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            preds = torch.argmax(model(batch_data), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
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