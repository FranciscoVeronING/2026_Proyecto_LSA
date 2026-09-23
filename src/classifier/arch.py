"""
TinySkeleton: Conv1D + Transformer encoder + attention pooling.

Entrada de inferencia: tensor (N, T, F) con T=MAX_FRAMES (16) y F=225.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Suma encoding sinusoidal. ``x``: (batch, seq, d_model)."""
        return x + self.pe[:, : x.size(1), :]


class TinySkeletonClassifier(nn.Module):
    """
    Conv1D + Transformer para secuencias de landmarks.
    Usa attention pooling en lugar de mean pooling para enfatizar
    los frames más informativos (útil en señas cortas/estáticas).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout1d(p=0.2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.classifier_dropout = nn.Dropout(p=dropout_rate)
        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(batch, frames, features)`` — en vivo ``(1, 16, 225)``.

        Returns:
            Logits ``(batch, num_classes)``.
        """
        x = x.permute(0, 2, 1)
        x = self.conv_extractor(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        attn_scores = self.attention_pool(x).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x_pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)

        x_dropped = self.classifier_dropout(x_pooled)
        return self.classification_head(x_dropped)


def load_classifier_bundle(device: str = "cpu"):
    """Carga mapeo_clases.json + tinyskeleton_best.pth. num_classes = tamaño del mapeo."""
    import json
    from pathlib import Path

    from classifier import config as cfg

    class_to_idx = json.loads(Path(cfg.CLASSES_PATH).read_text(encoding="utf-8"))
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    n_labels = len(idx_to_class)
    if n_labels == 0:
        raise RuntimeError(f"mapeo_clases.json vacío: {cfg.CLASSES_PATH}")

    ckpt = torch.load(cfg.WEIGHTS_PATH, map_location=device, weights_only=True)
    if "classification_head.weight" not in ckpt:
        raise RuntimeError(f"Checkpoint sin classification_head: {cfg.WEIGHTS_PATH}")
    n_ckpt = int(ckpt["classification_head.weight"].shape[0])
    if n_ckpt != n_labels:
        raise RuntimeError(
            f"El mapeo tiene {n_labels} señas pero {cfg.WEIGHTS_PATH} tiene {n_ckpt} salidas. "
            "Copiá el tinyskeleton_best.pth entrenado con esas clases (mismo mapeo_clases.json)."
        )

    model = TinySkeletonClassifier(
        cfg.FRAME_FEATURES_DIM,
        cfg.HIDDEN_DIM,
        num_heads=cfg.NUM_HEADS,
        num_layers=cfg.NUM_LAYERS,
        num_classes=n_labels,
        dropout_rate=cfg.DROPOUT_RATE,
    )
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, idx_to_class
