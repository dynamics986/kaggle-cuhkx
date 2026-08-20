from __future__ import annotations

import torch
from torch import nn

from cuhkx_har.features import (
    IMU_DEVICES,
    IMU_FEATURES,
    IMU_FEATURES_PER_DEVICE,
    RADAR_FEATURES,
    SKELETON_FEATURES,
    SKELETON_JOINTS,
)
from cuhkx_har.model import TemporalEncoder, VisualEncoder

from .config import ModalityConfig
from .constants import MODALITIES, NUM_CLASSES, VISUAL_MODALITIES


class RelativeSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, max_distance: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.max_distance = max_distance
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.output = nn.Linear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_distance + 1, heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dim**-0.5)
        positions = torch.arange(length, device=inputs.device)
        offsets = positions[None, :] - positions[:, None]
        offsets = offsets.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.relative_bias[offsets].permute(2, 0, 1)
        scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        encoded = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, width)
        encoded = self.output(encoded)
        return encoded * valid.unsqueeze(-1)


class RelativeTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, max_distance: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RelativeSelfAttention(d_model, heads, max_distance, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm1(inputs), valid)
        inputs = inputs + self.mlp(self.norm2(inputs))
        return inputs * valid.unsqueeze(-1)


class RelativeTemporalTransformer(nn.Module):
    def __init__(
        self, d_model: int, layers: int, heads: int, max_distance: int, dropout: float
    ) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(d_model, heads, max_distance, dropout)
                for _ in range(layers)
            ]
        )
        self.output = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch = len(sequence)
        sequence = torch.cat([self.cls.expand(batch, -1, -1), sequence], dim=1)
        valid = torch.cat(
            [torch.ones(batch, 1, dtype=torch.bool, device=valid.device), valid], dim=1
        )
        for block in self.blocks:
            sequence = block(sequence, valid)
        return self.output(sequence[:, 0])


class IMUDeviceCNNRelativeTransformerEncoder(nn.Module):
    """Shared per-device local CNN plus masked device and temporal attention."""

    def __init__(self, config: ModalityConfig) -> None:
        super().__init__()
        width = config.d_model
        self.local1 = nn.Conv1d(IMU_FEATURES_PER_DEVICE, width, kernel_size=5, padding=2)
        self.local2 = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width)
        self.local_norm = nn.LayerNorm(width)
        self.device_embedding = nn.Parameter(torch.zeros(IMU_DEVICES, width))
        self.device_gate = nn.Linear(width, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.temporal = RelativeTemporalTransformer(
            width,
            config.transformer_layers,
            config.transformer_heads,
            config.relative_position_max_distance,
            config.dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, device_valid = inputs[..., :IMU_FEATURES_PER_DEVICE], inputs[..., -1].bool()
        batch, steps, devices, features = values.shape
        values = values * device_valid.unsqueeze(-1)
        local = values.permute(0, 2, 3, 1).reshape(batch * devices, features, steps)
        local = torch.nn.functional.silu(self.local1(local))
        local = torch.nn.functional.silu(self.local2(local))
        local = local.transpose(1, 2).reshape(batch, devices, steps, -1).transpose(1, 2)
        local = self.local_norm(local) + self.device_embedding[None, None]
        local = self.dropout(local) * device_valid.unsqueeze(-1)
        logits = self.device_gate(local).squeeze(-1)
        logits = logits.masked_fill(~device_valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        any_device = device_valid.any(dim=-1)
        weights = torch.where(any_device.unsqueeze(-1), weights, torch.zeros_like(weights))
        sequence = (local * weights.unsqueeze(-1)).sum(dim=2)
        return self.temporal(sequence, any_device)


class ModalityHAR(nn.Module):
    """A classifier with exactly one modality encoder and no fusion path."""

    def __init__(
        self, modality: str, config: ModalityConfig, num_classes: int = NUM_CLASSES
    ) -> None:
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality {modality!r}; choose from {MODALITIES}")
        self.modality = modality
        if modality in VISUAL_MODALITIES:
            if config.sensor_encoder != "tcn":
                raise ValueError("Visual modalities require sensor_encoder='tcn'")
            self.encoder: nn.Module = VisualEncoder(config.width_mult, config.d_model)
            self.temporal = TemporalEncoder(
                config.d_model, config.d_model, config.dropout, config.temporal_pooling
            )
        else:
            if config.sensor_encoder == "imu_device_cnn_rel_transformer":
                if modality != "IMU":
                    raise ValueError(
                        "imu_device_cnn_rel_transformer is only valid for the IMU modality"
                    )
                self.encoder = IMUDeviceCNNRelativeTransformerEncoder(config)
            else:
                input_dim = {
                    "Skeleton": SKELETON_JOINTS * SKELETON_FEATURES,
                    "IMU": IMU_FEATURES,
                    "Radar": RADAR_FEATURES,
                }[modality]
                self.encoder = TemporalEncoder(
                    input_dim, config.d_model, config.dropout, config.temporal_pooling
                )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.modality in VISUAL_MODALITIES:
            batch, frames, channels, height, width = inputs.shape
            features = self.encoder(inputs.reshape(batch * frames, channels, height, width))
            encoded = features.reshape(batch, frames, -1)
            encoded = self.temporal(encoded)
        else:
            encoded = self.encoder(inputs)
        return self.head(encoded)


def parameter_size_mb(model: nn.Module) -> float:
    byte_count = sum(item.numel() * item.element_size() for item in model.parameters())
    byte_count += sum(item.numel() * item.element_size() for item in model.buffers())
    return byte_count / (1024**2)
