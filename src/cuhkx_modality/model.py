from __future__ import annotations

import torch
from torch import nn

from cuhkx_har.features import IMU_FEATURES, RADAR_FEATURES, SKELETON_FEATURES, SKELETON_JOINTS
from cuhkx_har.model import TemporalEncoder, VisualEncoder

from .config import ModalityConfig
from .constants import MODALITIES, NUM_CLASSES, VISUAL_MODALITIES


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
            self.encoder: nn.Module = VisualEncoder(config.width_mult, config.d_model)
            self.temporal = TemporalEncoder(
                config.d_model, config.d_model, config.dropout, config.temporal_pooling
            )
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
