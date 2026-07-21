from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260719
    image_size: int = 128
    visual_frames: int = 8
    sensor_steps: int = 64
    width_mult: float = 1.0
    d_model: int = 192
    fusion_layers: int = 2
    fusion_heads: int = 4
    skeleton_graph: bool = False
    skeleton_motion: bool = False
    horizontal_flip_probability: float = 0.0
    preserve_aspect_ratio: bool = False
    shared_visual_sampling: bool = False
    temporal_pooling: str = "mean"
    dropout: float = 0.15
    modality_dropout: float = 0.15
    batch_size: int = 2
    grad_accum_steps: int = 8
    num_workers: int = 2
    epochs: int = 40
    learning_rate: float = 4e-4
    weight_decay: float = 0.03
    label_smoothing: float = 0.08
    class_balance_power: float = 0.0
    amp: bool = True
    early_stopping_patience: int = 8

    def __post_init__(self) -> None:
        if self.d_model % self.fusion_heads:
            raise ValueError("d_model must be divisible by fusion_heads")
        if self.skeleton_graph and self.skeleton_motion:
            raise ValueError("skeleton_graph and skeleton_motion are mutually exclusive")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if self.temporal_pooling not in {"mean", "directional"}:
            raise ValueError("temporal_pooling must be 'mean' or 'directional'")
        if not 0 <= self.modality_dropout < 1:
            raise ValueError("modality_dropout must be in [0, 1)")
        if not 0 <= self.class_balance_power <= 1:
            raise ValueError("class_balance_power must be in [0, 1]")
        for name in ("image_size", "visual_frames", "sensor_steps", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
