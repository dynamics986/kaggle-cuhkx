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
    skeleton_motion_residual: bool = False
    imu_encoder: str = "tcn"
    imu_structured_cache_dir: str | None = None
    imu_transformer_layers: int = 2
    imu_transformer_heads: int = 4
    imu_relative_position_max_distance: int = 63
    horizontal_flip_probability: float = 0.0
    preserve_aspect_ratio: bool = False
    shared_visual_sampling: bool = False
    visual_crop_mode: str = "none"
    visual_crop_metadata_root: str | None = None
    visual_crop_padding: float = 0.12
    temporal_pooling: str = "mean"
    dropout: float = 0.15
    modality_dropout: float = 0.15
    imu_device_dropout: float = 0.0
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
        if self.imu_encoder not in {"tcn", "device_cnn_rel_transformer"}:
            raise ValueError("imu_encoder must be 'tcn' or 'device_cnn_rel_transformer'")
        if self.imu_encoder == "device_cnn_rel_transformer" and not self.imu_structured_cache_dir:
            raise ValueError(
                "imu_structured_cache_dir is required for imu_encoder='device_cnn_rel_transformer'"
            )
        if self.imu_transformer_layers <= 0 or self.imu_transformer_heads <= 0:
            raise ValueError("IMU transformer layers and heads must be positive")
        if self.d_model % self.imu_transformer_heads:
            raise ValueError("d_model must be divisible by imu_transformer_heads")
        if self.imu_relative_position_max_distance <= 0:
            raise ValueError("imu_relative_position_max_distance must be positive")
        skeleton_modes = sum(
            (self.skeleton_graph, self.skeleton_motion, self.skeleton_motion_residual)
        )
        if skeleton_modes > 1:
            raise ValueError(
                "skeleton_graph, skeleton_motion, and skeleton_motion_residual "
                "are mutually exclusive"
            )
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if self.temporal_pooling not in {"mean", "directional"}:
            raise ValueError("temporal_pooling must be 'mean' or 'directional'")
        if self.visual_crop_mode not in {"none", "yolo_person"}:
            raise ValueError("visual_crop_mode must be 'none' or 'yolo_person'")
        if self.visual_crop_mode == "yolo_person" and not self.visual_crop_metadata_root:
            raise ValueError("visual_crop_metadata_root is required for yolo_person")
        if not 0 <= self.visual_crop_padding < 1:
            raise ValueError("visual_crop_padding must be in [0, 1)")
        if not 0 <= self.modality_dropout < 1:
            raise ValueError("modality_dropout must be in [0, 1)")
        if not 0 <= self.imu_device_dropout < 1:
            raise ValueError("imu_device_dropout must be in [0, 1)")
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
