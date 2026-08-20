from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig
from .constants import NUM_CLASSES, VISUAL_MODALITIES
from .features import (
    IMU_DEVICES,
    IMU_FEATURES,
    IMU_FEATURES_PER_DEVICE,
    RADAR_FEATURES,
    SKELETON_FEATURES,
    SKELETON_JOINTS,
)


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvNormAct(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, stride: int, groups: int = 1
    ) -> None:
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, in_channels, 3, stride, groups=in_channels),
            ConvNormAct(in_channels, out_channels, 1, 1),
        )
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.block(inputs)
        return output + inputs if self.residual else output


class VisualEncoder(nn.Module):
    """Small from-scratch CNN whose capacity scales with width_mult."""

    def __init__(self, width_mult: float, output_dim: int) -> None:
        super().__init__()
        channels = [make_divisible(base * width_mult) for base in (24, 32, 48, 72, 96)]
        layers: list[nn.Module] = [ConvNormAct(3, channels[0], 3, 2)]
        repeats = (1, 2, 2, 2)
        for stage, repeat in enumerate(repeats):
            in_channels, out_channels = channels[stage], channels[stage + 1]
            layers.append(DepthwiseSeparableBlock(in_channels, out_channels, stride=2))
            for _ in range(repeat - 1):
                layers.append(DepthwiseSeparableBlock(out_channels, out_channels, stride=1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(channels[-1], output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images)).flatten(1)
        return self.projection(features)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class TemporalEncoder(nn.Module):
    def __init__(
        self, input_dim: int, d_model: int, dropout: float, pooling: str = "mean"
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.temporal = nn.Sequential(
            TemporalResidualBlock(d_model, dropout, dilation=1),
            TemporalResidualBlock(d_model, dropout, dilation=2),
            TemporalResidualBlock(d_model, dropout, dilation=4),
        )
        self.pool_projection: nn.Module = (
            nn.Sequential(nn.Linear(d_model * 3, d_model), nn.SiLU())
            if pooling == "directional"
            else nn.Identity()
        )
        self.output = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        features = self.input(sequence).transpose(1, 2)
        features = self.temporal(features)
        if self.pooling == "directional":
            pooled = torch.cat(
                [
                    features.mean(dim=-1),
                    features.amax(dim=-1),
                    features[..., -1] - features[..., 0],
                ],
                dim=1,
            )
        else:
            pooled = features.mean(dim=-1)
        return self.output(self.pool_projection(pooled))


class RelativeSelfAttention(nn.Module):
    """Multi-head self-attention with an independently learned relative bias per head."""

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
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, width)
        return self.output(output) * valid.unsqueeze(-1)


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


class IMUDeviceCNNRelativeTransformerEncoder(nn.Module):
    """Shared device CNN, mask-aware device pooling, then relative temporal attention."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        width = config.d_model
        self.local1 = nn.Conv1d(IMU_FEATURES_PER_DEVICE, width, kernel_size=5, padding=2)
        self.local2 = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width)
        self.local_norm = nn.LayerNorm(width)
        self.device_embedding = nn.Parameter(torch.zeros(IMU_DEVICES, width))
        self.device_gate = nn.Linear(width, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(
                    width,
                    config.imu_transformer_heads,
                    config.imu_relative_position_max_distance,
                    config.dropout,
                )
                for _ in range(config.imu_transformer_layers)
            ]
        )
        self.output = nn.LayerNorm(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[2:] != (IMU_DEVICES, IMU_FEATURES_PER_DEVICE + 1):
            raise ValueError(
                "Structured IMU input must have shape "
                f"[batch, steps, {IMU_DEVICES}, {IMU_FEATURES_PER_DEVICE + 1}]"
            )
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
        time_valid = device_valid.any(dim=-1)
        weights = torch.where(time_valid.unsqueeze(-1), weights, torch.zeros_like(weights))
        sequence = (local * weights.unsqueeze(-1)).sum(dim=2)
        sequence = torch.cat([self.cls.expand(batch, -1, -1), sequence], dim=1)
        valid = torch.cat(
            [torch.ones(batch, 1, dtype=torch.bool, device=inputs.device), time_valid], dim=1
        )
        for block in self.blocks:
            sequence = block(sequence, valid)
        return self.output(sequence[:, 0])


class SkeletonMotionEncoder(nn.Module):
    """TCN over pose, joint velocity, and joint acceleration.

    Confidence scores remain in the pose features but are not differentiated: their
    frame-to-frame changes describe detector certainty rather than human motion.
    """

    def __init__(self, d_model: int, dropout: float, pooling: str = "mean") -> None:
        super().__init__()
        pose_dim = SKELETON_JOINTS * SKELETON_FEATURES
        coordinate_dim = SKELETON_JOINTS * (SKELETON_FEATURES - 1)
        self.temporal = TemporalEncoder(
            pose_dim + 2 * coordinate_dim, d_model, dropout, pooling
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = sequence.shape
        joints = sequence.reshape(batch, steps, SKELETON_JOINTS, SKELETON_FEATURES)
        coordinates = joints[..., : SKELETON_FEATURES - 1]
        velocity = torch.diff(coordinates, dim=1, prepend=coordinates[:, :1])
        acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1])
        motion = torch.cat(
            [sequence, velocity.flatten(2), acceleration.flatten(2)], dim=-1
        )
        return self.temporal(motion)


class SkeletonPoseMotionResidualEncoder(nn.Module):
    """Pose encoder with a gated velocity/acceleration residual branch.

    Pose remains the default representation.  The gate begins near 0.12, so
    motion only contributes when optimization finds it useful for a feature
    dimension and sample.  This avoids replacing a strong pose representation
    with the weaker standalone motion encoder.
    """

    def __init__(self, d_model: int, dropout: float, pooling: str = "mean") -> None:
        super().__init__()
        self.pose = TemporalEncoder(
            SKELETON_JOINTS * SKELETON_FEATURES, d_model, dropout, pooling
        )
        self.motion = SkeletonMotionEncoder(d_model, dropout, pooling)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.output = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        pose = self.pose(sequence)
        motion = self.motion(sequence)
        gate = torch.sigmoid(self.gate(torch.cat([pose, motion], dim=-1)))
        return self.output(pose + gate * motion)


def skeleton_adjacency() -> torch.Tensor:
    edges = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )
    adjacency = torch.eye(SKELETON_JOINTS, dtype=torch.float32)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1).pow(-0.5)
    return degree[:, None] * adjacency * degree[None, :]


class SkeletonGraphBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual: nn.Module
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph_mixed = torch.einsum("nctv,vw->nctw", inputs, adjacency)
        output = self.temporal(self.spatial(graph_mixed))
        return self.activation(output + self.residual(inputs))


class SkeletonGraphEncoder(nn.Module):
    """Small ST-GCN over the 17 COCO joints and their temporal trajectories."""

    def __init__(self, d_model: int, width_mult: float, dropout: float) -> None:
        super().__init__()
        channels = [make_divisible(base * width_mult) for base in (64, 96, 128)]
        self.input_norm = nn.BatchNorm1d(SKELETON_JOINTS * SKELETON_FEATURES)
        self.register_buffer("adjacency", skeleton_adjacency())
        self.blocks = nn.ModuleList(
            [
                SkeletonGraphBlock(SKELETON_FEATURES, channels[0], dropout),
                SkeletonGraphBlock(channels[0], channels[1], dropout),
                SkeletonGraphBlock(channels[1], channels[2], dropout),
            ]
        )
        self.output = nn.Sequential(nn.Linear(channels[-1], d_model), nn.LayerNorm(d_model))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = sequence.shape
        features = sequence.reshape(batch, steps, SKELETON_JOINTS, SKELETON_FEATURES)
        features = features.permute(0, 3, 1, 2).contiguous()
        normalized = self.input_norm(features.permute(0, 3, 1, 2).reshape(batch, -1, steps))
        features = normalized.reshape(batch, SKELETON_JOINTS, SKELETON_FEATURES, steps)
        features = features.permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            features = block(features, self.adjacency)
        return self.output(features.mean(dim=(2, 3)))


class MultimodalHAR(nn.Module):
    def __init__(self, config: ExperimentConfig, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.visual_encoder = VisualEncoder(config.width_mult, d_model)
        self.visual_temporal = TemporalEncoder(
            d_model, d_model, config.dropout, config.temporal_pooling
        )
        if config.skeleton_graph:
            self.skeleton_encoder: nn.Module = SkeletonGraphEncoder(
                d_model, config.width_mult, config.dropout
            )
        elif config.skeleton_motion:
            self.skeleton_encoder = SkeletonMotionEncoder(
                d_model, config.dropout, config.temporal_pooling
            )
        elif config.skeleton_motion_residual:
            self.skeleton_encoder = SkeletonPoseMotionResidualEncoder(
                d_model, config.dropout, config.temporal_pooling
            )
        else:
            self.skeleton_encoder = TemporalEncoder(
                SKELETON_JOINTS * SKELETON_FEATURES,
                d_model,
                config.dropout,
                config.temporal_pooling,
            )
        self.imu_encoder: nn.Module = (
            IMUDeviceCNNRelativeTransformerEncoder(config)
            if config.imu_encoder == "device_cnn_rel_transformer"
            else TemporalEncoder(IMU_FEATURES, d_model, config.dropout, config.temporal_pooling)
        )
        self.radar_encoder = TemporalEncoder(
            RADAR_FEATURES, d_model, config.dropout, config.temporal_pooling
        )
        self.modality_embedding = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.fusion_heads,
            dim_feedforward=d_model * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=config.fusion_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, num_classes),
        )

    def _apply_modality_dropout(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.modality_dropout <= 0:
            return mask
        active = mask.clone()
        drops = torch.rand(active.shape, device=active.device) < self.config.modality_dropout
        active &= ~drops
        empty_rows = ~active.any(dim=1)
        if empty_rows.any():
            rows = torch.where(empty_rows)[0]
            for row in rows.tolist():
                present = torch.where(mask[row])[0]
                if len(present):
                    active[row, present[0]] = True
        return active

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        visual = batch["visual"]
        batch_size, modalities, frames, channels, height, width = visual.shape
        if modalities != len(VISUAL_MODALITIES):
            raise ValueError(
                f"Expected {len(VISUAL_MODALITIES)} visual modalities, got {modalities}"
            )
        flat = visual.reshape(batch_size * modalities * frames, channels, height, width)
        encoded = self.visual_encoder(flat).reshape(batch_size, modalities, frames, -1)
        visual_tokens = [
            self.visual_temporal(encoded[:, modality]) for modality in range(modalities)
        ]
        tokens = torch.stack(
            [
                *visual_tokens,
                self.skeleton_encoder(batch["skeleton"]),
                self.imu_encoder(batch["imu"]),
                self.radar_encoder(batch["radar"]),
            ],
            dim=1,
        )
        tokens = tokens + self.modality_embedding
        mask = self._apply_modality_dropout(batch["modality_mask"].bool())
        fused = self.fusion(tokens, src_key_padding_mask=~mask)
        masked_sum = (fused * mask.unsqueeze(-1)).sum(dim=1)
        pooled = masked_sum / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(pooled)


def parameter_size_mb(model: nn.Module) -> float:
    bytes_total = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    bytes_total += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return bytes_total / (1024**2)
