from __future__ import annotations

import torch

from cuhkx_har.config import ExperimentConfig
from cuhkx_har.features import IMU_FEATURES, RADAR_FEATURES, SKELETON_FEATURES, SKELETON_JOINTS
from cuhkx_har.model import MultimodalHAR, parameter_size_mb


def test_scalable_model_forward_and_size() -> None:
    config = ExperimentConfig(
        image_size=64,
        visual_frames=2,
        sensor_steps=8,
        width_mult=0.5,
        d_model=64,
        fusion_layers=1,
        fusion_heads=4,
        batch_size=2,
        num_workers=0,
        epochs=1,
    )
    model = MultimodalHAR(config)
    batch = {
        "visual": torch.randn(2, 3, 2, 3, 64, 64),
        "skeleton": torch.randn(2, 8, SKELETON_JOINTS * SKELETON_FEATURES),
        "imu": torch.randn(2, 8, IMU_FEATURES),
        "radar": torch.randn(2, 8, RADAR_FEATURES),
        "modality_mask": torch.tensor(
            [[True, True, True, True, True, True], [True, True, False, True, True, False]]
        ),
    }
    output = model(batch)
    assert output.shape == (2, 40)
    assert parameter_size_mb(model) < 100


def test_graph_skeleton_encoder_forward() -> None:
    config = ExperimentConfig(
        image_size=64,
        visual_frames=2,
        sensor_steps=8,
        width_mult=0.5,
        d_model=64,
        fusion_layers=1,
        fusion_heads=4,
        skeleton_graph=True,
        batch_size=1,
        num_workers=0,
        epochs=1,
    )
    model = MultimodalHAR(config).eval()
    batch = {
        "visual": torch.randn(1, 3, 2, 3, 64, 64),
        "skeleton": torch.randn(1, 8, SKELETON_JOINTS * SKELETON_FEATURES),
        "imu": torch.randn(1, 8, IMU_FEATURES),
        "radar": torch.randn(1, 8, RADAR_FEATURES),
        "modality_mask": torch.ones(1, 6, dtype=torch.bool),
    }
    assert model(batch).shape == (1, 40)


def test_motion_skeleton_encoder_forward() -> None:
    config = ExperimentConfig(
        image_size=64,
        visual_frames=2,
        sensor_steps=8,
        width_mult=0.5,
        d_model=64,
        fusion_layers=1,
        fusion_heads=4,
        skeleton_motion=True,
        batch_size=1,
        num_workers=0,
        epochs=1,
    )
    model = MultimodalHAR(config).eval()
    batch = {
        "visual": torch.randn(1, 3, 2, 3, 64, 64),
        "skeleton": torch.randn(1, 8, SKELETON_JOINTS * SKELETON_FEATURES),
        "imu": torch.randn(1, 8, IMU_FEATURES),
        "radar": torch.randn(1, 8, RADAR_FEATURES),
        "modality_mask": torch.ones(1, 6, dtype=torch.bool),
    }
    assert model(batch).shape == (1, 40)


def test_pose_motion_residual_skeleton_encoder_forward() -> None:
    config = ExperimentConfig(
        image_size=64,
        visual_frames=2,
        sensor_steps=8,
        width_mult=0.5,
        d_model=64,
        fusion_layers=1,
        fusion_heads=4,
        skeleton_motion_residual=True,
        batch_size=1,
        num_workers=0,
        epochs=1,
    )
    model = MultimodalHAR(config).eval()
    batch = {
        "visual": torch.randn(1, 3, 2, 3, 64, 64),
        "skeleton": torch.randn(1, 8, SKELETON_JOINTS * SKELETON_FEATURES),
        "imu": torch.randn(1, 8, IMU_FEATURES),
        "radar": torch.randn(1, 8, RADAR_FEATURES),
        "modality_mask": torch.ones(1, 6, dtype=torch.bool),
    }
    assert model(batch).shape == (1, 40)


def test_skeleton_encoder_modes_are_exclusive() -> None:
    try:
        ExperimentConfig(skeleton_graph=True, skeleton_motion=True)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("Expected mutually exclusive skeleton modes to fail")


def test_horizontal_flip_probability_is_validated() -> None:
    try:
        ExperimentConfig(horizontal_flip_probability=1.1)
    except ValueError as error:
        assert "horizontal_flip_probability" in str(error)
    else:
        raise AssertionError("Expected an invalid flip probability to fail")


def test_pose_motion_residual_mode_is_exclusive() -> None:
    try:
        ExperimentConfig(skeleton_motion=True, skeleton_motion_residual=True)
    except ValueError as error:
        assert "exclusive" in str(error)
    else:
        raise AssertionError("Expected incompatible skeleton encoder modes to fail")


def test_directional_temporal_pooling_forward() -> None:
    config = ExperimentConfig(
        image_size=64,
        visual_frames=3,
        sensor_steps=8,
        width_mult=0.5,
        d_model=64,
        fusion_layers=1,
        fusion_heads=4,
        temporal_pooling="directional",
        batch_size=1,
        num_workers=0,
        epochs=1,
    )
    model = MultimodalHAR(config).eval()
    batch = {
        "visual": torch.randn(1, 3, 3, 3, 64, 64),
        "skeleton": torch.randn(1, 8, SKELETON_JOINTS * SKELETON_FEATURES),
        "imu": torch.randn(1, 8, IMU_FEATURES),
        "radar": torch.randn(1, 8, RADAR_FEATURES),
        "modality_mask": torch.ones(1, 6, dtype=torch.bool),
    }
    assert model(batch).shape == (1, 40)
