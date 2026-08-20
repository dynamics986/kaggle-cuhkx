from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cuhkx_modality import report as modality_report
from cuhkx_modality.config import ModalityConfig
from cuhkx_modality.constants import MODALITIES, VISUAL_MODALITIES
from cuhkx_modality.data import ModalityDataset, compute_normalizer, select_present_rows
from cuhkx_modality.frozen import read_frozen_manifest, validate_frozen_cv5
from cuhkx_modality.model import (
    IMUDeviceCNNRelativeTransformerEncoder,
    ModalityHAR,
    RelativeSelfAttention,
)
from cuhkx_har.features import cache_key


def test_every_modality_has_one_input_model() -> None:
    config = ModalityConfig(
        image_size=32,
        visual_frames=2,
        sensor_steps=4,
        width_mult=0.5,
        d_model=32,
        batch_size=1,
        num_workers=0,
        epochs=1,
    )
    for modality in MODALITIES:
        model = ModalityHAR(modality, config)
        inputs = (
            torch.randn(1, 2, 3, 32, 32)
            if modality in VISUAL_MODALITIES
            else torch.randn(1, 4, {"Skeleton": 68, "IMU": 80, "Radar": 13}[modality])
        )
        assert model(inputs).shape == (1, 40)
        assert not hasattr(model, "fusion")
        assert not hasattr(model, "modality_embedding")


def test_imu_device_cnn_relative_transformer_forward() -> None:
    config = ModalityConfig(
        sensor_steps=16,
        d_model=32,
        sensor_encoder="imu_device_cnn_rel_transformer",
        batch_size=2,
        num_workers=0,
        epochs=1,
    )
    model = ModalityHAR("IMU", config)
    assert isinstance(model.encoder, IMUDeviceCNNRelativeTransformerEncoder)
    inputs = torch.randn(2, 16, 5, 17)
    inputs[..., -1] = 1
    inputs[0, :, 2:, -1] = 0
    output = model(inputs)
    assert output.shape == (2, 40)
    assert torch.isfinite(output).all()


def test_sensor_encoder_rejects_wrong_modality() -> None:
    config = ModalityConfig(sensor_encoder="imu_device_cnn_rel_transformer")
    with pytest.raises(ValueError, match="only valid for the IMU"):
        ModalityHAR("Radar", config)


def test_relative_attention_bias_changes_output() -> None:
    attention = RelativeSelfAttention(16, heads=4, max_distance=8, dropout=0)
    inputs = torch.randn(2, 6, 16)
    valid = torch.ones(2, 6, dtype=torch.bool)
    baseline = attention(inputs, valid)
    with torch.no_grad():
        attention.relative_bias[8 + 1].fill_(5)
    changed = attention(inputs, valid)
    assert not torch.allclose(baseline, changed)


def test_structured_normalizers_ignore_padding(tmp_path: Path) -> None:
    manifest = pd.DataFrame([{"clip_id": "clip"}])
    imu = np.zeros((4, 5, 16), dtype=np.float32)
    imu_mask = np.zeros((4, 5), dtype=np.bool_)
    imu[0, 0] = 2
    imu[1, 0] = 4
    imu_mask[:2, 0] = True
    np.savez_compressed(
        tmp_path / cache_key("train", "clip"),
        imu_synced=imu,
        imu_device_mask=imu_mask,
    )
    imu_mean, _ = compute_normalizer(
        manifest, "IMU", tmp_path, sensor_encoder="imu_device_cnn_rel_transformer"
    )
    np.testing.assert_allclose(imu_mean, 3)


def test_structured_datasets_append_masks_and_reject_legacy_cache(tmp_path: Path) -> None:
    manifest = pd.DataFrame([{"clip_id": "clip", "label": 2}])
    path = tmp_path / cache_key("train", "clip")
    imu = np.ones((4, 5, 16), dtype=np.float32)
    imu_mask = np.ones((4, 5), dtype=np.bool_)
    np.savez_compressed(
        path,
        imu_synced=imu,
        imu_device_mask=imu_mask,
    )
    imu_dataset = ModalityDataset(
        manifest,
        "IMU",
        tmp_path,
        tmp_path,
        8,
        2,
        4,
        training=False,
        sensor_encoder="imu_device_cnn_rel_transformer",
    )
    assert imu_dataset[0]["inputs"].shape == (4, 5, 17)

    np.savez_compressed(path, imu=np.zeros((4, 80), dtype=np.float32))
    with pytest.raises(ValueError, match="rebuild"):
        imu_dataset[0]


def test_visual_dataset_only_uses_selected_modality(tmp_path: Path) -> None:
    image_dir = tmp_path / "depth"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_dir / "frame.png")
    manifest = pd.DataFrame(
        [{"clip_id": "clip", "label": 3, "Depth_Color": "depth", "IR": "does-not-exist"}]
    )
    selected = select_present_rows(manifest, "Depth_Color", tmp_path)
    dataset = ModalityDataset(selected, "Depth_Color", tmp_path, tmp_path, 8, 2, 4, training=False)
    item = dataset[0]
    assert item["inputs"].shape == (2, 3, 8, 8)
    assert item["label"].item() == 3


def test_visual_dataset_applies_fold_specific_yolo_crop(tmp_path: Path) -> None:
    image_dir = tmp_path / "depth"
    image_dir.mkdir()
    image = Image.new("RGB", (8, 8), color="white")
    for x in range(4):
        for y in range(8):
            image.putpixel((x, y), (255, 0, 0))
    image.save(image_dir / "frame.png")
    metadata = tmp_path / "bboxes.json"
    metadata.write_text(
        json.dumps(
            {"schema_version": 1, "clips": {"clip": {"frames": [{"bbox": [0, 0, 0.5, 1]}]}}}
        ),
        encoding="utf-8",
    )
    manifest = pd.DataFrame([{"clip_id": "clip", "label": 3, "Depth_Color": "depth"}])
    dataset = ModalityDataset(
        manifest,
        "Depth_Color",
        tmp_path,
        tmp_path,
        8,
        1,
        4,
        training=False,
        visual_crop_mode="yolo_person",
        visual_crop_metadata_path=metadata,
        visual_crop_padding=0,
    )
    inputs = dataset[0]["inputs"]
    assert float(inputs[0, 0].mean()) > 0.9
    assert float(inputs[0, 1].mean()) < -0.9


def test_frozen_cv5_rejects_mapping_change() -> None:
    manifest = read_frozen_manifest("manifests/cv5/train.csv")
    validate_frozen_cv5(manifest)
    changed = manifest.copy()
    changed.loc[0, "fold"] = (int(changed.loc[0, "fold"]) + 1) % 5
    with pytest.raises(ValueError, match="mapping differs"):
        validate_frozen_cv5(changed)


def _prediction(clip_id: str, label: int) -> pd.DataFrame:
    probabilities = np.zeros((1, 40), dtype=np.float64)
    probabilities[0, label] = 1.0
    frame = pd.DataFrame(probabilities, columns=[f"prob_{index}" for index in range(40)])
    frame.insert(0, "prediction", label)
    frame.insert(0, "label", label)
    frame.insert(0, "clip_id", clip_id)
    return frame


def test_modality_oof_requires_exact_available_fold_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = pd.DataFrame(
        {
            "clip_id": [f"clip-{fold}" for fold in range(5)],
            "label": list(range(5)),
            "fold": list(range(5)),
            "action_name": [f"action-{fold}" for fold in range(5)],
            "Depth_Color": ["present"] * 5,
        }
    )
    monkeypatch.setattr(modality_report, "read_frozen_manifest", lambda _: manifest)
    paths = []
    for fold in range(5):
        path = tmp_path / f"fold-{fold}.csv"
        _prediction(f"clip-{fold}", fold).to_csv(path, index=False)
        paths.append(path)
    report, oof = modality_report.evaluate_modality_oof("ignored", tmp_path, "Depth_Color", paths)
    assert report["oof_accuracy"] == 1.0
    assert len(oof) == 5
    _prediction("clip-0", 0).to_csv(paths[1], index=False)
    with pytest.raises(ValueError, match="More than one prediction file"):
        modality_report.evaluate_modality_oof("ignored", tmp_path, "Depth_Color", paths)
