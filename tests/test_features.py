from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image

from cuhkx_har.data import (
    MultimodalDataset,
    compute_sensor_normalizer,
    flip_imu_devices,
    flip_radar,
    flip_skeleton,
    resize_visual,
    temporal_indices,
)
from cuhkx_har.features import (
    IMU_FEATURES,
    RADAR_MAX_POINTS,
    RADAR_FEATURES,
    SKELETON_FEATURES,
    SKELETON_JOINTS,
    build_feature_file,
    cache_key,
    read_imu,
    read_imu_synced,
    read_radar,
    read_radar_points,
    read_skeleton,
    resample_sequence,
)


def test_resample_sequence_shape_and_endpoints() -> None:
    source = np.asarray([[0.0, 2.0], [10.0, 12.0]], dtype=np.float32)
    output = resample_sequence(source, 5)
    assert output.shape == (5, 2)
    np.testing.assert_allclose(output[0], source[0])
    np.testing.assert_allclose(output[-1], source[-1])


def test_sensor_parsers_handle_valid_and_empty_inputs(tmp_path) -> None:
    skeleton_dir = tmp_path / "Skeleton" / "predictions"
    skeleton_dir.mkdir(parents=True)
    person = {
        "keypoints": np.arange(SKELETON_JOINTS * 3, dtype=float).reshape(17, 3).tolist(),
        "keypoint_scores": [0.9] * SKELETON_JOINTS,
    }
    (skeleton_dir / "frame.json").write_text(json.dumps([person]), encoding="utf-8")
    skeleton, skeleton_ok = read_skeleton(tmp_path / "Skeleton", steps=8)
    assert skeleton_ok
    assert skeleton.shape == (8, SKELETON_JOINTS * SKELETON_FEATURES)
    assert np.isfinite(skeleton).all()

    imu_dir = tmp_path / "IMU"
    imu_dir.mkdir()
    imu = pd.DataFrame(
        [
            ["2026-01-01 00:00:00", "WTC(00:00)", *range(16)],
            ["2026-01-01 00:00:01", "WTRA(11:11)", *range(16)],
        ],
        columns=["time", "device", *[f"f{i}" for i in range(16)]],
    )
    imu.to_csv(imu_dir / "sensors.csv", index=False)
    imu_array, imu_ok = read_imu(imu_dir, steps=8)
    assert imu_ok
    assert imu_array.shape == (8, IMU_FEATURES)
    assert np.any(imu_array[:, :16])
    assert not np.any(imu_array[:, 16:48])
    assert np.any(imu_array[:, 48:64])
    assert not np.any(imu_array[:, 64:])

    radar_dir = tmp_path / "Radar"
    radar_dir.mkdir()
    radar = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "z": [1, 2, 3],
            "v": [1, 2, 3],
            "snr": [1, 2, 3],
            "noise": [1, 2, 3],
        }
    )
    radar.to_csv(radar_dir / "radar.csv", index=False)
    radar_array, radar_ok = read_radar(radar_dir, steps=8)
    assert radar_ok
    assert radar_array.shape == (8, RADAR_FEATURES)

    empty, empty_ok = read_radar(tmp_path / "missing", steps=8)
    assert not empty_ok
    assert not empty.any()


def test_imu_sync_sorts_timestamps_averages_duplicates_and_masks_devices(tmp_path) -> None:
    directory = tmp_path / "IMU"
    directory.mkdir()
    rows = [
        ["2026-01-01 00:00:02", "WTC(a)", *([20.0] * 16)],
        ["2026-01-01 00:00:00", "WTC(a)", *([0.0] * 16)],
        ["2026-01-01 00:00:01", "WTC(a)", *([8.0] * 16)],
        ["2026-01-01 00:00:01", "WTC(a)", *([12.0] * 16)],
        ["2026-01-01 00:00:01", "WTLA(b)", *([5.0] * 16)],
        ["2026-01-01 00:00:02", "WTLA(b)", *([9.0] * 16)],
    ]
    pd.DataFrame(rows, columns=["time", "device", *[f"f{i}" for i in range(16)]]).to_csv(
        directory / "sensors.csv", index=False
    )
    values, mask, available = read_imu_synced(directory, steps=3)
    assert available
    assert values.shape == (3, 5, 16)
    np.testing.assert_allclose(values[:, 0, 0], [0, 10, 20])
    np.testing.assert_array_equal(mask[:, 0], [True, True, True])
    np.testing.assert_array_equal(mask[:, 1], [False, True, True])
    assert not mask[:, 2:].any()


def test_radar_points_preserve_empty_frames_and_truncate_by_snr(tmp_path) -> None:
    directory = tmp_path / "Radar"
    directory.mkdir()
    rows = []
    for point in range(RADAR_MAX_POINTS + 3):
        rows.append([0, point, point, 0, 0, 0, point, 0])
    rows.append([2, 0, 99, 1, 1, 1, 1, 1])
    pd.DataFrame(
        rows, columns=["frame", "DetObj#", "x", "y", "z", "v", "snr", "noise"]
    ).to_csv(directory / "radar.csv", index=False)
    points, mask, available = read_radar_points(directory, steps=3)
    assert available
    assert points.shape == (3, RADAR_MAX_POINTS, 6)
    assert mask[0].sum() == RADAR_MAX_POINTS
    assert not mask[1].any()
    assert mask[2].sum() == 1
    assert points[0, 0, 4] == RADAR_MAX_POINTS + 2
    assert 0 not in points[0, :, 4]


def test_feature_cache_keeps_radar_statistics_without_raw_point_fields(tmp_path) -> None:
    radar_dir = tmp_path / "Radar"
    radar_dir.mkdir()
    pd.DataFrame(
        [[0, 1, 2, 3, 4, 5, 6]],
        columns=["frame", "x", "y", "z", "v", "snr", "noise"],
    ).to_csv(radar_dir / "radar.csv", index=False)
    row = pd.Series({"clip_id": "clip", "Skeleton": "", "IMU": "", "Radar": "Radar"})
    path = build_feature_file(row, tmp_path, tmp_path / "cache", "train", steps=4)
    with np.load(path) as cache:
        assert cache["radar"].shape == (4, RADAR_FEATURES)
        assert "radar_points" not in cache
        assert "radar_point_mask" not in cache


def test_multimodal_structured_imu_keeps_legacy_radar_cache(tmp_path) -> None:
    manifest = pd.DataFrame(
        [{"clip_id": "clip", "label": 0, "Depth_Color": "", "IR": "", "Thermal": ""}]
    )
    legacy_cache, structured_cache = tmp_path / "legacy", tmp_path / "structured"
    legacy_cache.mkdir()
    structured_cache.mkdir()
    key = cache_key("train", "clip")
    radar = np.full((4, RADAR_FEATURES), 7, dtype=np.float32)
    np.savez_compressed(
        legacy_cache / key,
        skeleton=np.zeros((4, SKELETON_JOINTS * SKELETON_FEATURES), dtype=np.float32),
        imu=np.zeros((4, IMU_FEATURES), dtype=np.float32),
        radar=radar,
        sensor_mask=np.asarray([True, True, True]),
    )
    synced = np.zeros((4, 5, 16), dtype=np.float32)
    for device in range(5):
        synced[:, device] = device + 1
    device_mask = np.ones((4, 5), dtype=np.bool_)
    device_mask[:, 2] = False
    np.savez_compressed(
        structured_cache / key, imu_synced=synced, imu_device_mask=device_mask
    )
    normalizer = compute_sensor_normalizer(
        manifest,
        legacy_cache,
        imu_encoder="device_cnn_rel_transformer",
        imu_structured_cache_dir=structured_cache,
    )
    dataset = MultimodalDataset(
        manifest,
        tmp_path,
        legacy_cache,
        "train",
        image_size=8,
        visual_frames=2,
        sensor_steps=4,
        training=False,
        normalizer={"imu": normalizer["imu"]},
        imu_encoder="device_cnn_rel_transformer",
        imu_structured_cache_dir=structured_cache,
    )
    item = dataset[0]
    assert item["imu"].shape == (4, 5, 17)
    assert not item["imu"][:, 2, -1].any()
    np.testing.assert_allclose(item["radar"].numpy(), radar)


def test_visual_decoder_skips_corrupted_frame(tmp_path) -> None:
    bad = tmp_path / "bad.png"
    good = tmp_path / "good.png"
    bad.write_bytes(bytes(128))
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(good)
    decoded = MultimodalDataset._open_nearest_valid([bad, good], 0)
    assert decoded is not None
    assert decoded.size == (8, 8)


def test_letterbox_preserves_full_aspect_ratio() -> None:
    image = Image.new("RGB", (8, 4), color=(255, 0, 0))
    padded = resize_visual(image, 8, preserve_aspect_ratio=True)
    cropped = resize_visual(image, 8, preserve_aspect_ratio=False)
    assert padded.size == cropped.size == (8, 8)
    assert padded.getpixel((0, 0)) == (0, 0, 0)
    assert padded.getpixel((4, 4)) == (255, 0, 0)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)


def test_shared_temporal_phases_align_relative_positions() -> None:
    phases = np.asarray([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    short = temporal_indices(100, 4, training=True, phases=phases)
    long = temporal_indices(200, 4, training=True, phases=phases)
    np.testing.assert_allclose(short / 100, long / 200, atol=0.01)


def test_synchronized_sensor_flip_is_an_involution() -> None:
    skeleton = np.arange(3 * SKELETON_JOINTS * SKELETON_FEATURES, dtype=np.float32).reshape(3, -1)
    imu = np.arange(3 * IMU_FEATURES, dtype=np.float32).reshape(3, -1)
    radar = np.arange(3 * RADAR_FEATURES, dtype=np.float32).reshape(3, -1)

    np.testing.assert_allclose(flip_skeleton(flip_skeleton(skeleton)), skeleton)
    np.testing.assert_allclose(flip_imu_devices(flip_imu_devices(imu)), imu)
    np.testing.assert_allclose(flip_radar(flip_radar(radar)), radar)


def test_synchronized_sensor_flip_has_expected_mapping() -> None:
    skeleton = np.zeros((1, SKELETON_JOINTS, SKELETON_FEATURES), dtype=np.float32)
    skeleton[0, :, 0] = np.arange(SKELETON_JOINTS) + 1
    flipped_skeleton = flip_skeleton(skeleton.reshape(1, -1)).reshape(skeleton.shape)
    assert flipped_skeleton[0, 1, 0] == -3  # new left eye comes from old right eye
    assert flipped_skeleton[0, 2, 0] == -2

    imu = np.zeros((1, 5, 16), dtype=np.float32)
    for device in range(5):
        imu[:, device] = device
    flipped_imu = flip_imu_devices(imu.reshape(1, -1)).reshape(imu.shape)
    np.testing.assert_array_equal(flipped_imu[0, :, 0], [0, 3, 4, 1, 2])

    radar = np.ones((1, RADAR_FEATURES), dtype=np.float32)
    flipped_radar = flip_radar(radar)
    assert flipped_radar[0, 1] == -1
    np.testing.assert_array_equal(flipped_radar[0, 2:], radar[0, 2:])
