from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf

from .constants import VISUAL_MODALITIES
from .features import (
    IMU_DEVICES,
    IMU_FEATURES_PER_DEVICE,
    SKELETON_FEATURES,
    SKELETON_JOINTS,
    cache_key,
    resample_sequence,
)

COCO_LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
IMU_MIRROR_ORDER = (0, 3, 4, 1, 2)  # center, RA->LA, RL->LL, LA->RA, LL->RL
RADAR_MEAN_X_INDEX = 1  # count is column 0; mean(x) is the first radar statistic


def flip_skeleton(skeleton: np.ndarray) -> np.ndarray:
    joints = skeleton.reshape(-1, SKELETON_JOINTS, SKELETON_FEATURES).copy()
    joints[..., 0] *= -1
    source = joints.copy()
    for left, right in COCO_LEFT_RIGHT_PAIRS:
        joints[:, left] = source[:, right]
        joints[:, right] = source[:, left]
    return joints.reshape(skeleton.shape)


def flip_imu_devices(imu: np.ndarray) -> np.ndarray:
    devices = imu.reshape(-1, IMU_DEVICES, IMU_FEATURES_PER_DEVICE)
    return devices[:, IMU_MIRROR_ORDER].reshape(imu.shape).copy()


def flip_radar(radar: np.ndarray) -> np.ndarray:
    mirrored = radar.copy()
    mirrored[:, RADAR_MEAN_X_INDEX] *= -1
    return mirrored


def temporal_indices(
    length: int,
    count: int,
    training: bool,
    view_index: int = 0,
    num_views: int = 1,
    phases: np.ndarray | None = None,
) -> np.ndarray:
    if length <= 0:
        return np.zeros(count, dtype=np.int64)
    edges = np.linspace(0, length, count + 1)
    indices = []
    if phases is not None and len(phases) != count:
        raise ValueError("phases length must match the requested frame count")
    if phases is not None and (np.any(phases < 0) or np.any(phases >= 1)):
        raise ValueError("phases must be in [0, 1)")
    for segment, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        low, high = int(np.floor(left)), max(int(np.ceil(right)) - 1, int(np.floor(left)))
        if training and phases is None:
            index = random.randint(low, high)
        else:
            phase = (
                float(phases[segment])
                if phases is not None
                else (view_index + 0.5) / max(num_views, 1)
            )
            index = int(np.floor(left + phase * max(right - left, 1.0)))
        indices.append(min(max(index, 0), length - 1))
    return np.asarray(indices, dtype=np.int64)


def _has_path(value: object) -> bool:
    return not pd.isna(value) and bool(str(value))


def resize_visual(image: Image.Image, image_size: int, preserve_aspect_ratio: bool) -> Image.Image:
    size = (image_size, image_size)
    if preserve_aspect_ratio:
        return ImageOps.pad(
            image, size, method=Image.Resampling.BILINEAR, color=(0, 0, 0)
        )
    return ImageOps.fit(image, size, method=Image.Resampling.BILINEAR)


class MultimodalDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        data_root: str | Path,
        cache_dir: str | Path,
        split: str,
        image_size: int,
        visual_frames: int,
        sensor_steps: int,
        training: bool,
        normalizer: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        view_index: int = 0,
        num_views: int = 1,
        horizontal_flip_probability: float = 0.0,
        preserve_aspect_ratio: bool = False,
        shared_visual_sampling: bool = False,
        imu_device_dropout: float = 0.0,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.data_root = Path(data_root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.split = split
        self.image_size = image_size
        self.visual_frames = visual_frames
        self.sensor_steps = sensor_steps
        self.training = training
        self.normalizer = normalizer or {}
        self.view_index = view_index
        self.num_views = num_views
        self.horizontal_flip_probability = horizontal_flip_probability
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.shared_visual_sampling = shared_visual_sampling
        self.imu_device_dropout = imu_device_dropout

    def __len__(self) -> int:
        return len(self.manifest)

    @staticmethod
    def _open_nearest_valid(files: list[Path], index: int) -> Image.Image | None:
        offsets = [0]
        for distance in range(1, len(files)):
            offsets.extend((-distance, distance))
        visited: set[int] = set()
        for offset in offsets:
            candidate = min(max(index + offset, 0), len(files) - 1)
            if candidate in visited:
                continue
            visited.add(candidate)
            try:
                with Image.open(files[candidate]) as source:
                    return source.convert("RGB")
            except (OSError, UnidentifiedImageError):
                continue
        return None

    def _load_visual(
        self,
        row: pd.Series,
        modality: str,
        flip: bool,
        phases: np.ndarray | None,
    ) -> tuple[torch.Tensor, bool]:
        value = row[modality]
        if not _has_path(value):
            shape = (self.visual_frames, 3, self.image_size, self.image_size)
            return torch.zeros(shape, dtype=torch.float32), False
        directory = self.data_root / Path(str(value))
        suffixes = {".jpg", ".jpeg", ".png"}
        files = sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes
        )
        if not files:
            shape = (self.visual_frames, 3, self.image_size, self.image_size)
            return torch.zeros(shape, dtype=torch.float32), False
        indices = temporal_indices(
            len(files),
            self.visual_frames,
            self.training,
            self.view_index,
            self.num_views,
            phases,
        )
        frames = []
        brightness = random.uniform(0.9, 1.1) if self.training else 1.0
        contrast = random.uniform(0.9, 1.1) if self.training else 1.0
        for index in indices:
            image = self._open_nearest_valid(files, int(index))
            if image is None:
                shape = (self.visual_frames, 3, self.image_size, self.image_size)
                return torch.zeros(shape, dtype=torch.float32), False
            image = resize_visual(image, self.image_size, self.preserve_aspect_ratio)
            if flip:
                image = ImageOps.mirror(image)
            if self.training:
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
            tensor = tvf.pil_to_tensor(image).float().div_(255.0)
            frames.append(tensor.sub_(0.5).div_(0.5))
        return torch.stack(frames), True

    def _normalize(self, name: str, array: np.ndarray) -> np.ndarray:
        if name not in self.normalizer:
            return array
        mean, std = self.normalizer[name]
        return (array - mean) / np.maximum(std, 1e-5)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        flip = self.training and random.random() < self.horizontal_flip_probability
        phases = None
        if self.training and self.shared_visual_sampling:
            phases = np.asarray([random.random() for _ in range(self.visual_frames)])
        visual_tensors, visual_mask = [], []
        for modality in VISUAL_MODALITIES:
            tensor, present = self._load_visual(row, modality, flip, phases)
            visual_tensors.append(tensor)
            visual_mask.append(present)

        feature_path = self.cache_dir / cache_key(self.split, str(row["clip_id"]))
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing sensor cache {feature_path}; run cuhkx-cache first")
        with np.load(feature_path) as cached:
            skeleton = cached["skeleton"].astype(np.float32)
            imu = cached["imu"].astype(np.float32)
            radar = cached["radar"].astype(np.float32)
            sensor_mask = cached["sensor_mask"].astype(np.bool_)
        if flip:
            skeleton = flip_skeleton(skeleton)
            imu = flip_imu_devices(imu)
            radar = flip_radar(radar)
        skeleton = self._normalize("skeleton", skeleton)
        imu = self._normalize("imu", imu)
        radar = self._normalize("radar", radar)
        if self.training and self.imu_device_dropout and random.random() < self.imu_device_dropout:
            # IMU is stored in five fixed 16-feature device slots.  Drop one
            # slot after normalization so missing-device robustness does not
            # alter normalization statistics or the inference data path.
            device = random.randrange(IMU_DEVICES)
            start = device * IMU_FEATURES_PER_DEVICE
            imu[:, start : start + IMU_FEATURES_PER_DEVICE] = 0.0
        if len(skeleton) != self.sensor_steps:
            skeleton = resample_sequence(skeleton, self.sensor_steps)
            imu = resample_sequence(imu, self.sensor_steps)
            radar = resample_sequence(radar, self.sensor_steps)

        modality_mask = np.concatenate([np.asarray(visual_mask), sensor_mask])
        return {
            "visual": torch.stack(visual_tensors),
            "skeleton": torch.from_numpy(skeleton),
            "imu": torch.from_numpy(imu),
            "radar": torch.from_numpy(radar),
            "modality_mask": torch.from_numpy(modality_mask),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "clip_id": str(row["clip_id"]),
            "submission_path": str(row.get("submission_path", "")),
        }


def compute_sensor_normalizer(
    manifest: pd.DataFrame, cache_dir: str | Path, split: str = "train"
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache = Path(cache_dir)
    names = ("skeleton", "imu", "radar")
    sums: dict[str, np.ndarray] = {}
    squares: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {name: 0 for name in names}
    for _, row in manifest.iterrows():
        with np.load(cache / cache_key(split, str(row["clip_id"]))) as item:
            mask = item["sensor_mask"]
            for index, name in enumerate(names):
                if not bool(mask[index]):
                    continue
                values = item[name].astype(np.float64)
                sums[name] = sums.get(name, np.zeros(values.shape[1])) + values.sum(axis=0)
                current_squares = squares.get(name, np.zeros(values.shape[1]))
                squares[name] = current_squares + np.square(values).sum(axis=0)
                counts[name] += len(values)
    result = {}
    for name in names:
        if counts[name] == 0:
            raise ValueError(f"No present {name} features in training partition")
        mean = sums[name] / counts[name]
        variance = np.maximum(squares[name] / counts[name] - np.square(mean), 1e-8)
        result[name] = (mean.astype(np.float32), np.sqrt(variance).astype(np.float32))
    return result
