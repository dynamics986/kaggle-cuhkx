from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cuhkx_har.data import (
    _has_path,
    flip_imu_devices,
    flip_radar,
    flip_skeleton,
    resize_visual,
    temporal_indices,
)
from cuhkx_har.features import cache_key, resample_sequence

from .constants import SENSOR_CACHE_INDEX, SENSOR_CACHE_KEY, SENSOR_MODALITIES, VISUAL_MODALITIES


def select_present_rows(
    manifest: pd.DataFrame, modality: str, cache_dir: str | Path, split: str = "train"
) -> pd.DataFrame:
    """Return only clips whose selected modality is actually available."""
    if modality in VISUAL_MODALITIES:
        return manifest.loc[manifest[modality].map(_has_path)].reset_index(drop=True)
    cache = Path(cache_dir)
    index = SENSOR_CACHE_INDEX[modality]
    available: list[bool] = []
    for clip_id in manifest["clip_id"]:
        path = cache / cache_key(split, str(clip_id))
        if not path.is_file():
            raise FileNotFoundError(f"Missing sensor cache {path}; run cuhkx-cache first")
        with np.load(path) as item:
            available.append(bool(item["sensor_mask"][index]))
    return manifest.loc[available].reset_index(drop=True)


def compute_normalizer(
    manifest: pd.DataFrame, modality: str, cache_dir: str | Path, split: str = "train"
) -> tuple[np.ndarray, np.ndarray] | None:
    if modality not in SENSOR_MODALITIES:
        return None
    key, cache = SENSOR_CACHE_KEY[modality], Path(cache_dir)
    total: np.ndarray | None = None
    squares: np.ndarray | None = None
    count = 0
    for clip_id in manifest["clip_id"]:
        with np.load(cache / cache_key(split, str(clip_id))) as item:
            values = item[key].astype(np.float64)
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        squares = np.square(values).sum(axis=0) if squares is None else squares + np.square(values).sum(axis=0)
        count += len(values)
    if count == 0 or total is None or squares is None:
        raise ValueError(f"No present {modality} examples in the training partition")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - np.square(mean), 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class ModalityDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        modality: str,
        data_root: str | Path,
        cache_dir: str | Path,
        image_size: int,
        visual_frames: int,
        sensor_steps: int,
        training: bool,
        normalizer: tuple[np.ndarray, np.ndarray] | None = None,
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.modality = modality
        self.data_root, self.cache_dir = Path(data_root), Path(cache_dir)
        self.image_size, self.visual_frames, self.sensor_steps = image_size, visual_frames, sensor_steps
        self.training, self.normalizer = training, normalizer
        self.horizontal_flip_probability = horizontal_flip_probability

    def __len__(self) -> int:
        return len(self.manifest)

    def _visual(self, row: pd.Series, flip: bool) -> torch.Tensor:
        directory = self.data_root / Path(str(row[self.modality]))
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise FileNotFoundError(f"No readable image files for {self.modality}: {directory}")
        indices = temporal_indices(len(files), self.visual_frames, self.training)
        brightness = random.uniform(0.9, 1.1) if self.training else 1.0
        contrast = random.uniform(0.9, 1.1) if self.training else 1.0
        from PIL import Image, ImageEnhance, ImageOps
        from torchvision.transforms import functional as tvf

        frames = []
        for index in indices:
            with Image.open(files[int(index)]) as source:
                image = resize_visual(source.convert("RGB"), self.image_size, False)
            if flip:
                image = ImageOps.mirror(image)
            if self.training:
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
            frames.append(tvf.pil_to_tensor(image).float().div_(255.0).sub_(0.5).div_(0.5))
        return torch.stack(frames)

    def _sensor(self, row: pd.Series, flip: bool) -> torch.Tensor:
        key = SENSOR_CACHE_KEY[self.modality]
        with np.load(self.cache_dir / cache_key("train", str(row["clip_id"]))) as item:
            values = item[key].astype(np.float32)
        if flip:
            values = {"Skeleton": flip_skeleton, "IMU": flip_imu_devices, "Radar": flip_radar}[self.modality](values)
        if self.normalizer is not None:
            mean, std = self.normalizer
            values = (values - mean) / np.maximum(std, 1e-5)
        return torch.from_numpy(resample_sequence(values, self.sensor_steps))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        flip = self.training and random.random() < self.horizontal_flip_probability
        inputs = self._visual(row, flip) if self.modality in VISUAL_MODALITIES else self._sensor(row, flip)
        return {"inputs": inputs, "label": torch.tensor(int(row["label"]), dtype=torch.long), "clip_id": str(row["clip_id"])}
