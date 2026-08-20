from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cuhkx_har.data import (
    _has_path,
    crop_person,
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
    manifest: pd.DataFrame,
    modality: str,
    cache_dir: str | Path,
    split: str = "train",
    sensor_encoder: str = "tcn",
) -> tuple[np.ndarray, np.ndarray] | None:
    if modality not in SENSOR_MODALITIES:
        return None
    key, cache = SENSOR_CACHE_KEY[modality], Path(cache_dir)
    structured_key = {
        "imu_device_cnn_rel_transformer": "imu_synced",
    }.get(sensor_encoder)
    mask_key = {
        "imu_device_cnn_rel_transformer": "imu_device_mask",
    }.get(sensor_encoder)
    total: np.ndarray | None = None
    squares: np.ndarray | None = None
    count = 0
    for clip_id in manifest["clip_id"]:
        with np.load(cache / cache_key(split, str(clip_id))) as item:
            if structured_key is not None:
                if structured_key not in item or mask_key not in item:
                    raise ValueError(
                        f"Cache lacks {structured_key}/{mask_key}; rebuild it with cuhkx-cache "
                        "into a new cache-64-synced-points directory"
                    )
                values = item[structured_key].astype(np.float64)
                valid = item[mask_key].astype(np.bool_)
                values = values[valid]
            else:
                values = item[key].astype(np.float64)
        if not len(values):
            continue
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        squares = (
            np.square(values).sum(axis=0)
            if squares is None
            else squares + np.square(values).sum(axis=0)
        )
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
        visual_crop_mode: str = "none",
        visual_crop_metadata_path: str | Path | None = None,
        visual_crop_padding: float = 0.12,
        sensor_encoder: str = "tcn",
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).fillna("")
        self.modality = modality
        self.data_root, self.cache_dir = Path(data_root), Path(cache_dir)
        self.image_size, self.visual_frames, self.sensor_steps = (
            image_size,
            visual_frames,
            sensor_steps,
        )
        self.training, self.normalizer = training, normalizer
        self.horizontal_flip_probability = horizontal_flip_probability
        self.visual_crop_mode = visual_crop_mode
        self.visual_crop_padding = visual_crop_padding
        self.sensor_encoder = sensor_encoder
        self.crop_metadata: dict[str, dict[str, Any]] = {}
        if visual_crop_mode == "yolo_person":
            if modality not in VISUAL_MODALITIES:
                raise ValueError("yolo_person crops are only valid for visual modalities")
            if visual_crop_metadata_path is None:
                raise ValueError("visual_crop_metadata_path is required for yolo_person")
            path = Path(visual_crop_metadata_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing YOLO crop metadata: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError(f"Unsupported crop metadata schema: {path}")
            self.crop_metadata = payload["clips"]

    def __len__(self) -> int:
        return len(self.manifest)

    def _visual(self, row: pd.Series, flip: bool) -> torch.Tensor:
        directory = self.data_root / Path(str(row[self.modality]))
        files = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
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
                image = source.convert("RGB")
            if self.visual_crop_mode == "yolo_person":
                clip = self.crop_metadata.get(str(row["clip_id"]))
                if clip is None:
                    raise KeyError(f"No crop metadata for clip {row['clip_id']}")
                depth_frames = clip.get("frames", [])
                if depth_frames:
                    mapped = int(
                        round(int(index) * (len(depth_frames) - 1) / max(len(files) - 1, 1))
                    )
                    bbox = depth_frames[mapped].get("bbox")
                    if bbox is not None:
                        image = crop_person(
                            image,
                            tuple(float(value) for value in bbox),
                            self.visual_crop_padding,
                        )
            image = resize_visual(image, self.image_size, False)
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
            if self.sensor_encoder == "imu_device_cnn_rel_transformer":
                required = ("imu_synced", "imu_device_mask")
                if any(name not in item for name in required):
                    raise ValueError(
                        "Cache lacks synchronized IMU fields; rebuild cache-64-synced-points"
                    )
                values = item["imu_synced"].astype(np.float32)
                valid = item["imu_device_mask"].astype(np.bool_)
            else:
                values = item[key].astype(np.float32)
                valid = None
        if flip:
            if self.sensor_encoder == "imu_device_cnn_rel_transformer":
                order = (0, 3, 4, 1, 2)
                values, valid = values[:, order].copy(), valid[:, order].copy()
            else:
                values = {
                    "Skeleton": flip_skeleton,
                    "IMU": flip_imu_devices,
                    "Radar": flip_radar,
                }[self.modality](values)
        if self.normalizer is not None:
            mean, std = self.normalizer
            values = (values - mean) / np.maximum(std, 1e-5)
        if valid is not None:
            if len(values) != self.sensor_steps:
                raise ValueError(
                    f"Structured cache has {len(values)} steps but config requests "
                    f"{self.sensor_steps}; rebuild the cache with matching --steps"
                )
            values = np.where(valid[..., None], values, 0.0)
            return torch.from_numpy(
                np.concatenate([values, valid[..., None].astype(np.float32)], axis=-1)
            )
        return torch.from_numpy(resample_sequence(values, self.sensor_steps))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        flip = self.training and random.random() < self.horizontal_flip_probability
        inputs = (
            self._visual(row, flip)
            if self.modality in VISUAL_MODALITIES
            else self._sensor(row, flip)
        )
        return {
            "inputs": inputs,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "clip_id": str(row["clip_id"]),
        }
