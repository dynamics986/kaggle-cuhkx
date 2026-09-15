from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import cache_path, digest


def validate_cache(root, split, manifest_path, fold, detector, config=None):
    meta = json.loads((Path(root) / f"{split}_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "split": split,
        "fold": str(fold),
        "manifest_sha256": digest(manifest_path),
        "detector_sha256": digest(detector),
    }
    if config:
        expected.update(frames=config["frames"], image_size=config["image_size"])
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Crop cache {key}: expected {value}, got {meta.get(key)}")
    return meta


class CroppedDataset(Dataset):
    def __init__(self, rows, root, split, training=False, augmentation=None):
        self.rows = rows.reset_index(drop=True)
        self.root, self.split, self.training = root, split, training
        # ``None`` means the historical Sep12 policy: flip plus brightness only.
        # Keep this distinct from an explicit Sep13 augmentation dictionary so a
        # future Sep12 rerun remains comparable with its recorded result.
        self.augmentation = augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        path = cache_path(self.root, self.split, row.clip_id)
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["clip_id"]) != row.clip_id:
                raise ValueError(f"Cache clip identity mismatch: {path}")
            images = torch.from_numpy(saved["images"].copy()).permute(0, 1, 4, 2, 3).float() / 255
            mask = torch.from_numpy(saved["mask"].copy()).bool()
        if self.training:
            # Every geometric/photometric draw is made once per clip, then shared by
            # every modality and frame.  Independent per-frame augmentation would
            # turn augmentation noise into artificial motion.
            options = self.augmentation or {}
            if torch.rand(()) < float(options.get("flip_probability", 0.5)):
                images = images.flip(-1)
            crop_scale = float(options.get("crop_scale", 0.0))
            crop_scale_min = float(options.get("crop_scale_min", 0.0))
            if crop_scale or crop_scale_min:
                height, width = images.shape[-2:]
                amount = crop_scale_min + (crop_scale - crop_scale_min) * torch.rand(()).item()
                scale = 1.0 - amount
                side_h, side_w = max(2, round(height * scale)), max(2, round(width * scale))
                top = int(torch.randint(height - side_h + 1, ()).item())
                left = int(torch.randint(width - side_w + 1, ()).item())
                crop = images[..., top : top + side_h, left : left + side_w]
                images = torch.nn.functional.interpolate(
                    crop.reshape(-1, *crop.shape[-3:]), size=(height, width), mode="bilinear",
                    align_corners=False,
                ).reshape_as(images)
            brightness = float(options.get("brightness", 0.1))
            contrast = float(options.get("contrast", 0.0))
            if brightness or contrast:
                images = images * (1 + brightness * (2 * torch.rand(()) - 1))
                mean = images.mean(dim=(-2, -1), keepdim=True)
                images = (images - mean) * (1 + contrast * (2 * torch.rand(()) - 1)) + mean
                images = images.clamp(0, 1)
            erase = float(options.get("erase_fraction", 0.0))
            if erase and torch.rand(()) < float(options.get("erase_probability", 0.5)):
                height, width = images.shape[-2:]
                erase_h, erase_w = max(1, round(height * erase)), max(1, round(width * erase))
                top = int(torch.randint(height - erase_h + 1, ()).item())
                left = int(torch.randint(width - erase_w + 1, ()).item())
                images[..., top : top + erase_h, left : left + erase_w] = 0
            offset = int(options.get("temporal_offset", 0))
            speed_range = options.get("temporal_speed_range")
            if offset or speed_range:
                shift = int(torch.randint(-offset, offset + 1, ()).item())
                positions = torch.arange(images.shape[1], dtype=torch.float32)
                if speed_range:
                    low, high = map(float, speed_range)
                    if not 0 < low <= high:
                        raise ValueError("temporal_speed_range must be positive and ordered")
                    speed = low + (high - low) * torch.rand(()).item()
                    center = (images.shape[1] - 1) / 2
                    positions = center + (positions - center) * speed
                index = positions.add(shift).round().long().clamp(0, images.shape[1] - 1)
                images, mask = images[:, index], mask[:, index]
            if float(options.get("drop_ir_probability", 0.0)) and torch.rand(()) < float(
                options["drop_ir_probability"]
            ):
                # The model then sees a valid Depth-only clip, rather than black IR
                # pixels incorrectly marked as an observed stream.
                images[0] = 0
                mask[0] = False
        images = (images - 0.5) / 0.5
        images *= mask[:, :, None, None, None]
        return images, mask, int(row.label)
