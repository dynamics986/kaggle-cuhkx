from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, UnidentifiedImageError
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from cuhkx_har.splits import assert_no_subject_leakage
from cuhkx_modality.frozen import read_frozen_manifest

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def files_in(path):
    return (
        sorted((p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS), key=natural)
        if path.is_dir()
        else []
    )


def sample_indices(n, k, training, jitter=0, quantile=0.5):
    if n <= 0:
        return []
    result = []
    for a, b in zip(np.linspace(0, n, k + 1)[:-1], np.linspace(0, n, k + 1)[1:], strict=True):
        lo, hi = min(int(a), n - 1), min(max(int(np.ceil(b)) - 1, int(a)), n - 1)
        index = random.randint(lo, hi) if training else int(lo + (hi - lo) * quantile)
        result.append(min(max(index + jitter, 0), n - 1))
    return result


def open_rgb(path):
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError):
        return None


def index_thermal(manifest_path, root, split="frozen", test=False):
    table = pd.read_csv(manifest_path).fillna("") if test else read_frozen_manifest(manifest_path)
    table["frames"] = [
        files_in(Path(root) / relative) if relative else [] for relative in table.Thermal
    ]
    if not test:
        table = table.loc[table.frames.map(bool)].reset_index(drop=True)
        if split == "notebook":
            groups = table.user.str.extract(r"(\d+)")[0].astype(int)
            for fold, (_, valid) in enumerate(GroupKFold(5).split(table, table.label, groups)):
                table.loc[valid, "fold"] = fold
        assert_no_subject_leakage(table)
    return table


def split_digest(table):
    payload = table[["clip_id", "label", "user", "fold"]].sort_values("clip_id").to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class ThermalDataset(Dataset):
    def __init__(self, table, config, training=False, view=0):
        self.table = table.reset_index(drop=True)
        self.config, self.training, self.view = config, training, view

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        cfg = self.config
        source_tta = cfg["recipe"] == "r3_thermal_specialist" and self.view > 0
        random_sample = self.training or source_tta
        options = cfg.get("augmentation") or {}
        flip = random_sample and random.random() < float(options.get("flip_probability", 0.5))
        if cfg["improved"] and not self.training:
            flip = self.view == 1
        jitter = {2: 1, 3: -1}.get(self.view, 0) if source_tta else 0
        if self.training:
            jitter += random.randint(-int(options.get("temporal_offset", 0)), int(options.get("temporal_offset", 0)))
        quantile = {2: 0.25, 3: 0.75}.get(self.view, 0.5) if cfg["improved"] else 0.5
        indices = sample_indices(len(row.frames), cfg["frames"], random_sample, jitter, quantile)
        speed_range = options.get("temporal_speed_range") if self.training else None
        if speed_range and indices:
            low, high = map(float, speed_range)
            speed = random.uniform(low, high)
            center = (len(row.frames) - 1) / 2
            indices = [min(max(round(center + (item - center) * speed), 0), len(row.frames) - 1) for item in indices]
        crop_amount = 0.0
        crop_side = crop_left = crop_top = 0
        if self.training and (options.get("crop_scale") or options.get("crop_scale_min")):
            crop_amount = random.uniform(float(options.get("crop_scale_min", 0.0)), float(options.get("crop_scale", 0.0)))
            crop_side = max(2, round(cfg["image_size"] * (1 - crop_amount)))
            crop_left = random.randint(0, cfg["image_size"] - crop_side)
            crop_top = random.randint(0, cfg["image_size"] - crop_side)
        brightness = random.uniform(1 - float(options.get("brightness", 0.0)), 1 + float(options.get("brightness", 0.0))) if self.training else 1.0
        contrast = random.uniform(1 - float(options.get("contrast", 0.0)), 1 + float(options.get("contrast", 0.0))) if self.training else 1.0
        decoded, bad, images = {}, 0, []
        for selected in indices:
            image = None
            # Invalid sampled files are replaced by the nearest decodable frame in this clip.
            for candidate in sorted(range(len(row.frames)), key=lambda j: abs(j - selected)):
                if candidate not in decoded:
                    decoded[candidate] = open_rgb(row.frames[candidate])
                image = decoded[candidate]
                if image is not None:
                    break
            if decoded.get(selected) is None:
                bad += 1
            if image is None:
                image = Image.new("RGB", (cfg["image_size"], cfg["image_size"]))
            image = image.resize((cfg["image_size"], cfg["image_size"]))
            if crop_amount:
                image = image.crop((crop_left, crop_top, crop_left + crop_side, crop_top + crop_side)).resize((cfg["image_size"], cfg["image_size"]))
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.training:
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
            value = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255
            images.append((value - 0.5) / 0.25)
        if not images:
            images = [torch.full((3, cfg["image_size"], cfg["image_size"]), -2.0)] * cfg["frames"]
            bad = cfg["frames"]
        stacked = torch.stack(images)
        erase = float(options.get("erase_fraction", 0.0))
        if self.training and erase and random.random() < float(options.get("erase_probability", 0.5)):
            side = max(1, round(cfg["image_size"] * erase))
            left, top = random.randint(0, cfg["image_size"] - side), random.randint(0, cfg["image_size"] - side)
            stacked[..., top : top + side, left : left + side] = -2.0
        return stacked, int(row.label), bad
