from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import MultimodalDataset
from .model import MultimodalHAR, parameter_size_mb
from .train import move_batch


def normalize_weights(count: int, weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("Number of ensemble weights must match checkpoints")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Ensemble weights must be non-negative with a positive sum")
    total = sum(weights)
    return [weight / total for weight in weights]


@torch.inference_mode()
def predict_checkpoint(
    checkpoint_path: str | Path,
    manifest: pd.DataFrame,
    data_root: str | Path,
    cache_dir: str | Path,
    device: torch.device,
    views: int,
) -> tuple[np.ndarray, float]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    model = MultimodalHAR(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    size_mb = parameter_size_mb(model)
    probabilities = np.zeros((len(manifest), 40), dtype=np.float64)
    for view in range(views):
        dataset = MultimodalDataset(
            manifest,
            data_root,
            cache_dir,
            split="test",
            image_size=config.image_size,
            visual_frames=config.visual_frames,
            sensor_steps=config.sensor_steps,
            training=False,
            normalizer=checkpoint["normalizer"],
            view_index=view,
            num_views=views,
            preserve_aspect_ratio=config.preserve_aspect_ratio,
            shared_visual_sampling=config.shared_visual_sampling,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
        )
        offset = 0
        for batch in tqdm(loader, desc=f"{Path(checkpoint_path).stem}:view{view + 1}", leave=False):
            batch = move_batch(batch, device)
            with torch.amp.autocast(
                device_type=device.type, enabled=config.amp and device.type == "cuda"
            ):
                output = torch.softmax(model(batch), dim=1).float().cpu().numpy()
            probabilities[offset : offset + len(output)] += output / views
            offset += len(output)
    return probabilities, size_mb


def ensemble_predict(
    checkpoints: list[str],
    manifest_path: str | Path,
    data_root: str | Path,
    cache_dir: str | Path,
    output_csv: str | Path,
    views: int = 3,
    device_name: str = "cuda",
    weights: list[float] | None = None,
) -> Path:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    manifest = pd.read_csv(manifest_path).fillna("")
    ensemble = np.zeros((len(manifest), 40), dtype=np.float64)
    total_size = 0.0
    normalized_weights = normalize_weights(len(checkpoints), weights)
    for checkpoint, weight in zip(checkpoints, normalized_weights, strict=True):
        probabilities, size_mb = predict_checkpoint(
            checkpoint, manifest, data_root, cache_dir, device, views
        )
        ensemble += probabilities * weight
        total_size += size_mb
    if total_size > 100:
        raise ValueError(f"Ensemble size {total_size:.2f} MB exceeds the 100 MB limit")
    output = Path(output_csv).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "path": manifest["submission_path"],
            "prediction": ensemble.argmax(axis=1).astype(int),
        }
    ).to_csv(output, index=False)
    print(f"Wrote {len(manifest)} predictions to {output}; ensemble_size_mb={total_size:.2f}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble CUHK-X fold checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", nargs="+", type=float)
    args = parser.parse_args()
    ensemble_predict(
        args.checkpoints,
        args.manifest,
        args.data_root,
        args.cache_dir,
        args.output,
        args.views,
        args.device,
        args.weights,
    )


if __name__ == "__main__":
    main()
