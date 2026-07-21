from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .constants import MODALITIES
from .data import MultimodalDataset
from .model import MultimodalHAR
from .splits import fold_partition
from .train import move_batch


@torch.inference_mode()
def evaluate_masked(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    disabled_index: int | None,
) -> float:
    model.eval()
    correct = 0
    count = 0
    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        if disabled_index is not None:
            batch["modality_mask"][:, disabled_index] = False
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(batch).argmax(dim=1)
        correct += int((predictions == batch["label"]).sum())
        count += len(predictions)
    return correct / count


def run_ablation(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    data_root: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    device_name: str = "cuda",
) -> dict[str, float]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    manifest = pd.read_csv(manifest_path).fillna("")
    _, valid = fold_partition(manifest, int(checkpoint["fold"]))
    dataset = MultimodalDataset(
        valid,
        data_root,
        cache_dir,
        split="train",
        image_size=config.image_size,
        visual_frames=config.visual_frames,
        sensor_steps=config.sensor_steps,
        training=False,
        normalizer=checkpoint["normalizer"],
        preserve_aspect_ratio=config.preserve_aspect_ratio,
        shared_visual_sampling=config.shared_visual_sampling,
    )
    device = torch.device(device_name)
    model = MultimodalHAR(config).to(device)
    model.load_state_dict(checkpoint["model"])
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    use_amp = config.amp and device.type == "cuda"
    results = {"full": evaluate_masked(model, loader, device, use_amp, None)}
    for index, modality in enumerate(MODALITIES):
        results[f"without_{modality}"] = evaluate_masked(model, loader, device, use_amp, index)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.6f} (delta={accuracy - results['full']:+.6f})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop-one-modality validation ablation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_ablation(
        args.checkpoint,
        args.manifest,
        args.data_root,
        args.cache_dir,
        args.output,
        args.device,
    )


if __name__ == "__main__":
    main()
