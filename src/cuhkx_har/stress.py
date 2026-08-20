"""Subject-held-out missingness and input-quality stress evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import MultimodalDataset
from .model import MultimodalHAR
from .splits import fold_partition
from .train import move_batch

STRESSES = (
    "none",
    "drop_thermal",
    "drop_radar",
    "drop_imu",
    "drop_skeleton",
    "visual_first_frame",
)


def apply_stress(
    batch: dict[str, Any], stress: str
) -> tuple[dict[str, Any], torch.Tensor]:
    """Return a copied batch and flags for samples meaningfully affected.

    Missing-modality cases remove a present token and its input.  The quality
    case preserves modality masks but collapses every available visual sequence
    to its first frame, exposing reliance on temporal visual evidence.
    """
    if stress not in STRESSES:
        raise ValueError(f"Unknown stress '{stress}'; expected one of {STRESSES}")
    result = dict(batch)
    mask = batch["modality_mask"].bool()
    if stress == "none":
        return result, torch.ones(len(mask), dtype=torch.bool, device=mask.device)
    if stress == "drop_thermal":
        result["visual"] = batch["visual"].clone()
        result["visual"][:, 2] = 0
        index = 2
    elif stress == "drop_skeleton":
        result["skeleton"] = torch.zeros_like(batch["skeleton"])
        index = 3
    elif stress == "drop_imu":
        result["imu"] = torch.zeros_like(batch["imu"])
        index = 4
    elif stress == "drop_radar":
        result["radar"] = torch.zeros_like(batch["radar"])
        index = 5
    else:  # visual_first_frame
        result["visual"] = batch["visual"].clone()
        result["visual"][:, :, 1:] = result["visual"][:, :, :1]
        return result, mask[:, :3].any(dim=1)
    result["modality_mask"] = mask.clone()
    result["modality_mask"][:, index] = False
    return result, mask[:, index]


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest: pd.DataFrame,
    data_root: str | Path,
    cache_dir: str | Path,
    device: torch.device,
    stresses: list[str],
) -> tuple[int, dict[str, dict[str, int]]]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fold = int(checkpoint["fold"])
    _, valid = fold_partition(manifest, fold)
    config = checkpoint["config"]
    dataset = MultimodalDataset(
        valid,
        data_root,
        cache_dir,
        split="train",
        image_size=int(config["image_size"]),
        visual_frames=int(config["visual_frames"]),
        sensor_steps=int(config["sensor_steps"]),
        training=False,
        normalizer=checkpoint["normalizer"],
        preserve_aspect_ratio=bool(config.get("preserve_aspect_ratio", False)),
        shared_visual_sampling=bool(config.get("shared_visual_sampling", False)),
        imu_encoder=str(config.get("imu_encoder", "tcn")),
        imu_structured_cache_dir=config.get("imu_structured_cache_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
    )
    # Rebuild via the dataclass to preserve defaults absent from older checkpoints.
    model = MultimodalHAR(ExperimentConfig(**config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"examples": 0, "correct": 0, "affected_examples": 0, "affected_correct": 0}
    )
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    for batch in tqdm(loader, desc=f"fold{fold}:{checkpoint_path.stem}", leave=False):
        batch = move_batch(batch, device)
        labels = batch["label"]
        for stress in stresses:
            stressed, affected = apply_stress(batch, stress)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(stressed).argmax(dim=1)
            correct = prediction.eq(labels)
            totals[stress]["examples"] += len(labels)
            totals[stress]["correct"] += int(correct.sum())
            totals[stress]["affected_examples"] += int(affected.sum())
            totals[stress]["affected_correct"] += int(correct[affected].sum())
    return fold, dict(totals)


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    affected = counts["affected_examples"]
    return {
        **counts,
        "accuracy": counts["correct"] / counts["examples"],
        "affected_accuracy": counts["affected_correct"] / affected if affected else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress held-out folds without using test labels")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--stresses", nargs="+", choices=STRESSES, default=list(STRESSES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = pd.read_csv(args.manifest).fillna("")
    device = torch.device(args.device)
    expected_folds = set(int(fold) for fold in manifest["fold"].unique())
    by_fold: dict[str, dict[str, dict[str, float | int]]] = {}
    pooled: dict[str, dict[str, int]] = defaultdict(
        lambda: {"examples": 0, "correct": 0, "affected_examples": 0, "affected_correct": 0}
    )
    for path in args.checkpoints:
        fold, counts = evaluate_checkpoint(
            path, manifest, args.data_root, args.cache_dir, device, args.stresses
        )
        if str(fold) in by_fold:
            raise ValueError(f"Duplicate checkpoint fold {fold}")
        by_fold[str(fold)] = {stress: _metrics(values) for stress, values in counts.items()}
        for stress, values in counts.items():
            for key, value in values.items():
                pooled[stress][key] += value
    if set(int(fold) for fold in by_fold) != expected_folds:
        raise ValueError("Checkpoints must cover every manifest fold exactly once")
    report = {
        "stresses": args.stresses,
        "pooled": {stress: _metrics(values) for stress, values in pooled.items()},
        "folds": by_fold,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for stress in args.stresses:
        metrics = report["pooled"][stress]
        print(
            f"{stress}: accuracy={metrics['accuracy']:.5f}; "
            f"affected_accuracy={metrics['affected_accuracy']:.5f} "
            f"({metrics['affected_examples']} affected)"
        )


if __name__ == "__main__":
    main()
