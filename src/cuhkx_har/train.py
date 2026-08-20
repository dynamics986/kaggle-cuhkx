from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import ExperimentConfig
from .data import MultimodalDataset, compute_sensor_normalizer
from .model import MultimodalHAR, parameter_size_mb
from .splits import fold_partition


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    progress_path: Path | None = None,
    epoch: int = 0,
    total_epochs: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    examples = 0
    epoch_started = time.time()
    progress = tqdm(loader, leave=False, desc="train" if training else "valid")
    for step, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        with (
            torch.set_grad_enabled(training),
            torch.amp.autocast(device_type=device.type, enabled=use_amp),
        ):
            logits = model(batch)
            raw_loss = criterion(logits, batch["label"])
            loss = raw_loss / grad_accum_steps if training else raw_loss
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        count = len(batch["label"])
        loss_sum += float(raw_loss.detach()) * count
        correct += int((logits.argmax(dim=1) == batch["label"]).sum())
        examples += count
        progress.set_postfix(loss=f"{loss_sum / max(examples, 1):.4f}")
        if progress_path is not None and (step == 1 or step % 5 == 0 or step == len(loader)):
            progress_path.write_text(
                json.dumps(
                    {
                        "phase": "train" if training else "valid",
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "step": step,
                        "total_steps": len(loader),
                        "loss": loss_sum / max(examples, 1),
                        "accuracy": correct / max(examples, 1),
                        "examples": examples,
                        "elapsed_seconds": time.time() - epoch_started,
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return {"loss": loss_sum / examples, "accuracy": correct / examples}


@torch.inference_mode()
def collect_validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, object]] = []
    for batch in tqdm(loader, leave=False, desc="collect-valid"):
        clip_ids = list(batch["clip_id"])
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            probabilities = torch.softmax(model(batch), dim=1).float().cpu().numpy()
        labels = batch["label"].cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for row_index, clip_id in enumerate(clip_ids):
            record: dict[str, object] = {
                "clip_id": clip_id,
                "label": int(labels[row_index]),
                "prediction": int(predictions[row_index]),
            }
            record.update(
                {f"prob_{label}": float(probabilities[row_index, label]) for label in range(40)}
            )
            records.append(record)
    return pd.DataFrame(records)


def train_fold(
    config: ExperimentConfig,
    manifest_path: str | Path,
    data_root: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    fold: int,
    device_name: str = "cuda",
    max_clips_per_class: int | None = None,
    resume_path: str | Path | None = None,
) -> Path:
    seed_everything(config.seed + fold)
    manifest = pd.read_csv(manifest_path).fillna("")
    train_frame, valid_frame = fold_partition(manifest, fold)
    if max_clips_per_class is not None:
        if max_clips_per_class <= 0:
            raise ValueError("max_clips_per_class must be positive")
        train_frame = (
            train_frame.sort_values("clip_id")
            .groupby("label", group_keys=False)
            .head(max_clips_per_class)
            .reset_index(drop=True)
        )
        valid_frame = (
            valid_frame.sort_values("clip_id")
            .groupby("label", group_keys=False)
            .head(max_clips_per_class)
            .reset_index(drop=True)
        )
    output = Path(output_dir).resolve() / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    normalizer = compute_sensor_normalizer(
        train_frame,
        cache_dir,
        split="train",
        imu_encoder=config.imu_encoder,
        imu_structured_cache_dir=config.imu_structured_cache_dir,
    )

    common = dict(
        data_root=data_root,
        cache_dir=cache_dir,
        split="train",
        image_size=config.image_size,
        visual_frames=config.visual_frames,
        sensor_steps=config.sensor_steps,
        normalizer=normalizer,
        horizontal_flip_probability=config.horizontal_flip_probability,
        preserve_aspect_ratio=config.preserve_aspect_ratio,
        shared_visual_sampling=config.shared_visual_sampling,
        imu_device_dropout=config.imu_device_dropout,
        visual_crop_mode=config.visual_crop_mode,
        visual_crop_metadata_path=(
            Path(config.visual_crop_metadata_root) / f"fold_{fold}" / "bboxes.json"
            if config.visual_crop_mode == "yolo_person"
            else None
        ),
        visual_crop_padding=config.visual_crop_padding,
        imu_encoder=config.imu_encoder,
        imu_structured_cache_dir=config.imu_structured_cache_dir,
    )
    train_set = MultimodalDataset(train_frame, training=True, **common)
    valid_set = MultimodalDataset(valid_frame, training=False, **common)
    generator = torch.Generator().manual_seed(config.seed + fold)
    loader_common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device_name.startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    sampler = None
    if config.class_balance_power > 0:
        class_counts = train_frame["label"].value_counts()
        sample_weights = train_frame["label"].map(
            lambda label: float(class_counts[label]) ** (-config.class_balance_power)
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights.to_numpy(copy=True), dtype=torch.double),
            num_samples=len(train_frame),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader_common)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_common)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    model = MultimodalHAR(config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    size_mb = parameter_size_mb(model)
    if size_mb > 100:
        raise ValueError(f"Model is {size_mb:.2f} MB, exceeding the 100 MB rule")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float | int]] = []
    best_accuracy = -math.inf
    epochs_without_improvement = 0
    start_epoch = 1
    checkpoint_path = output / "best.pt"
    last_checkpoint_path = output / "last.pt"
    progress_path = output / "progress.json"
    if resume_path is not None:
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        resumed_config = ExperimentConfig(**resume["config"]).to_dict()
        if resumed_config != config.to_dict() or int(resume["fold"]) != fold:
            raise ValueError("Resume checkpoint config or fold does not match this run")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = resume["history"]
        best_accuracy = float(resume["best_accuracy"])
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        start_epoch = int(resume["epoch"]) + 1
    started = time.time()
    print(f"fold={fold} train_users={sorted(train_frame['user'].unique())}")
    print(f"fold={fold} valid_users={sorted(valid_frame['user'].unique())}")
    print(f"model_size_mb={size_mb:.2f}, parameters={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            use_amp,
            optimizer,
            scaler,
            config.grad_accum_steps,
            progress_path,
            epoch,
            config.epochs,
        )
        valid_metrics = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
            use_amp,
            progress_path=progress_path,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        scheduler.step()
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if valid_metrics["accuracy"] > best_accuracy:
            best_accuracy = valid_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.to_dict(),
                    "fold": fold,
                    "valid_accuracy": best_accuracy,
                    "model_size_mb": size_mb,
                    "normalizer": normalizer,
                    "train_users": sorted(train_frame["user"].unique()),
                    "valid_users": sorted(valid_frame["user"].unique()),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config.to_dict(),
                "fold": fold,
                "normalizer": normalizer,
                "epoch": epoch,
                "history": history,
                "best_accuracy": best_accuracy,
                "epochs_without_improvement": epochs_without_improvement,
            },
            last_checkpoint_path,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping after epoch {epoch}")
            break

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    validation_predictions = collect_validation_predictions(model, valid_loader, device, use_amp)
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False)
    per_class = (
        validation_predictions.assign(correct=lambda frame: frame["label"] == frame["prediction"])
        .groupby("label")["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    per_class.to_csv(output / "per_class_accuracy.csv", index=False)
    summary = {
        "fold": fold,
        "imu_encoder": config.imu_encoder,
        "imu_structured_cache_dir": config.imu_structured_cache_dir,
        "best_valid_accuracy": best_accuracy,
        "model_size_mb": size_mb,
        "elapsed_minutes": (time.time() - started) / 60,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
        "checkpoint": str(checkpoint_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one leakage-safe CUHK-X subject fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-clips-per-class",
        type=int,
        help="Pipeline smoke test only: cap each partition after subject-safe splitting",
    )
    parser.add_argument("--resume", help="Resume from a matching last.pt checkpoint")
    args = parser.parse_args()
    train_fold(
        ExperimentConfig.load(args.config),
        args.manifest,
        args.data_root,
        args.cache_dir,
        args.output_dir,
        args.fold,
        args.device,
        args.max_clips_per_class,
        args.resume,
    )


if __name__ == "__main__":
    main()
