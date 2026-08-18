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

from cuhkx_har.splits import fold_partition
from cuhkx_har.train import seed_worker

from .config import ModalityConfig
from .constants import MODALITIES, NUM_CLASSES
from .data import ModalityDataset, compute_normalizer, select_present_rows
from .frozen import read_frozen_manifest
from .model import ModalityHAR, parameter_size_mb


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, use_amp: bool,
    optimizer: torch.optim.Optimizer | None = None, scaler: torch.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = examples = 0
    for step, batch in enumerate(loader, start=1):
        inputs, labels = batch["inputs"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            raw_loss = criterion(logits, labels)
            loss = raw_loss / grad_accum_steps if training else raw_loss
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        count = len(labels)
        loss_sum += float(raw_loss.detach()) * count
        correct += int((logits.argmax(dim=1) == labels).sum())
        examples += count
    return {"loss": loss_sum / examples, "accuracy": correct / examples}


@torch.inference_mode()
def validation_predictions(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> pd.DataFrame:
    model.eval()
    records: list[dict[str, object]] = []
    for batch in loader:
        clip_ids = list(batch["clip_id"])
        inputs, labels = batch["inputs"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            probabilities = torch.softmax(model(inputs), dim=1).float().cpu().numpy()
        for index, clip_id in enumerate(clip_ids):
            row: dict[str, object] = {"clip_id": clip_id, "label": int(labels[index]), "prediction": int(probabilities[index].argmax())}
            row.update({f"prob_{label}": float(probabilities[index, label]) for label in range(NUM_CLASSES)})
            records.append(row)
    return pd.DataFrame(records)


def train_fold(
    config: ModalityConfig, modality: str, manifest_path: str | Path, data_root: str | Path,
    cache_dir: str | Path, output_dir: str | Path, fold: int, device_name: str = "cuda",
    max_clips_per_class: int | None = None,
) -> Path:
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}; choose from {MODALITIES}")
    if fold not in range(5):
        raise ValueError("fold must be in [0, 4] for frozen CV5")
    seed_everything(config.seed + fold)
    manifest = read_frozen_manifest(manifest_path)
    train_frame, valid_frame = fold_partition(manifest, fold)
    train_frame = select_present_rows(train_frame, modality, cache_dir)
    valid_frame = select_present_rows(valid_frame, modality, cache_dir)
    if max_clips_per_class is not None:
        if max_clips_per_class <= 0:
            raise ValueError("max_clips_per_class must be positive")
        train_frame = train_frame.sort_values("clip_id").groupby("label", group_keys=False).head(max_clips_per_class).reset_index(drop=True)
        valid_frame = valid_frame.sort_values("clip_id").groupby("label", group_keys=False).head(max_clips_per_class).reset_index(drop=True)
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"{modality} has no usable training or validation clips in fold {fold}")
    output = Path(output_dir).resolve() / modality / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    normalizer = compute_normalizer(train_frame, modality, cache_dir)
    common = dict(modality=modality, data_root=data_root, cache_dir=cache_dir, image_size=config.image_size,
                  visual_frames=config.visual_frames, sensor_steps=config.sensor_steps, normalizer=normalizer,
                  horizontal_flip_probability=config.horizontal_flip_probability)
    train_set, valid_set = ModalityDataset(train_frame, training=True, **common), ModalityDataset(valid_frame, training=False, **common)
    generator = torch.Generator().manual_seed(config.seed + fold)
    loader_common = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=device_name.startswith("cuda"), worker_init_fn=seed_worker, generator=generator,
                         persistent_workers=config.num_workers > 0)
    sampler = None
    if config.class_balance_power > 0:
        counts = train_frame["label"].value_counts()
        weights = train_frame["label"].map(lambda label: float(counts[label]) ** (-config.class_balance_power))
        sampler = WeightedRandomSampler(torch.as_tensor(weights.to_numpy(), dtype=torch.double), len(train_frame), replacement=True, generator=generator)
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader_common)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_common)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    amp_enabled = config.amp and device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    print(
        json.dumps(
            {"device": str(device), "gpu_name": gpu_name, "amp_enabled": amp_enabled}
        ),
        flush=True,
    )
    model = ModalityHAR(modality, config).to(device)
    size_mb = parameter_size_mb(model)
    if size_mb > 100:
        raise ValueError(f"Model is {size_mb:.2f} MB, exceeding the 100 MB rule")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best, stale, history, started = -math.inf, 0, [], time.time()
    checkpoint = output / "best.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, amp_enabled, optimizer, scaler,
            config.grad_accum_steps,
        )
        valid_metrics = run_epoch(model, valid_loader, criterion, device, amp_enabled)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "valid_loss": valid_metrics["loss"], "valid_accuracy": valid_metrics["accuracy"], "learning_rate": scheduler.get_last_lr()[0]}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if valid_metrics["accuracy"] > best:
            best, stale = valid_metrics["accuracy"], 0
            torch.save({"model": model.state_dict(), "config": config.to_dict(), "modality": modality, "fold": fold,
                        "normalizer": normalizer, "valid_accuracy": best, "model_size_mb": size_mb}, checkpoint)
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    validation_predictions(model, valid_loader, device, amp_enabled).to_csv(
        output / "validation_predictions.csv", index=False
    )
    summary = {
        "modality": modality,
        "fold": fold,
        "device": str(device),
        "gpu_name": gpu_name,
        "amp_enabled": amp_enabled,
        "best_valid_accuracy": best,
        "model_size_mb": size_mb,
        "train_examples": len(train_frame),
        "valid_examples": len(valid_frame),
        "elapsed_minutes": (time.time() - started) / 60,
        "checkpoint": str(checkpoint),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one frozen-CV5 CUHK-X single-modality fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modality", required=True, choices=MODALITIES)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/modality")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-clips-per-class", type=int)
    args = parser.parse_args()
    train_fold(ModalityConfig.load(args.config), args.modality, args.manifest, args.data_root, args.cache_dir, args.output_dir, args.fold, args.device, args.max_clips_per_class)


if __name__ == "__main__":
    main()
