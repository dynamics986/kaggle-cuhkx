"""Complete selected-fold training, validation and official submission for methods 2-8."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cuhkx_har.splits import fold_partition
from cuhkx_modality.frozen import read_frozen_manifest

from .common import check_size, digest, submit, write_json, write_predictions
from .data import CroppedDataset, validate_cache
from .model import VisualHAR
from .prepare import audit_detector


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(rows, cache, split, config, training=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        CroppedDataset(rows, cache, split, training, config.get("augmentation")),
        batch_size=config["batch_size"],
        shuffle=training,
        num_workers=config["workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def epoch(model, batches, device, optimizer=None, scaler=None, context=""):
    training = optimizer is not None
    model.train(training)
    total, losses, correct, all_probabilities = 0, 0.0, 0, []
    with torch.set_grad_enabled(training):
        for batch_number, (images, mask, labels) in enumerate(batches, 1):
            images, mask, labels = images.to(device), mask.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(images, mask)
                loss = model.loss(output, labels) if training or (labels >= 0).all() else None
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            p = output["probabilities"].detach().float().cpu().numpy()
            all_probabilities.append(p)
            correct += int((p.argmax(1) == labels.cpu().numpy()).sum())
            if loss is not None:
                losses += float(loss.detach()) * len(labels)
            total += len(labels)
    if not total:
        raise ValueError("Empty data partition")
    return {"loss": losses / total, "accuracy": correct / total}, np.concatenate(all_probabilities)


def fit_fold(args, config, manifest, fold, device):
    seed_all(config["seed"])
    cache = Path(args.cache_root) / f"fold_{fold}"
    detector = Path(args.detector2 if fold == 2 else args.detector4)
    audit_detector(detector, manifest, str(fold))
    meta = validate_cache(cache, "train", args.manifest, fold, detector, config)
    train, valid = fold_partition(manifest, fold)
    output = Path(args.output) / f"fold_{fold}"
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        if (
            summary["config"] != config
            or summary["crop_metadata"] != meta
            or summary["manifest_sha256"] != digest(args.manifest)
        ):
            raise ValueError("Existing run differs from requested configuration; use new output")
        if not args.reuse_completed:
            raise FileExistsError(f"{output} completed; use --reuse-completed or a new --output")
        if digest(output / "best.pt") != summary["checkpoint_sha256"]:
            raise ValueError("Existing checkpoint hash changed")
        return summary
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete run at {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    model = VisualHAR(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    train_loader = loader(train, cache, "train", config, True)
    valid_loader = loader(valid, cache, "train", config)
    best, best_epoch, history = -1.0, 0, []
    for number in range(1, config["epochs"] + 1):
        train_metrics, _ = epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            context=f"method={config['method']} fold={fold} epoch={number} train",
        )
        valid_metrics, probabilities = epoch(
            model,
            valid_loader,
            device,
            context=f"method={config['method']} fold={fold} epoch={number} validation",
        )
        scheduler.step()
        history.append({"epoch": number, "train": train_metrics, "validation": valid_metrics})
        write_json(output / "history.json", history)
        print(
            f"method={config['method']} fold={fold} epoch={number}/{config['epochs']} "
            f"train_loss={train_metrics['loss']:.4f} train_accuracy={train_metrics['accuracy']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_accuracy={valid_metrics['accuracy']:.4f}",
            flush=True,
        )
        if valid_metrics["accuracy"] > best:
            best, best_epoch = valid_metrics["accuracy"], number
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": config,
                    "fold": fold,
                    "crop_metadata": meta,
                },
                output / "best.pt",
            )
            write_predictions(valid, probabilities, output / "validation_predictions.csv")
        if number - best_epoch >= config["patience"]:
            break
    size = check_size([output / "best.pt", detector])
    summary = {
        "fold": fold,
        "method": config["method"],
        "config": config,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_users": sorted(set(train.user)),
        "validation_users": sorted(set(valid.user)),
        "best_epoch": best_epoch,
        "validation_accuracy": best,
        "inference_bytes_including_detector": size,
        "checkpoint_sha256": digest(output / "best.pt"),
        "manifest_sha256": digest(args.manifest),
        "crop_metadata": meta,
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(output / "summary.json", summary)
    return summary


def predict_fold(args, config, test, fold, device):
    output = Path(args.output) / f"fold_{fold}"
    detector = Path(args.detector2 if fold == 2 else args.detector4)
    cache = Path(args.cache_root) / f"fold_{fold}"
    test_meta = validate_cache(cache, "test", args.test_manifest, fold, detector, config)
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["fold"] != fold:
        raise ValueError("Checkpoint configuration/fold mismatch")
    summary = json.loads((output / "summary.json").read_text())
    if summary["checkpoint_sha256"] != digest(output / "best.pt"):
        raise ValueError("Checkpoint hash differs from training summary")
    for key in (
        "detector_sha256",
        "frames",
        "image_size",
        "confidence",
        "padding",
        "missing_policy",
        "alignment",
    ):
        if test_meta.get(key) != checkpoint["crop_metadata"].get(key):
            raise ValueError(f"Train/test crop setting differs: {key}")
    model = VisualHAR(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    _, probabilities = epoch(model, loader(test, cache, "test", config), device)
    check_size([output / "best.pt", detector])
    write_predictions(test, probabilities, output / "test_predictions.csv")
    submit(test, probabilities, args.test_csv, output / "submission.csv")
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default="manifests/cv5/train.csv")
    parser.add_argument("--test-manifest", default="manifests/cv5/test.csv")
    parser.add_argument("--test-csv", default="../Small-Model-Track/Testing/test_file/test.csv")
    parser.add_argument("--cache-root", default="cache-sep12")
    parser.add_argument("--detector2", default="artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt")
    parser.add_argument("--detector4", default="artifacts/sep12/yolo/fold_4/yolov8n_4.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", nargs="+", type=int, choices=[2, 4], default=[2, 4])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--stage", choices=["all", "train", "predict"], default="all")
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds):
        parser.error("Duplicate folds")
    config = json.loads(Path(args.config).read_text())
    if config["method"] not in range(2, 9):
        parser.error("Neural method must be 2..8")
    if min(config[k] for k in ("epochs", "frames", "batch_size", "patience")) < 1:
        parser.error("epochs, frames, batch_size and patience must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    device = torch.device(args.device)
    manifest = read_frozen_manifest(args.manifest)
    summaries = []
    for fold in args.folds:
        if args.stage != "predict":
            summaries.append(fit_fold(args, config, manifest, fold, device))
        else:
            summary = json.loads((Path(args.output) / f"fold_{fold}/summary.json").read_text())
            if summary["config"] != config or summary["manifest_sha256"] != digest(args.manifest):
                raise ValueError("Prediction run configuration/manifest mismatch")
            summaries.append(summary)
    # Partial CV only: no five-fold OOF claims and no five-model ensemble.
    selected = max(summaries, key=lambda x: (x["validation_accuracy"], -x["fold"]))["fold"]
    report = {
        "evaluation": "selected-fold validation (not full five-fold OOF)",
        "folds": args.folds,
        "fold_results": summaries,
        "deployment_fold": selected,
        "selection": "highest selected-fold validation accuracy; ties use smaller fold",
        "weighted_validation_accuracy": sum(
            s["validation_accuracy"] * s["validation_rows"] for s in summaries
        )
        / sum(s["validation_rows"] for s in summaries),
    }
    write_json(Path(args.output) / "comparison.json", report)
    if args.stage != "train":
        test = pd.read_csv(args.test_manifest).fillna("")
        for fold in args.folds:
            probabilities = predict_fold(args, config, test, fold, device)
            if fold == selected:
                submit(test, probabilities, args.test_csv, Path(args.output) / "submission.csv")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
