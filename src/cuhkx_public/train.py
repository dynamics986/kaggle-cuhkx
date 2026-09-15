"""Train or fine-tune either supplied Thermal notebook, with explicit reproduction deviations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cuhkx_sep12.common import check_size, digest, submit, write_json, write_predictions
from cuhkx_sep12.train import seed_all, seed_worker

from .data import ThermalDataset, index_thermal, split_digest
from .models import make_model


def batches(table, cfg, training=False, view=0):
    return DataLoader(
        ThermalDataset(table, cfg, training, view),
        batch_size=cfg["micro_batch"],
        shuffle=training,
        num_workers=cfg["workers"],
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg["seed"] + view),
    )


def train_epoch(model, loader, optimizer, scaler, cfg, device, description):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    losses = correct = count = bad = updates = 0
    optimizer.zero_grad(set_to_none=True)
    accumulation = max(1, cfg["effective_batch"] // cfg["micro_batch"])
    for step, (x, y, missing) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        # Original specialist BatchNorm1d cannot train with a final batch of one.
        singleton_bn = []
        if len(y) == 1:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d) and module.training:
                    module.eval()
                    singleton_bn.append(module)
        # Account for a final, possibly partial accumulation group by sample count.
        start = ((step - 1) // accumulation) * accumulation * cfg["micro_batch"]
        group_count = min(cfg["effective_batch"], len(loader.dataset) - start)
        with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss * len(y) / group_count).backward()
        for module in singleton_bn:
            module.train()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            if cfg["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(scaler.get_scale() >= old_scale)
            optimizer.zero_grad(set_to_none=True)
        count += len(y)
        losses += float(loss.detach()) * len(y)
        correct += int((logits.argmax(1) == y).sum())
        bad += int(missing.sum())
    return {
        "loss": losses / count,
        "accuracy": correct / count,
        "replaced_frames": bad,
        "optimizer_updates": updates,
    }


@torch.inference_mode()
def predict(model, table, cfg, device, tta=False):
    model.eval()
    total = np.zeros((len(table), 40), np.float64)
    views = cfg["tta"] if tta else 1
    for view in range(views):
        random.seed(cfg["seed"] + view)
        np.random.seed(cfg["seed"] + view)
        offset = 0
        for x, _, _ in batches(table, cfg, view=view):
            x = x.to(device)
            # Notebook specialist inference is fp32; preserve that behavior.
            logits = model(x)
            if cfg["recipe"] == "r3_thermal_specialist" and view == 1:
                logits = (logits + model(x.flip(-1))) / 2
            total[offset : offset + len(x)] += logits.float().cpu().numpy()
            offset += len(x)
        print(f"inference view={view + 1}/{views} clips={offset}", flush=True)
    if cfg["recipe"] != "r3_thermal_specialist":
        total /= views
    shifted = total - total.max(1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(1, keepdims=True)


def fit(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("epochs", "micro_batch", "workers"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if cfg["effective_batch"] % cfg["micro_batch"]:
        raise ValueError("micro-batch must divide effective_batch")
    if args.split:
        cfg["split"] = args.split
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed_all(cfg["seed"])
    table = index_thermal(args.manifest, args.train_root, cfg["split"])
    identity = split_digest(table)
    train = table.loc[table.fold != args.fold].reset_index(drop=True)
    valid = table.loc[table.fold == args.fold].reset_index(drop=True)
    if args.smoke:
        train, valid = train.head(4), valid.head(3)
        cfg.update(epochs=1, workers=0)
    if train.empty or valid.empty:
        raise ValueError("Empty training/validation partition")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = make_model(cfg["recipe"]).to(device)
    initialization = None
    if cfg["improved"]:
        if not args.init:
            raise ValueError("Fine-tuning requires --init from the matching reproduction fold")
        parent = torch.load(args.init, map_location="cpu", weights_only=True)
        expected = (
            "r1_thermal_baseline" if cfg["recipe"].startswith("i1") else "r3_thermal_specialist"
        )
        if (
            parent["config"]["recipe"] != expected
            or parent["fold"] != args.fold
            or parent["split_digest"] != identity
            or parent.get("smoke", False) != args.smoke
        ):
            raise ValueError(
                "Fine-tuning checkpoint has incompatible recipe, split, fold or smoke mode"
            )
        incompatible = model.load_state_dict(parent["model"], strict=False)
        allowed = (
            ("temporal.", "attention.") if cfg["recipe"].startswith("i1") else ("channel_gate.",)
        )
        if incompatible.unexpected_keys or any(
            not k.startswith(allowed) for k in incompatible.missing_keys
        ):
            raise ValueError(f"Unexpected checkpoint mismatch: {incompatible}")
        initialization = {"path": str(Path(args.init).resolve()), "sha256": digest(args.init)}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=1e-6)
        if cfg["cosine"]
        else None
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=cfg["amp"] and device.type == "cuda", init_scale=128.0
    )
    history, best = [], -1.0
    early_stopping_patience = int(cfg.get("early_stopping_patience", 0))
    stale_epochs = 0
    loader = batches(train, cfg, training=True)
    for epoch in range(1, cfg["epochs"] + 1):
        metrics = train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            cfg,
            device,
            f"{cfg['recipe']} fold={args.fold} epoch={epoch}/{cfg['epochs']}",
        )
        if not metrics["optimizer_updates"]:
            raise RuntimeError(
                "No optimizer update completed; inspect AMP overflow before continuing"
            )
        if scheduler:
            scheduler.step()
        probabilities = predict(model, valid, cfg, device)
        accuracy = float((probabilities.argmax(1) == valid.label.to_numpy()).mean())
        history.append({"epoch": epoch, "train": metrics, "validation_accuracy": accuracy})
        write_json(output / "history.json", history)
        print(f"fold={args.fold} epoch={epoch} validation_accuracy={accuracy:.4f}", flush=True)
        checkpoint = {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": cfg,
            "fold": args.fold,
            "split_digest": identity,
            "epoch": epoch,
            "validation_accuracy": accuracy,
            "smoke": args.smoke,
        }
        improved = accuracy > best
        if improved:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if improved or cfg["checkpoint"] == "last":
            best = accuracy
            torch.save(checkpoint, output / "model.pt")
        if early_stopping_patience and stale_epochs >= early_stopping_patience:
            print(
                f"{cfg['recipe']} fold={args.fold} early_stopping "
                f"epoch={epoch} stale_epochs={stale_epochs} patience={early_stopping_patience}",
                flush=True,
            )
            break
    selected = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    probabilities = predict(model, valid, cfg, device, tta=True)
    write_predictions(valid, probabilities, output / "validation_predictions.csv")
    size = check_size([output / "model.pt"])
    reproduction_notes = [
        "Selected folds only, not original full fold run",
        "Bad images replaced with nearest decodable frame; missing clip uses black frames",
        "Final singleton classifier BN uses running statistics",
    ]
    if cfg["micro_batch"] != cfg["effective_batch"]:
        reproduction_notes.insert(
            1,
            "Microbatch gradient accumulation changes BatchNorm statistics versus the notebook batch size",
        )
    summary = {
        "recipe": cfg["recipe"],
        "config": cfg,
        "fold": args.fold,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_users": sorted(set(train.user)),
        "validation_users": sorted(set(valid.user)),
        "selected_epoch": selected["epoch"],
        "single_view_accuracy": selected["validation_accuracy"],
        "validation_accuracy": float((probabilities.argmax(1) == valid.label.to_numpy()).mean()),
        "split_digest": identity,
        "initialization": initialization,
        "inference_bytes": size,
        "model_sha256": digest(output / "model.pt"),
        "smoke": args.smoke,
        "reproduction_notes": reproduction_notes,
        "torch": torch.__version__,
        "device": str(device),
    }
    write_json(output / "summary.json", summary)
    # Validation artifacts survive independently if test decoding/inference later fails.
    test = index_thermal(args.test_manifest, args.test_root, test=True)
    if args.smoke:
        test = test.head(2)
    p_test = predict(model, test, cfg, device, tta=True)
    write_predictions(test, p_test, output / "test_predictions.csv")
    official = args.test_csv
    if args.smoke:
        official = output / "smoke_official.csv"
        test[["submission_path"]].rename(columns={"submission_path": "path"}).to_csv(
            official, index=False
        )
    submit(test, p_test, official, output / "submission.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init")
    parser.add_argument("--split", choices=["notebook", "frozen"])
    parser.add_argument("--manifest", default="manifests/cv5/train.csv")
    parser.add_argument("--test-manifest", default="manifests/cv5/test.csv")
    parser.add_argument("--test-csv", default="../Small-Model-Track/Testing/test_file/test.csv")
    parser.add_argument("--train-root", default="../Small-Model-Track/Training/extracted/HAR/data")
    parser.add_argument(
        "--test-root", default="../Small-Model-Track/Testing/data/small_model_track_test"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if any(
        getattr(args, key) is not None and getattr(args, key) < minimum
        for key, minimum in [("micro_batch", 1), ("workers", 0), ("epochs", 1)]
    ):
        parser.error("Invalid training configuration")
    fit(args)


if __name__ == "__main__":
    main()
