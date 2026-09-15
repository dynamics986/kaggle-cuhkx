"""Full-data thermal r3 training followed by official test submission generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cuhkx_sep12.common import check_size, digest, submit, write_json, write_predictions
from cuhkx_sep12.train import seed_all

from .data import index_thermal, split_digest
from .models import make_model
from .train import batches, predict, train_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="manifests/cv5/train.csv")
    parser.add_argument("--test-manifest", default="manifests/cv5/test.csv")
    parser.add_argument("--test-csv", default="../Small-Model-Track/Testing/test_file/test.csv")
    parser.add_argument("--train-root", default="../Small-Model-Track/Training/extracted/HAR/data")
    parser.add_argument("--test-root", default="../Small-Model-Track/Testing/data/small_model_track_test")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg["recipe"] != "r3_thermal_specialist" or cfg["improved"]:
        raise ValueError("Full training currently supports the unmodified r3 Thermal specialist only")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    seed_all(cfg["seed"])
    train = index_thermal(args.manifest, args.train_root, cfg["split"])
    if train.empty:
        raise ValueError("No decodable Thermal training clips")
    model = make_model(cfg["recipe"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=1e-6) if cfg["cosine"] else None
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda", init_scale=128.0)
    history = []
    loader = batches(train, cfg, training=True)
    for epoch in range(1, cfg["epochs"] + 1):
        metrics = train_epoch(model, loader, optimizer, scaler, cfg, device, f"r3_full epoch={epoch}/{cfg['epochs']}")
        if not metrics["optimizer_updates"]:
            raise RuntimeError("No optimizer update completed; inspect AMP overflow")
        if scheduler:
            scheduler.step()
        row = {"epoch": epoch, "train": metrics, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); write_json(output / "history.json", history)
        print(f"r3_full epoch={epoch}/{cfg['epochs']} train_loss={metrics['loss']:.4f} train_accuracy={metrics['accuracy']:.4f}", flush=True)
    checkpoint = {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "config": cfg, "training_rows": len(train), "split_digest": split_digest(train), "epoch": cfg["epochs"]}
    torch.save(checkpoint, output / "model.pt")
    model.load_state_dict(checkpoint["model"])
    test = index_thermal(args.test_manifest, args.test_root, test=True)
    p = predict(model, test, cfg, device, tta=True)
    write_predictions(test, p, output / "test_predictions.csv")
    submit(test, p, args.test_csv, output / "submission.csv")
    write_json(output / "summary.json", {"recipe": cfg["recipe"], "full_training": True, "config": cfg, "training_rows": len(train), "training_users": sorted(set(train.user)), "epoch": cfg["epochs"], "inference_bytes": check_size([output / "model.pt"]), "model_sha256": digest(output / "model.pt"), "manifest_sha256": digest(args.manifest), "device": str(device)})


if __name__ == "__main__":
    main()
