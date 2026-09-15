"""Port of YOLO v9 inference plus frozen-backbone classifier fine-tuning.

The supplied notebook does NOT contain its training recipe or original subject splits.
Public-backbone fine-tuning validation is explicitly diagnostic, not leakage-safe CV.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cuhkx_modality.frozen import read_frozen_manifest
from cuhkx_sep12.common import cache_path, check_size, digest, submit, write_json, write_predictions

from .data import files_in, open_rgb

MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)
CLASSIFIER_ASSETS = ("ensemble_packed.pt",)
DEFAULT_DETECTOR = "artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt"
DEFAULT_MODEL_DEFINITION = Path(__file__).with_name("ig65m_models.py")


def require_assets(root):
    missing = [name for name in CLASSIFIER_ASSETS if not (Path(root) / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "YOLO notebook classifier assets missing: "
            + ", ".join(missing)
            + f". Put the ORIGINAL attached Kaggle assets in {root}. "
            "The notebook contains inference only; no substitute classifier weights are used."
        )


def unpack_signed(packed, shape, bits):
    if not 2 <= bits <= 8:
        raise ValueError("Unsupported packed precision")
    count = math.prod(shape)
    starts = torch.arange(count, dtype=torch.int64) * bits
    codes = torch.zeros(count, dtype=torch.int16)
    source = packed.to(torch.int16)
    for bit in range(bits):
        positions = starts + bit
        codes |= ((source[positions >> 3] >> (positions & 7)) & 1) << bit
    sign, modulus = 1 << (bits - 1), 1 << bits
    return torch.where(codes >= sign, codes - modulus, codes).to(torch.int8).reshape(shape)


def dequantize_state(state):
    output = {}
    for key, value in state.items():
        if isinstance(value, Mapping):
            shape = tuple(int(item) for item in value["shape"])
            output[key] = (
                unpack_signed(value["packed"], shape, int(value["bits"])).float()
                * value["scale"].float()
            )
        else:
            output[key] = value.float() if value.is_floating_point() else value
    return output


def adapt_input_conv(conv, channels=4):
    replacement = type(conv)(
        channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        replacement.weight[:, :3] = conv.weight
        replacement.weight[:, 3:] = conv.weight.mean(1, keepdim=True).expand(
            -1, channels - 3, -1, -1, -1
        )
        if conv.bias is not None:
            replacement.bias.copy_(conv.bias)
    return replacement


class R2Plus1D34(nn.Module):
    def __init__(self, source):
        super().__init__()
        spec = importlib.util.spec_from_file_location("ig65m_models", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        network = module.r2plus1d_34_32_kinetics(num_classes=400, pretrained=False)
        network.stem[0] = adapt_input_conv(network.stem[0])
        features = network.fc.in_features
        network.fc = nn.Identity()
        self.encoder = network
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features, 40))

    def features(self, x):
        return self.encoder(x.permute(0, 2, 1, 3, 4))

    def forward(self, x):
        return self.head(self.features(x))


def window_from_boxes(boxes):
    values = np.asarray(boxes, dtype=np.float64)
    cx = float(np.median((values[:, 0] + values[:, 2]) / 2))
    cy = float(np.median((values[:, 1] + values[:, 3]) / 2))
    side = max(
        max((values[:, 2] - values[:, 0]).max(), (values[:, 3] - values[:, 1]).max()) * 1.4 * 640,
        0.35 * 640,
    )
    hx, hy = side / 640 / 2, side / 480 / 2
    return max(cx - hx, 0), max(cy - hy, 0), min(cx + hx, 1), min(cy + hy, 1)


def pick_indices(length, count):
    return np.linspace(0, length - 1, count).round().astype(int) if length else []


def prepare_frames(table, root, weights, cache, split, device):
    from ultralytics import YOLO

    identity = {
        "detector_sha256": digest(weights),
        "frames": 16,
        "size": 128,
        "margin": 1.4,
        "rows": table.clip_id.tolist(),
        "paths": table[["Depth_Color", "IR"]].fillna("").values.tolist(),
        "root": str(Path(root).resolve()),
        "pipeline": "notebook_v9_median_ir_depth_full_fallback_v1",
    }
    meta_path = Path(cache) / f"{split}_meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("YOLO cache provenance changed; use a new cache directory")
    write_json(meta_path, identity)
    detector = YOLO(str(weights))
    coverage = []
    for number, row in enumerate(table.fillna("").itertuples(index=False), 1):
        target = cache_path(cache, split, row.clip_id)
        if target.is_file():
            with np.load(target, allow_pickle=False) as saved:
                if (
                    saved["images"].shape != (16, 4, 128, 128)
                    or str(saved["clip_id"]) != row.clip_id
                ):
                    raise ValueError(f"Invalid YOLO cache: {target}")
                coverage.append(
                    {
                        "clip_id": row.clip_id,
                        "has_crop": bool(saved["has_crop"]),
                        "bad_frames": int(saved["bad_frames"]),
                    }
                )
            continue
        streams = [
            files_in(Path(root) / getattr(row, m)) if getattr(row, m) else []
            for m in ("Depth_Color", "IR")
        ]
        boxes = []
        for modality in (1, 0):
            probes = []
            for index in sorted(set(pick_indices(len(streams[modality]), 8))):
                image = open_rgb(streams[modality][index])
                if image is not None and np.asarray(image).any():
                    # Preserve the notebook RGB-ndarray channel convention for Ultralytics.
                    probes.append(np.asarray(image))
            if probes:
                results = detector.predict(
                    probes, classes=[0], conf=0.25, verbose=False, device=device, batch=8
                )
                for result in results:
                    h, w = result.orig_shape
                    for x1, y1, x2, y2 in result.boxes.xyxy.tolist():
                        boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))
            if boxes:
                break
        window = window_from_boxes(boxes) if boxes else None
        images = np.zeros((16, 4, 128, 128), np.uint8)
        bad = 0
        for m, files in enumerate(streams):
            if not files:
                bad += 16
                continue
            for frame, selected in enumerate(pick_indices(len(files), 16)):
                image = open_rgb(files[selected])
                if image is None:
                    bad += 1
                    continue  # Source notebook uses zero frames for undecodable PNGs.
                if m == 1:
                    image = image.convert("L")
                if window is not None:
                    w, h = image.size
                    bounds = tuple(
                        round(v * (w if i % 2 == 0 else h)) for i, v in enumerate(window)
                    )
                    image = image.crop(bounds)
                array = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR))
                if not array.any():
                    bad += 1
                    continue
                if m == 0:
                    images[frame, :3] = array.transpose(2, 0, 1)
                else:
                    images[frame, 3] = array
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                images=images,
                has_crop=window is not None,
                bad_frames=bad,
                clip_id=row.clip_id,
            )
        temporary.replace(target)
        coverage.append({"clip_id": row.clip_id, "has_crop": window is not None, "bad_frames": bad})
        if number % 25 == 0:
            print(f"YOLO {split}: {number}/{len(table)} clips", flush=True)
    pd.DataFrame(coverage).to_csv(Path(cache) / f"{split}_coverage.csv", index=False)


class YoloDataset(Dataset):
    def __init__(self, table, cache, split):
        self.table, self.cache, self.split = table.reset_index(drop=True), cache, split
        self.mean = torch.tensor((*MEAN, sum(MEAN) / 3)).view(1, 4, 1, 1)
        self.std = torch.tensor((*STD, sum(STD) / 3)).view(1, 4, 1, 1)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        with np.load(cache_path(self.cache, self.split, self.table.iloc[index].clip_id)) as data:
            image = torch.from_numpy(data["images"].copy()).float() / 255
        return (image - self.mean) / self.std, index


def views(x):
    # Preserve source circular temporal roll exactly for reproduction.
    return (x, x.flip(-1), torch.roll(x, 1, 1), torch.roll(x, -1, 1))


def load_checkpoint(assets, detector):
    require_assets(assets)
    if not Path(detector).is_file():
        raise FileNotFoundError(f"Detector weight does not exist: {detector}")
    if not Path(detector).is_file():
        raise FileNotFoundError(f"Detector weight does not exist: {detector}")
    checkpoint = torch.load(
        Path(assets) / "ensemble_packed.pt", map_location="cpu", weights_only=True
    )
    if (
        checkpoint["schema_version"] != "kuno-yolo-r2p1d-packed-ensemble/v1"
        or checkpoint["bits"] != [5, 6]
        or checkpoint["folds"] != [0, 1]
        or checkpoint["weights"] != [0.5, 0.5]
        or len(checkpoint["models_packed"]) != 2
    ):
        raise ValueError("Unexpected public checkpoint schema")
    check_size([Path(assets) / "ensemble_packed.pt", detector])
    return checkpoint


@torch.inference_mode()
def features_for(model, table, cache, split, device, micro_batch):
    model.eval()
    output = None
    loader = DataLoader(YoloDataset(table, cache, split), batch_size=micro_batch, num_workers=0)
    for x, indices in loader:
        x = x.to(device)
        encoded = torch.stack([model.features(view) for view in views(x)], 1).float().cpu().numpy()
        if output is None:
            output = np.zeros((len(table), 4, encoded.shape[-1]), np.float32)
        output[indices.numpy()] = encoded
        print(f"{split} backbone features: {int(indices[-1]) + 1}/{len(table)}", flush=True)
    return output


def fit_head(initial, features, labels, epochs, seed=2026):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features.shape[-1], 40))
    head.load_state_dict(initial)
    anchor = {k: v.detach().clone() for k, v in head.named_parameters()}
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-3)
    x, y = torch.from_numpy(features), torch.tensor(labels, dtype=torch.long)
    for epoch in range(epochs):
        head.train()
        total = correct = 0
        for indices in torch.randperm(len(y)).split(64):
            selected = torch.randint(0, 4, (len(indices),))
            logits = head(x[indices, selected])
            loss = nn.functional.cross_entropy(logits, y[indices], label_smoothing=0.05)
            loss = loss + 0.01 * sum(
                (p - anchor[k]).square().mean() for k, p in head.named_parameters()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(indices)
            correct += int((logits.argmax(1) == y[indices]).sum())
        print(
            f"head fine-tune epoch={epoch + 1}/{epochs} loss={total / len(y):.4f} "
            f"accuracy={correct / len(y):.4f}",
            flush=True,
        )
    return head.eval()


@torch.inference_mode()
def head_logits(head, features):
    x = torch.from_numpy(features)
    return head(x.reshape(-1, x.shape[-1])).reshape(len(x), 4, 40).mean(1).numpy()


def softmax(logits):
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(1, keepdims=True)


def run(args):
    assets, output = Path(args.assets), Path(args.output)
    detector = Path(args.detector)
    checkpoint = load_checkpoint(assets, detector)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    test = pd.read_csv("manifests/cv5/test.csv").fillna("")
    train = read_frozen_manifest("manifests/cv5/train.csv") if args.improve else None
    prepare_frames(
        test,
        args.test_root,
        detector,
        args.cache,
        "test",
        "0" if args.device == "cuda" else "cpu",
    )
    if train is not None:
        prepare_frames(
            train,
            args.train_root,
            detector,
            args.cache,
            "train",
            "0" if args.device == "cuda" else "cpu",
        )
    test_logits = np.zeros((len(test), 40), np.float64)
    diagnostic = (
        {fold: np.zeros((int((train.fold == fold).sum()), 40), np.float64) for fold in (2, 4)}
        if train is not None
        else {}
    )
    heads = []
    for index, packed in enumerate(checkpoint["models_packed"]):
        model = R2Plus1D34(args.model_definition)
        model.load_state_dict(dequantize_state(packed))
        model.to(device).eval()
        test_features = features_for(model, test, args.cache, "test", device, args.micro_batch)
        initial = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
        if args.improve:
            train_features = features_for(
                model, train, args.cache, "train", device, args.micro_batch
            )
            for fold in (2, 4):
                tr, va = (train.fold != fold).to_numpy(), (train.fold == fold).to_numpy()
                print(
                    f"public model={index} diagnostic fold={fold} (NOT independent CV)", flush=True
                )
                head = fit_head(
                    initial, train_features[tr], train.loc[tr, "label"].to_numpy(), args.epochs
                )
                diagnostic[fold] += 0.5 * head_logits(head, train_features[va])
            head = fit_head(initial, train_features, train.label.to_numpy(), args.epochs)
            heads.append(head.state_dict())
        else:
            head = model.head.cpu().eval()
        test_logits += 0.5 * head_logits(head, test_features)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    files = [assets / "ensemble_packed.pt", detector]
    if args.improve:
        torch.save(
            {"heads": heads, "base_sha256": digest(assets / "ensemble_packed.pt")},
            output / "heads.pt",
        )
        files.append(output / "heads.pt")
        for fold, logits in diagnostic.items():
            write_predictions(
                train.loc[train.fold == fold],
                softmax(logits),
                output / f"diagnostic_fold_{fold}.csv",
            )
    probabilities = softmax(test_logits)
    write_predictions(test, probabilities, output / "test_predictions.csv")
    submit(test, probabilities, args.test_csv, output / "submission.csv")
    write_json(
        output / "summary.json",
        {
            "recipe": "i2_yolo_head_finetune" if args.improve else "r2_yolo_v9",
            "inference_bytes": check_size(files),
            "asset_hashes": {p.name: digest(p) for p in files},
            "detector": {
                "path": str(detector.resolve()),
                "sha256": digest(detector),
                "source": "local fold_2 fine-tuned YOLOv8n; not the notebook's yolo11n.pt",
            },
            "validation_status": "diagnostic_only: public HAR backbone training subjects unknown",
            "diagnostic_accuracy": {
                str(f): float(
                    (p.argmax(1) == train.loc[train.fold == f, "label"].to_numpy()).mean()
                )
                for f, p in diagnostic.items()
            },
            "notes": [
                "Source notebook is inference-only; original training recipe unavailable",
                "Original packed ensemble, preprocessing and four-view logit mean retained",
                "Fine-tuning uses fixed epochs; diagnostic folds never select epochs or weights",
                "No test-prior adjustment is used in the main submission",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default="artifacts/public/assets/yolo_v9")
    parser.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        help="Local person detector; default is the existing fold_2 YOLOv8n checkpoint",
    )
    parser.add_argument(
        "--model-definition",
        default=str(DEFAULT_MODEL_DEFINITION),
        help="ig65m model definition; defaults to the vendored upstream-compatible module",
    )
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="cache-public/yolo_v9")
    parser.add_argument("--improve", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--train-root", default="../Small-Model-Track/Training/extracted/HAR/data")
    parser.add_argument(
        "--test-root", default="../Small-Model-Track/Testing/data/small_model_track_test"
    )
    parser.add_argument("--test-csv", default="../Small-Model-Track/Testing/test_file/test.csv")
    args = parser.parse_args()
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        expected = "i2_yolo_head_finetune" if args.improve else "r2_yolo_v9"
        if config["recipe"] != expected:
            parser.error("YOLO config/recipe mismatch")
        for key, value in {
            "frames": 16,
            "image_size": 128,
            "channels": 4,
            "tta": 4,
            "margin": 1.4,
            "weights": [0.5, 0.5],
        }.items():
            if config[key] != value:
                parser.error(f"Pretrained YOLO reproduction requires {key}={value}")
        if args.improve:
            args.epochs = config["epochs"]
    if min(args.epochs, args.micro_batch) < 1:
        parser.error("epochs and micro-batch must be positive")
    run(args)


if __name__ == "__main__":
    main()
