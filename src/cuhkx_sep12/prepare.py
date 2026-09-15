"""Fold-safe detector preparation and reusable, temporally aligned YOLO crops."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from cuhkx_har.splits import fold_partition
from cuhkx_modality.frozen import read_frozen_manifest

from .common import MODALITIES, cache_path, digest, write_json


def natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def image_files(root, relative):
    if not relative or pd.isna(relative):
        return []
    folder = Path(root) / relative
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    files = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}),
        key=natural_key,
    )
    if not files:
        raise ValueError(f"Present modality has no images: {folder}")
    return files


def timestamps(files):
    values = []
    for file in files:
        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)", file.name)
        if not match:
            return None
        values.append(datetime.strptime(match[1], "%Y-%m-%d_%H-%M-%S.%f").timestamp())
    values = np.asarray(values)
    return values if np.all(np.diff(values) >= 0) else None


def aligned_indices(streams, frames):
    """IR/depth absolute-clock overlap, thermal clip-relative rank alignment."""
    q = np.linspace(0, 1, frames)
    indices = [np.rint(q * (len(files) - 1)).astype(int) if files else None for files in streams]
    ir, depth = timestamps(streams[0]), timestamps(streams[1])
    mode = "shared_clip_relative"
    if ir is not None and depth is not None and len(ir) and len(depth):
        start, end = max(ir[0], depth[0]), min(ir[-1], depth[-1])
        if start > end:
            raise ValueError("IR/Depth timestamps have no overlap; cannot align this clip")
        targets = start + q * (end - start)
        indices[0] = np.abs(ir[:, None] - targets[None]).argmin(0)
        indices[1] = np.abs(depth[:, None] - targets[None]).argmin(0)
        # Map overlap timestamps to the full depth clip, then to thermal clip ranks.
        if streams[2] and depth[-1] > depth[0]:
            thermal_q = (targets - depth[0]) / (depth[-1] - depth[0])
            indices[2] = np.rint(thermal_q * (len(streams[2]) - 1)).astype(int)
        mode = "ir_depth_timestamp_thermal_relative"
    return indices, mode


def audit_detector(weights, manifest, fold):
    summary_path = Path(weights).parent / "detector_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_users = set(summary["training_users"])
    if not training_users or not training_users <= set(manifest.user):
        raise ValueError("Detector provenance has invalid training users")
    if fold != "full":
        held_out = set(manifest.loc[manifest.fold == int(fold), "user"])
        if training_users & held_out:
            raise ValueError(f"Detector leaks fold {fold}: {sorted(training_users & held_out)}")
    actual = digest(weights)
    if summary.get("weight_sha256", actual) != actual:
        raise ValueError("Detector weight hash does not match provenance")
    return summary


def detector(args):
    from ultralytics import YOLO

    manifest = read_frozen_manifest(args.manifest)
    train = manifest if args.fold == "full" else fold_partition(manifest, int(args.fold))[0]
    annotations_path = Path(args.annotations) / "annotation_manifest.csv"
    annotations = pd.read_csv(annotations_path, dtype={"sample_id": str})
    # Validate identity before relying on the user column to prevent held-out exposure.
    users = manifest.set_index("clip_id").user
    if not annotations.clip_id.isin(users.index).all():
        raise ValueError("Unknown annotated clip")
    if not annotations.user.equals(annotations.clip_id.map(users).rename("user")):
        raise ValueError("Annotation users do not match the frozen manifest")
    selected = annotations.loc[annotations.user.isin(train.user)].copy()
    if selected.empty:
        raise ValueError("No detector training annotations")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    image_dir, label_dir = output / "dataset/images/train", output / "dataset/labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for row in selected.itertuples():

        source = Path(args.annotations) / row.image_path
        label = Path(args.annotations) / row.label_path
        
        if not source.is_file() or not label.is_file():
            print(f"[WARNING] Missing file(s) for sample, skipping: source={source.name}, label={label.name}")
            continue
            
        if digest(source) != row.image_sha256:
            raise ValueError(f"Annotated image hash changed: {source}")
        
        shutil.copy2(source, image_dir / source.name)
        shutil.copy2(label, label_dir / (source.stem + ".txt"))
    dataset = output / "dataset.yaml"
    dataset.write_text(
        f"path: {json.dumps(str((output / 'dataset').resolve()))}\n"
        "train: images/train\nval: images/train\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    # Fixed epochs; validation uses training images only and is NOT a held-out metric.
    # Start from base weights, never the fold_2 fine-tuned detector for another fold.
    model = YOLO(args.base_weights)
    model.train(
        data=str(dataset.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch_size,
        device=args.device,
        seed=0,
        deterministic=True,
        patience=0,
        val=False,
        project=str((output / "runs").resolve()),
        name="train",
        exist_ok=False,
    )
    weight = output / f"yolov8n_{args.fold}.pt"
    shutil.copy2(Path(model.trainer.save_dir) / "weights/last.pt", weight)
    write_json(
        output / "detector_summary.json",
        {
            "fold": args.fold,
            "training_users": sorted(set(selected.user)),
            "held_out_users": sorted(set(manifest.user) - set(train.user)),
            "labeled_frames_used": len(selected),
            "epochs": args.epochs,
            "annotation_manifest_sha256": digest(annotations_path),
            "weight_sha256": digest(weight),
            "base_sha256": digest(args.base_weights),
            "selection": "fixed epochs, final checkpoint; no held-out validation",
        },
    )


def prepare(args):
    from ultralytics import YOLO

    manifest = read_frozen_manifest(args.manifest)
    audit_detector(args.weights, manifest, args.fold)
    source_manifest = args.manifest if args.split == "train" else args.test_manifest
    rows = pd.read_csv(source_manifest).fillna("")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": 1,
        "split": args.split,
        "fold": args.fold,
        "frames": args.frames,
        "image_size": args.image_size,
        "detector_sha256": digest(args.weights),
        "detector_bytes": Path(args.weights).stat().st_size,
        "manifest_sha256": digest(source_manifest),
        "confidence": args.confidence,
        "padding": 0.10,
        "missing_policy": "nearest_detected_frame_box_else_mask_zero",
        "alignment": "IR/Depth timestamps when available; Thermal normalized clip time",
    }
    meta_path = output / f"{args.split}_metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Cache settings/provenance changed; use a new output directory")
    write_json(meta_path, metadata)
    model = YOLO(args.weights)
    coverage = []

    def record_coverage(clip_id, mask, detected):
        record = {"clip_id": clip_id}
        for m, modality in enumerate(MODALITIES):
            record[f"{modality}_detected_frames"] = int(detected[m].sum())
            record[f"{modality}_cropped_frames"] = int(mask[m].sum())
        coverage.append(record)

    for number, row in enumerate(rows.itertuples(index=False), 1):
        path = cache_path(output, args.split, row.clip_id)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved["images"].shape != (3, args.frames, args.image_size, args.image_size, 3):
                    raise ValueError(f"Invalid existing cache: {path}")
                if str(saved["clip_id"]) != row.clip_id:
                    raise ValueError(f"Cache identity mismatch: {path}")
                record_coverage(row.clip_id, saved["mask"], saved["detected"])
            continue
        streams = [image_files(args.data_root, getattr(row, m)) for m in MODALITIES]
        indices, alignment = aligned_indices(streams, args.frames)
        images = np.zeros((3, args.frames, args.image_size, args.image_size, 3), dtype=np.uint8)
        mask = np.zeros((3, args.frames), dtype=bool)
        boxes = np.full((3, args.frames, 4), -1, dtype=np.float32)
        detected = np.zeros_like(mask)
        
        # ===== 1. cross modality alignment (Spatial Sharing from Depth) =====
        # extract Depth_Color bounding box for sharing 
        depth_lookup = {}
        if streams[1]:
            depth_selected = [streams[1][int(i)] for i in indices[1]]
            valid_depth = []
            for p in depth_selected:
                try:
                    with Image.open(p) as img: img.load()
                    valid_depth.append(p)
                except Exception: pass
            
            if valid_depth:
                predictions = model.predict(
                    source=[str(p) for p in valid_depth],
                    device=args.device,
                    conf=args.confidence,
                    classes=[0],
                    verbose=False,
                    batch=args.batch_size,
                )
                for file, prediction in zip(valid_depth, predictions, strict=True):
                    if len(prediction.boxes):
                        best = int(prediction.boxes.conf.argmax())
                        box_coords = prediction.boxes.xyxyn[best].cpu().numpy()
                        depth_lookup[file] = box_coords
                        for t, s_file in enumerate(depth_selected):
                            if s_file == file:
                                boxes[1, t] = box_coords
                                detected[1, t] = True

        # ===== 2. Set Bounding Box and Align all modalities =====
        for m, files in enumerate(streams):
            if not files:
                continue
            selected = [files[int(i)] for i in indices[m]]
            
            # if IR (m=0) or Thermal (m=2)，use Depth (m=1) Bounding Box
            if m in (0, 2) and depth_lookup and streams[1]:
                depth_selected = [streams[1][int(i)] for i in indices[1]]
                for t, file in enumerate(selected):
                    d_file = depth_selected[t]
                    if d_file in depth_lookup:
                        boxes[m, t] = depth_lookup[d_file]
                        detected[m, t] = True
            
            # if not shared（Depth does not detect any person），run its yolo of own modality
            if not detected[m].any():
                print("Francis Attention!")
                unique = list(dict.fromkeys(selected))
                valid_unique = [p for p in unique if p.is_file()]
                if valid_unique:
                    predictions = model.predict(
                        source=[str(p) for p in valid_unique],
                        device=args.device,
                        conf=args.confidence,
                        classes=[0],
                        verbose=False,
                        batch=args.batch_size,
                    )
                    for file, prediction in zip(valid_unique, predictions, strict=True):
                        if len(prediction.boxes):
                            best = int(prediction.boxes.conf.argmax())
                            for t, s_file in enumerate(selected):
                                if s_file == file and not detected[m, t]:
                                    boxes[m, t] = prediction.boxes.xyxyn[best].cpu().numpy()
                                    detected[m, t] = True

            # ===== 3. time backward by frames (Temporal Backtracking / Fallback) =====
            available = np.flatnonzero(detected[m])
            for t, file in enumerate(selected):
                if len(available) > 0:
                    # Time nearest neighbor interpolation to complete missing frames
                    nearest = available[np.abs(available - t).argmin()]
                    box = boxes[m, t] if detected[m, t] else boxes[m, nearest].copy()
                else:
                    # The central bottom box when missing throughout the entire time period (Center Crop: 20%~80% regions)
                    box = np.array([0.2, 0.2, 0.8, 0.8], dtype=np.float32)

                boxes[m, t] = box
                try:
                    with Image.open(file) as source:
                        source = source.convert("RGB")
                        width, height = source.size
                        x1, y1, x2, y2 = box
                        dx, dy = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
                        bounds = (
                            max(0, int((x1 - dx) * width)),
                            max(0, int((y1 - dy) * height)),
                            min(width, int(np.ceil((x2 + dx) * width))),
                            min(height, int(np.ceil((y2 + dy) * height))),
                        )
                        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                            continue
                        crop = ImageOps.pad(source.crop(bounds), (args.image_size, args.image_size))
                        images[m, t] = np.asarray(crop)
                        mask[m, t] = True
                except Exception:
                    continue

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
                images=images,
                mask=mask,
                boxes=boxes,
                detected=detected,
                alignment=alignment,
                clip_id=row.clip_id,
            )
        temporary.replace(path)
        record_coverage(row.clip_id, mask, detected)
        if number % 25 == 0:
            print(f"{args.split}: {number}/{len(rows)} cropped clips", flush=True)

    report = pd.DataFrame(coverage)
    report.to_csv(output / f"{args.split}_coverage.csv", index=False)
    write_json(
        output / f"{args.split}_coverage.json",
        {
            "clips": len(rows),
            "zero_crop_clips": {
                m: int((report[f"{m}_cropped_frames"] == 0).sum()) for m in MODALITIES
            },
            "detected_frames": {m: int(report[f"{m}_detected_frames"].sum()) for m in MODALITIES},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    det = sub.add_parser("detector")
    det.add_argument("--annotations", default="artifacts/yolo/annotations_800")
    det.add_argument("--base-weights", default="yolov8n.pt")
    det.add_argument("--epochs", type=int, default=60)
    crop = sub.add_parser("crops")
    crop.add_argument("--weights", required=True)
    crop.add_argument("--split", choices=["train", "test"], required=True)
    crop.add_argument("--test-manifest", default="manifests/cv5/test.csv")
    crop.add_argument("--data-root", required=True)
    crop.add_argument("--frames", type=int, default=16)
    crop.add_argument("--image-size", type=int, default=128)
    crop.add_argument("--confidence", type=float, default=0.25)
    for item in (det, crop):
        item.add_argument("--manifest", default="manifests/cv5/train.csv")
        item.add_argument("--fold", choices=["2", "4", "full"], required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--device", default="0", help="Ultralytics device: 0 or cpu")
        item.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "detector":
        detector(args)
    else:
        if args.frames < 1 or args.image_size < 16:
            parser.error("frames must be positive and image-size >=16")
        prepare(args)


if __name__ == "__main__":
    main()
