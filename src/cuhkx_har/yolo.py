"""Auditable YOLOv8n person-crop preprocessing for the visual HAR streams.

The commands in this module intentionally never inspect test labels (and the
inspection command rejects test-style clip ids).  Detection is precomputed so
the HAR DataLoader stays deterministic and has no Ultralytics dependency at
runtime beyond the preprocessing stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .constants import TRAIN_USERS, VISUAL_MODALITIES
from .data import crop_person
from .splits import fold_partition

SEED = 20260719
PERSON_CLASS = 0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _require_train_manifest(frame: pd.DataFrame) -> None:
    users = set(frame["user"].astype(str))
    unexpected = users - set(TRAIN_USERS)
    if unexpected or not users:
        raise ValueError(
            "Annotation/detector manifest must contain training users only; "
            f"got {sorted(unexpected)}"
        )
    if any(str(clip).startswith("SM_test_") for clip in frame["clip_id"]):
        raise ValueError("Test clips must never enter the YOLO annotation/detector pipeline")


def assert_frozen_cv5_manifest(path: str | Path) -> None:
    """Reject a re-folded or otherwise altered training manifest at CLI boundaries."""
    expected_path = Path(__file__).resolve().parents[2] / "manifests" / "cv5" / "train.csv"
    if not expected_path.is_file():
        raise FileNotFoundError(f"Missing frozen CV5 manifest: {expected_path}")
    expected = pd.read_csv(expected_path, usecols=["clip_id", "fold"])
    supplied = pd.read_csv(path, usecols=["clip_id", "fold"])
    if (
        expected["clip_id"].duplicated().any()
        or supplied["clip_id"].duplicated().any()
        or set(expected["clip_id"]) != set(supplied["clip_id"])
        or not expected.set_index("clip_id")["fold"].equals(supplied.set_index("clip_id")["fold"])
    ):
        raise ValueError(
            "Manifest does not match frozen manifests/cv5/train.csv clip_id-to-fold mapping"
        )


def _candidates(manifest: pd.DataFrame, data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in manifest.sort_values(["label", "user", "clip_id"]).iterrows():
        relative = str(row.get("Depth_Color", ""))
        if not relative:
            continue
        directory = data_root / relative
        files = _files(directory) if directory.is_dir() else []
        if not files:
            continue
        positions = sorted(
            {
                min(len(files) - 1, max(0, int(round(p * (len(files) - 1)))))
                for p in (0.15, 0.5, 0.85)
            }
        )
        for rank, index in enumerate(positions):
            rows.append(
                {
                    "clip_id": str(row["clip_id"]),
                    "action_name": str(row["action_name"]),
                    "label": int(row["label"]),
                    "user": str(row["user"]),
                    "frame_index": index,
                    "time_position": (index / max(len(files) - 1, 1)),
                    "source_path": str(files[index].resolve()),
                    "position_rank": rank,
                }
            )
    return rows


def stratified_annotation_sample(
    manifest: pd.DataFrame, data_root: str | Path, count: int = 800
) -> pd.DataFrame:
    """Take a deterministic round-robin sample over action/user strata."""
    _require_train_manifest(manifest)
    if count <= 0:
        raise ValueError("count must be positive")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in _candidates(manifest, Path(data_root).resolve()):
        groups[(candidate["label"], candidate["user"])].append(candidate)
    if len(groups) < min(40, count):
        raise ValueError("Too few action/user strata with usable Depth_Color frames")
    ordered_groups = [groups[key] for key in sorted(groups)]
    selected: list[dict[str, Any]] = []
    rank = 0
    while len(selected) < count:
        added = 0
        for group in ordered_groups:
            if rank < len(group) and len(selected) < count:
                selected.append(group[rank])
                added += 1
        if not added:
            break
        rank += 1
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} usable stratified frames exist; requested {count}")
    result = pd.DataFrame(selected)
    result.insert(0, "sample_id", [f"{index:04d}" for index in range(len(result))])
    return result


def export_annotations(
    manifest_path: str | Path, data_root: str | Path, output_dir: str | Path, count: int = 800
) -> Path:
    manifest = pd.read_csv(manifest_path).fillna("")
    output = Path(output_dir).resolve()
    images, labels = output / "images", output / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    sample = stratified_annotation_sample(manifest, data_root, count)
    image_paths, label_paths, hashes = [], [], []
    for row in sample.itertuples(index=False):
        source = Path(row.source_path)
        destination = images / f"{row.sample_id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        image_paths.append(destination.relative_to(output).as_posix())
        label_paths.append((labels / f"{row.sample_id}.txt").relative_to(output).as_posix())
        hashes.append(_sha256(destination))
    sample["image_path"] = image_paths
    sample["label_path"] = label_paths
    sample["image_sha256"] = hashes
    manifest_output = output / "annotation_manifest.csv"
    sample.to_csv(manifest_output, index=False)
    (output / "README_LABELIMG.txt").write_text(
        "Label only training images. In LabelImg choose YOLO format, one class named "
        "person (class id 0), "
        f"and save .txt labels to {labels}. Do not open or annotate test images.\n",
        encoding="utf-8",
    )
    return manifest_output


def _parse_label(path: Path) -> tuple[float, float, float, float] | None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        # A zero-byte YOLO label is a valid explicit negative image.
        return None
    if len(lines) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty person annotation")
    pieces = lines[0].split()
    if len(pieces) != 5 or pieces[0] not in {"0", "0.0"}:
        raise ValueError(f"{path}: expected YOLO '0 x_center y_center width height'")
    x, y, width, height = (float(value) for value in pieces[1:])
    if (
        not all(math.isfinite(value) for value in (x, y, width, height))
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{path}: non-finite or non-positive bbox")
    # LabelImg writes decimal values; permit insignificant rounding at an edge.
    epsilon = 1e-4
    if not (
        -epsilon <= x - width / 2 <= 1 + epsilon
        and -epsilon <= x + width / 2 <= 1 + epsilon
        and -epsilon <= y - height / 2 <= 1 + epsilon
        and -epsilon <= y + height / 2 <= 1 + epsilon
    ):
        raise ValueError(f"{path}: bbox is outside normalized image boundaries")
    return x, y, width, height


def audit_labels(annotation_dir: str | Path) -> dict[str, Any]:
    root = Path(annotation_dir).resolve()
    annotation_manifest = root / "annotation_manifest.csv"
    if not annotation_manifest.is_file():
        raise FileNotFoundError(f"Missing {annotation_manifest}")
    frame = pd.read_csv(annotation_manifest).fillna("")
    _require_train_manifest(frame)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate annotation sample ids")
    positives = 0
    negatives = 0
    for row in frame.itertuples(index=False):
        image, label = root / row.image_path, root / row.label_path
        if not image.is_file():
            raise FileNotFoundError(f"Missing annotation image: {image}")
        if _sha256(image) != row.image_sha256:
            raise ValueError(f"Image has changed since annotation sampling: {image}")
        parsed = _parse_label(label) if label.is_file() else None
        positives += parsed is not None
        negatives += parsed is None
    # LabelImg emits this class-name sidecar in the save directory.  It is
    # metadata, not a YOLO image label, and must not be staged for training.
    label_files = {
        path.resolve()
        for path in (root / "labels").glob("*.txt")
        if path.name.lower() != "classes.txt"
    }
    expected = {(root / value).resolve() for value in frame["label_path"]}
    if not label_files <= expected:
        raise ValueError("Label directory has label files not belonging to annotation_manifest.csv")
    return {
        "annotation_manifest": str(annotation_manifest),
        "frames": len(frame),
        "positive_frames": positives,
        "negative_frames": negatives,
        "manifest_sha256": _sha256(annotation_manifest),
        "status": "passed",
    }


def _import_yolo() -> Any:
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "ultralytics is required; run 'uv sync' before using YOLO commands"
        ) from error
    return YOLO, ultralytics.__version__


def _stage_dataset(
    annotation_dir: Path, train_frame: pd.DataFrame, output: Path, seed: int
) -> tuple[Path, pd.DataFrame]:
    annotations = pd.read_csv(annotation_dir / "annotation_manifest.csv").fillna("")
    eligible = annotations[annotations["user"].isin(set(train_frame["user"]))].copy()
    if eligible.empty:
        raise ValueError("No human-labeled frames belong to this fold's training users")
    audit_labels(annotation_dir)
    rng = np.random.default_rng(seed)
    eligible["detector_validation"] = False
    # A detector-internal split stays in fold-training users; HAR validation
    # users remain unseen.
    for _, indices in eligible.groupby("label").groups.items():
        index_list = list(indices)
        if len(index_list) > 1:
            eligible.loc[
                rng.choice(index_list, size=max(1, round(len(index_list) * 0.1)), replace=False),
                "detector_validation",
            ] = True
    for split in ("train", "val"):
        subset = eligible[eligible["detector_validation"] == (split == "val")]
        if subset.empty:
            subset = eligible
        for row in subset.itertuples(index=False):
            source_image = annotation_dir / row.image_path
            source_label = annotation_dir / row.label_path
            image_target = output / "dataset" / "images" / split / source_image.name
            label_target = output / "dataset" / "labels" / split / source_label.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, image_target)
            if source_label.is_file():
                shutil.copy2(source_label, label_target)
            else:
                # YOLO treats a missing-object image as an empty label file.
                label_target.write_text("", encoding="utf-8")
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        "path: "
        + str((output / "dataset").as_posix())
        + "\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
        encoding="utf-8",
    )
    return yaml_path, eligible


def train_detector(
    annotation_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    fold: int | None,
    epochs: int,
    image_size: int,
    batch: int,
    device: str,
) -> Path:
    annotation_root, output = Path(annotation_dir).resolve(), Path(output_dir).resolve()
    train_manifest = pd.read_csv(manifest_path).fillna("")
    _require_train_manifest(train_manifest)
    if fold is not None:
        train_frame, valid_frame = fold_partition(train_manifest, fold)
        if set(train_frame["user"]) & set(valid_frame["user"]):
            raise RuntimeError("Subject leakage while preparing YOLO fold")
        tag = f"fold_{fold}"
    else:
        train_frame, valid_frame, tag = train_manifest, pd.DataFrame(), "final"
    run = output / tag
    yaml_path, eligible = _stage_dataset(
        annotation_root, train_frame, run, SEED + (fold if fold is not None else 99)
    )
    YOLO, version = _import_yolo()
    model = YOLO("yolov8n.pt")
    result = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        project=str(run / "runs"),
        name="train",
        exist_ok=True,
        pretrained=True,
        seed=SEED,
        deterministic=True,
        verbose=True,
    )
    del result
    best = run / "runs" / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Ultralytics did not create expected weight file {best}")
    final_weight = run / f"yolov8n_{tag}.pt"
    shutil.copy2(best, final_weight)
    info = {
        "fold": fold,
        "tag": tag,
        "source_weights": "COCO pretrained yolov8n.pt",
        "ultralytics_version": version,
        "license": "AGPL-3.0 (verify intended competition use)",
        "weight": str(final_weight),
        "weight_bytes": final_weight.stat().st_size,
        "epochs": epochs,
        "image_size": image_size,
        "batch": batch,
        "device": device,
        "annotation_manifest_sha256": _sha256(annotation_root / "annotation_manifest.csv"),
        "training_users": sorted(train_frame["user"].unique()),
        "held_out_users": sorted(valid_frame["user"].unique()),
        "labeled_frames_used": len(eligible),
    }
    (run / "detector_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return final_weight


def _xyxy(box: Iterable[float], width: int, height: int) -> list[float]:
    left, top, right, bottom = (float(value) for value in box)
    return [
        max(0.0, min(1.0, left / width)),
        max(0.0, min(1.0, top / height)),
        max(0.0, min(1.0, right / width)),
        max(0.0, min(1.0, bottom / height)),
    ]


def _smooth_boxes(
    raw: list[list[float] | None], confidence: list[float], threshold: float, window: int = 5
) -> tuple[list[list[float] | None], list[str]]:
    valid = [
        box if box is not None and score >= threshold else None
        for box, score in zip(raw, confidence, strict=True)
    ]
    if not any(valid):
        return [None] * len(raw), ["full_frame"] * len(raw)
    filled, status = [], []
    for index, box in enumerate(valid):
        if box is not None:
            filled.append(box)
            status.append("detected")
        else:
            nearest = min(
                (candidate for candidate in range(len(valid)) if valid[candidate] is not None),
                key=lambda candidate: abs(candidate - index),
            )
            filled.append(valid[nearest])
            status.append("neighbor")
    smoothed: list[list[float] | None] = []
    for index in range(len(filled)):
        nearby = [
            filled[candidate]
            for candidate in range(
                max(0, index - window // 2), min(len(filled), index + window // 2 + 1)
            )
        ]
        smoothed.append(
            np.median(np.asarray(nearby, dtype=np.float32), axis=0).astype(float).tolist()
        )
    return smoothed, status


def create_crop_metadata(
    manifest_path: str | Path,
    data_root: str | Path,
    weights: str | Path,
    output_dir: str | Path,
    confidence: float = 0.25,
    device: str = "cuda",
    allow_test: bool = False,
) -> Path:
    frame, root, output = (
        pd.read_csv(manifest_path).fillna(""),
        Path(data_root).resolve(),
        Path(output_dir).resolve(),
    )
    has_test = any(str(clip).startswith("SM_test_") for clip in frame["clip_id"])
    if has_test and not allow_test:
        raise ValueError(
            "Test crop inference requires --allow-test; it must use the final detector only"
        )
    if has_test and set(frame["user"].astype(str)) - {""}:
        raise ValueError("A crop manifest cannot mix training and test clips")
    if not has_test:
        _require_train_manifest(frame)
    YOLO, version = _import_yolo()
    model = YOLO(str(weights))
    clips: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        directory = root / row.Depth_Color
        files = _files(directory) if directory.is_dir() else []
        raw: list[list[float] | None] = []
        scores: list[float] = []
        for file in files:
            result = model.predict(
                str(file), conf=confidence, classes=[PERSON_CLASS], device=device, verbose=False
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                raw.append(None)
                scores.append(0.0)
                continue
            index = int(result.boxes.conf.argmax().item())
            with Image.open(file) as image:
                raw.append(_xyxy(result.boxes.xyxy[index].tolist(), *image.size))
            scores.append(float(result.boxes.conf[index].item()))
        boxes, status = _smooth_boxes(raw, scores, confidence)
        clips[str(row.clip_id)] = {
            "frames": [
                {"bbox": box, "confidence": score, "status": frame_status}
                for box, score, frame_status in zip(boxes, scores, status, strict=True)
            ],
            "depth_frame_count": len(files),
        }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "bboxes.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "detector_weights": str(Path(weights).resolve()),
                "detector_sha256": _sha256(Path(weights)),
                "ultralytics_version": version,
                "confidence_threshold": confidence,
                "clips": clips,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return target


def inspection_sheets(
    manifest_path: str | Path,
    data_root: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    fold: int,
    count: int = 100,
    padding: float = 0.12,
) -> Path:
    frame = pd.read_csv(manifest_path).fillna("")
    _require_train_manifest(frame)
    held = frame[frame["fold"] == fold].sort_values("clip_id")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))["clips"]
    candidates = [
        (row, index, item)
        for row in held.itertuples(index=False)
        for index, item in enumerate(metadata.get(str(row.clip_id), {}).get("frames", []))
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} held-out frames available; need {count}")
    rng = np.random.default_rng(SEED + fold)
    picked = [
        candidates[index] for index in sorted(rng.choice(len(candidates), count, replace=False))
    ]
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    thumbs: list[Image.Image] = []
    crop_tiles: dict[str, list[Image.Image]] = {modality: [] for modality in VISUAL_MODALITIES}
    for row, index, item in picked:
        if str(row.clip_id).startswith("SM_test_"):
            raise ValueError("Inspection must not contain test clips")
        depth_files = _files(root / row.Depth_Color)
        if index >= len(depth_files):
            continue
        with Image.open(depth_files[index]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        bbox = item["bbox"]
        if bbox:
            left, top, right, bottom = bbox
            draw.rectangle(
                (
                    left * image.width,
                    top * image.height,
                    right * image.width,
                    bottom * image.height,
                ),
                outline="lime",
                width=max(2, image.width // 160),
            )
        draw.text(
            (3, 3),
            f"{row.clip_id} #{index} {item['status']}",
            fill="yellow",
            stroke_width=1,
            stroke_fill="black",
        )
        image.thumbnail((320, 240))
        thumbs.append(image.copy())
        modality_indices: dict[str, int] = {}
        for modality in VISUAL_MODALITIES:
            files = _files(root / getattr(row, modality))
            if not files:
                continue
            mapped = int(round(index * (len(files) - 1) / max(len(depth_files) - 1, 1)))
            modality_indices[modality] = mapped
            with Image.open(files[mapped]) as source:
                original = source.convert("RGB")
            cropped = crop_person(original, tuple(bbox), padding) if bbox else original
            original = ImageOps.pad(original, (160, 120), color="black")
            cropped = ImageOps.pad(cropped, (160, 120), color="black")
            original_draw = ImageDraw.Draw(original)
            original_draw.text(
                (3, 3),
                f"raw #{mapped}",
                fill="yellow",
                stroke_width=1,
                stroke_fill="black",
            )
            crop_draw = ImageDraw.Draw(cropped)
            crop_draw.text(
                (3, 3),
                "crop",
                fill="yellow",
                stroke_width=1,
                stroke_fill="black",
            )
            tile = Image.new("RGB", (320, 120), "black")
            tile.paste(original, (0, 0))
            tile.paste(cropped, (160, 0))
            crop_tiles[modality].append(tile)
        records.append(
            {
                "clip_id": row.clip_id,
                "depth_frame_index": index,
                "ir_frame_index": modality_indices.get("IR", ""),
                "thermal_frame_index": modality_indices.get("Thermal", ""),
                "confidence": item["confidence"],
                "fallback_status": item["status"],
                "bbox": json.dumps(bbox),
            }
        )
    for page in range(10):
        sheet = Image.new("RGB", (5 * 320, 2 * 240), "black")
        for offset, image in enumerate(thumbs[page * 10 : (page + 1) * 10]):
            sheet.paste(image, ((offset % 5) * 320, (offset // 5) * 240))
        sheet.save(output / f"overlay_{page + 1:02d}.png")
        for modality, tiles in crop_tiles.items():
            sheet = Image.new("RGB", (5 * 320, 2 * 120), "black")
            for offset, tile in enumerate(tiles[page * 10 : (page + 1) * 10]):
                sheet.paste(tile, ((offset % 5) * 320, (offset // 5) * 120))
            sheet.save(output / f"crop_{modality.lower()}_{page + 1:02d}.png")
    report = output / "inspection.csv"
    pd.DataFrame(records).to_csv(report, index=False)
    return report


def audit_weights(
    yolo_weight: str | Path, har_weights: list[str | Path], output: str | Path | None = None
) -> dict[str, Any]:
    paths = [Path(yolo_weight), *(Path(path) for path in har_weights)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing weights: {missing}")
    total = sum(path.stat().st_size for path in paths)
    report = {
        "files": [{"path": str(path), "bytes": path.stat().st_size} for path in paths],
        "total_bytes": total,
        "total_mb_decimal": total / 1_000_000,
        "limit_mb": 100,
        "passes": total <= 100_000_000,
    }
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passes"]:
        raise ValueError(
            f"Final detector + HAR ensemble is {report['total_mb_decimal']:.2f} MB, over 100 MB"
        )
    return report


def _parser(name: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=name)


def main_annotations() -> None:
    p = _parser("cuhkx-yolo-annotations")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--count", type=int, default=800)
    a = p.parse_args()
    assert_frozen_cv5_manifest(a.manifest)
    print(export_annotations(a.manifest, a.data_root, a.output_dir, a.count))


def main_audit_labels() -> None:
    p = _parser("cuhkx-yolo-audit-labels")
    p.add_argument("--annotation-dir", required=True)
    a = p.parse_args()
    print(json.dumps(audit_labels(a.annotation_dir), indent=2))


def main_train() -> None:
    p = _parser("cuhkx-yolo-train")
    p.add_argument("--annotation-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fold", type=int)
    p.add_argument(
        "--final",
        action="store_true",
        help="train with all labels after CV acceptance; never use this model for CV",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if (a.fold is None) == (not a.final):
        p.error("provide exactly one of --fold <0..4> or --final")
    if a.fold is not None and a.fold not in range(5):
        p.error("--fold must be 0..4")
    assert_frozen_cv5_manifest(a.manifest)
    print(
        train_detector(
            a.annotation_dir,
            a.manifest,
            a.output_dir,
            a.fold,
            a.epochs,
            a.image_size,
            a.batch,
            a.device,
        )
    )


def main_crops() -> None:
    p = _parser("cuhkx-yolo-crops")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--confidence", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--allow-test",
        action="store_true",
        help="final detector inference only; never use this output for CV",
    )
    a = p.parse_args()
    if not a.allow_test:
        assert_frozen_cv5_manifest(a.manifest)
    print(
        create_crop_metadata(
            a.manifest, a.data_root, a.weights, a.output_dir, a.confidence, a.device, a.allow_test
        )
    )


def main_inspect() -> None:
    p = _parser("cuhkx-yolo-inspect")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--padding", type=float, default=0.12)
    a = p.parse_args()
    assert_frozen_cv5_manifest(a.manifest)
    print(
        inspection_sheets(
            a.manifest,
            a.data_root,
            a.metadata,
            a.output_dir,
            a.fold,
            a.count,
            a.padding,
        )
    )


def main_weight_audit() -> None:
    p = _parser("cuhkx-weight-audit")
    p.add_argument("--yolo", required=True)
    p.add_argument("--har", required=True, nargs="+")
    p.add_argument("--output")
    a = p.parse_args()
    print(json.dumps(audit_weights(a.yolo, a.har, a.output), indent=2))
