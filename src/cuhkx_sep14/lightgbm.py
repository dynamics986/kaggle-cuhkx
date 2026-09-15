"""Sep14 LightGBM: add leak-safe YOLO box trajectory features and narrow tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cuhkx_har.splits import fold_partition
from cuhkx_modality.frozen import read_frozen_manifest
from cuhkx_sep12.common import cache_path, check_size, digest, submit, write_json, write_predictions
from cuhkx_sep12.data import validate_cache
from cuhkx_sep12.lightgbm import feature_table
from cuhkx_sep12.prepare import audit_detector
from cuhkx_sep13.lightgbm import BASE, train


def box_feature_table(rows, crop_cache, split):
    """Detection coverage plus normalized box geometry and motion for each visual stream."""
    records = []
    for row in rows.itertuples(index=False):
        record = {}
        with np.load(cache_path(crop_cache, split, row.clip_id), allow_pickle=False) as saved:
            boxes, mask, detected = saved["boxes"].astype(np.float32), saved["mask"].astype(bool), saved["detected"].astype(bool)
        for index, name in enumerate(("ir", "depth_color", "thermal")):
            valid = mask[index] & (boxes[index, :, 0] >= 0)
            record[f"{name}_crop_coverage"] = float(mask[index].mean())
            record[f"{name}_detection_coverage"] = float(detected[index].mean())
            if not valid.any():
                values = np.zeros((0, 4), dtype=np.float32)
            else:
                value = boxes[index, valid]
                width = np.maximum(value[:, 2] - value[:, 0], 0)
                height = np.maximum(value[:, 3] - value[:, 1], 0)
                # prepare.py persists YOLO ``xyxyn`` coordinates, already normalized
                # to [0, 1]; do not divide by the 128-pixel crop size again.
                values = np.stack(
                    [(value[:, 0] + value[:, 2]) / 2, (value[:, 1] + value[:, 3]) / 2,
                     width, height], axis=1
                )
            for column, label in enumerate(("cx", "cy", "w", "h")):
                sequence = values[:, column] if len(values) else np.zeros(1, dtype=np.float32)
                record[f"{name}_box_{label}_mean"] = float(sequence.mean())
                record[f"{name}_box_{label}_std"] = float(sequence.std())
                record[f"{name}_box_{label}_first"] = float(sequence[0])
                record[f"{name}_box_{label}_last"] = float(sequence[-1])
                record[f"{name}_box_{label}_delta"] = float(sequence[-1] - sequence[0])
                record[f"{name}_box_{label}_motion"] = float(np.abs(np.diff(sequence)).mean()) if len(sequence) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def features(rows, sensor_cache, crop_cache, split, include_boxes):
    base = feature_table(rows, sensor_cache, crop_cache, split).reset_index(drop=True)
    if include_boxes:
        base = pd.concat([base, box_feature_table(rows, crop_cache, split)], axis=1)
    return base


def score(parameters, partitions, include_boxes):
    result = {}
    for fold, item in partitions.items():
        model = train(item["train"][include_boxes], item["rows_train"].label, parameters)
        probabilities = model.predict(item["valid"][include_boxes])
        result[str(fold)] = float((probabilities.argmax(1) == item["rows_valid"].label.to_numpy()).mean())
    weighted = sum(result[str(f)] * len(partitions[f]["rows_valid"]) for f in partitions) / sum(len(v["rows_valid"]) for v in partitions.values())
    return result, weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="configs/sep14/m01_yolo_box_grid.json")
    parser.add_argument("--manifest", default="manifests/cv5/train.csv")
    parser.add_argument("--test-manifest", default="manifests/cv5/test.csv")
    parser.add_argument("--test-csv", default="../Small-Model-Track/Testing/test_file/test.csv")
    parser.add_argument("--sensor-cache", default="artifacts/sep12/serial_depth_align/sensors")
    parser.add_argument("--cache-root", default="artifacts/sep12/serial_depth_align/cache")
    parser.add_argument("--detector2", default="artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt")
    parser.add_argument("--detector4", default="artifacts/sep12/serial_depth_align/jobs/detector_4/attempt_1/yolov8n_4.pt")
    parser.add_argument("--detector-full", default="artifacts/sep12/serial_depth_align/jobs/detector_full/attempt_1/yolov8n_full.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True)
    grid, manifest = json.loads(Path(args.grid).read_text()), read_frozen_manifest(args.manifest)
    partitions = {}
    for fold, detector in ((2, args.detector2), (4, args.detector4)):
        audit_detector(detector, manifest, str(fold))
        cache = Path(args.cache_root) / f"fold_{fold}"
        validate_cache(cache, "train", args.manifest, fold, detector)
        rows_train, rows_valid = fold_partition(manifest, fold)
        print(f"m01_box fold={fold}: building visual, sensor, and box features", flush=True)
        partitions[fold] = {"rows_train": rows_train, "rows_valid": rows_valid,
            "train": {False: features(rows_train, args.sensor_cache, cache, "train", False), True: features(rows_train, args.sensor_cache, cache, "train", True)},
            "valid": {False: features(rows_valid, args.sensor_cache, cache, "train", False), True: features(rows_valid, args.sensor_cache, cache, "train", True)}}
    reference = grid["reference"]
    legacy_folds, legacy_weighted = score(reference, partitions, False)
    box_baseline_folds, box_baseline_weighted = score(reference, partitions, True)
    expected = grid["reference_folds"]
    if legacy_folds != expected:
        raise ValueError(f"Legacy feature baseline changed: {legacy_folds} != {expected}")
    # The box-feature baseline is itself a valid candidate.  It must take part
    # in selection; otherwise a later grid point can replace a better measured
    # baseline merely because the old implementation initialized from legacy
    # features instead of from this result.
    trials = []
    if box_baseline_weighted > legacy_weighted:
        winner, winner_folds, winner_weighted = reference, box_baseline_folds, box_baseline_weighted
    else:
        winner, winner_folds, winner_weighted = reference, legacy_folds, legacy_weighted
    for number, candidate in enumerate(grid["candidates"], 1):
        folds, weighted = score(candidate, partitions, True)
        accepted = all(folds[str(f)] >= legacy_folds[str(f)] for f in partitions) and weighted > legacy_weighted
        trials.append({"parameters": candidate, "folds": folds, "weighted": weighted, "accepted": accepted})
        print(json.dumps({"m01_box_trial": number, **trials[-1]}), flush=True)
        if accepted and weighted > winner_weighted:
            winner, winner_folds, winner_weighted = candidate, folds, weighted
    use_boxes = winner_weighted > legacy_weighted
    for fold, item in partitions.items():
        directory = output / f"fold_{fold}"; directory.mkdir()
        model = train(item["train"][use_boxes], item["rows_train"].label, winner)
        p = model.predict(item["valid"][use_boxes]); model.save_model(str(directory / "model.txt"))
        write_predictions(item["rows_valid"], p, directory / "validation_predictions.csv")
        write_json(directory / "summary.json", {"fold": fold, "validation_accuracy": winner_folds[str(fold)], "parameters": {**BASE, **winner}, "uses_box_features": use_boxes})
    audit_detector(args.detector_full, manifest, "full")
    cache = Path(args.cache_root) / "full"; validate_cache(cache, "train", args.manifest, "full", args.detector_full); validate_cache(cache, "test", args.test_manifest, "full", args.detector_full)
    full = output / "full"; full.mkdir(); model = train(features(manifest, args.sensor_cache, cache, "train", use_boxes), manifest.label, winner); model.save_model(str(full / "model.txt"))
    test = pd.read_csv(args.test_manifest).fillna(""); p = model.predict(features(test, args.sensor_cache, cache, "test", use_boxes)); write_predictions(test, p, full / "test_predictions.csv"); submit(test, p, args.test_csv, output / "submission.csv")
    write_json(output / "comparison.json", {"legacy": {"folds": legacy_folds, "weighted": legacy_weighted}, "box_baseline": {"folds": box_baseline_folds, "weighted": box_baseline_weighted}, "selected": {"parameters": {**BASE, **winner}, "folds": winner_folds, "weighted": winner_weighted, "uses_box_features": use_boxes}, "trials": trials, "inference_bytes_including_detector": check_size([full / "model.txt", args.detector_full]), "manifest_sha256": digest(args.manifest)})


if __name__ == "__main__":
    main()
