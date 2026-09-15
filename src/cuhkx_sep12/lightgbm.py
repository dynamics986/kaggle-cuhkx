"""Method 1: fixed-hyperparameter LightGBM, folds 2/4 then all-training refit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cuhkx_har.automl import _sequence_summary
from cuhkx_har.features import cache_key
from cuhkx_har.splits import fold_partition
from cuhkx_modality.frozen import read_frozen_manifest

from .common import cache_path, check_size, digest, submit, write_json, write_predictions
from .data import validate_cache
from .prepare import audit_detector

PARAMETERS = {"learning_rate": 0.035, "num_boost_round": 1000, "num_leaves": 48, "seed": 0}


def feature_table(rows, sensor_cache, crop_cache, split):
    records = []
    for row in rows.itertuples(index=False):
        record = {}
        with np.load(
            Path(sensor_cache) / cache_key(split, row.clip_id), allow_pickle=False
        ) as saved:
            for name in ("skeleton", "imu", "radar"):
                record.update(_sequence_summary(saved[name], name))
            for index, present in enumerate(saved["sensor_mask"]):
                record[f"sensor_present_{index}"] = int(present)
        with np.load(cache_path(crop_cache, split, row.clip_id), allow_pickle=False) as saved:
            if str(saved["clip_id"]) != row.clip_id:
                raise ValueError("Crop cache clip identity mismatch")
            for m, name in enumerate(("ir", "depth_color", "thermal")):
                present = saved["mask"][m].astype(bool)
                images = saved["images"][m].astype(np.float32) / 255
                valid = images[present]
                adjacent = present[:-1] & present[1:]
                summary = {
                    "mean": valid.mean((0, 1, 2)) if len(valid) else np.zeros(3),
                    "std": valid.std((0, 1, 2)) if len(valid) else np.zeros(3),
                    "motion": np.abs(np.diff(images, axis=0))[adjacent].mean((0, 1, 2))
                    if adjacent.any()
                    else np.zeros(3),
                }
                record[f"{name}_available"] = float(present.any())
                for stat, vector in summary.items():
                    for c, value in enumerate(vector):
                        record[f"{name}_{stat}_{c}"] = float(value)
        records.append(record)
    features = pd.DataFrame(records)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Non-finite features")
    return features


def fit(features, labels, rounds=1000):
    import lightgbm as lgb

    if set(labels) != set(range(40)):
        raise ValueError("LightGBM training requires all 40 classes")
    params = {k: v for k, v in PARAMETERS.items() if k != "num_boost_round"}
    params.update(
        objective="multiclass",
        num_class=40,
        metric="multi_logloss",
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        num_threads=8,
    )
    # No hidden AutoGluon holdout or early stopping: every supplied row is used.
    dataset = lgb.Dataset(features, label=labels)
    return lgb.train(
        params,
        dataset,
        num_boost_round=rounds,
        valid_sets=[dataset],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=50)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifests/cv5/train.csv")
    parser.add_argument("--test-manifest", default="manifests/cv5/test.csv")
    parser.add_argument("--test-csv", default="../Small-Model-Track/Testing/test_file/test.csv")
    parser.add_argument("--sensor-cache", default="cache-64")
    parser.add_argument("--cache-root", default="cache-sep12")
    parser.add_argument("--detector2", default="artifacts/yolo/detectors/fold_2/yolov8n_fold_2.pt")
    parser.add_argument("--detector4", default="artifacts/sep12/yolo/fold_4/yolov8n_4.pt")
    parser.add_argument("--detector-full", default="artifacts/sep12/yolo/full/yolov8n_full.pt")
    parser.add_argument("--output", default="artifacts/sep12/m01_lightgbm")
    parser.add_argument("--stage", choices=["all", "cv", "full", "predict"], default="all")
    args = parser.parse_args()
    manifest = read_frozen_manifest(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    if args.stage in ("all", "cv"):
        for fold in (2, 4):
            print(f"method=1 fold={fold}: building features and training", flush=True)
            detector = args.detector2 if fold == 2 else args.detector4
            audit_detector(detector, manifest, str(fold))
            cache = Path(args.cache_root) / f"fold_{fold}"
            meta = validate_cache(cache, "train", args.manifest, fold, detector)
            train, valid = fold_partition(manifest, fold)
            directory = output / f"fold_{fold}"
            directory.mkdir(parents=True, exist_ok=False)
            x_train = feature_table(train, args.sensor_cache, cache, "train")
            x_valid = feature_table(valid, args.sensor_cache, cache, "train")
            booster = fit(x_train, train.label)
            booster.save_model(str(directory / "model.txt"))
            probabilities = booster.predict(x_valid)
            write_predictions(valid, probabilities, directory / "validation_predictions.csv")
            accuracy = float((probabilities.argmax(1) == valid.label.to_numpy()).mean())
            summary = {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "validation_accuracy": accuracy,
                "parameters": PARAMETERS,
                "crop_metadata": meta,
                "inference_bytes_including_detector": check_size(
                    [directory / "model.txt", detector]
                ),
            }
            write_json(directory / "summary.json", summary)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        write_json(
            output / "comparison.json",
            {"evaluation": "selected folds, not full OOF", "fold_results": summaries},
        )
    full = output / "full"
    if args.stage in ("all", "full"):
        print("method=1 full: building features and training all rows", flush=True)
        audit_detector(args.detector_full, manifest, "full")
        cache = Path(args.cache_root) / "full"
        meta = validate_cache(cache, "train", args.manifest, "full", args.detector_full)
        full.mkdir(parents=True, exist_ok=False)
        features = feature_table(manifest, args.sensor_cache, cache, "train")
        booster = fit(features, manifest.label)
        booster.save_model(str(full / "model.txt"))
        write_json(
            full / "summary.json",
            {
                "training_rows": len(manifest),
                "training_users": sorted(set(manifest.user)),
                "parameters": PARAMETERS,
                "features": list(features.columns),
                "crop_metadata": meta,
                "manifest_sha256": digest(args.manifest),
                "model_sha256": digest(full / "model.txt"),
                "sensor_cache": str(Path(args.sensor_cache).resolve()),
                "inference_bytes_including_detector": check_size(
                    [full / "model.txt", args.detector_full]
                ),
                "validation": "No validation score for full refit; consult fold_2/fold_4 summaries",
            },
        )
    if args.stage in ("all", "predict"):
        import lightgbm as lgb

        summary = json.loads((full / "summary.json").read_text())
        if summary["manifest_sha256"] != digest(args.manifest):
            raise ValueError("Training manifest differs from full fit")
        if summary["model_sha256"] != digest(full / "model.txt"):
            raise ValueError("Full model hash changed")
        if summary["crop_metadata"]["detector_sha256"] != digest(args.detector_full):
            raise ValueError("Full detector differs from training")
        test = pd.read_csv(args.test_manifest).fillna("")
        cache = Path(args.cache_root) / "full"
        meta = validate_cache(cache, "test", args.test_manifest, "full", args.detector_full)
        for key in ("frames", "image_size", "confidence", "padding", "missing_policy", "alignment"):
            if meta[key] != summary["crop_metadata"][key]:
                raise ValueError(f"Train/test crop setting differs: {key}")
        features = feature_table(test, args.sensor_cache, cache, "test")
        if list(features.columns) != summary["features"]:
            raise ValueError("Test feature schema differs from training")
        booster = lgb.Booster(model_file=str(full / "model.txt"))
        probabilities = booster.predict(features)
        check_size([full / "model.txt", args.detector_full])
        write_predictions(test, probabilities, full / "test_predictions.csv")
        submit(test, probabilities, args.test_csv, output / "submission.csv")


if __name__ == "__main__":
    main()
