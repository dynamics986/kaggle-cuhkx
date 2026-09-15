"""Robust LightGBM selection: both selected folds must hold before a refit is accepted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cuhkx_har.splits import fold_partition
from cuhkx_modality.frozen import read_frozen_manifest
from cuhkx_sep12.common import check_size, digest, submit, write_json, write_predictions
from cuhkx_sep12.data import validate_cache
from cuhkx_sep12.lightgbm import feature_table
from cuhkx_sep12.prepare import audit_detector

BASE = {"learning_rate": 0.035, "num_boost_round": 1000, "seed": 0}


def train(features, labels, parameters):
    import lightgbm as lgb

    if set(labels) != set(range(40)):
        raise ValueError("Every LightGBM fit must contain all 40 classes")
    params = {
        **BASE,
        **parameters,
        "objective": "multiclass",
        "num_class": 40,
        "metric": "multi_logloss",
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 8,
    }
    rounds = params.pop("num_boost_round")
    return lgb.train(params, lgb.Dataset(features, label=labels), num_boost_round=rounds)


def evaluate(candidate, partitions):
    result = {}
    for fold, item in partitions.items():
        model = train(item["x_train"], item["train"].label, candidate)
        probabilities = model.predict(item["x_valid"])
        result[str(fold)] = float((probabilities.argmax(1) == item["valid"].label.to_numpy()).mean())
    weighted = sum(result[str(f)] * len(partitions[f]["valid"]) for f in partitions) / sum(
        len(partitions[f]["valid"]) for f in partitions
    )
    return result, weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="configs/sep13/m01_robust_grid.json")
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
    grid = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    baseline = grid["baseline"]
    manifest = read_frozen_manifest(args.manifest)
    partitions = {}
    for fold, detector in ((2, args.detector2), (4, args.detector4)):
        audit_detector(detector, manifest, str(fold))
        cache = Path(args.cache_root) / f"fold_{fold}"
        validate_cache(cache, "train", args.manifest, fold, detector)
        train_rows, valid_rows = fold_partition(manifest, fold)
        print(f"m01 fold={fold}: materialising train/validation features once", flush=True)
        partitions[fold] = {
            "train": train_rows,
            "valid": valid_rows,
            "x_train": feature_table(train_rows, args.sensor_cache, cache, "train"),
            "x_valid": feature_table(valid_rows, args.sensor_cache, cache, "train"),
        }
    baseline_folds, baseline_score = evaluate(baseline, partitions)
    trials = [{"parameters": baseline, "folds": baseline_folds, "weighted": baseline_score, "baseline": True}]
    winner, winner_folds, winner_score = baseline, baseline_folds, baseline_score
    for number, candidate in enumerate(grid["candidates"], 1):
        folds, score = evaluate(candidate, partitions)
        accepted = all(folds[str(f)] >= baseline_folds[str(f)] for f in partitions) and score > baseline_score
        trial = {"parameters": candidate, "folds": folds, "weighted": score, "accepted": accepted}
        trials.append(trial)
        print(json.dumps({"m01_trial": number, **trial}), flush=True)
        if accepted and score > winner_score:
            winner, winner_folds, winner_score = candidate, folds, score

    # Persist selected-fold predictions from fresh models, then perform the only full-data fit.
    for fold, item in partitions.items():
        directory = output / f"fold_{fold}"
        directory.mkdir()
        model = train(item["x_train"], item["train"].label, winner)
        probabilities = model.predict(item["x_valid"])
        model.save_model(str(directory / "model.txt"))
        write_predictions(item["valid"], probabilities, directory / "validation_predictions.csv")
        write_json(directory / "summary.json", {
            "fold": fold, "validation_accuracy": winner_folds[str(fold)], "parameters": {**BASE, **winner},
            "train_rows": len(item["train"]), "validation_rows": len(item["valid"]),
        })

    audit_detector(args.detector_full, manifest, "full")
    full_cache = Path(args.cache_root) / "full"
    validate_cache(full_cache, "train", args.manifest, "full", args.detector_full)
    validate_cache(full_cache, "test", args.test_manifest, "full", args.detector_full)
    x_full = feature_table(manifest, args.sensor_cache, full_cache, "train")
    full = output / "full"
    full.mkdir()
    model = train(x_full, manifest.label, winner)
    model.save_model(str(full / "model.txt"))
    test = pd.read_csv(args.test_manifest).fillna("")
    probabilities = model.predict(feature_table(test, args.sensor_cache, full_cache, "test"))
    write_predictions(test, probabilities, full / "test_predictions.csv")
    submit(test, probabilities, args.test_csv, output / "submission.csv")
    report = {
        "rule": "accept only candidates with no selected-fold decrease and strictly higher weighted score",
        "baseline": {"parameters": {**BASE, **baseline}, "folds": baseline_folds, "weighted": baseline_score},
        "selected": {"parameters": {**BASE, **winner}, "folds": winner_folds, "weighted": winner_score,
                     "improved": winner_score > baseline_score},
        "trials": trials,
        "inference_bytes_including_detector": check_size([full / "model.txt", args.detector_full]),
        "manifest_sha256": digest(args.manifest),
    }
    write_json(output / "comparison.json", report)
    print(json.dumps(report["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
