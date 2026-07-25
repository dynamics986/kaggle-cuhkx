"""Strict, subject-held-out evaluation for CUHK-X experiments.

The competition metric is clip-level top-1 accuracy.  This module reports the
same metric on out-of-fold (OOF) predictions, while also proving that every
labelled clip was evaluated exactly once by a model that did not train on its
subject.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import NUM_CLASSES
from .splits import assert_no_subject_leakage

PROBABILITY_COLUMNS = [f"prob_{index}" for index in range(NUM_CLASSES)]
REQUIRED_PREDICTION_COLUMNS = {"clip_id", "label", "prediction", *PROBABILITY_COLUMNS}
REQUIRED_MANIFEST_COLUMNS = {"clip_id", "label", "user", "fold"}


def _read_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    frame = frame.loc[:, ["clip_id", "label", "prediction", *PROBABILITY_COLUMNS]].copy()
    if frame["clip_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate clip_id values")
    if not frame["label"].between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"{path} contains an invalid label")
    if not frame["prediction"].between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"{path} contains an invalid prediction")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < -1e-8).any():
        raise ValueError(f"{path} contains invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{path} probabilities must sum to one per clip")
    if not np.array_equal(probabilities.argmax(axis=1), frame["prediction"].to_numpy()):
        raise ValueError(f"{path} prediction must equal argmax(prob_0..prob_39)")
    return frame


def evaluate_oof(manifest_path: str | Path, prediction_paths: list[str | Path]) -> dict[str, object]:
    """Validate OOF coverage and return competition-aligned CV metrics.

    A prediction file must correspond to one complete validation fold.  Files
    may be passed in any order; fold identity is inferred from their clip ids.
    """
    manifest = pd.read_csv(manifest_path).fillna("")
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["clip_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate clip_id values")
    if not manifest["label"].between(0, NUM_CLASSES - 1).all():
        raise ValueError("Manifest contains an invalid label")
    assert_no_subject_leakage(manifest)

    folds = sorted(int(value) for value in manifest["fold"].unique())
    if len(prediction_paths) != len(folds):
        raise ValueError(
            f"Expected one OOF file for each of {len(folds)} folds, got {len(prediction_paths)}"
        )

    expected_by_id = manifest.set_index("clip_id")
    assigned_folds: set[int] = set()
    parts: list[pd.DataFrame] = []
    sources: dict[str, str] = {}
    for path in prediction_paths:
        predictions = _read_predictions(path)
        unknown = set(predictions["clip_id"]) - set(expected_by_id.index)
        if unknown:
            raise ValueError(f"{path} contains clips absent from the manifest")
        expected = expected_by_id.loc[predictions["clip_id"]]
        if not np.array_equal(predictions["label"].to_numpy(), expected["label"].to_numpy()):
            raise ValueError(f"{path} labels do not match the manifest")
        file_folds = expected["fold"].unique()
        if len(file_folds) != 1:
            raise ValueError(f"{path} mixes validation folds")
        fold = int(file_folds[0])
        if fold in assigned_folds:
            raise ValueError(f"More than one prediction file was supplied for fold {fold}")
        expected_ids = set(manifest.loc[manifest["fold"] == fold, "clip_id"])
        if set(predictions["clip_id"]) != expected_ids:
            raise ValueError(f"{path} does not cover exactly the held-out clips of fold {fold}")
        assigned_folds.add(fold)
        predictions["fold"] = fold
        predictions["user"] = expected["user"].to_numpy()
        parts.append(predictions)
        sources[str(fold)] = str(Path(path).resolve())

    if assigned_folds != set(folds):
        raise ValueError(f"Missing OOF folds: {sorted(set(folds) - assigned_folds)}")
    oof = pd.concat(parts, ignore_index=True).sort_values("clip_id").reset_index(drop=True)
    correct = oof["prediction"].eq(oof["label"])
    fold_accuracy = {
        str(fold): float(correct.loc[oof["fold"] == fold].mean()) for fold in folds
    }
    per_class = (
        oof.assign(correct=correct)
        .groupby("label", sort=True)["correct"]
        .agg([("accuracy", "mean"), ("examples", "count")])
        .reset_index()
    )
    confusion = pd.crosstab(oof["label"], oof["prediction"], dropna=False)
    confusion = confusion.reindex(index=range(NUM_CLASSES), columns=range(NUM_CLASSES), fill_value=0)
    return {
        "metric": "clip_top1_accuracy",
        "metric_definition": "mean(prediction == label); matches Kaggle clip-accuracy scoring",
        "examples": int(len(oof)),
        "correct": int(correct.sum()),
        "oof_accuracy": float(correct.mean()),
        "fold_accuracy": fold_accuracy,
        "fold_mean_accuracy": float(np.mean(list(fold_accuracy.values()))),
        "fold_std_accuracy": float(np.std(list(fold_accuracy.values()), ddof=0)),
        "subjects_per_fold": {
            str(fold): sorted(manifest.loc[manifest["fold"] == fold, "user"].unique().tolist())
            for fold in folds
        },
        "prediction_sources": sources,
        "per_class": per_class.to_dict(orient="records"),
        "confusion_matrix": confusion.to_numpy(dtype=int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict subject-held-out OOF evaluation")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", nargs="+", required=True, type=Path)
    parser.add_argument("--name", default="experiment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_oof(args.manifest, args.predictions)
    report["experiment"] = args.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.name}: OOF accuracy={report['oof_accuracy']:.5f} "
        f"({report['correct']}/{report['examples']}), "
        f"fold std={report['fold_std_accuracy']:.5f}"
    )


if __name__ == "__main__":
    main()
