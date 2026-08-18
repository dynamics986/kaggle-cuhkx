"""Strict OOF validation and action-level comparison for modality experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import MODALITIES, NUM_CLASSES
from .data import select_present_rows
from .frozen import read_frozen_manifest

PROBABILITY_COLUMNS = [f"prob_{index}" for index in range(NUM_CLASSES)]
REQUIRED_COLUMNS = {"clip_id", "label", "prediction", *PROBABILITY_COLUMNS}


def _read_predictions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if missing := REQUIRED_COLUMNS - set(frame.columns):
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    frame = frame.loc[:, ["clip_id", "label", "prediction", *PROBABILITY_COLUMNS]].copy()
    if frame["clip_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate clip_id values")
    if not frame["label"].between(0, NUM_CLASSES - 1).all() or not frame["prediction"].between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"{path} contains an invalid label or prediction")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < -1e-8).any():
        raise ValueError(f"{path} contains invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{path} probabilities must sum to one per clip")
    if not np.array_equal(probabilities.argmax(axis=1), frame["prediction"].to_numpy()):
        raise ValueError(f"{path} prediction must equal probability argmax")
    return frame


def evaluate_modality_oof(
    manifest_path: str | Path, cache_dir: str | Path, modality: str, prediction_paths: list[str | Path]
) -> tuple[dict[str, object], pd.DataFrame]:
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}")
    manifest = read_frozen_manifest(manifest_path)
    expected = select_present_rows(manifest, modality, cache_dir).set_index("clip_id")
    folds = range(5)
    if len(prediction_paths) != len(folds):
        raise ValueError("Expected one OOF prediction file for each frozen CV5 fold")
    parts: list[pd.DataFrame] = []
    assigned: set[int] = set()
    for path in prediction_paths:
        predictions = _read_predictions(path)
        unknown = set(predictions["clip_id"]) - set(expected.index)
        if unknown:
            raise ValueError(f"{path} includes clips without available {modality}")
        expected_rows = expected.loc[predictions["clip_id"]]
        if not np.array_equal(predictions["label"].to_numpy(), expected_rows["label"].to_numpy()):
            raise ValueError(f"{path} labels do not match the frozen manifest")
        file_folds = expected_rows["fold"].unique()
        if len(file_folds) != 1:
            raise ValueError(f"{path} mixes validation folds")
        fold = int(file_folds[0])
        expected_ids = set(expected.index[expected["fold"] == fold])
        if set(predictions["clip_id"]) != expected_ids:
            raise ValueError(f"{path} does not cover exactly the available {modality} clips in fold {fold}")
        if fold in assigned:
            raise ValueError(f"More than one prediction file was supplied for fold {fold}")
        assigned.add(fold)
        predictions["fold"] = fold
        predictions["action_name"] = expected_rows["action_name"].to_numpy()
        parts.append(predictions)
    if assigned != set(folds):
        raise ValueError(f"Missing OOF folds: {sorted(set(folds) - assigned)}")
    oof = pd.concat(parts, ignore_index=True).sort_values("clip_id").reset_index(drop=True)
    oof.insert(0, "modality", modality)
    correct = oof["prediction"].eq(oof["label"])
    report: dict[str, object] = {
        "modality": modality,
        "examples": int(len(oof)),
        "correct": int(correct.sum()),
        "oof_accuracy": float(correct.mean()),
        "fold_accuracy": {str(fold): float(correct.loc[oof["fold"] == fold].mean()) for fold in folds},
    }
    return report, oof


def write_modality_report(
    manifest_path: str | Path, cache_dir: str | Path, modality: str, prediction_paths: list[str | Path], output: str | Path
) -> dict[str, object]:
    report, oof = evaluate_modality_oof(manifest_path, cache_dir, modality, prediction_paths)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    oof_path = output_path.with_suffix(".oof.csv")
    oof.to_csv(oof_path, index=False)
    report["oof_path"] = str(oof_path)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def summarize_actions(oof_paths: list[str | Path], output: str | Path) -> pd.DataFrame:
    if len(oof_paths) != len(MODALITIES):
        raise ValueError("Provide exactly six OOF files, one for each modality")
    frames = [pd.read_csv(path) for path in oof_paths]
    by_modality = {str(frame["modality"].iloc[0]): frame for frame in frames if not frame.empty}
    if set(by_modality) != set(MODALITIES):
        raise ValueError("OOF files must contain every modality exactly once")
    labels = pd.concat([frame[["label", "action_name"]] for frame in frames]).drop_duplicates().sort_values("label")
    result = labels.reset_index(drop=True)
    accuracy_columns: list[str] = []
    for modality in MODALITIES:
        frame = by_modality[modality].assign(correct=lambda values: values["label"] == values["prediction"])
        metrics = frame.groupby("label")["correct"].agg([("accuracy", "mean"), ("samples", "count")]).reset_index()
        result = result.merge(metrics.rename(columns={"accuracy": f"{modality}_accuracy", "samples": f"{modality}_samples"}), on="label", how="left")
        accuracy_columns.append(f"{modality}_accuracy")
    def winners(row: pd.Series) -> str:
        available = row[accuracy_columns].dropna()
        if available.empty:
            return ""
        maximum = available.max()
        return "|".join(column.removesuffix("_accuracy") for column in available.index if np.isclose(available[column], maximum))
    result["best_modality"] = result.apply(winners, axis=1)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main_report() -> None:
    parser = argparse.ArgumentParser(description="Validate and combine one modality's frozen-CV5 OOF predictions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--modality", required=True, choices=MODALITIES)
    parser.add_argument("--predictions", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = write_modality_report(args.manifest, args.cache_dir, args.modality, args.predictions, args.output)
    print(f"{args.modality}: OOF accuracy={report['oof_accuracy']:.5f} ({report['correct']}/{report['examples']})")


def main_summary() -> None:
    parser = argparse.ArgumentParser(description="Create per-action accuracy table for all six CUHK-X modalities")
    parser.add_argument("--oof", required=True, nargs="+", help="Six *.oof.csv paths, one per modality")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    table = summarize_actions(args.oof, args.output)
    print(f"Wrote {len(table)} actions to {Path(args.output).resolve()}")
