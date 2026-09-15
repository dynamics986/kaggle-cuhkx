"""Leakage-safe AutoGluon baseline for the CUHK-X Small Model Track.

This is deliberately a *clip* classifier: it converts each variable-length
sensor recording (and a small set of inexpensive visual summaries) into one
row, then trains AutoGluon Tabular on subject-held-out folds.  It is useful as
an independent model to compare with, or later blend with, the neural model;
do not select it from the public leaderboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .features import cache_key
from .splits import fold_partition
from .submission import validate_submission

LABEL = "label"
NUM_CLASSES = 40
SEQUENCE_NAMES = ("skeleton", "imu", "radar")
DEFAULT_MIN_FREE_MEMORY_GB = 6.0


def _sequence_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return temporal statistics without depending on clip length or timing."""
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D {prefix} sequence, received {values.shape}")
    first, last = values[0], values[-1]
    delta = np.diff(values, axis=0) if len(values) > 1 else np.zeros_like(values)
    # Mean spectral magnitude captures repetitive motions while preserving a compact table.
    spectrum = np.abs(np.fft.rfft(values - values.mean(axis=0), axis=0))[1:]
    spectral = spectrum.mean(axis=0) if len(spectrum) else np.zeros(values.shape[1])
    summaries = {
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "first": first,
        "last": last,
        "change": last - first,
        "abs_delta": np.abs(delta).mean(axis=0),
        "spectral": spectral,
    }
    return {
        f"{prefix}_{stat}_{index}": float(value)
        for stat, vector in summaries.items()
        for index, value in enumerate(vector)
    }


def _visual_summary(directory: Path, prefix: str, frames: int) -> dict[str, float]:
    """Summarize uniformly sampled images; absent/corrupt streams become zeros."""
    files = (
        sorted(
            path
            for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            for path in directory.glob(extension)
        )
        if directory.is_dir()
        else []
    )
    output = {
        f"{prefix}_{name}_{channel}": 0.0
        for name in ("mean", "std", "motion")
        for channel in range(3)
    }
    output[f"{prefix}_available"] = float(bool(files))
    if not files:
        return output
    indices = np.linspace(0, len(files) - 1, min(frames, len(files)), dtype=int)
    images: list[np.ndarray] = []
    for index in indices:
        try:
            with Image.open(files[index]) as image:
                images.append(np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0)
        except (OSError, ValueError):
            continue
    if not images:
        return output
    stack = np.stack(images)
    means, stds = stack.mean(axis=(0, 1, 2)), stack.std(axis=(0, 1, 2))
    motion = np.abs(np.diff(stack, axis=0)).mean(axis=(0, 1, 2)) if len(stack) > 1 else np.zeros(3)
    for name, values in (("mean", means), ("std", stds), ("motion", motion)):
        for channel, value in enumerate(values):
            output[f"{prefix}_{name}_{channel}"] = float(value)
    return output


def make_feature_table(
    manifest_path: str | Path,
    cache_dir: str | Path,
    data_root: str | Path,
    split: str,
    include_visual: bool = True,
    visual_frames: int = 12,
) -> pd.DataFrame:
    """Build a feature table from an existing sensor cache and manifest.

    The cache must be built with the same ``split`` and sensor step count used
    by the neural pipeline.  No labels enter this transformation.
    """
    manifest = pd.read_csv(manifest_path).fillna("")
    cache, root = Path(cache_dir), Path(data_root)
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        record: dict[str, Any] = {
            "clip_id": str(item["clip_id"]),
            LABEL: int(item[LABEL]),
            # Retained solely for fold_partition's subject-leakage assertion;
            # it is explicitly removed before the model is fitted.
            "user": str(item["user"]),
        }
        feature_file = cache / cache_key(split, record["clip_id"])
        if not feature_file.is_file():
            raise FileNotFoundError(f"Missing feature cache for {record['clip_id']}: {feature_file}")
        with np.load(feature_file) as arrays:
            for name in SEQUENCE_NAMES:
                record.update(_sequence_summary(arrays[name], name))
            for index, present in enumerate(arrays["sensor_mask"]):
                record[f"sensor_present_{index}"] = int(present)
        if include_visual:
            for modality in ("Depth_Color", "IR", "Thermal"):
                record.update(_visual_summary(root / str(item[modality]), modality.lower(), visual_frames))
        rows.append(record)
    return pd.DataFrame(rows)


def _import_predictor() -> Any:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as error:  # Keep feature generation usable without AutoGluon.
        raise RuntimeError(
            "AutoGluon is required for training. Install it in this environment, "
            "for example: pip install 'autogluon.tabular[all]'"
        ) from error
    return TabularPredictor


def _gpu_available(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if device == "cuda" and not available:
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable to PyTorch")
    return available


def _hyperparameters(use_gpu: bool) -> dict[str, dict[str, Any]]:
    """Compact candidates with GPU assigned only to backends that support it here."""
    # CatBoost's 40-class GPU training exceeds the available 8 GB VRAM.  Its
    # CPU implementation remains a strong and much more stable candidate.
    cat: dict[str, Any] = {
        "iterations": 1200,
        "depth": 8,
        "learning_rate": 0.04,
        "ag_args_fit": {"num_gpus": 0},
    }
    xgb: dict[str, Any] = {"n_estimators": 1000, "max_depth": 8, "learning_rate": 0.035}
    if use_gpu:
        xgb["device"] = "cuda"
    xgb["ag_args_fit"] = {"num_gpus": int(use_gpu)}
    return {
        "CAT": cat,
        "XGB": xgb,
        # Windows' standard LightGBM wheel is CPU-only. CPU is also normally
        # higher quality than GPU LightGBM on a small tabular dataset.
        "GBM": {
            "num_boost_round": 1000,
            "learning_rate": 0.035,
            "num_leaves": 48,
            "ag_args_fit": {"num_gpus": 0},
        },
        "NN_TORCH": {
            "num_epochs": 80,
            "learning_rate": 2e-3,
            "dropout_prob": 0.1,
            "ag_args_fit": {"num_gpus": int(use_gpu)},
        },
    }


def _assert_free_memory(minimum_gb: float) -> None:
    """Stop before AutoGluon silently skips its strongest candidate models."""
    if minimum_gb <= 0:
        return
    try:
        import psutil
    except ImportError:
        return
    available_gb = psutil.virtual_memory().available / 1024**3
    if available_gb < minimum_gb:
        raise RuntimeError(
            f"Only {available_gb:.2f} GB system RAM is currently free; this AutoGluon run needs "
            f"at least {minimum_gb:.1f} GB to train CatBoost and XGBoost instead of skipping them. "
            "Close memory-heavy programs or reboot, then retry. Use --min-free-memory-gb 0 only "
            "for a deliberately reduced-memory smoke test."
        )


def _directory_size_mb(directory: Path) -> float:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) / 1024**2


def _write_model_card(predictor: Any, leaderboard: pd.DataFrame, fold_dir: Path) -> dict[str, Any]:
    """Save the chosen AutoGluon model's reproducible, human-readable details."""
    best_model = predictor.model_best
    model_row = leaderboard.loc[leaderboard["model"] == best_model]
    if model_row.empty:
        raise RuntimeError(f"AutoGluon best model {best_model!r} is absent from its leaderboard")
    info = predictor.info()
    model_info = info.get("model_info", {}).get(best_model, {})
    model_path = Path(predictor.path) / "models" / best_model
    card: dict[str, Any] = {
        "best_model": best_model,
        "validation_accuracy": float(model_row.iloc[0]["score_val"]),
        "fit_time_seconds": float(model_row.iloc[0]["fit_time"]),
        "validation_prediction_time_seconds": float(model_row.iloc[0]["pred_time_val"]),
        "best_model_size_mb": _directory_size_mb(model_path),
        "all_predictors_size_mb": _directory_size_mb(Path(predictor.path)),
        "model_type": model_info.get("model_type"),
        "hyperparameters": model_info.get("hyperparameters", {}),
        "predictor_path": str(Path(predictor.path).resolve()),
    }
    (fold_dir / "best_model.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return card


def fit_fold(
    features: pd.DataFrame,
    fold: int,
    output_dir: Path,
    device: str,
    time_limit: int | None,
    min_free_memory_gb: float,
) -> pd.DataFrame:
    """Fit one held-out-subject fold and return its class probabilities."""
    train, valid = fold_partition(features, fold)
    TabularPredictor = _import_predictor()
    model_dir = output_dir / f"fold_{fold}" / "predictor"
    if model_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing AutoGluon predictor: {model_dir}. "
            "Choose a new --output-dir."
        )
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    use_gpu = _gpu_available(device)
    _assert_free_memory(min_free_memory_gb)
    predictor = TabularPredictor(
        label=LABEL,
        problem_type="multiclass",
        eval_metric="accuracy",
        path=str(model_dir),
        verbosity=2,
        learner_kwargs={"label_count_threshold": 1},
    ).fit(
        train_data=train.drop(columns=["clip_id", "user", "fold"]),
        # Do not use high_quality: it enables refit_full and can retain an
        # additional model copy. Both are unsuitable for a 100 MB deployment.
        presets="medium_quality",
        hyperparameters=_hyperparameters(use_gpu),
        auto_stack=False,
        num_bag_folds=0,
        num_stack_levels=0,
        fit_weighted_ensemble=False,
        refit_full=False,
        time_limit=time_limit,
        num_gpus=1 if use_gpu else 0,
    )
    probabilities = predictor.predict_proba(
        valid.drop(columns=[LABEL, "clip_id", "user", "fold"]), as_pandas=True
    )
    probabilities = probabilities.reindex(columns=range(NUM_CLASSES), fill_value=0.0)
    result = valid[["clip_id", LABEL]].copy()
    result["prediction"] = probabilities.to_numpy().argmax(axis=1)
    for class_id in range(NUM_CLASSES):
        result[f"prob_{class_id}"] = probabilities[class_id].to_numpy()
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(fold_dir / "validation_predictions.csv", index=False)
    leaderboard = predictor.leaderboard(
        valid.drop(columns=["clip_id", "user", "fold"]), silent=True
    )
    leaderboard.to_csv(fold_dir / "leaderboard.csv", index=False)
    _write_model_card(predictor, leaderboard, fold_dir)
    return result


def run_cv(
    features_path: Path,
    output_dir: Path,
    device: str,
    time_limit: int | None,
    min_free_memory_gb: float,
) -> None:
    features = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    if "fold" not in features or set(features["fold"].unique()) != set(range(5)):
        raise ValueError("Training feature table must contain the frozen folds 0..4")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = [
        fit_fold(features, fold, output_dir, device, time_limit, min_free_memory_gb)
        for fold in range(5)
    ]
    oof = pd.concat(all_predictions, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    cards = [
        json.loads((output_dir / f"fold_{fold}" / "best_model.json").read_text(encoding="utf-8"))
        for fold in range(5)
    ]
    summary = {
        "oof_accuracy": float((oof[LABEL] == oof["prediction"]).mean()),
        "rows": len(oof),
        "fold_winners": [
            {
                "fold": fold,
                "model": card["best_model"],
                "validation_accuracy": card["validation_accuracy"],
                "size_mb": card["best_model_size_mb"],
            }
            for fold, card in enumerate(cards)
        ],
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def make_submission(model_dir: Path, test_features_path: Path, test_csv_path: Path, output: Path) -> None:
    """Average the five CV predictors and write the official, order-preserving CSV."""
    TabularPredictor = _import_predictor()
    features = (
        pd.read_parquet(test_features_path)
        if test_features_path.suffix == ".parquet"
        else pd.read_csv(test_features_path)
    )
    model_columns = [column for column in features if column not in {LABEL, "clip_id", "user", "fold"}]
    mean_probabilities = np.zeros((len(features), NUM_CLASSES), dtype=np.float64)
    for fold in range(5):
        predictor = TabularPredictor.load(str(model_dir / f"fold_{fold}" / "predictor"))
        probabilities = predictor.predict_proba(features[model_columns], as_pandas=True)
        mean_probabilities += probabilities.reindex(columns=range(NUM_CLASSES), fill_value=0.0).to_numpy()
    predictions = mean_probabilities.argmax(axis=1)
    by_clip = dict(zip(features["clip_id"], predictions, strict=True))
    official = pd.read_csv(test_csv_path)
    submission = pd.DataFrame(
        {
            "path": official["path"],
            "prediction": [by_clip[Path(str(path).rstrip("/\\")).name] for path in official["path"]],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    validate_submission(output, test_csv_path)
    print(f"Wrote valid five-fold ensemble submission to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGluon CUHK-X clip classifier")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("features", help="build a reusable clip-level feature table")
    build.add_argument("--manifest", required=True)
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--data-root", required=True)
    build.add_argument("--split", choices=("train", "test"), required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--no-visual", action="store_true")
    build.add_argument("--visual-frames", type=int, default=12)
    cv = subcommands.add_parser("cv", help="run all five frozen subject-held-out folds")
    cv.add_argument("--features", required=True)
    cv.add_argument("--output-dir", default="artifacts/automl")
    cv.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    cv.add_argument("--time-limit", type=int, help="seconds per fold; omit for no limit")
    cv.add_argument(
        "--min-free-memory-gb",
        type=float,
        default=DEFAULT_MIN_FREE_MEMORY_GB,
        help="fail before training if less system RAM is free (default: 6 GB; 0 disables the guard)",
    )
    submit = subcommands.add_parser("submit", help="average five trained folds into an official submission")
    submit.add_argument("--model-dir", required=True, help="output directory previously used by cv")
    submit.add_argument("--test-features", required=True)
    submit.add_argument("--test-csv", required=True)
    submit.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "features":
        table = make_feature_table(args.manifest, args.cache_dir, args.data_root, args.split, not args.no_visual, args.visual_frames)
        if args.split == "train":
            source = pd.read_csv(args.manifest)
            table["fold"] = source["fold"].to_numpy()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(output, index=False) if output.suffix == ".parquet" else table.to_csv(output, index=False)
        print(f"Wrote {len(table)} clips and {len(table.columns)} columns to {output.resolve()}")
    elif args.command == "cv":
        run_cv(
            Path(args.features),
            Path(args.output_dir),
            args.device,
            args.time_limit,
            args.min_free_memory_gb,
        )
    else:
        make_submission(
            Path(args.model_dir), Path(args.test_features), Path(args.test_csv), Path(args.output)
        )


if __name__ == "__main__":
    main()
