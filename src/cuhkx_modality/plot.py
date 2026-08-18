"""Render loss and accuracy curves from a single-modality training history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_METRICS = ("epoch", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy")


def read_history(path: str | Path) -> list[dict[str, float]]:
    history_path = Path(path)
    try:
        payload: Any = json.loads(history_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing training history: {history_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON training history: {history_path}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Training history must be a non-empty JSON list: {history_path}")

    result: list[dict[str, float]] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"History entry {index} must be an object")
        missing = set(REQUIRED_METRICS) - set(row)
        if missing:
            raise ValueError(f"History entry {index} is missing metrics: {sorted(missing)}")
        converted: dict[str, float] = {}
        for metric in REQUIRED_METRICS:
            try:
                value = float(row[metric])
            except (TypeError, ValueError) as error:
                raise ValueError(f"History entry {index} has a non-numeric {metric}") from error
            if not math.isfinite(value):
                raise ValueError(f"History entry {index} has a non-finite {metric}")
            converted[metric] = value
        if not converted["epoch"].is_integer() or converted["epoch"] <= 0:
            raise ValueError(f"History entry {index} has an invalid epoch")
        result.append(converted)
    return result


def plot_training_curves(run_dir: str | Path, output: str | Path | None = None) -> Path:
    run_path = Path(run_dir).resolve()
    history = read_history(run_path / "history.json")
    output_path = Path(output).resolve() if output else run_path / "training_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    best = max(history, key=lambda row: row["valid_accuracy"])
    title = f"{run_path.parent.name} | {run_path.name}"
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    figure.suptitle(f"Single-modality training curves: {title}", fontsize=13)

    loss_axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    loss_axis.plot(epochs, [row["valid_loss"] for row in history], marker="o", label="Valid loss")
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")

    accuracy_axis.plot(
        epochs, [row["train_accuracy"] for row in history], marker="o", label="Train accuracy"
    )
    accuracy_axis.plot(
        epochs, [row["valid_accuracy"] for row in history], marker="o", label="Valid accuracy"
    )
    accuracy_axis.scatter(
        [best["epoch"]], [best["valid_accuracy"]], color="tab:red", zorder=3
    )
    accuracy_axis.annotate(
        f"best {best['valid_accuracy']:.3f}\n(epoch {int(best['epoch'])})",
        (best["epoch"], best["valid_accuracy"]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    accuracy_axis.set(title="Accuracy", xlabel="Epoch", ylabel="Top-1 accuracy", ylim=(0, 1))

    for axis in (loss_axis, accuracy_axis):
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot single-modality loss and accuracy curves")
    parser.add_argument("--run-dir", required=True, help="Fold directory containing history.json")
    parser.add_argument("--output", help="Output PNG path; defaults to <run-dir>/training_curves.png")
    args = parser.parse_args()
    output = plot_training_curves(args.run_dir, args.output)
    print(f"Wrote training curves to {output}")
