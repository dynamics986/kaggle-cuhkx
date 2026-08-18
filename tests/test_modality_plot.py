from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuhkx_modality.plot import plot_training_curves, read_history


def _history() -> list[dict[str, float]]:
    return [
        {
            "epoch": 1,
            "train_loss": 2.1,
            "valid_loss": 2.3,
            "train_accuracy": 0.2,
            "valid_accuracy": 0.1,
        },
        {
            "epoch": 2,
            "train_loss": 1.5,
            "valid_loss": 1.8,
            "train_accuracy": 0.5,
            "valid_accuracy": 0.4,
        },
    ]


def test_plot_training_curves_writes_default_and_requested_png(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(json.dumps(_history()), encoding="utf-8")

    default_output = plot_training_curves(tmp_path)
    requested_output = plot_training_curves(tmp_path, tmp_path / "nested" / "curves.png")

    assert default_output == tmp_path / "training_curves.png"
    assert default_output.stat().st_size > 0
    assert requested_output.is_file()
    assert requested_output.stat().st_size > 0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("[]", "non-empty"),
        (json.dumps([{"epoch": 1}]), "missing metrics"),
        (json.dumps([{**_history()[0], "train_loss": "nan"}]), "non-finite"),
    ],
)
def test_read_history_rejects_invalid_metrics(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "history.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_history(path)


def test_read_history_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing training history"):
        read_history(tmp_path / "history.json")
