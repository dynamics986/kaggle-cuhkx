from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cuhkx_har.cv import evaluate_oof


def _prediction_frame(rows: list[tuple[str, int]]) -> pd.DataFrame:
    probabilities = np.zeros((len(rows), 40))
    for index, (_, label) in enumerate(rows):
        probabilities[index, label] = 1.0
    frame = pd.DataFrame(probabilities, columns=[f"prob_{index}" for index in range(40)])
    frame.insert(0, "prediction", [label for _, label in rows])
    frame.insert(0, "label", [label for _, label in rows])
    frame.insert(0, "clip_id", [clip_id for clip_id, _ in rows])
    return frame


def test_evaluate_oof_requires_exact_subject_fold_coverage(tmp_path) -> None:
    manifest = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "user": ["user1", "user1", "user2", "user2"],
            "fold": [0, 0, 1, 1],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    fold0, fold1 = tmp_path / "fold0.csv", tmp_path / "fold1.csv"
    _prediction_frame([("a", 0), ("b", 1)]).to_csv(fold0, index=False)
    _prediction_frame([("c", 0), ("d", 1)]).to_csv(fold1, index=False)

    report = evaluate_oof(manifest_path, [fold1, fold0])

    assert report["metric"] == "clip_top1_accuracy"
    assert report["oof_accuracy"] == 1.0
    assert report["fold_accuracy"] == {"0": 1.0, "1": 1.0}


def test_evaluate_oof_rejects_partial_fold(tmp_path) -> None:
    manifest = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "user": ["user1", "user1", "user2", "user2"],
            "fold": [0, 0, 1, 1],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    fold0, fold1 = tmp_path / "fold0.csv", tmp_path / "fold1.csv"
    _prediction_frame([("a", 0)]).to_csv(fold0, index=False)
    _prediction_frame([("c", 0), ("d", 1)]).to_csv(fold1, index=False)

    with pytest.raises(ValueError, match="exactly the held-out clips"):
        evaluate_oof(manifest_path, [fold0, fold1])
