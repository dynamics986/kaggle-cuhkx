from __future__ import annotations

import pandas as pd
import pytest

from cuhkx_har.predict import normalize_weights
from cuhkx_har.submission import validate_submission


def test_submission_requires_exact_path_order_and_valid_labels(tmp_path) -> None:
    test_csv = tmp_path / "test.csv"
    valid_csv = tmp_path / "valid.csv"
    bad_csv = tmp_path / "bad.csv"
    paths = ["small_model_track_test/SM_test_0001/", "small_model_track_test/SM_test_0002/"]
    pd.DataFrame({"path": paths, "prediction": [None, None]}).to_csv(test_csv, index=False)
    pd.DataFrame({"path": paths, "prediction": [0, 39]}).to_csv(valid_csv, index=False)
    validate_submission(valid_csv, test_csv)

    pd.DataFrame({"path": paths[::-1], "prediction": [0, 40]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        validate_submission(bad_csv, test_csv)


def test_ensemble_weights_are_normalized_and_checked() -> None:
    assert normalize_weights(2, None) == [0.5, 0.5]
    assert normalize_weights(2, [2.0, 1.0]) == pytest.approx([2 / 3, 1 / 3])
    with pytest.raises(ValueError):
        normalize_weights(2, [1.0])
