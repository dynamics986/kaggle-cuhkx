from __future__ import annotations

import pandas as pd

from cuhkx_har.constants import TRAIN_USERS
from cuhkx_har.splits import assert_no_subject_leakage, assign_subject_folds, fold_partition


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for user in TRAIN_USERS:
        for label in range(40):
            rows.append({"clip_id": f"{user}/{label}", "user": user, "label": label})
    return pd.DataFrame(rows)


def test_subjects_never_cross_folds() -> None:
    folded = assign_subject_folds(synthetic_manifest(), n_splits=3, seed=7)
    assert_no_subject_leakage(folded)
    assert set(folded["fold"]) == {0, 1, 2}
    for fold in range(3):
        train, valid = fold_partition(folded, fold)
        assert set(train["user"]).isdisjoint(valid["user"])
        assert set(train["label"]) == set(range(40))
        assert set(valid["label"]) == set(range(40))


def test_unknown_training_subject_is_rejected() -> None:
    manifest = synthetic_manifest()
    manifest.loc[0, "user"] = "user10"
    try:
        assign_subject_folds(manifest)
    except ValueError as error:
        assert "Unexpected users" in str(error)
    else:
        raise AssertionError("Unknown subject should have been rejected")

