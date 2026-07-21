from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import TRAIN_USERS


def assign_subject_folds(
    manifest: pd.DataFrame, n_splits: int = 3, seed: int = 20260719
) -> pd.DataFrame:
    """Assign folds without ever placing one subject in train and validation."""
    required = {"label", "user"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest lacks required columns: {sorted(missing)}")
    users = set(manifest["user"].unique())
    unexpected = users - set(TRAIN_USERS)
    if unexpected:
        raise ValueError(f"Unexpected users in training data: {sorted(unexpected)}")
    if n_splits < 2 or n_splits > len(users):
        raise ValueError("n_splits must be between 2 and the number of subjects")

    output = manifest.copy()
    output["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(output), dtype=np.uint8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(dummy, output["label"].to_numpy(), output["user"].to_numpy())
    ):
        output.loc[output.index[validation_indices], "fold"] = fold

    if (output["fold"] < 0).any():
        raise RuntimeError("At least one training row was not assigned to a fold")
    assert_no_subject_leakage(output)
    return output


def assert_no_subject_leakage(manifest: pd.DataFrame) -> None:
    fold_counts = manifest.groupby("user")["fold"].nunique()
    leaked = fold_counts[fold_counts != 1]
    if not leaked.empty:
        raise ValueError(f"Subject leakage detected for: {leaked.index.tolist()}")


def fold_partition(manifest: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_no_subject_leakage(manifest)
    train = manifest.loc[manifest["fold"] != fold].reset_index(drop=True)
    valid = manifest.loc[manifest["fold"] == fold].reset_index(drop=True)
    overlap = set(train["user"]) & set(valid["user"])
    if overlap:
        raise RuntimeError(f"Subject leakage in requested fold {fold}: {sorted(overlap)}")
    return train, valid

