from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .constants import NUM_CLASSES


def validate_submission(submission_path: str | Path, test_csv_path: str | Path) -> None:
    submission = pd.read_csv(submission_path)
    expected = pd.read_csv(test_csv_path)
    if list(submission.columns) != ["path", "prediction"]:
        raise ValueError("Submission columns must be exactly: path,prediction")
    if len(submission) != len(expected):
        raise ValueError(f"Expected {len(expected)} rows, found {len(submission)}")
    if not submission["path"].equals(expected["path"]):
        raise ValueError("Submission paths or row order do not match test.csv")
    numeric = pd.to_numeric(submission["prediction"], errors="coerce")
    if numeric.isna().any() or not (numeric == numeric.astype(int)).all():
        raise ValueError("Every prediction must be an integer")
    if not numeric.between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"Predictions must be between 0 and {NUM_CLASSES - 1}")
    if submission.duplicated("path").any():
        raise ValueError("Submission contains duplicate paths")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CUHK-X submission CSV")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--test-csv", required=True)
    args = parser.parse_args()
    validate_submission(args.submission, args.test_csv)
    print(f"Submission is valid: {Path(args.submission).resolve()}")


if __name__ == "__main__":
    main()
