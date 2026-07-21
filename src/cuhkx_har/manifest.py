from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .constants import MODALITIES, NUM_CLASSES, TRAIN_USERS
from .splits import assign_subject_folds

TEST_ID_PATTERN = re.compile(r"^SM_test_\d{4}$")


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _assert_separate_roots(train_root: Path, test_root: Path) -> None:
    roots_overlap = (
        train_root == test_root
        or train_root in test_root.parents
        or test_root in train_root.parents
    )
    if roots_overlap:
        raise ValueError("Training and test roots must be separate directories")


def build_train_manifest(data_root: str | Path, class_mapping: str | Path) -> pd.DataFrame:
    root = _resolve(data_root)
    mapping = pd.read_csv(class_mapping, encoding="utf-8-sig")
    if len(mapping) != NUM_CLASSES or set(mapping["action_id"]) != set(range(NUM_CLASSES)):
        raise ValueError("class_mapping.csv must contain action_id 0..39 exactly once")
    name_to_id = dict(zip(mapping["action_name"], mapping["action_id"], strict=True))

    records: dict[tuple[str, str, str], dict[str, object]] = {}
    for modality in MODALITIES:
        modality_root = root / modality
        if not modality_root.is_dir():
            raise FileNotFoundError(f"Missing training modality directory: {modality_root}")
        for action_dir in sorted(p for p in modality_root.iterdir() if p.is_dir()):
            if action_dir.name not in name_to_id:
                raise ValueError(f"Unknown action directory: {action_dir}")
            for user_dir in sorted(p for p in action_dir.iterdir() if p.is_dir()):
                if user_dir.name not in TRAIN_USERS:
                    raise ValueError(f"Unexpected training subject: {user_dir.name}")
                for trial_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                    key = (action_dir.name, user_dir.name, trial_dir.name)
                    record = records.setdefault(
                        key,
                        {
                            "clip_id": "/".join(key),
                            "action_name": action_dir.name,
                            "label": int(name_to_id[action_dir.name]),
                            "user": user_dir.name,
                            "trial": trial_dir.name,
                        },
                    )
                    record[modality] = trial_dir.relative_to(root).as_posix()

    frame = pd.DataFrame(records.values()).fillna("")
    for modality in MODALITIES:
        if modality not in frame:
            frame[modality] = ""
    if frame["clip_id"].duplicated().any():
        raise RuntimeError("Duplicate training clip identifiers found")
    return frame.sort_values(["label", "user", "trial"]).reset_index(drop=True)


def build_test_manifest(test_root: str | Path, test_csv: str | Path) -> pd.DataFrame:
    root = _resolve(test_root)
    submission_rows = pd.read_csv(test_csv)
    records = []
    for path_text in submission_rows["path"]:
        clip_name = Path(str(path_text).rstrip("/\\")).name
        if not TEST_ID_PATTERN.fullmatch(clip_name):
            raise ValueError(f"Invalid test clip id in test.csv: {clip_name}")
        clip_dir = root / clip_name
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"Missing test clip directory: {clip_dir}")
        record: dict[str, object] = {
            "clip_id": clip_name,
            "submission_path": path_text,
            "label": -1,
            "user": "",
            "trial": clip_name,
        }
        for modality in MODALITIES:
            path = clip_dir / modality
            record[modality] = path.relative_to(root).as_posix() if path.is_dir() else ""
        records.append(record)
    output = pd.DataFrame(records)
    if output["clip_id"].duplicated().any():
        raise ValueError("Duplicate test clip ids in test.csv")
    return output


def create_manifests(
    train_root: str | Path,
    test_root: str | Path,
    class_mapping: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    n_splits: int = 3,
    seed: int = 20260719,
) -> tuple[Path, Path]:
    train_path, test_path = _resolve(train_root), _resolve(test_root)
    _assert_separate_roots(train_path, test_path)
    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train = assign_subject_folds(
        build_train_manifest(train_path, class_mapping), n_splits=n_splits, seed=seed
    )
    test = build_test_manifest(test_path, test_csv)
    train_file, test_file = output / "train.csv", output / "test.csv"
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    return train_file, test_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe CUHK-X manifests")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--class-mapping", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", default="manifests")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    train_file, test_file = create_manifests(
        args.train_root,
        args.test_root,
        args.class_mapping,
        args.test_csv,
        args.output_dir,
        args.folds,
        args.seed,
    )
    train, test = pd.read_csv(train_file), pd.read_csv(test_file)
    print(f"Wrote {len(train)} training clips to {train_file}")
    print(f"Wrote {len(test)} test clips to {test_file}")
    for fold, group in train.groupby("fold"):
        print(f"fold={fold}: clips={len(group)}, users={sorted(group['user'].unique())}")


if __name__ == "__main__":
    main()
