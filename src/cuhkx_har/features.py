from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SKELETON_JOINTS = 17
SKELETON_FEATURES = 4
IMU_DEVICES = 5
IMU_FEATURES_PER_DEVICE = 16
IMU_FEATURES = IMU_DEVICES * IMU_FEATURES_PER_DEVICE
IMU_DEVICE_PREFIXES = ("WTC", "WTLA", "WTLL", "WTRA", "WTRL")
RADAR_COLUMNS = ("x", "y", "z", "v", "snr", "noise")
RADAR_FEATURES = 1 + 2 * len(RADAR_COLUMNS)


def cache_key(split: str, clip_id: str) -> str:
    digest = hashlib.sha1(f"{split}:{clip_id}".encode()).hexdigest()[:20]
    return f"{split}_{digest}.npz"


def resample_sequence(array: np.ndarray, steps: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got {array.shape}")
    if len(array) == 0:
        return np.zeros((steps, array.shape[1]), dtype=np.float32)
    if len(array) == 1:
        return np.repeat(array.astype(np.float32), steps, axis=0)
    old = np.linspace(0.0, 1.0, len(array), dtype=np.float32)
    new = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    output = np.empty((steps, array.shape[1]), dtype=np.float32)
    for column in range(array.shape[1]):
        output[:, column] = np.interp(new, old, array[:, column])
    return output


def read_skeleton(directory: Path, steps: int) -> tuple[np.ndarray, bool]:
    files = sorted(directory.rglob("*.json")) if directory.is_dir() else []
    frames: list[np.ndarray] = []
    for path in files:
        try:
            people = json.loads(path.read_text(encoding="utf-8"))
            if not people:
                continue
            person = people[0]
            points = np.asarray(person["keypoints"], dtype=np.float32)
            scores = np.asarray(
                person.get("keypoint_scores", np.ones(len(points))), dtype=np.float32
            )
            if points.shape != (SKELETON_JOINTS, 3):
                continue
            scores = scores.reshape(SKELETON_JOINTS, 1)
            pelvis = (points[11] + points[12]) * 0.5
            shoulders = np.linalg.norm(points[5] - points[6])
            hips = np.linalg.norm(points[11] - points[12])
            scale = max(float(shoulders), float(hips), 1e-3)
            normalized = (points - pelvis) / scale
            frames.append(np.concatenate([normalized, scores], axis=1).reshape(-1))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    width = SKELETON_JOINTS * SKELETON_FEATURES
    if not frames:
        return np.zeros((steps, width), dtype=np.float32), False
    return resample_sequence(np.stack(frames), steps), True


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_csv(path, encoding="gb18030", encoding_errors="replace", low_memory=False)


def read_imu(directory: Path, steps: int) -> tuple[np.ndarray, bool]:
    tables = []
    for path in sorted(directory.glob("*.csv")) if directory.is_dir() else []:
        try:
            frame = _read_csv(path)
            if frame.shape[1] < 3 or frame.empty:
                continue
            values = frame.iloc[:, 2 : 2 + IMU_FEATURES_PER_DEVICE].apply(
                pd.to_numeric, errors="coerce"
            )
            values.columns = range(values.shape[1])
            if values.shape[1] < IMU_FEATURES_PER_DEVICE:
                for index in range(values.shape[1], IMU_FEATURES_PER_DEVICE):
                    values[index] = 0.0
            values = values.iloc[:, :IMU_FEATURES_PER_DEVICE]
            values.insert(0, "device", frame.iloc[:, 1].astype(str).to_numpy())
            tables.append(values)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    output = np.zeros((steps, IMU_FEATURES), dtype=np.float32)
    if not tables:
        return output, False
    combined = pd.concat(tables, ignore_index=True)
    device_prefix = combined["device"].str.extract(
        r"^(WTC|WTLA|WTLL|WTRA|WTRL)", expand=False
    )
    found_device = False
    for device_index, device in enumerate(IMU_DEVICE_PREFIXES):
        values = combined.loc[device_prefix == device].iloc[:, 1:].to_numpy(np.float32)
        if not len(values):
            continue
        found_device = True
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        start = device_index * IMU_FEATURES_PER_DEVICE
        output[:, start : start + IMU_FEATURES_PER_DEVICE] = resample_sequence(values, steps)
    return output, found_device


def read_radar(directory: Path, steps: int) -> tuple[np.ndarray, bool]:
    frame_features: list[np.ndarray] = []
    for path in sorted(directory.glob("*.csv")) if directory.is_dir() else []:
        try:
            table = _read_csv(path)
            if table.empty or "frame" not in table:
                continue
            available = [name for name in RADAR_COLUMNS if name in table]
            if len(available) != len(RADAR_COLUMNS):
                continue
            numeric = table.loc[:, ["frame", *RADAR_COLUMNS]].copy()
            for name in numeric.columns:
                numeric[name] = pd.to_numeric(numeric[name], errors="coerce")
            for _, group in numeric.dropna(subset=["frame"]).groupby("frame", sort=True):
                values = group.loc[:, list(RADAR_COLUMNS)].to_numpy(np.float32)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                means = values.mean(axis=0)
                stds = values.std(axis=0)
                frame_features.append(
                    np.concatenate([np.asarray([len(values)], dtype=np.float32), means, stds])
                )
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if not frame_features:
        return np.zeros((steps, RADAR_FEATURES), dtype=np.float32), False
    return resample_sequence(np.stack(frame_features), steps), True


def _path_from_row(root: Path, value: object) -> Path:
    if pd.isna(value) or not str(value):
        return root / "__missing__"
    return root / Path(str(value))


def build_feature_file(
    row: pd.Series,
    data_root: Path,
    output_dir: Path,
    split: str,
    steps: int,
    overwrite: bool = False,
) -> Path:
    output = output_dir / cache_key(split, str(row["clip_id"]))
    if output.exists() and not overwrite:
        return output
    skeleton, skeleton_ok = read_skeleton(_path_from_row(data_root, row["Skeleton"]), steps)
    imu, imu_ok = read_imu(_path_from_row(data_root, row["IMU"]), steps)
    radar, radar_ok = read_radar(_path_from_row(data_root, row["Radar"]), steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        skeleton=skeleton,
        imu=imu,
        radar=radar,
        sensor_mask=np.asarray([skeleton_ok, imu_ok, radar_ok], dtype=np.bool_),
    )
    return output


def build_cache(
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    split: str,
    steps: int,
    workers: int = 2,
    overwrite: bool = False,
) -> None:
    manifest = pd.read_csv(manifest_path).fillna("")
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()

    def process(record: tuple[int, pd.Series]) -> Path:
        _, row = record
        return build_feature_file(row, root, output, split, steps, overwrite)

    records: Iterable[tuple[int, pd.Series]] = manifest.iterrows()
    if workers <= 1:
        for record in tqdm(records, total=len(manifest), desc=f"cache-{split}"):
            process(record)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(process, records), total=len(manifest), desc=f"cache-{split}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache CUHK-X sensor features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="cache")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_cache(
        args.manifest,
        args.data_root,
        args.output_dir,
        args.split,
        args.steps,
        args.workers,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
