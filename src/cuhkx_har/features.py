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
RADAR_MAX_POINTS = 32


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


def read_imu_synced(directory: Path, steps: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Read five IMU devices on one timestamp-aligned grid.

    Values outside a device's observed time range are zero and marked invalid.
    Duplicate timestamps are averaged before interpolation.
    """
    tables: list[pd.DataFrame] = []
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
            values.insert(0, "timestamp", pd.to_datetime(frame.iloc[:, 0], errors="coerce"))
            tables.append(values)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    output = np.zeros((steps, IMU_DEVICES, IMU_FEATURES_PER_DEVICE), dtype=np.float32)
    mask = np.zeros((steps, IMU_DEVICES), dtype=np.bool_)
    if not tables:
        return output, mask, False
    combined = pd.concat(tables, ignore_index=True)
    device_prefix = combined["device"].str.extract(
        r"^(WTC|WTLA|WTLL|WTRA|WTRL)", expand=False
    )
    combined = combined.loc[combined["timestamp"].notna()].copy()
    if combined.empty:
        return output, mask, False
    combined["device_prefix"] = device_prefix.loc[combined.index]
    timestamps = combined["timestamp"].astype("int64").to_numpy()
    grid = np.linspace(timestamps.min(), timestamps.max(), steps, dtype=np.float64)
    for device_index, device in enumerate(IMU_DEVICE_PREFIXES):
        selected = combined.loc[combined["device_prefix"] == device]
        if selected.empty:
            continue
        feature_columns = list(range(IMU_FEATURES_PER_DEVICE))
        selected = selected.loc[:, ["timestamp", *feature_columns]]
        selected = selected.groupby("timestamp", sort=True, as_index=False).mean(numeric_only=True)
        times = selected["timestamp"].astype("int64").to_numpy(dtype=np.float64)
        values = selected.loc[:, feature_columns].to_numpy(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if len(times) == 1:
            output[:, device_index] = values[0]
            mask[:, device_index] = True
            continue
        valid = (grid >= times[0]) & (grid <= times[-1])
        mask[:, device_index] = valid
        for column in range(IMU_FEATURES_PER_DEVICE):
            output[valid, device_index, column] = np.interp(
                grid[valid], times, values[:, column]
            )
    return output, mask, bool(mask.any())


def read_imu(directory: Path, steps: int) -> tuple[np.ndarray, bool]:
    values, _, available = read_imu_synced(directory, steps)
    return values.reshape(steps, IMU_FEATURES), available


def read_radar_points(
    directory: Path, steps: int, max_points: int = RADAR_MAX_POINTS
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Read raw Radar detections while preserving empty frame numbers."""
    frame_points: dict[int, list[np.ndarray]] = {}
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
            for frame_id, group in numeric.dropna(subset=["frame"]).groupby("frame", sort=True):
                values = group.loc[:, list(RADAR_COLUMNS)].to_numpy(np.float32)
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                frame_points.setdefault(int(frame_id), []).append(values)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    output = np.zeros((steps, max_points, len(RADAR_COLUMNS)), dtype=np.float32)
    mask = np.zeros((steps, max_points), dtype=np.bool_)
    if not frame_points:
        return output, mask, False
    first, last = min(frame_points), max(frame_points)
    sampled_frames = np.rint(np.linspace(first, last, steps)).astype(np.int64)
    for output_index, frame_id in enumerate(sampled_frames):
        groups = frame_points.get(int(frame_id))
        if not groups:
            continue
        values = np.concatenate(groups, axis=0)
        if len(values) > max_points:
            # Stable top-SNR selection keeps truncation deterministic.
            order = np.argsort(-values[:, RADAR_COLUMNS.index("snr")], kind="stable")
            values = values[order[:max_points]]
        count = len(values)
        output[output_index, :count] = values
        mask[output_index, :count] = True
    return output, mask, bool(mask.any())


def radar_statistics(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros((len(points), RADAR_FEATURES), dtype=np.float32)
    for index in range(len(points)):
        values = points[index, mask[index]]
        if not len(values):
            continue
        output[index] = np.concatenate(
            [np.asarray([len(values)], dtype=np.float32), values.mean(axis=0), values.std(axis=0)]
        )
    return output


def read_radar(directory: Path, steps: int) -> tuple[np.ndarray, bool]:
    points, mask, available = read_radar_points(directory, steps)
    return radar_statistics(points, mask), available


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
    imu_synced, imu_device_mask, imu_ok = read_imu_synced(
        _path_from_row(data_root, row["IMU"]), steps
    )
    imu = imu_synced.reshape(steps, IMU_FEATURES)
    radar_points, radar_point_mask, radar_ok = read_radar_points(
        _path_from_row(data_root, row["Radar"]), steps
    )
    radar = radar_statistics(radar_points, radar_point_mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        skeleton=skeleton,
        imu=imu,
        imu_synced=imu_synced,
        imu_device_mask=imu_device_mask,
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
