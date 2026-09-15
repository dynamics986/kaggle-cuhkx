from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cuhkx_har.submission import validate_submission

MODALITIES = ("IR", "Depth_Color", "Thermal")
LIMIT_BYTES = 100_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_path(root, split, clip_id):
    return Path(root) / split / (hashlib.sha256(clip_id.encode()).hexdigest() + ".npz")


def probabilities_valid(probabilities, rows):
    p = np.asarray(probabilities)
    if p.shape != (rows, 40) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid class probabilities")
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")


def write_predictions(rows, probabilities, output):
    probabilities_valid(probabilities, len(rows))
    result = rows[["clip_id", "label"]].reset_index(drop=True).copy()
    result["prediction"] = probabilities.argmax(1)
    for c in range(40):
        result[f"prob_{c}"] = probabilities[:, c]
    result.to_csv(output, index=False)


def submit(rows, probabilities, official_path, output):
    probabilities_valid(probabilities, len(rows))
    if rows.clip_id.duplicated().any() or rows.submission_path.duplicated().any():
        raise ValueError("Duplicate test identity")
    official = pd.read_csv(official_path)
    mapping = dict(zip(rows.submission_path, probabilities.argmax(1), strict=True))
    if set(mapping) != set(official.path):
        raise ValueError("Test manifest paths differ from official test.csv")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": official.path, "prediction": official.path.map(mapping)}).to_csv(
        output, index=False
    )
    validate_submission(output, official_path)


def check_size(paths):
    size = sum(Path(p).stat().st_size for p in paths)
    if size >= LIMIT_BYTES:
        raise ValueError(f"Inference weights {size:,} bytes exceed 100 MB: {paths}")
    return size
