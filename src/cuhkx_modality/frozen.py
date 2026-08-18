"""Identity check for the repository's frozen CV5 split."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

FROZEN_CV5_ROWS = 3036
FROZEN_CV5_FOLDS = frozenset(range(5))
FROZEN_CV5_MAPPING_SHA256 = "3a6f40155f19964e83b65dad6436ce0202d3739c298df15e65b59c7a8fa5f4e3"


def mapping_fingerprint(manifest: pd.DataFrame) -> str:
    required = {"clip_id", "fold"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["clip_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate clip_id values")
    rows = manifest.loc[:, ["clip_id", "fold"]].copy()
    try:
        rows["fold"] = rows["fold"].astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("Manifest fold values must be integers") from error
    payload = "".join(
        "{}\t{}\n".format(clip_id, fold)
        for clip_id, fold in rows.sort_values("clip_id").itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_frozen_cv5(manifest: pd.DataFrame) -> None:
    required = {"clip_id", "fold"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if len(manifest) != FROZEN_CV5_ROWS:
        raise ValueError(f"Frozen CV5 manifest must contain {FROZEN_CV5_ROWS} rows")
    if set(manifest["fold"]) != FROZEN_CV5_FOLDS:
        raise ValueError("Manifest is not the frozen five-fold CV5 split")
    if mapping_fingerprint(manifest) != FROZEN_CV5_MAPPING_SHA256:
        raise ValueError("Manifest clip_id-to-fold mapping differs from frozen CV5")


def read_frozen_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path).fillna("")
    validate_frozen_cv5(manifest)
    return manifest
