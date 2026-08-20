
# CUHK-X Small Model Track

Current, leakage-safe multimodal HAR pipeline for the CUHK-X Small Model Track.
All model selection is driven by subject-held-out local cross-validation; the
Kaggle public leaderboard is only a final external check.

## Current training approach

Current local-CV model: a `0.25 / 0.75` probability blend of
[`synced_flip_imu_dropout`](configs/synced_flip_imu_dropout.json) and
[`pose_motion_residual`](configs/pose_motion_residual.json): pooled OOF
`0.50198`, cross-fitted OOF `0.49736`, 51.7 MB.

Both use lightweight visual CNNs, temporal sensor encoders, missing-modality
masks, and Transformer fusion. The residual model replaces the Skeleton TCN
with pose + motion residual encoding. Synchronized flip and 15% training-only
IMU slot dropout are retained. YOLO three-stream crop and Radar PointNet are
rejected; the IMU Transformer fusion probe gained only `+0.00185` on fold 2,
so it is not promoted.

Large pretrained backbones are prohibited. Auditable lightweight pretrained
models are permitted when their source, version, licence, training data scope,
and weight size are recorded; no pseudo-label, manual test label, or
test-derived normalization/statistics is used. The final detector plus HAR
inference ensemble must remain below the 100 MB competition limit.

## Evaluation contract

The competition metric is clip-level top-1 accuracy.  Local validation uses a
frozen five-fold `StratifiedGroupKFold` split grouped by subject:

- A subject occurs in exactly one validation fold.
- Sensor normalization is fitted on that fold's training subjects only.
- Every candidate needs one complete OOF prediction file per fold.
- `cuhkx-cv-report` rejects incomplete, duplicated, misaligned, or invalid OOF
  predictions before reporting OOF accuracy, fold stability, per-class metrics,
  and a confusion matrix.
- Ensemble ideas are accepted only when their leave-one-fold-out, cross-fitted
  score improves, not merely because a weight search improves pooled OOF.

Detailed setup, training, monitoring, resume rules, OOF audit, and ensemble
selection commands are in [`docs/July25.md`](docs/July25.md).  The next
CV-gated visual upgrade is specified in [`docs/July26.md`](docs/July26.md).

## Setup

Run from `C:\Users\dynam\Documents\CUHK-X\har-solution`:

```powershell
uv sync --dev
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
uv run pytest
```

## Submission format

Kaggle receives a CSV, not checkpoints.  Validate a generated file before
uploading:

```powershell
uv run cuhkx-check-submission `
  --submission artifacts\submission.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```

The file must have exactly `path,prediction`, preserve official test-row order,
and contain integer labels from 0 to 39.
