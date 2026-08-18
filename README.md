# CUHK-X Small Model Track

Current, leakage-safe multimodal HAR pipeline for the CUHK-X Small Model Track.
All model selection is driven by subject-held-out local cross-validation; the
Kaggle public leaderboard is only a final external check.

## Current training approach

The current five-fold baseline is [`configs/synced_flip_imu_dropout.json`](configs/synced_flip_imu_dropout.json):

- A lightweight, from-scratch visual encoder processes Depth Color, IR, and
  Thermal streams over time.
- Separate temporal encoders process normalized 17-joint skeleton, five fixed
  IMU device slots, and frame-aggregated mmWave radar.
- A compact Transformer fuses modality embeddings with explicit
  missing-modality masks.
- One synchronized horizontal flip is applied to the visual streams, Skeleton,
  IMU device order, and radar horizontal position.
- During training only, one IMU device slot is dropped with 15% probability to
  improve robustness to partially missing IMU devices.  Validation and
  inference never use this augmentation.

All weights are trained from scratch.  No pretrained backbone, pseudo-label,
manual test label, or test-derived normalization/statistics is used.  The model
and any inference ensemble must remain below the 100 MB competition limit.

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

## Current workflow

1. Build and freeze `manifests/cv5/train.csv` using the commands in
   [`docs/July25.md`](docs/July25.md).
2. Ensure `cache-64` exists for both the train and test manifests.
3. Train the same configuration on folds 0–4, using a dedicated artifact
   directory per experiment.
4. Run `cuhkx-cv-report` only after all five validation-prediction files exist.
5. Record the configuration, all fold results, OOF score, and decision in
   [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
6. Generate and validate a Kaggle CSV only after CV selects a model or
   cross-fitted ensemble.

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
