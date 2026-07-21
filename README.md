# CUHK-X Small Model Track solution

Leakage-safe, from-scratch multimodal HAR code for the CUHK-X Small Model Track. The code is
kept in `har-solution/`; the official files remain untouched in the sibling
`Small-Model-Track/` directory.

## Method

- Visual streams (`Depth_Color`, `IR`, `Thermal`): shared lightweight CNN plus temporal TCN.
- Sensor streams: separate temporal encoders for 17-joint skeleton, five-device IMU, and
  frame-aggregated mmWave radar.
- Fusion: modality embeddings and a compact Transformer with explicit missing-modality masks.
- All weights are trained from scratch. No pretrained backbone, LLM inference, pseudo-labeling,
  or manual test labeling is used.

Capacity is controlled by `width_mult`, `d_model`, `fusion_layers`, `image_size`, and
`visual_frames` in the JSON configs. A checkpoint is rejected if it exceeds 100 MB; inference
also rejects an ensemble whose combined parameter size exceeds 100 MB.

`configs/balanced.json` uses square-root inverse-frequency sampling. This is intentionally milder
than fully uniform class sampling, because the official metric is clip accuracy and the real class
distribution is not uniform.

`configs/graph.json` replaces the flattened skeleton TCN with a small from-scratch ST-GCN that
uses the 17-joint body graph. Old checkpoints retain the original encoder through their saved
configuration.

`configs/motion.json` keeps a temporal skeleton encoder and augments each pose with first-order
joint velocity and second-order joint acceleration. Confidence scores are not differentiated,
because their changes measure detector certainty rather than body motion. The option is disabled
by default and is mutually exclusive with the ST-GCN option.

`configs/synced_flip.json` applies one shared horizontal-flip decision to all three visual
streams, mirrors Skeleton x coordinates and swaps COCO left/right joints, swaps left/right IMU
device slots, and mirrors the Radar mean-x feature. IMU sensor-local axis signs are preserved
because the dataset does not publish a mounting-axis mirror calibration.

## Leakage policy

The unit of validation is a **subject**, never a clip or frame. The manifest builder only accepts
the official training subjects (users 1-9 and 16-24), assigns each subject to exactly one fold,
and checks train/validation user disjointness whenever a fold is loaded. Sensor normalization is
computed from the training subjects of that fold only. Test data never contributes statistics,
model selection, or hyperparameter decisions.

## Setup

Run these commands from `C:\Users\dynam\Documents\CUHK-X\har-solution` in PowerShell:

```powershell
uv sync --dev
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
uv run pytest
```

## 1. Build manifests

```powershell
uv run cuhkx-index `
  --train-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --test-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --class-mapping ..\Small-Model-Track\class_mapping.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv `
  --output-dir manifests `
  --folds 3
```

Inspect the printed user list for every fold. A user must occur in only one validation fold.

## 2. Cache non-visual features

The base config uses 64 steps. Caches are derived artifacts; deleting them never affects raw data.

```powershell
uv run cuhkx-cache --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --output-dir cache-64 --split train --steps 64 --workers 2

uv run cuhkx-cache --manifest manifests\test.csv `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --output-dir cache-64 --split test --steps 64 --workers 2
```

The smoke config automatically downsamples this cache from 64 to 32 steps in memory, so it does
not require a second copy.

## 3. Train one fold

Start with the smoke config and fold 0. This validates the pipeline; its score is not meaningful.

```powershell
uv run cuhkx-train --config configs\smoke.json --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 --output-dir artifacts\smoke --fold 0 `
  --max-clips-per-class 2
```

After the smoke run and a one-fold baseline are reviewed, train the base configuration. Do not
launch all three folds until runtime and validation quality from fold 0 are known.

`configs/base_probe.json` has the base architecture but only one epoch. Use it with
`--max-clips-per-class 1` when re-checking a new GPU or a changed architecture; it is not a
competitive training run.

```powershell
uv run cuhkx-train --config configs\base.json --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 --output-dir artifacts\base --fold 0
```

The motion experiment uses the same manifest, cache, and subject split:

```powershell
uv run cuhkx-train --config configs\motion.json --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 --output-dir artifacts\motion --fold 0
```

Visual V2 keeps the full image with letterboxing, uses 192 x 192 inputs and 12 synchronized visual
samples, and retains temporal direction during pooling:

```powershell
uv run cuhkx-train --config configs\visual_v2.json --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 --output-dir artifacts\visual_v2 --fold 0
```

If a run is interrupted, resume the same fold with the same command plus:

```powershell
--resume artifacts\base\fold_0\last.pt
```

### Live loss monitoring

Keep training in the first PowerShell window. Open a second PowerShell window in the project
directory and point the monitor at that run's fold directory:

```powershell
uv run cuhkx-monitor --run-dir artifacts\base\fold_0 --patience 8
```

The dashboard refreshes every second. During an epoch it shows the train/validation phase,
batch progress, running loss and accuracy. After each epoch it shows the last eight train/valid
losses, accuracies, learning rate, best validation epoch, early-stop wait, and ASCII loss trends.
Pressing `Ctrl+C` in the monitor window stops only the dashboard; training continues in the first
window. Use a new output directory for a new experiment so an old `summary.json` is not mistaken
for a completed current run.

Each fold writes `history.json`, `summary.json`, `validation_predictions.csv`, and
`per_class_accuracy.csv`. While training, it also updates `progress.json` every five batches.
These validation artifacts contain only held-out training subjects.

After both model families have all three folds, reproduce the leakage-safe pooled OOF weight scan:

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\synced_flip\fold_0\validation_predictions.csv artifacts\synced_flip\fold_1\validation_predictions.csv artifacts\synced_flip\fold_2\validation_predictions.csv `
  --candidate artifacts\visual_v2\fold_0\validation_predictions.csv artifacts\visual_v2\fold_1\validation_predictions.csv artifacts\visual_v2\fold_2\validation_predictions.csv `
  --steps 40 --output artifacts\visual_v2\oof_ensemble_scan.json
```

The current shared optimum is `0.575 synchronized flip + 0.425 Visual V2`, giving `0.49539` OOF.
One common weight is used for all folds; test labels and test-derived statistics are never used.

To measure whether each modality contributes, run a drop-one ablation on a trained fold:

```powershell
uv run cuhkx-ablate --checkpoint artifacts\base\fold_0\best.pt `
  --manifest manifests\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 --output artifacts\base\fold_0\ablation.json
```

## 4. Ensemble and validate a submission

```powershell
uv run cuhkx-predict `
  --checkpoints artifacts\base\fold_0\best.pt artifacts\base\fold_1\best.pt artifacts\base\fold_2\best.pt `
  --manifest manifests\test.csv `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --cache-dir cache-64 --views 3 --output artifacts\submission.csv

uv run cuhkx-check-submission --submission artifacts\submission.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```

Only upload a CSV after the validator reports success. Kaggle submission remains a manual action.
When validation supports a non-uniform ensemble, pass one weight per checkpoint with `--weights`.
The current fold-0 analysis supports weights `0.30 0.30 0.40` for base, balanced, and motion,
respectively. Re-estimate robustness across folds before treating these as final weights. The
participant should run final test inference and upload the validated CSV manually.

For the six cross-validation checkpoints, the validated family weights can be distributed equally
across the three folds: use `0.1916667` for each synchronized-flip checkpoint and `0.1416667` for
each Visual V2 checkpoint. The six files total about 32.5 MB. Keep checkpoint ordering consistent
with the supplied weights, and have the participant run final test inference and submission checks.
