# Reproducing the reported results

Run all commands from `path\to\CUHK-X\har-solution` in
PowerShell. The raw competition data must be available at:

```text
..\Small-Model-Track\Training\extracted\HAR\data
..\Small-Model-Track\Testing\data\small_model_track_test
..\Small-Model-Track\Testing\test_file\test.csv
```

Install the locked environment once:

```powershell
uv sync --extra sep12 --dev
```

The frozen splits are `manifests\cv3\` for the historical protocol and
`manifests\cv5\` for current work. Do not regenerate either split.

## 1. m01 LightGBM with YOLO-box temporal features

This is the Sep14 box baseline, evaluated on cv5 Fold 2 and Fold 4. It needs
the fold-safe Sep12 crop caches, sensor cache and three detector files. If they
already exist, use these paths unchanged:

```text
artifacts\sep12\serial_depth_align\sensors
artifacts\sep12\serial_depth_align\cache\fold_{2,4,full}
artifacts\yolo\detectors\fold_2\yolov8n_fold_2.pt
artifacts\sep12\serial_depth_align\jobs\detector_4\attempt_1\yolov8n_4.pt
artifacts\sep12\serial_depth_align\jobs\detector_full\attempt_1\yolov8n_full.pt
```

If these prerequisites are absent, rebuild them with the Sep12 workflow in
[Sep12.md](Sep12.md). It trains the Fold-4 and full detectors from original
YOLO weights, creates the crop caches, and creates 64-step sensor caches. Do
not use the Fold-2 detector for Fold 4.

```powershell
uv run --extra sep12 python -m cuhkx_sep14.lightgbm `
  --output artifacts\repro\m01_yolo_box_baseline
```

It validates Fold 2 and Fold 4, selects the actual box-feature baseline when
it has the highest weighted score, refits it on all cv5 training clips, and
writes `comparison.json` and `submission.csv`. Expected values: Fold 2
0.55926, Fold 4 0.48594, weighted 0.51949.

## 2. Dual ResNet-18 with moderate augmentation

This reproduces Sep13 m05 using the same Sep12 crop prerequisites above:

```powershell
uv run --extra sep12 python -m cuhkx_sep12.train `
  --config configs\sep13\m05_dual_resnet18.json `
  --output artifacts\repro\m05_sep13_moderate `
  --folds 2 4 `
  --cache-root artifacts\sep12\serial_depth_align\cache `
  --detector2 artifacts\yolo\detectors\fold_2\yolov8n_fold_2.pt `
  --detector4 artifacts\sep12\serial_depth_align\jobs\detector_4\attempt_1\yolov8n_4.pt `
  --device cuda
```

Expected scores are 0.44259 on Fold 2 and 0.35938 on Fold 4. The augmentation
is shared across each clip's frames; do not replace it with the stronger Sep14
policy.

## 3. Synced-flip baseline plus pose-motion residual blend

Build the sensor cache if it is absent:

```powershell
uv run python -m cuhkx_har.features --manifest manifests\cv5\train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data --output-dir cache-64 --split train --steps 64 --workers 6
uv run python -m cuhkx_har.features --manifest manifests\cv5\test.csv --data-root ..\Small-Model-Track\Testing\data\small_model_track_test --output-dir cache-64 --split test --steps 64 --workers 6
```

Train all five folds for each family:

```powershell
foreach ($fold in 0..4) {
  uv run cuhkx-train --config configs\synced_flip_imu_dropout.json --manifest manifests\cv5\train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data --cache-dir cache-64 --output-dir artifacts\repro\synced_flip --fold $fold --device cuda
  uv run cuhkx-train --config configs\pose_motion_residual.json --manifest manifests\cv5\train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data --cache-dir cache-64 --output-dir artifacts\repro\pose_motion --fold $fold --device cuda
}
```

Then run the pooled and cross-fitted blend analysis:

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\repro\synced_flip\fold_0\validation_predictions.csv artifacts\repro\synced_flip\fold_1\validation_predictions.csv artifacts\repro\synced_flip\fold_2\validation_predictions.csv artifacts\repro\synced_flip\fold_3\validation_predictions.csv artifacts\repro\synced_flip\fold_4\validation_predictions.csv `
  --candidate artifacts\repro\pose_motion\fold_0\validation_predictions.csv artifacts\repro\pose_motion\fold_1\validation_predictions.csv artifacts\repro\pose_motion\fold_2\validation_predictions.csv artifacts\repro\pose_motion\fold_3\validation_predictions.csv artifacts\repro\pose_motion\fold_4\validation_predictions.csv `
  --steps 40 --output artifacts\repro\pose_motion\ensemble_scan.json
```

Expected cross-fitted OOF is 0.49736. The submission command, checkpoint list
and 0.25 / 0.75 deployment weights are in [July26.md](July26.md).

## 4. Historical Base + balanced + pose-motion blend

This is a cv3 Fold-0 development result. Matching the published 0.54938
requires the original pre-correction cache; the current fixed-slot IMU cache
intentionally defines a new experiment. It must not be used for cv5 selection.

```powershell
uv run cuhkx-train --config configs\base.json --manifest manifests\cv3\train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data --cache-dir <historical-cache> --output-dir artifacts\repro\cv3_base --fold 0 --device cuda
uv run cuhkx-train --config configs\balanced.json --manifest manifests\cv3\train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data --cache-dir <historical-cache> --output-dir artifacts\repro\cv3_balanced --fold 0 --device cuda
uv run cuhkx-train --config configs\motion.json --manifest manifests\cv3\train.csv --data-root ..\Small-Model-Track\Training\extracted\HAR\data --cache-dir <historical-cache> --output-dir artifacts\repro\cv3_motion --fold 0 --device cuda
```

Blend Fold-0 probabilities as 0.30 Base, 0.30 balanced and 0.40 motion. The
exact historical cache is not in Git; a run using the corrected cache must be
recorded as a new result.

## Submission validation

Every submission must retain the official row order and contain only
`path,prediction`:

```powershell
uv run cuhkx-check-submission --submission <submission.csv> --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```
