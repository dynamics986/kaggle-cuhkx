# Reproduction from a clean checkout

Run every command from `path\to\CUHK-X\har-solution` in PowerShell.

The raw competition data must be at these paths relative to the repository:

```text
..\Small-Model-Track\Training\extracted\HAR\data
..\Small-Model-Track\Testing\data\small_model_track_test
..\Small-Model-Track\Testing\test_file\test.csv
```

`manifests\cv5\train.csv`, `manifests\cv5\test.csv`, and the configuration files are inputs. 

Install the locked environment once. The first YOLO run may download the public COCO-pretrained `yolov8n.pt` weight if it is not already present.

```powershell
uv sync --extra sep12 --dev
```

## Reproducibility boundary for the YOLO results

The Sep13 and Sep14 visual results depend on 800 **human-labelled**
Depth_Color frames. Those bounding boxes are not derivable from the raw HAR
data, and are deliberately not stored in Git. Therefore a repository with
only raw data and code cannot bit-for-bit reproduce the published YOLO-based
scores (`0.51949` and `0.39748`) without the original annotation package.

To reproduce those exact numbers, obtain the original `annotations_800`
directory and place it at `artifacts\repro\annotations_800`. It must contain
`images\`, `labels\`, and `annotation_manifest.csv`.

For a new, independently labelled reproduction, first create the deterministic
800-frame sample:

```powershell
uv run python -c "from cuhkx_har.yolo import export_annotations; print(export_annotations('manifests/cv5/train.csv', r'..\Small-Model-Track\Training\extracted\HAR\data', r'artifacts\repro\annotations_800', 800))"
```

Label every exported image in YOLO format with the sole class `person` (class
id `0`) and save labels to `artifacts\repro\annotations_800\labels`. This
creates a valid new experiment, but its detector, cache hashes, and measured
scores can differ from the published run. The Sep14 score guard intentionally
rejects a changed legacy baseline instead of silently calling it historical.

## 1. Build all clean-run prerequisites

The following commands create fresh fold-safe YOLO detectors, all required
YOLO crop caches, and the 64-step sensor cache. They never read a previous
`artifacts/` directory.

Train a detector independently for Fold 2, Fold 4, and the final full-data
fit. Fold 2 and Fold 4 detectors exclude their respective validation users;
the full detector is used only for final training and test inference.

```powershell
$annotations = "artifacts\repro\annotations_800"
$detectors = "artifacts\repro\detectors"

foreach ($fold in @(2, 4, "full")) {
  uv run cuhkx-sep12-prepare detector `
    --annotations $annotations `
    --manifest manifests\cv5\train.csv `
    --fold $fold `
    --output "$detectors\fold_$fold" `
    --device 0
}
```

Create the three-modality, temporally aligned crop caches. The output names
are distinct by detector fold, so a validation clip is never cropped by a
detector trained on that clip's subject.

```powershell
$trainRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$testRoot = "..\Small-Model-Track\Testing\data\small_model_track_test"
$cropRoot = "artifacts\repro\sep12\cache"

foreach ($fold in @(2, 4, "full")) {
  $weight = "artifacts\repro\detectors\fold_$fold\yolov8n_$fold.pt"
  $tag = if ($fold -eq "full") { "full" } else { "fold_$fold" }

  uv run cuhkx-sep12-prepare crops `
    --manifest manifests\cv5\train.csv `
    --fold $fold --weights $weight --split train --data-root $trainRoot `
    --output "$cropRoot\$tag" --device 0

  uv run cuhkx-sep12-prepare crops `
    --manifest manifests\cv5\train.csv --test-manifest manifests\cv5\test.csv `
    --fold $fold --weights $weight --split test --data-root $testRoot `
    --output "$cropRoot\$tag" --device 0
}
```

Build the sensor cache once. This cache is shared by the LightGBM and
non-YOLO multimodal experiments.

```powershell
$sensorCache = "artifacts\repro\sensors"

uv run cuhkx-cache `
  --manifest manifests\cv5\train.csv --data-root $trainRoot `
  --output-dir $sensorCache --split train --steps 64 --workers 6

uv run cuhkx-cache `
  --manifest manifests\cv5\test.csv --data-root $testRoot `
  --output-dir $sensorCache --split test --steps 64 --workers 6
```

## 2. m01 LightGBM with YOLO-box temporal features

With the original annotation package, this reproduces the Sep14 box-feature
baseline. It validates Folds 2 and 4, refits using all labelled clips, and
writes a submission.

```powershell
uv run --extra sep12 python -m cuhkx_sep14.lightgbm `
  --output artifacts\repro\m01_yolo_box_baseline `
  --sensor-cache artifacts\repro\sensors `
  --cache-root artifacts\repro\sep12\cache `
  --detector2 artifacts\repro\detectors\fold_2\yolov8n_2.pt `
  --detector4 artifacts\repro\detectors\fold_4\yolov8n_4.pt `
  --detector-full artifacts\repro\detectors\fold_full\yolov8n_full.pt
```

The output is `artifacts\repro\m01_yolo_box_baseline\submission.csv`.
With the original annotations and the recorded environment, the reference
scores are Fold 2 `0.55926`, Fold 4 `0.48594`, and weighted `0.51949`.

## 3. Sep13 m05 dual ResNet-18 with moderate augmentation

This trains a YOLO-cropped IR + Depth_Color dual ResNet-18 on Folds 2 and 4
and writes a submission after each fold.

```powershell
uv run --extra sep12 python -m cuhkx_sep12.train `
  --config configs\sep13\m05_dual_resnet18.json `
  --output artifacts\repro\m05_sep13_moderate `
  --folds 2 4 `
  --cache-root artifacts\repro\sep12\cache `
  --detector2 artifacts\repro\detectors\fold_2\yolov8n_2.pt `
  --detector4 artifacts\repro\detectors\fold_4\yolov8n_4.pt `
  --device cuda
```

Its two results are stored under `fold_2\summary.json` and
`fold_4\summary.json`; the recorded reference values are `0.44259` and
`0.35938`. The selected-fold submission is
`artifacts\repro\m05_sep13_moderate\submission.csv`.

## 4. Synced-flip baseline plus pose-motion residual blend

This result needs only raw data, source-controlled manifests/configuration,
and the clean sensor cache created above. Train all five folds of both
families:

```powershell
foreach ($fold in 0..4) {
  uv run cuhkx-train `
    --config configs\synced_flip_imu_dropout.json `
    --manifest manifests\cv5\train.csv --data-root $trainRoot `
    --cache-dir artifacts\repro\sensors `
    --output-dir artifacts\repro\synced_flip --fold $fold --device cuda

  uv run cuhkx-train `
    --config configs\pose_motion_residual.json `
    --manifest manifests\cv5\train.csv --data-root $trainRoot `
    --cache-dir artifacts\repro\sensors `
    --output-dir artifacts\repro\pose_motion --fold $fold --device cuda
}
```

Measure the cross-fitted OOF blend before creating the test submission:

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\repro\synced_flip\fold_0\validation_predictions.csv artifacts\repro\synced_flip\fold_1\validation_predictions.csv artifacts\repro\synced_flip\fold_2\validation_predictions.csv artifacts\repro\synced_flip\fold_3\validation_predictions.csv artifacts\repro\synced_flip\fold_4\validation_predictions.csv `
  --candidate artifacts\repro\pose_motion\fold_0\validation_predictions.csv artifacts\repro\pose_motion\fold_1\validation_predictions.csv artifacts\repro\pose_motion\fold_2\validation_predictions.csv artifacts\repro\pose_motion\fold_3\validation_predictions.csv artifacts\repro\pose_motion\fold_4\validation_predictions.csv `
  --steps 40 --output artifacts\repro\pose_motion\ensemble_scan.json
```

The reported cross-fitted OOF is `0.49736`. Produce the final 0.25 / 0.75
blend submission as follows:

```powershell
uv run cuhkx-predict `
  --checkpoints artifacts\repro\synced_flip\fold_0\best.pt artifacts\repro\synced_flip\fold_1\best.pt artifacts\repro\synced_flip\fold_2\best.pt artifacts\repro\synced_flip\fold_3\best.pt artifacts\repro\synced_flip\fold_4\best.pt artifacts\repro\pose_motion\fold_0\best.pt artifacts\repro\pose_motion\fold_1\best.pt artifacts\repro\pose_motion\fold_2\best.pt artifacts\repro\pose_motion\fold_3\best.pt artifacts\repro\pose_motion\fold_4\best.pt `
  --weights 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 `
  --manifest manifests\cv5\test.csv --data-root $testRoot `
  --cache-dir artifacts\repro\sensors --views 3 `
  --output artifacts\repro\submission_pose_motion_blend.csv
```

## 5. Historical cv3 Base + balanced + pose-motion blend

The historical `0.54938` is **not reproducible from raw data and source code
alone**. It used a pre-correction cache that was not retained. The current
cache implementation intentionally produces a different experiment; do not
report its score as the historical result.

## Submission validation

Validate every output before uploading it manually to Kaggle:

```powershell
uv run cuhkx-check-submission `
  --submission <submission.csv> `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```
