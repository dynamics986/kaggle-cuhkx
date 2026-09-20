# Second Round of IMU / Radar Temporal-Model Experiments

This round trains only genuine single-modality models and does not modify the six-modality fusion network. All training continues to use the frozen
`manifests\cv5\train.csv`. The old `cache-64` and historical artifacts are
retained; the new experiments must use `cache-64-synced-points`, otherwise the
candidate explicitly raises an error.

## 1. Experiments

- IMU baseline: corrected time-synchronized 80-dimensional input plus the original TCN.
- IMU candidate: shared CNN across five devices, device embedding/gated pooling, and a two-layer relative-position Transformer.
- Radar baseline: 13-dimensional per-frame statistics from the same new cache plus the original TCN.

IMU first sorts timestamps, merges duplicate timestamps, and interpolates the
five devices onto a shared 64-point time grid. Positions outside a device's
observed range are excluded by the mask.

## 2. PowerShell variables and GPU preflight

Run from the `har-solution` root:

```powershell
$manifest = "manifests\cv5\train.csv"
$dataRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$cacheDir = "cache-64-synced-points"
$baselineRoot = "artifacts\modality_sequence_v2\baseline"
$candidateRoot = "artifacts\modality_sequence_v2\candidate"

uv run python -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'cuda': torch.version.cuda})"
```

The output must contain `cuda_available: True`.

## 3. Build the new synchronized / Radar-statistics cache

Do not change `$cacheDir` back to the old `cache-64`, and do not run
`--overwrite` on the old cache.

```powershell
uv run cuhkx-cache `
  --manifest $manifest `
  --data-root $dataRoot `
  --output-dir $cacheDir `
  --split train `
  --steps 64 `
  --workers 4
```

Inspect the fields and shapes of any cache file:

```powershell
uv run python -c "from pathlib import Path; import numpy as np; p=next(Path(r'cache-64-synced-points').glob('train_*.npz')); z=np.load(p); print(p); print({k:z[k].shape for k in z.files})"
```

The output must include:

```text
imu                 (64, 80)
imu_synced          (64, 5, 16)
imu_device_mask     (64, 5)
radar               (64, 13)
```

## 4. Run the Fold-2 smoke/probe first

First confirm that all three paths can train, write checkpoints, and predict.
`--max-clips-per-class 2` is only for workflow verification; these outputs
cannot support accuracy conclusions.

```powershell
uv run cuhkx-modality-train `
  --config configs\modality_imu_synced_tcn.json `
  --modality IMU --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
  --output-dir "artifacts\modality_sequence_v2_smoke\baseline" `
  --fold 2 --device cuda --max-clips-per-class 2

uv run cuhkx-modality-train `
  --config configs\modality_imu_device_cnn_rel_transformer.json `
  --modality IMU --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
  --output-dir "artifacts\modality_sequence_v2_smoke\candidate" `
  --fold 2 --device cuda --max-clips-per-class 2

uv run cuhkx-modality-train `
  --config configs\modality_radar_statistics_tcn.json `
  --modality Radar --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
  --output-dir "artifacts\modality_sequence_v2_smoke\baseline" `
  --fold 2 --device cuda --max-clips-per-class 2

```

## 5. Full CV5: synchronized TCN baseline

```powershell
$baselineConfigs = @{
  IMU = "configs\modality_imu_synced_tcn.json"
  Radar = "configs\modality_radar_statistics_tcn.json"
}

foreach ($modality in @("IMU", "Radar")) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-train `
      --config $baselineConfigs[$modality] `
      --modality $modality `
      --manifest $manifest `
      --data-root $dataRoot `
      --cache-dir $cacheDir `
      --output-dir $baselineRoot `
      --fold $fold `
      --device cuda
  }
}
```

The IMU baseline must be rerun because the input changed after timestamp repair;
the old 0.27593 cannot be used directly as a strict comparison.

## 6. Full CV5: IMU Transformer candidate

```powershell
$candidateConfigs = @{
  IMU = "configs\modality_imu_device_cnn_rel_transformer.json"
}

foreach ($modality in @("IMU")) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-train `
      --config $candidateConfigs[$modality] `
      --modality $modality `
      --manifest $manifest `
      --data-root $dataRoot `
      --cache-dir $cacheDir `
      --output-dir $candidateRoot `
      --fold $fold `
      --device cuda
  }
}
```

The Radar PointNet candidate was rejected after full CV5. Its implementation
and configuration have been removed, while its historical artifacts remain
available locally for audit.

Each fold writes:

```text
artifacts\modality_sequence_v2\<baseline|candidate>\<IMU|Radar>\fold_<0..4>\
  best.pt
  history.json
  summary.json
  validation_predictions.csv
```

## 7. Plot training curves for each fold

```powershell
foreach ($modality in @("IMU", "Radar")) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-plot `
      --run-dir (Join-Path $baselineRoot "$modality\fold_$fold")
  }
}
foreach ($fold in 0..4) {
  uv run cuhkx-modality-plot `
    --run-dir (Join-Path $candidateRoot "IMU\fold_$fold")
}
  }
}
```

Each fold generates `training_curves.png`, which can be used to assess
overfitting, underfitting, and the best epoch.

## 8. Generate the strict OOF report

```powershell
foreach ($modality in @("IMU", "Radar")) {
  $predictions = 0..4 | ForEach-Object {
    Join-Path $baselineRoot "$modality\fold_$($_)\validation_predictions.csv"
  }
  uv run cuhkx-modality-report `
    --manifest $manifest `
    --cache-dir $cacheDir `
    --modality $modality `
    --predictions $predictions `
    --output (Join-Path $baselineRoot "$modality\report.json")
}
$imuPredictions = 0..4 | ForEach-Object {
  Join-Path $candidateRoot "IMU\fold_$($_)\validation_predictions.csv"
}
uv run cuhkx-modality-report `
  --manifest $manifest `
  --cache-dir $cacheDir `
  --modality IMU `
  --predictions $imuPredictions `
  --output (Join-Path $candidateRoot "IMU\report.json")
```

## 9. Compare baseline and candidate

```powershell
$baseline = Get-Content (Join-Path $baselineRoot "IMU\report.json") | ConvertFrom-Json
$candidate = Get-Content (Join-Path $candidateRoot "IMU\report.json") | ConvertFrom-Json
[pscustomobject]@{
  modality = "IMU"
  examples = $candidate.examples
  baseline_oof = [math]::Round($baseline.oof_accuracy, 5)
  candidate_oof = [math]::Round($candidate.oof_accuracy, 5)
  improvement = [math]::Round($candidate.oof_accuracy - $baseline.oof_accuracy, 5)
}

$comparison | Format-Table -AutoSize
$comparison | Export-Csv `
  "artifacts\modality_sequence_v2\comparison.csv" `
  -NoTypeInformation -Encoding UTF8
```

Inspect every fold as well as pooled OOF:

```powershell
$baseline = Get-Content (Join-Path $baselineRoot "IMU\report.json") | ConvertFrom-Json
$candidate = Get-Content (Join-Path $candidateRoot "IMU\report.json") | ConvertFrom-Json
0..4 | ForEach-Object {
  [pscustomobject]@{
    modality = "IMU"
    fold = $_
    baseline = $baseline.fold_accuracy."$_"
    candidate = $candidate.fold_accuracy."$_"
    delta = $candidate.fold_accuracy."$_" - $baseline.fold_accuracy."$_"
  }
}
```

Only recommend integrating the new encoder into the six-modality fusion model
when pooled OOF improves and most folds do not decline.

## 10. Single-modality conclusion: retain the IMU candidate and reject Radar PointNet

The full five-fold OOF results are:

| Modality | Baseline OOF | Candidate OOF | Delta | Decision |
|---|---:|---:|---:|---|
| IMU | 0.25009 | 0.33461 | +0.08453 | Retain the device-aware CNN + relative Transformer |
| Radar | 0.17033 | 0.16395 | -0.00639 | Reject PointNet + Transformer |

The IMU candidate improves all five held-out folds. The Radar candidate improves
only Fold 4 slightly and declines overall. Fusion experiments continue to use
the original 13-dimensional Radar-statistics TCN in `cache-64`; they do not use
PointNet or the Radar fields with a changed sampling strategy in
`cache-64-synced-points`.

## 11. Six-modality IMU candidate: fixed Fold-2 gate

This probe replaces only IMU: it reads the synchronized five-device tensor from
`cache-64-synced-points`; vision, Skeleton motion residual, the fusion
Transformer, and the old Radar TCN continue to use the original `cache-64`.

```powershell
$manifest = "manifests\cv5\train.csv"
$dataRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$legacyCache = "cache-64"
$probeRoot = "artifacts\cv5_pose_motion_residual_imu_candidate_probe"

uv run cuhkx-train `
  --config configs\pose_motion_residual_imu_candidate_probe.json `
  --manifest $manifest `
  --data-root $dataRoot `
  --cache-dir $legacyCache `
  --output-dir $probeRoot `
  --fold 2 `
  --device cuda
```

Monitor progress during training:

```powershell
uv run cuhkx-monitor `
  --run-dir "$probeRoot\fold_2" `
  --patience 9
```

After training, plot the curve and compare the Fold-2 gate:

```powershell
uv run cuhkx-modality-plot `
  --run-dir "$probeRoot\fold_2" `
  --output "$probeRoot\fold_2\training_curves.png"

$baseline = Get-Content "artifacts\cv5_pose_motion_residual\fold_2\summary.json" | ConvertFrom-Json
$candidate = Get-Content "$probeRoot\fold_2\summary.json" | ConvertFrom-Json

[pscustomobject]@{
  baseline = [math]::Round($baseline.best_valid_accuracy, 5)
  candidate = [math]::Round($candidate.best_valid_accuracy, 5)
  delta = [math]::Round($candidate.best_valid_accuracy - $baseline.best_valid_accuracy, 5)
} | Format-List
```

The current baseline is `0.52963`. The candidate must exceed it, and should
preferably gain at least `+0.01`, before starting full CV5; otherwise retain
the current final ensemble and do not train the remaining four folds.
