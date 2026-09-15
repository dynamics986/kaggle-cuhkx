# Single-Modality Control Experiments

These experiments use the frozen `manifests/cv5/train.csv` as the only training manifest. `cuhkx_modality` checks the full `clip_id → fold` mapping in this file. It cannot be replaced with the historical three-fold manifest at `manifests/cv3/train.csv` or a newly split file.

All six modalities are trained with five folds, for 30 independent models in total:

- `Depth_Color`, `IR`, `Thermal`: separate visual temporal models;
- `Skeleton`, `IMU`, `Radar`: separate sensor temporal models.

If a clip does not contain the target modality, it is excluded from training and validation for that modality. The number of usable samples for each modality is recorded in its `summary.json` and in the final per-action comparison table.


## Train Five Folds for Six Modalities

Run from the `har-solution` root directory. For the first run, it is recommended to replace `configs/modality_base.json` with `configs/modality_smoke.json` and validate one fold only. Use the base configuration for the formal experiment.

```powershell
$modalities = @("Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar")

foreach ($modality in $modalities) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-train `
      --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
      --manifest manifests/cv5/train.csv `
      --config configs/modality_base.json `
      --cache-dir cache-64 `
      --output-dir artifacts\modality `
      --fold $fold `
      --modality $modality `
      --device cuda
  }
}
```

Each training job writes:

```text
artifacts/modality/<modality>/fold_<0..4>/
  best.pt
  history.json
  summary.json
  validation_predictions.csv
```

## Report for Each Modality

After all five folds finish, use the five validation predictions to create one strictly checked OOF file. The command rejects missing folds, duplicate clips, cross-fold predictions, inconsistent labels, and clips where the target modality is actually missing.

```powershell
foreach ($modality in $modalities) {
  $predictions = 0..4 | ForEach-Object {
    Join-Path artifacts\modality "$modality\fold_$($_)\validation_predictions.csv"
  }

  uv run cuhkx-modality-report `
    --manifest manifests/cv5/train.csv `
    --cache-dir cache-64 `
    --modality $modality `
    --predictions $predictions `
    --output (Join-Path artifacts\modality "$modality\report.json")
}
```

Each modality produces `artifacts/modality/<modality>/report.json` and `report.oof.csv`.

## Compare Six Modalities by Action

After all six OOF files are created, run:

```powershell
$oof = $modalities | ForEach-Object {
  Join-Path artifacts\modality "$_\report.oof.csv"
}

uv run cuhkx-modality-summary `
  --oof $oof `
  --output (Join-Path artifacts\modality "per_action_modality_accuracy.csv")
```

The output `artifacts/modality/per_action_modality_accuracy.csv` has one row per action. It contains accuracy and usable sample count for each of the six modalities, plus `best_modality`. When accuracies are tied, all best modalities are listed with `|`.

## Training Curve Plots

Each fold updates `history.json` after every epoch. After training finishes, or once at least one full epoch is available during training, generate loss and accuracy curves with:

```powershell
uv run cuhkx-modality-plot `
  --run-dir artifacts\modality\IMU\fold_0
```

By default this creates `artifacts/modality/IMU/fold_0/training_curves.png`. The left plot shows training and validation loss. The right plot shows training and validation accuracy, and marks the highest validation accuracy and its epoch.

To choose an output path:

```powershell
uv run cuhkx-modality-plot `
  --run-dir artifacts\modality\IMU\fold_0 `
  --output artifacts\modality\IMU\fold_0\curves_final.png
```

## Historical Record

Earlier three-fold experiments, rejected architectures, and their scores are intentionally not presented as the current solution. Their manifests are retained at `manifests/cv3/{train,test}.csv`, and their results are retained in [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) for reproducibility and audit.
