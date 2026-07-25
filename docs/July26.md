# July 26: supervised screen, unattended overnight CV

## Phase A failure diagnosis

The attempted 192px / 12-frame Visual V2 run aborted during its first CUDA
allocation (`c10_cuda_check` / CUDA allocator). The GPU was otherwise idle and
had 8,151 MiB free. The relevant new setting was `batch_size=24`; the older
Visual V2 used 16. Treat this as a probable CUDA out-of-memory failure, not a
cache or Python-code failure.

The revised Visual V2 regularization configuration uses **batch 16** and **6
workers**. Six workers still improves visual decoding throughput over the prior
four; batch 16 is already known to fit this 8 GB GPU.

## Baseline and scope

The current comparable baseline is `cv5_synced_flip_imu_dropout`:

| Metric | Result |
| --- | ---: |
| 5-fold OOF clip accuracy | 0.47332 (1,437 / 3,036) |
| Fold 2 accuracy | 0.48333 |
| Fold standard deviation | 0.03834 |

Tonight tests Visual V2 with stronger regularization. It preserves the
full-frame, 12-frame, directional design while targeting the observed
cross-subject overfitting. The larger V3 model is explicitly deferred: an
unattended job must not start an unverified model that may OOM or fail workers.

## Now: supervised Phase 0 reliability probe (5–15 minutes)

This has the exact architecture, batch size, workers, and augmentations planned
for tonight; only the epoch count is shortened to two.

```powershell
cd C:\Users\dynam\Documents\CUHK-X\har-solution

$probe = "artifacts\probe_visual_v2_regularized_b16_w6"
uv run cuhkx-train `
  --config configs\visual_v2_regularized_probe.json `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir $probe `
  --fold 2
```

Monitor from another PowerShell window:

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\probe_visual_v2_regularized_b16_w6\fold_2 `
  --patience 2
```

Proceed only when a `summary.json` is written with no CUDA abort or worker
failure and peak GPU memory is comfortably below 8 GB. If it fails, edit both
V2 configs to `batch_size: 12`, use a new probe directory, and repeat.

## Now: supervised Phase 1 fold-2 screen (up to 2 hours)

The short probe cannot be resumed because it has a different epoch count. Run a
full, fresh fold-2 screen after Phase 0 passes:

```powershell
$screen = "artifacts\cv5_visual_v2_regularized_b16_w6"
uv run cuhkx-train `
  --config configs\visual_v2_regularized.json `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir $screen `
  --fold 2
```

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\cv5_visual_v2_regularized_b16_w6\fold_2 `
  --patience 10
```

Before leaving, inspect the fold-2 `summary.json`. Start the overnight job only
if best validation accuracy is at least `0.48333` and the best epoch is not an
obvious one-epoch spike. This is a screening gate, not a CV result.

**Result (2026-07-25):** fold 2 completed at `0.44626` in 71.8 minutes. It did
not pass this gate. Phase 2 must not be started for
`cv5_visual_v2_regularized_b16_w6`.

## Tonight: unattended Phase 2 (8–10 hours)

If and only if a future Phase 1 passes, start this sequential loop. It runs the four
remaining folds; fold 2 is excluded because it already used the exact same
configuration. Do not add `--resume` to this loop.

```powershell
$screen = "artifacts\cv5_visual_v2_regularized_b16_w6"
foreach ($fold in @(0, 1, 3, 4)) {
  uv run cuhkx-train `
    --config configs\visual_v2_regularized.json `
    --manifest manifests\cv5\train.csv `
    --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
    --cache-dir cache-64 `
    --output-dir $screen `
    --fold $fold
}
```

If a fold is interrupted, resume it the next day with its own matching fold,
configuration, and `last.pt`; never use one fold's checkpoint in this loop.

## Morning: OOF audit and ensemble test

First confirm every fold has `summary.json` and `validation_predictions.csv`.
Then run the strict audit:

```powershell
uv run cuhkx-cv-report `
  --manifest manifests\cv5\train.csv `
  --predictions `
    artifacts\cv5_visual_v2_regularized_b16_w6\fold_0\validation_predictions.csv `
    artifacts\cv5_visual_v2_regularized_b16_w6\fold_1\validation_predictions.csv `
    artifacts\cv5_visual_v2_regularized_b16_w6\fold_2\validation_predictions.csv `
    artifacts\cv5_visual_v2_regularized_b16_w6\fold_3\validation_predictions.csv `
    artifacts\cv5_visual_v2_regularized_b16_w6\fold_4\validation_predictions.csv `
  --name cv5_visual_v2_regularized_b16_w6 `
  --output artifacts\cv5_visual_v2_regularized_b16_w6\cv_report.json
```

Evaluate complementary errors only after the candidate completes five folds:

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\cv5_synced_flip_imu_dropout\fold_0\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_1\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_2\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_3\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_4\validation_predictions.csv `
  --candidate artifacts\cv5_visual_v2_regularized_b16_w6\fold_0\validation_predictions.csv artifacts\cv5_visual_v2_regularized_b16_w6\fold_1\validation_predictions.csv artifacts\cv5_visual_v2_regularized_b16_w6\fold_2\validation_predictions.csv artifacts\cv5_visual_v2_regularized_b16_w6\fold_3\validation_predictions.csv artifacts\cv5_visual_v2_regularized_b16_w6\fold_4\validation_predictions.csv `
  --steps 40 `
  --output artifacts\cv5_visual_v2_regularized_b16_w6\ensemble_scan.json
```

Keep the candidate only if its full OOF improves on 0.47332 or its
cross-fitted blend improves on the baseline. Record the result in
`docs/EXPERIMENTS.md`; do not submit before that decision.

## Replacement plan after the failed Visual V2 screen: Pose + Motion residual gate

Visual V2 regularization failed fold 2 (`0.44626`), so it must not consume the
overnight window. The next experiment targets the modality with the strongest
existing evidence: removing Skeleton previously caused the largest accuracy
drop, while the standalone motion encoder was weaker but had potentially
complementary errors.

### Method

`skeleton_motion_residual` keeps the original pose TCN as the main Skeleton
representation. A second TCN receives pose plus first-order velocity and
second-order acceleration. Their embeddings are combined as:

```text
output = LayerNorm(pose + sigmoid(gate(pose, motion)) * motion)
```

The gate is initialized with bias `-2`, so the initial motion contribution is
about 12%. Training must earn a larger motion contribution; the proven pose
path is never replaced. The change is intentionally confined to the Skeleton
encoder. Visual streams, IMU, radar, fusion, subject folds, and normalizer
rules remain unchanged.

### Now: two-epoch reliability probe (5–15 minutes)

```powershell
cd C:\Users\dynam\Documents\CUHK-X\har-solution

$probe = "artifacts\probe_pose_motion_residual"
uv run cuhkx-train `
  --config configs\pose_motion_residual_probe.json `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir $probe `
  --fold 2
```

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\probe_pose_motion_residual\fold_2 `
  --patience 2
```

Require a normal `summary.json` and a model size below 100 MB before continuing.

### Now: supervised fold-2 screen (about 1–2 hours)

```powershell
$residual = "artifacts\cv5_pose_motion_residual"
uv run cuhkx-train `
  --config configs\pose_motion_residual.json `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir $residual `
  --fold 2
```

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\cv5_pose_motion_residual\fold_2 `
  --patience 9
```

Start the unattended run only if this screen reaches at least `0.48333`, the
current baseline's fold-2 score, and the validation curve is not a one-epoch
spike. Do not combine this model with the failed Visual V2 candidate.

### Tonight: unattended remaining folds (8–10 hours)

If the fold-2 gate passes, run the following before leaving. Fold 2 is omitted
because it already completed with the identical configuration.

```powershell
$residual = "artifacts\cv5_pose_motion_residual"
foreach ($fold in @(0, 1, 3, 4)) {
  uv run cuhkx-train `
    --config configs\pose_motion_residual.json `
    --manifest manifests\cv5\train.csv `
    --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
    --cache-dir cache-64 `
    --output-dir $residual `
    --fold $fold
}
```

Do not add `--resume` to the loop. A later interruption is resumed manually for
that same fold with the same config and its own `last.pt`.

### Morning: OOF audit and cross-fitted ensemble

```powershell
uv run cuhkx-cv-report `
  --manifest manifests\cv5\train.csv `
  --predictions `
    artifacts\cv5_pose_motion_residual\fold_0\validation_predictions.csv `
    artifacts\cv5_pose_motion_residual\fold_1\validation_predictions.csv `
    artifacts\cv5_pose_motion_residual\fold_2\validation_predictions.csv `
    artifacts\cv5_pose_motion_residual\fold_3\validation_predictions.csv `
    artifacts\cv5_pose_motion_residual\fold_4\validation_predictions.csv `
  --name cv5_pose_motion_residual `
  --output artifacts\cv5_pose_motion_residual\cv_report.json
```

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\cv5_synced_flip_imu_dropout\fold_0\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_1\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_2\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_3\validation_predictions.csv artifacts\cv5_synced_flip_imu_dropout\fold_4\validation_predictions.csv `
  --candidate artifacts\cv5_pose_motion_residual\fold_0\validation_predictions.csv artifacts\cv5_pose_motion_residual\fold_1\validation_predictions.csv artifacts\cv5_pose_motion_residual\fold_2\validation_predictions.csv artifacts\cv5_pose_motion_residual\fold_3\validation_predictions.csv artifacts\cv5_pose_motion_residual\fold_4\validation_predictions.csv `
  --steps 40 `
  --output artifacts\cv5_pose_motion_residual\ensemble_scan.json
```

The next engineering option, only if this gate fails, is Skeleton-specific
augmentation (small joint jitter and body-scale variation) after a coordinate
system audit. It is a better next bet than filling the 100 MB parameter limit
or launching another unvalidated Visual V2 family unattended.
