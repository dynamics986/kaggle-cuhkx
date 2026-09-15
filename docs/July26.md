# July 26 Training and Results

> uses `batch 16` and `6 workers`

## Baseline and scope

The current comparable baseline is `cv5_synced_flip_imu_dropout`:

| Metric | Result |
| --- | ---: |
| 5-fold OOF clip accuracy | 0.47332 (1,437 / 3,036) |
| Fold 2 accuracy | 0.48333 |
| Fold standard deviation | 0.03834 |

## Pose + Motion residual gate

Visual V2 regularization failed fold 2 (`0.44626`). So the next experiment targets the modality with the strongest existing evidence: removing Skeleton previously caused the largest accuracy drop, while the standalone motion encoder was weaker but had potentially complementary errors.

### Method

`skeleton_motion_residual` keeps the original pose TCN as the main Skeleton representation. A second TCN receives pose plus first-order velocity and second-order acceleration. Their embeddings are combined as:

```text
output = LayerNorm(pose + sigmoid(gate(pose, motion)) * motion)
```


### supervised fold-2 screen (about 1–2 hours)

```powershell
uv run cuhkx-train `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --manifest manifests\cv5\train.csv `
  --config configs\pose_motion_residual.json `
  --cache-dir cache-64 `
  --output-dir artifacts\cv5_pose_motion_residual `
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

### Night: unattended remaining folds (8–10 hours)

If the fold-2 gate passes, run the following before leaving. Fold 2 is omitted
because it already completed with the identical configuration.

```powershell
foreach ($fold in @(0, 1, 3, 4)) {
  uv run cuhkx-train `
    --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
    --manifest manifests\cv5\train.csv `
    --config configs\pose_motion_residual.json `
    --cache-dir cache-64 `
    --output-dir artifacts\cv5_pose_motion_residual `
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

## Result and Kaggle Submission

The pose + motion residual gate completed all five folds successfully.

| Model / evaluation | OOF clip accuracy | Change vs. baseline |
| --- | ---: | ---: |
| Synced flip + IMU device dropout | 0.47332 | — |
| Pose + motion residual gate | 0.48650 | +0.01318 |
| Cross-fitted two-family blend | 0.49736 | +0.02404 |

The cross-fitted blend improved every held-out fold, which is much stronger
evidence than a pooled OOF gain alone. The pooled OOF scan selects the final
common deployment mix of **0.25 baseline + 0.75 pose-motion residual**. Its
pooled OOF is 0.50198; use 0.49736 as the less-optimistic validation estimate.

The ten checkpoints total 51.7 MB, below the 100 MB limit. Generate predictions
with equal weight within each family: 0.05 for each of the five baseline folds
and 0.15 for each of the five pose-motion residual folds.

```powershell
uv run cuhkx-predict `
  --checkpoints `
    artifacts\cv5_synced_flip_imu_dropout\fold_0\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_1\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_2\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_3\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_4\best.pt `
    artifacts\cv5_pose_motion_residual\fold_0\best.pt `
    artifacts\cv5_pose_motion_residual\fold_1\best.pt `
    artifacts\cv5_pose_motion_residual\fold_2\best.pt `
    artifacts\cv5_pose_motion_residual\fold_3\best.pt `
    artifacts\cv5_pose_motion_residual\fold_4\best.pt `
  --weights 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 `
  --manifest manifests\cv5\test.csv `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --cache-dir cache-64 `
  --views 3 `
  --output artifacts\submission_pose_motion_blend.csv
```

Validate the file before uploading it manually on Kaggle:

```powershell
uv run cuhkx-check-submission `
  --submission artifacts\submission_pose_motion_blend.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv
```

After this submission, we scored `0.41791`, lower than the prior public score `0.42786`.


### Difficult actions and user analysis

The following results are from five-fold OOF: each user's samples are predicted only by models that have not seen that user.

The changes for difficult actions are not consistent:

| Action | Baseline | Gating Model | Final Ensemble |
|---|---:|---:|---:|
| Take and use tableware | 11/97 = 11.3% | 9/97 = 9.3% | 7/97 = 7.2% |
| Write | 1/39 = 2.6% | 6/39 = 15.4% | 5/39 = 12.8% |
| Make a phone call | 5/43 = 11.6% | 7/43 = 16.3% | 7/43 = 16.3% |
| Take medicine | 13/72 = 18.1% | 9/72 = 12.5% | 9/72 = 12.5% |

So the gating branch significantly improves `Write` and `Make a phone call`, but regresses on `Take and use tableware` and `Take medicine`. The overall improvement comes from accumulated gains across multiple actions, not all fine-grained hand actions becoming better.

Across users, the final ensemble improves over baseline for 13 out of 18 held-out users, with 5 declining. Notable improvements include:

- `user16`: 50.5% → 60.2%
- `user9`: 42.8% → 51.9%
- `user5`: 30.6% → 40.0%
- `user4`: 42.0% → 46.9%

Users that remain difficult include `user3` (35.6%), `user5` (40.0%), and `user23` (42.6%). The standalone gating model actually shows larger fold-to-fold fluctuation, but the cross-fitted ensemble results are better than baseline across all five folds, which is the primary basis for keeping the ensemble.

## Missingness and input-quality stress validation

The public-score gap makes robustness a diagnostic priority. This evaluation
uses **only the labelled training manifest**: each fold checkpoint predicts its
own held-out users, and each input degradation is applied only at evaluation
time. It does not inspect test labels, tune to the public leaderboard, or
replace the main OOF score.

The `cuhkx-stress-test` command evaluates six conditions:

| Condition | Intervention | What it diagnoses |
| --- | --- | --- |
| `none` | No change | The one-view held-out reference for this command |
| `drop_thermal` | Mask Thermal and set its pixels to zero | Robustness if an otherwise-present Thermal stream is unavailable |
| `drop_radar` | Mask Radar and set its features to zero | Reliance on Radar |
| `drop_imu` | Mask IMU and set its features to zero | Reliance on IMU |
| `drop_skeleton` | Mask Skeleton and set its features to zero | Reliance on pose |
| `visual_first_frame` | Repeat the first sampled visual frame across time | Sensitivity to loss of visual temporal information |

`affected_accuracy` is calculated only on clips that originally had the
intervened modality (or any visual modality for `visual_first_frame`). This
avoids falsely treating a clip whose modality was already missing as an
additional stress-test sample. These are intentionally severe, controlled
counterfactuals: they estimate relative dependence, not the exact corruption
rate of Kaggle test data.

Run the baseline first. Cached sensor features must already exist; this command
does not train or overwrite checkpoints.

```powershell
uv run cuhkx-stress-test `
  --checkpoints `
    artifacts\cv5_synced_flip_imu_dropout\fold_0\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_1\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_2\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_3\best.pt `
    artifacts\cv5_synced_flip_imu_dropout\fold_4\best.pt `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output artifacts\cv5_synced_flip_imu_dropout\stress_report.json
```

Then run the same controlled cases for the pose-motion model.

```powershell
uv run cuhkx-stress-test `
  --checkpoints `
    artifacts\cv5_pose_motion_residual\fold_0\best.pt `
    artifacts\cv5_pose_motion_residual\fold_1\best.pt `
    artifacts\cv5_pose_motion_residual\fold_2\best.pt `
    artifacts\cv5_pose_motion_residual\fold_3\best.pt `
    artifacts\cv5_pose_motion_residual\fold_4\best.pt `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output artifacts\cv5_pose_motion_residual\stress_report.json
```

For a quick diagnosis of the two shifts most relevant to the competition, add
`--stresses none drop_thermal drop_radar`. Use the full six-case run before a
major architecture change.
