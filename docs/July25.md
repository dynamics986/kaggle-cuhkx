
# Iteration workflow (updated 2026-07-25)

## One-time setup: freeze a five-fold protocol

Use five subject-held-out folds for all new experiments.  The older
`manifests/train.csv` has three folds and is useful for historical comparison,
but do not mix it with five-fold scores.

```powershell
cd C:\Users\dynam\Documents\CUHK-X\har-solution

uv run cuhkx-index `
  --train-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --test-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --class-mapping ..\Small-Model-Track\class_mapping.csv `
  --test-csv ..\Small-Model-Track\Testing\test_file\test.csv `
  --output-dir manifests\cv5 `
  --folds 5 `
  --seed 20260719
```

Commit or archive `manifests/cv5/train.csv` after creating it.  Do not change
its seed, number of folds, or subject assignments while comparing experiments.
The cache remains valid because feature caching is independent of fold
assignment.  If it does not already exist, make the training cache once:

```powershell
uv run cuhkx-cache `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --output-dir cache-64 --split train --steps 64 --workers 2
```

## Run one experiment across all folds

Each fold writes a checkpoint, held-out probabilities, class metrics, and
history below one dedicated directory.  Give every experiment a new directory;
never overwrite an accepted baseline.

Run this command once for each `fold` from 0 through 4.  The commands below use
the small, targeted IMU device-dropout candidate.  For a clean control, replace
the config with `configs/synced_flip.json`.

```powershell
$experiment = "artifacts\cv5_synced_flip_imu_dropout"
foreach ($fold in 0..4) {
  uv run cuhkx-train `
    --config configs\synced_flip_imu_dropout.json `
    --manifest manifests\cv5\train.csv `
    --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
    --cache-dir cache-64 `
    --output-dir $experiment `
    --fold $fold
}
```

This loop is intentionally sequential: the current program uses one GPU.  An
interrupted fold can be resumed only with the same fold and exactly the same
configuration (including `batch_size` and `num_workers`), for example:

```powershell
uv run cuhkx-train `
  --config configs\synced_flip_imu_dropout.json `
  --manifest manifests\cv5\train.csv `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --cache-dir cache-64 `
  --output-dir artifacts\cv5_synced_flip_imu_dropout `
  --fold 3 `
  --resume artifacts\cv5_synced_flip_imu_dropout\fold_3\last.pt
```

Do not put one fold's `--resume` path inside a `0..4` loop.  A throughput
experiment that changes `batch_size` or `num_workers` is a new experiment and
must use a new artifact directory without `--resume`.



## Produce the only score used for model selection

After all five folds finish, run the strict OOF audit:

```powershell
uv run cuhkx-cv-report `
  --manifest manifests\cv5\train.csv `
  --predictions `
    artifacts\cv5_synced_flip_imu_dropout\fold_0\validation_predictions.csv `
    artifacts\cv5_synced_flip_imu_dropout\fold_1\validation_predictions.csv `
    artifacts\cv5_synced_flip_imu_dropout\fold_2\validation_predictions.csv `
    artifacts\cv5_synced_flip_imu_dropout\fold_3\validation_predictions.csv `
    artifacts\cv5_synced_flip_imu_dropout\fold_4\validation_predictions.csv `
  --name cv5_synced_flip_imu_dropout `
  --output artifacts\cv5_synced_flip_imu_dropout\cv_report.json
```

The command fails rather than issuing a plausible score if any of these are
wrong: missing/duplicate OOF clips, mixed folds, labels that differ from the
manifest, probability vectors that are invalid, or subject leakage.  The JSON
report contains the Kaggle-aligned `oof_accuracy`, the accuracy of each fold,
fold standard deviation, subject allocation, per-class accuracy, and a 40×40
confusion matrix.

Compare experiments primarily by `oof_accuracy`, then use lower
`fold_std_accuracy` as a robustness tie-breaker.  A change that only wins one
fold is a hypothesis, not an accepted improvement.

## Ensemble selection without OOF weight leakage

An ordinary OOF weight search selects its best weight after observing every OOF
label, so its top score is optimistic.  The ensemble tool now prints two views:

1. The pooled OOF grid, used only to pick one final common deployment weight.
2. `Cross-fitted blend accuracy`, where each held-out fold's weight is selected
   exclusively using the other folds.  Use this second number to decide whether
   the ensemble is genuinely better.

For example, compare a five-fold baseline and candidate:

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\cv5_baseline\fold_0\validation_predictions.csv artifacts\cv5_baseline\fold_1\validation_predictions.csv artifacts\cv5_baseline\fold_2\validation_predictions.csv artifacts\cv5_baseline\fold_3\validation_predictions.csv artifacts\cv5_baseline\fold_4\validation_predictions.csv `
  --candidate artifacts\cv5_candidate\fold_0\validation_predictions.csv artifacts\cv5_candidate\fold_1\validation_predictions.csv artifacts\cv5_candidate\fold_2\validation_predictions.csv artifacts\cv5_candidate\fold_3\validation_predictions.csv artifacts\cv5_candidate\fold_4\validation_predictions.csv `
  --steps 40 `
  --output artifacts\cv5_candidate\ensemble_scan.json
```

Keep only an ensemble that improves the cross-fitted score and does not collapse
on an individual subject fold.  Use one common family weight across all folds;
do not tune a distinct final weight per fold.

## Iteration rules

1. Establish the clean `synced_flip` five-fold control on the corrected IMU
   cache.  Historical three-fold results are not a comparable baseline.
2. Change one related idea at a time.  Start with
   `synced_flip_imu_dropout.json`, which drops one of the five fixed IMU device
   slots for 15% of training samples.  It is disabled during validation and
   inference, and does not change normalization statistics.
3. Record config path, git revision, per-fold scores, OOF score, and decision
   in `docs/EXPERIMENTS.md`.  Promote a candidate only after all five folds.
4. Diagnose errors with `per_class` and `confusion_matrix` in `cv_report.json`.
   Target a pattern demonstrated across folds, not a single leaderboard move.
5. Only after choosing the architecture and ensemble by CV, train final models,
   create `submission.csv`, validate it with `cuhkx-check-submission`, and make
   at most the planned Kaggle submissions.

## Critical adoption of `Gemini_Advice.md`

Adopted now:

- A clean control is mandatory because old checkpoints used the pre-fix IMU
  cache and inconsistent visual-only flip history.
- Subject-held-out OOF, per-fold reporting, and cross-fitted ensemble selection
  are the decision gate.
- IMU device dropout is a low-risk robustness experiment because the cache has
  five fixed device slots and known partial-device clips.

Deferred until the clean five-fold baseline exists:

- Skeleton augmentation needs an explicit coordinate-system audit and a
  controlled ablation.  Random 3D rotation is not automatically valid for
  camera-aligned or gravity-aligned modalities.
- Rank averaging and stacking need a nested/CV-safe meta-model protocol; they
  should not be selected on the same OOF labels used to report their score.

Rejected for the current small-model track:

- Pretrained X3D, Video Swin, SlowFast, and similar backbones conflict with the
  existing from-scratch/small-model constraints and would obscure the source of
  any gain.  A larger architecture can be reconsidered only after a documented
  capacity and rule audit.
- The proposed “winning architecture” is a hypothesis, not evidence.  It is
  not a substitute for a stable subject-held-out CV gain.
