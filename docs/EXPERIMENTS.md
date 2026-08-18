
# Experiment log

## Current experiment (updated 2026-07-26)

All new model decisions use the frozen five-fold subject-held-out manifest at
`manifests/cv5/train.csv` with seed `20260719`.  The reported selection metric
is pooled OOF clip top-1 accuracy; an experiment is not accepted until all five
folds complete and `cuhkx-cv-report` validates their OOF coverage.

| Experiment | Config | Status | Available result | Decision |
| --- | --- | --- | --- | --- |
| Synced flip + IMU device dropout | `configs/synced_flip_imu_dropout.json` | Complete | 5-fold OOF `0.47332` (1,437/3,036); fold scores `0.40952`, `0.51509`, `0.48333`, `0.50718`, `0.45625`; fold std `0.03834` | Current comparable baseline and only complete five-fold model. |

This candidate uses 15% training-only dropout of one fixed IMU device slot.
Its checkpoint was created with `batch_size=16` and `num_workers=4`; changing
either setting creates a separate throughput experiment and cannot resume this
checkpoint.

### Failed Visual V2 regularization launch (2026-07-25)

`cv5_visual_v2_regularized` with 192px inputs, 12 frames, `batch_size=24`, and
`num_workers=8` aborted before its first training batch in the CUDA allocator.
The idle RTX 5060 Laptop GPU had 8 GB VRAM; the most likely cause is the larger
batch exceeding usable VRAM. This is not a model-quality result. The revised
trial uses `batch_size=16`, `num_workers=6`, and a two-epoch reliability probe.

### Visual V2 regularization fold-2 screen (2026-07-25)

`cv5_visual_v2_regularized_b16_w6`, fold 2, completed successfully after the
reliability probe with `batch_size=16` and `num_workers=6`. Its best validation
accuracy was `0.44626` after 71.8 minutes. This fails the pre-registered
fold-2 gate of `0.48333` (the current five-fold baseline on the same fold), so
the candidate is rejected for now. Do not train its remaining four folds and do
not use it in an ensemble.

### Pose + motion residual gate (2026-07-26)

The gated pose-plus-motion Skeleton encoder completed the frozen five-fold CV.
Its standalone OOF clip accuracy was `0.48650` (1,477/3,036), a `+0.01318`
improvement over the synced-flip + IMU-device-dropout baseline (`0.47332`).
The 0.25 baseline / 0.75 residual common-weight blend achieved pooled OOF
`0.50198` (1,524/3,036). More importantly, leave-one-fold-out weight selection
produced cross-fitted OOF `0.49736` (1,510/3,036), improving every held-out
fold. The ten-checkpoint inference ensemble is 51.7 MB and is approved for a
validated Kaggle submission.

### Public leaderboard check: pose-motion blend (2026-07-26)

The validated 0.25 baseline / 0.75 residual submission scored `0.41791` on the
public leaderboard, versus `0.42786` for the prior submission. This is a
`-0.00995` change despite a `+0.02404` cross-fitted CV gain. Treat it as a
weak distribution/noise signal, not as a reason to select future models on the
public leaderboard: if the public set were all 405 clips, the difference is
approximately four predictions. The previous submission remains the current
public-LB safe pick; the pose-motion blend remains the local-CV pick pending
further robustness checks.

The next diagnostic is a modality-availability audit and a missingness-stress
validation, not a public-score-driven weight search. Existing cache masks show
test sensor patterns `110: 206`, `111: 198`, `100: 1`; their dominant patterns
also occur in training. This rules out a wholly novel sensor-mask pattern but
does not rule out visual quality or subject-distribution shift.

## Historical three-fold development record

The results below predate the frozen five-fold protocol.  They are useful for
understanding prior decisions but are not comparable to current CV scores and
must not be used as the final model-selection baseline.

All validation scores below use fold 0 with held-out subjects `user16`, `user2`, `user20`,
`user21`, `user23`, and `user7`. Test data was not used for model selection, normalization,
pseudo-labeling, or manual labeling.

| Experiment | Best epoch | Accuracy | Size | Decision |
| --- | ---: | ---: | ---: | --- |
| Base multimodal TCN | 28 | 0.52675 | 4.57 MB | Strongest single model |
| Inverse-sqrt balanced sampling | 26 | 0.50617 | 4.57 MB | Keep only for ensemble diversity |
| Skeleton ST-GCN | 26 | 0.41152 | 4.45 MB | Reject; not ensemble-complementary |
| Pose + velocity + acceleration TCN | 40 | 0.51132 | 4.65 MB | Keep for ensemble diversity |
| Clean Base, flip disabled (old IMU cache) | 11 | 0.47428 | 4.57 MB | Correct augmentation but weak |
| Synchronized flip (fixed IMU cache) | 24 | 0.50823 | 4.57 MB | Better than Clean Base; needs controlled comparison |
| Visual V2 (fixed IMU cache) | 23 | 0.50103 | 6.26 MB | Keep for cross-fold ensemble diversity |

Drop-one ablation on the base model reduced accuracy most when Skeleton was removed
(`-0.21605`), followed by IMU (`-0.02058`) and IR (`-0.01440`). This makes cross-subject
skeleton generalization the highest-priority modeling target.

## Ensemble checks

- Base alone: `0.52675`.
- Base/balanced best coarse blend: `0.53704`.
- Base/motion best blend (`0.60/0.40`): `0.53601`.
- Base/balanced/motion (`0.30/0.30/0.40`): `0.54938`.
- A local 2.5% grid reached `0.55041`, but improved only one sample and had two tied asymmetric
  solutions. The simpler `0.30/0.30/0.40` blend is retained to reduce fold-0 weight overfitting.

The graph model decreased every base/graph blend tested, so it is excluded from inference.
A 5-7.5% synchronized-flip weight improved the retained three-model blend from `0.54938` to
`0.55041`, only one additional validation clip. This is too small to justify multi-fold training
by itself.

## Visual V2 cross-fold result

Visual V2 uses 192 x 192 letterboxed images, 12 shared-phase visual samples, and directional
`mean + max + last - first` temporal pooling. Single-model accuracy was `0.50103`, `0.42871`, and
`0.45373` on folds 0, 1, and 2. Its pooled OOF accuracy was `0.46014`, below synchronized flip's
`0.46838`, so it does not replace that model.

A single weight shared across all folds was scanned in 2.5% increments. The blend
`0.575 synchronized flip + 0.425 Visual V2` reached `0.49539` pooled OOF (1,504/3,036), with fold
accuracies `0.53395`, `0.48347`, and `0.47065`. This is a `+0.02701` absolute OOF gain and improves
all three folds. The scan is saved in `artifacts/visual_v2/oof_ensemble_scan.json` and can be
reproduced with `cuhkx-ensemble-oof`.

## Reproducibility caveat

The original base checkpoint was trained before visual-only horizontal flip was removed. That
augmentation was inconsistent with the unmirrored Skeleton and IMU streams. Balanced, graph, and
motion checkpoints were trained after the correction. This history is preserved rather than
hidden; a clean no-flip base retrain is required before making a strict single-variable claim.

These fold-0 results are development evidence, not a leaderboard estimate. Architecture and
ensemble decisions must be checked on the other subject folds before final training.

## IMU cache correction

IMU devices are now assigned to fixed slots in the order `WTC, WTLA, WTLL, WTRA, WTRL`. The old
parser compacted whichever devices were present, which could shift device identity when one was
missing. Among 2,863 training clips with IMU, 78 have at least one missing device. The cache was
rebuilt after this correction. Consequently, the synchronized-flip run used the corrected cache,
while earlier Base, Clean Base, Balanced, Graph, and Motion checkpoints used the old cache. A new
no-flip Base on the corrected cache is required to isolate the augmentation effect.
