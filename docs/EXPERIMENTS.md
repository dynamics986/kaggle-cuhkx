# Experiment log

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
