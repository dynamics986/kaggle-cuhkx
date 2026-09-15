# August 17 Summary

## Data and Training Pipeline Improvements

- Synchronized data augmentation: flip all six modalities together, including:
  - mirror Skeleton and swap left/right joints;
  - swap left/right IMU devices;
  - flip the sign of the Radar horizontal coordinate.
- IMU cache device slots are fixed as `WTC, WTLA, WTLL, WTRA, WTRL`.
- Add IMU device dropout during training: randomly mask one IMU device slot to reduce reliance on a single wearable device.
- Sensor normalization is computed only from the training users in each fold, to avoid validation leakage.
- Upgrade from the old three-fold protocol to frozen five-fold subject-held-out CV. Historical three-fold manifests are retained under `manifests/cv3/{train,test}.csv`; all later experiments use `manifests/cv5/train.csv` and fixed seed `20260719`.
- Add training monitoring, resume-consistency checks, OOF completeness audits, and submission CSV validation.

## Models and Methods Tried

| Method | Result and conclusion |
|---|---|
| Base multimodal model: visual CNN + TCN for each sensor + Transformer modality fusion | Historical fold-0 performance was strong, but the early cache and augmentation settings differ from later ones, so it cannot be directly compared with the new five-fold results. |
| Inverse-sqrt class balancing | Overall accuracy decreased, so it was not continued as the main direction. Minority classes lack cross-user diversity, and resampling cannot create new user variation. |
| Skeleton ST-GCN / graph encoder | Performance was clearly worse and it did not provide a useful complementary ensemble, so it was rejected. |
| Skeleton motion TCN (pose velocity and acceleration) | It does not always beat the base model alone, but it showed that motion and raw pose provide complementary information. This led to the later residual-gating design. |
| Visual V2: 192 px, letterbox, 12 frames, shared sampling phase for three visual modalities, directional pooling `mean + max + last-first` | Old three-fold single-model OOF was `0.46014`, below synced-flip at `0.46838`; however, its old three-fold blend with the baseline reached `0.49539`. A later regularized fold-2 screen was only `0.44626`, below the threshold, so the remaining folds were not trained. |
| Larger Visual V3 / larger batch attempts | CUDA memory allocation failed before. CV evidence also did not support insufficient parameter count, so the model was not blindly expanded to 100 MB. |
| Sync flip + IMU device dropout | Current strict five-fold baseline: OOF `0.47332`. |
| Raw pose + Motion residual gate | Current best single-model family: five-fold OOF `0.48650`, which is `+0.01318` above the baseline. It keeps the original Skeleton representation and adds velocity/acceleration through a learned gate. |
| Baseline + pose-motion probability blend | A shared `0.25 / 0.75` weight gives pooled OOF `0.50198`; the more conservative cross-fitted OOF is `0.49736`, and every held-out fold is better than the baseline. The current recommended solution is a weighted ensemble of ten checkpoints, about `51.7 MB` in total. |

## Evaluation Features Added

- `cuhkx-cv-report`: checks whether every OOF file covers the correct fold, clips, and labels.
- `cuhkx-ensemble-oof`: searches blend weights and uses cross-fitting to avoid overfitting by selecting weights on the same OOF data.
- `cuhkx-predict`: supports one checkpoint or a probability-weighted ensemble of checkpoints, with 3-view inference.
- `cuhkx-check-submission`: checks CSV paths and prediction format.
- `cuhkx-stress-test`: simulates missing modalities and reduced quality on held-out validation folds, without using test labels.

## Results of Completed Stress Tests

The two current model families show the same clear pattern:

- Thermal missing: about `1.3 pt` lower;
- Radar missing: about `0.9 pt` lower;
- IMU missing: about `4.7–5.6 pt` lower;
- visual temporal degradation: about `3.2–3.5 pt` lower;
- Skeleton missing: about `19.4 pt` lower.

Therefore, the models are not especially fragile when Thermal or Radar is missing. Their largest dependency is Skeleton. Pose-motion residual is better than the baseline on normal data and under every stress setting, but it does not remove the basic dependency on Skeleton.

## Known Issues and Open Directions

- The main difficulties are still cross-user generalization and fine hand/small-object actions: `Write`, `Make a phone call`, `Take medicine`, and `Take and use tableware`.
- The next training direction with the strongest evidence is not increasing parameters, but:
  1. audit the Skeleton coordinate system;
  2. use root-centered and scale normalization;
  3. add small cross-user augmentation such as joint jitter and bone-length scaling;
  4. keep the current Pose + Motion residual;
  5. screen on fold 2 first, then run full five-fold CV and a cross-fitted blend audit.
