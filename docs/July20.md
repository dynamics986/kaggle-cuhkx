
# CUHK-X HAR: Three-Fold Results and Next Improvements

## Current Conclusions

The three subject-disjoint folds of Synchronized-flip have been trained:

| Fold | Validation Users | Best Epoch | Stopped Epoch | Validation Accuracy |
| --- | --- | ---: | ---: | ---: |
| 0 | user2, user7, user16, user20, user21, user23 | 24 | 32 | 0.50823 |
| 1 | user1, user4, user6, user9, user18, user22 | 37 | 38 | 0.45798 |
| 2 | user3, user5, user8, user17, user19, user24 | 11 | 19 | 0.44080 |

Across three folds, 3,036 held-out training clips, with 1,422 correct:

```text
OOF micro accuracy = 0.46838
Three-fold accuracy average = 0.46900
```

Fold 0 is noticeably easier than folds 1/2. From now on, fold 0 alone cannot be used to make model decisions; three-fold OOF or three-fold average must be reported. Do not re-partition to create easier validation users solely to boost scores.

## Difficult Users

| User | Fold | Number of Clips | Accuracy |
| --- | ---: | ---: | ---: |
| user5 | 2 | 160 | 0.33125 |
| user4 | 1 | 143 | 0.36364 |
| user8 | 2 | 169 | 0.37278 |
| user3 | 2 | 163 | 0.38650 |
| user1 | 1 | 153 | 0.43137 |

This indicates the primary risk is cross-subject generalization, not insufficient model capacity. When training accuracy is high but accuracy on new users is low, continuing to widen the model usually only exacerbates overfitting.

## Difficult Actions

| Action | Number of Clips | Users Involved | OOF Accuracy |
| --- | ---: | ---: | ---: |
| Play games | 40 | 6 | 0.00000 |
| Watch TV | 12 | 3 | 0.00000 |
| Wipe bowls | 35 | 11 | 0.05714 |
| Turn pages | 59 | 13 | 0.08475 |
| Make a phone call | 43 | 12 | 0.09302 |
| Take medicine | 72 | 17 | 0.12500 |
| Do lunges | 27 | 8 | 0.14815 |
| Write | 39 | 11 | 0.15385 |
| Take body temperature | 57 | 13 | 0.15789 |

Primary confusions include:

```text
Take/use tableware -> Pour drinks
Eat food <-> Drink water
Stir drinks -> Eat food / Pour drinks
Turn pages -> Read documents
Sit down -> Stand up
```

The first four categories require stronger small-object and fine-grained hand visual features; Sit down/Stand up require clearer temporal directionality.

## Experimental Interpretation Limitations

Synchronized-flip uses the repaired IMU cache; earlier Base, Clean Base, Balanced, Graph, and Motion checkpoints use the old cache. Therefore they are not strictly controlled univariate comparisons. Currently there is no `artifacts/base_fixed_cache`, so a no-flip Base needs to be trained on the repaired cache before the true contribution of synchronous flipping can be isolated.

The old IMU parser would compress remaining devices when a device was missing, causing device identity misalignment. The fixed slot order is now:

```text
WTC, WTLA, WTLL, WTRA, WTRL
```

Among 2,863 training clips with IMU data, 78 are missing at least one device. The current `cache-64` has been rebuilt with fixed slots.

## First Priority: Visual V2

Original visual resolutions are typically:

```text
Depth_Color: 640 x 480
IR:          640 x 480
Thermal:     320 x 240
```

Current input is 128 x 128, 8 frames per modality, using `ImageOps.fit` to center-crop the 4:3 image into a square.
This may crop off left/right regions, and tends to lose small objects such as phones, medicine, book pages, and tableware.

Visual V2 plan:

1. Use letterbox/padding to preserve the full 4:3 image without center cropping.
2. Increase image resolution to 192 x 192.
3. Increase each visual modality from 8 frames to 12 frames.
4. Use the same relative temporal sampling positions for Depth_Color, IR, and Thermal to reduce cross-modal temporal misalignment.
5. Keep strict separation of training/validation users; all normalization statistics are computed only from training users of the current fold.

Measured peak GPU memory for Visual V2 is about 2.19 GB, and RTX 5060 Laptop 8 GB can run it stably.
Model file size is essentially unaffected by input resolution; the three folds took 76.1, 53.7, and 46.2 minutes respectively.

## Second Priority: Preserve Action Temporal Direction

The current `TemporalEncoder` directly averages over the temporal dimension at the end, which tends to weaken the sequential distinction between "sit down" and "stand up."
It is recommended to change the temporal aggregation to:

```text
mean feature
+ max feature
+ last feature - first feature
-> lightweight projection
```

`last - first` explicitly preserves action direction and is expected to help with:

```text
Sit down <-> Stand up
Squat <-> Sit down
Pick up <-> Put down and similar actions
```

## Third Priority: Raw Pose + Motion Residual Branch

Motion-TCN alone did not exceed Base, but its errors are complementary to Base's errors. In the next version, the raw skeleton branch should not be completely replaced; instead, use:

```text
Raw Pose TCN -----------+
                        +-> Learnable gating/residual fusion
Velocity & Acceleration TCN -------+
```

Initialize gating to favor the raw pose, so that the model increases motion feature weights only when validation evidence supports it.

## Class Balancing Strategy

`class_balance_power=0.5` has already reduced overall clip accuracy, so do not continue using strong balanced sampling for now.
After Visual V2 is completed, if minority classes remain very poor, test:

```json
"class_balance_power": 0.25
```

Watch TV has only 12 clips from 3 users. Sampling weights cannot create new subject diversity, so class balancing is not the highest-priority improvement right now.

## Evaluation Order for New Experiments

1. First screen new configurations on the hardest fold 2.
2. Once fold 2 shows a clear improvement of at least 0.02-0.03 relative to the current 0.44080, then train fold 1.
3. If fold 1 also improves, then train fold 0.
4. The final decision to keep a configuration is based on three-fold OOF/average, not on the best-looking single fold.
5. Change only one related set of factors per stage, and record the configuration, results, and rejection reasons in `EXPERIMENTS.md`.

The currently recommended next experiment is:

```text
Visual V2
+ letterbox
+ 192 x 192
+ 12 frames
+ synchronized relative temporal sampling
+ mean/max/last-first temporal pooling
```

Run fold 2 first, expected to take about 1-2 hours, not an 8-12 hour overnight cross-validation run.

## Visual V2 Completed Results (2026-07-20)

In this round, the following have been implemented and validated:

1. Visual input changed to 192 x 192 letterbox, preserving the full 4:3 image.
2. Each visual modality increased from 8 frames to 12 frames.
3. Depth_Color, IR, and Thermal share relative temporal sampling positions within a single sample.
4. Temporal aggregation uses `mean + max + last - first`, then projects back to the original feature dimension.
5. Synchronized multi-modal horizontal flipping is still used during training, and fixed IMU device slots are maintained.

Model size is 6.26 MB, with 1,630,488 parameters, well below the 100 MB limit. Single-model results:

| Fold | Synced Flip | Visual V2 | Visual V2 best epoch |
| --- | ---: | ---: | ---: |
| 0 | 0.50823 | 0.50103 | 23 |
| 1 | 0.45798 | 0.42871 | 9 |
| 2 | 0.44080 | 0.45373 | 11 |
| OOF | 0.46838 | 0.46014 | — |

Visual V2 alone does not exceed Synced Flip, but its errors have stable complementarity. Using unified weights selected jointly across all three folds:

```text
Synced Flip: 0.575
Visual V2:   0.425
```

Yields:

| Fold | Synced Flip | Unified-Weight Ensemble | Improvement |
| --- | ---: | ---: | ---: |
| 0 | 0.50823 | 0.53395 | +0.02572 |
| 1 | 0.45798 | 0.48347 | +0.02550 |
| 2 | 0.44080 | 0.47065 | +0.02985 |
| OOF | 0.46838 | **0.49539** | **+0.02701** |

Total improved from 1,422/3,036 correct clips to 1,504/3,036 correct clips, an additional 82 clips. Candidate weights of 0.40, 0.425, 0.45, and 0.475 all give OOF scores in the 0.4937-0.4954 range, indicating the gain is not confined to a single sharp weight point. The choice of 0.425 is the unified result from pooled three-fold OOF, not tuned per fold and not using test labels.

At the class level, the most noticeable improvement is for `Sit down`: from 0.6190 to 0.8299, with 31 more samples correctly recognized. This aligns with the design goal of `last - first` preserving action direction. `Do jumping jacks`, `Wipe bowls`, `Stir drinks`, and `Check the time` also improved. Current critical issues remain for `Make a phone call`, `Write`, `Take medicine`, and `Take and use tableware` — small objects or fine-grained hand actions; these categories did not improve or slightly degraded after fusion.

Reproduce the weight scan:

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\synced_flip\fold_0\validation_predictions.csv artifacts\synced_flip\fold_1\validation_predictions.csv artifacts\synced_flip\fold_2\validation_predictions.csv `
  --candidate artifacts\visual_v2\fold_0\validation_predictions.csv artifacts\visual_v2\fold_1\validation_predictions.csv artifacts\visual_v2\fold_2\validation_predictions.csv `
  --steps 40 --output artifacts\visual_v2\oof_ensemble_scan.json
```

### Why Keep Rather Than Replace the Old Model

The new visual settings improve some categories that rely on full-frame information, small objects, or action direction, but overfit more easily on fold 1.
Therefore the current correct usage is to treat it as a second model family with a different error pattern, rather than directly replacing Synced Flip.
The unified ensemble improves across all three folds, which is more reliable than looking at the best result from a single fold.

### Next Recommended Item

The next low-risk experiment should be to strengthen regularization for Visual V2, rather than continuing to increase parameters: training accuracy reaches about 90% in later epochs, while cross-user validation remains noticeably lower, so capacity is not the primary bottleneck. Prioritize testing higher dropout/weight decay, or implement the "raw pose + Motion residual gating" only on fold 2 first; new experiments must continue to be compared against the current 0.49539 OOF ensemble baseline.

## Final Submission Training Strategy

Cross-validation is used for architecture selection; fold 0's best score cannot be used as a test-set estimate. Once the architecture is determined, enable a "full training users" mode:

1. Use all 18 labeled training users.
2. Do not use the test set or test statistics any further.
3. Determine the fixed number of training epochs based on the best epochs from the three folds, without looking at test performance.
4. Train full-data models with multiple random seeds.
5. Ensemble the full-data models and the validated fold models at the probability level.
6. Total model size must remain within 100 MB.

This allows the final model to leverage all labeled users while avoiding data leakage. Final test inference, CSV validation, and Kaggle submission are to be performed by the participant themselves.