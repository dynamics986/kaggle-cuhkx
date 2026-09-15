# July 20 Summary

## Visual V2 Completed Results

What I did:

```text
Visual V2
+ letterbox
+ 192 x 192
+ 12 frames
+ synchronized relative temporal sampling
+ mean/max/last-first temporal pooling
```

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

### Analysis

The new visual settings improve some categories that rely on full-frame info, small objects, or action direction, but overfit more easily on fold 1.
Therefore I treat it as a second model family with a different error pattern.
The unified ensemble improves across all three folds, which is more reliable than looking at the best result from a single fold.

### Next Recommended Item

The next low-risk experiment should be to strengthen regularization for Visual V2: training accuracy reaches about 90% in later epochs, while cross-user validation remains noticeably lower, so capacity is not the primary bottleneck. Prioritize `testing higher dropout/weight decay`, or implement the "raw pose + Motion residual gating" only on fold 2 first.