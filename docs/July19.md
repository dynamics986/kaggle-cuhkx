# July 19 Training and Results

The synchronized-flip experiment uses `configs\synced_flip.json`. Fold 0 completed at `0.50823` validation accuracy. Each fold trains for at most 40 epochs and stops after 8 consecutive epochs without a validation-accuracy improvement. Always use `best.pt` for inference.

## Fold 1

Training Terminal:

```powershell
uv run cuhkx-train `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --manifest manifests\cv3\train.csv `
  --config configs\synced_flip.json ` # --config path/to/config/file
  --cache-dir cache-64 `
  --output-dir artifacts\synced_flip `
  --fold 1
```

Monitoring Terminal:

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\synced_flip\fold_1 `
  --patience 8
```

If fold 1 is interrupted, resume it with the original training command plus:

```powershell
--resume artifacts\synced_flip\fold_1\last.pt # --resume path/to/last.pt
```

## Fold 2

Training Terminal:

```powershell
uv run cuhkx-train `
  --data-root ..\Small-Model-Track\Training\extracted\HAR\data `
  --manifest manifests\cv3\train.csv `
  --config configs\synced_flip.json `
  --cache-dir cache-64 `
  --output-dir artifacts\synced_flip `
  --fold 2
```

Monitoring Terminal:

```powershell
uv run cuhkx-monitor `
  --run-dir artifacts\synced_flip\fold_2 `
  --patience 8
```

## Results

The three subject-disjoint folds of synchronized-flip have been trained:

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

## Lessons

Some of the actions are less common so I ccame up with an idea to use balanced sampling, which is to have less common actions sampled more frequently during training, and less common actions sampled relatively less. Therefore, I set `class_balance_power=0.5`.

But the experimental results showed that after using this balanced sampling, the overall clip accuracy on the validation set decreased. Thus I would rather use ordinary random sampling.
