# CUHK-X HAR experiment record

This report separates the two validation protocols used by the project. Scores
between the sections are **not comparable**.

| Protocol | Manifest | Meaning |
| -------- | ------ | ------ |
| Historical 3-fold | `manifests/cv3/{train,test}.csv` | July development split with three subject-held-out folds. |
| Current 5-fold | `manifests/cv5/{train,test}.csv` | Frozen subject-held-out split: 3,036 clips, 18 users and 40 classes. Sep12–Sep14 report Fold 2 and Fold 4 only, so they are selected-fold screens, not full OOF. |

The split definitions are tracked Git inputs. Raw data, checkpoints, caches and
submissions remain ignored under `artifacts/`.

## Historical 3-fold experiments

These experiments used the earlier cache and preprocessing pipeline. They are
kept as design evidence, but cannot be used as a baseline for five-fold work.
The single-model entries are Fold-0 development results unless stated as OOF.

| Experiment | Evaluation | Accuracy | Size | Finding |
| --- | --- | ---: | ---: | --- |
| Base multimodal TCN | Fold 0 | 0.52675 | 4.57 MB | Best early single model. |
| Inverse-sqrt balanced sampler | Fold 0 | 0.50617 | 4.57 MB | Lower standalone result; checked only for ensemble diversity. |
| Skeleton ST-GCN | Fold 0 | 0.41152 | 4.45 MB | Rejected: weak and not complementary. |
| Pose + velocity + acceleration TCN | Fold 0 | 0.51132 | 4.65 MB | Motion was complementary to raw pose. |
| Clean base, visual flip disabled | Fold 0 | 0.47428 | 4.57 MB | Corrected flip, but still used the old IMU cache. |
| Synchronized flip with fixed IMU slots | Fold 0 | 0.50823 | 4.57 MB | Established the corrected augmentation direction. |
| Visual V2 | Fold 0 | 0.50103 | 6.26 MB | Retained for complementary errors. |
| Visual V2 | 3-fold OOF | 0.46014 | 6.26 MB | Below synchronized flip alone at 0.46838. |
| Synchronized flip + Visual V2 | 3-fold OOF | 0.49539 | — | Shared 0.575 / 0.425 blend improved all folds. |

### Historical blend checks

| Blend | Fold-0 accuracy |
| --- | ---: |
| Base | 0.52675 |
| Base + balanced sampler | 0.53704 |
| Base + pose-motion (0.60 / 0.40) | 0.53601 |
| Base + balanced + pose-motion (0.30 / 0.30 / 0.40) | 0.54938 |
| Fine 2.5% grid maximum | 0.55041 |

The fine-grid result added one clip and had tied asymmetric weights, so the
simpler 0.30 / 0.30 / 0.40 blend was retained. Dropping Skeleton caused the
largest Base-model ablation loss (0.21605), then IMU (0.02058) and IR
(0.01440). This made cross-user skeleton robustness the main modelling target.

### Historical limitations

The old IMU parser compacted available devices, which could shift device
identity when one device was missing. The corrected cache fixes the slots to
`WTC`, `WTLA`, `WTLL`, `WTRA`, `WTRL`. Early visual-only horizontal flip was
also inconsistent with Skeleton and IMU. These issues prevent strict causal
comparison between the earliest runs and corrected runs.

## 5-fold protocol

The current split is frozen and subject-held-out. Completed five-fold results
use pooled OOF over all 3,036 clips. Selected-fold results use Fold 2 (540
validation clips) and Fold 4 (640); their weighted score is over only these
1,180 clips, not full five-fold OOF.

### Completed full 5-fold OOF runs

| Experiment | Pooled OOF | Detail | Decision |
| --- | ---: | --- | --- |
| Synchronized flip + 15% IMU-device dropout | 0.47332 (1,437 / 3,036) | Fold accuracies: 0.40952, 0.51509, 0.48333, 0.50718, 0.45625 | Five-fold baseline. |
| Pose + motion residual gate | 0.48650 (1,477 / 3,036) | Best completed single-model family. | Keep. |
| Baseline + residual blend | 0.50198 (1,524 / 3,036) | Common probability weight: 0.25 / 0.75. | Pooled estimate. |
| Same blend with leave-one-fold-out weight selection | 0.49736 (1,510 / 3,036) | Improved every held-out fold. | Conservative preferred estimate. |

The ten-checkpoint baseline/residual inference package is 51.7 MB. Its public
LB score was 0.41791, below an older 0.42786 submission. That small public
signal is not used for model selection because it conflicts with the
cross-fitted local result.

### Rejected or incomplete 5-fold screens

| Experiment | Evaluation | Result | Decision |
| --- | --- | --- | --- |
| Visual V2 regularized, 192 px, batch 24 | Start-up | CUDA allocation failure before the first batch | No quality conclusion. |
| Same Visual V2, batch 16 / workers 6 | Fold 2 | 0.44626 | Rejected against the Fold-2 baseline gate, 0.48333. |
| Sep13 m04 depth-weighted probability sum | Fold 2 and 4 | Failed during output normalization | No score; fix before retesting. |

## Sep12: YOLO-cropped screens

Sep12 introduced fold-safe YOLO person crops for IR, Depth_Color and Thermal.
The Fold-2 detector was used only for Fold 2; Fold 4 and full-data detectors
were trained separately. The early `artifacts/sep12/serial` run was an
incomplete, pre-alignment Fold-4-only pass (m02 0.28594, m03 0.28438, m04
0.28906, m05 0.30781, m06 0.26875, m07 0.31250, m08 0.28125). It is superseded
by the depth-aligned result below and is not used for selection.

| Method | Architecture | Fold 2 | Fold 4 | Weighted | Finding |
| --- | --- | ---: | ---: | ---: | --- |
| m01 | LightGBM: sensors + three-modality crop summaries | 0.53148 | 0.44531 | 0.48475 | Strongest Sep12 result; full-data refit was created. |
| m02 | Three CNN branches, temporal attention, concat head and vote | 0.35185 | 0.30469 | 0.32644 | Best initial visual method. |
| m03 | IR + Depth_Color CNN attention and concat | 0.32037 | 0.28438 | 0.30085 | Thermal removal did not help. |
| m04 | Per-modality attention and probability sum | 0.31667 | 0.30781 | 0.31186 | Below m02. |
| m05 | Dual IR/Depth ResNet-18 with temporal attention | 0.37593 | 0.33125 | 0.35127 | Best initial visual encoder. |
| m06 | Independent CNN encoders and concat control | 0.32037 | 0.27656 | 0.29661 | Temporal attention was useful. |
| m07 | Lightweight visual temporal Transformer | 0.34074 | 0.30781 | 0.32373 | No advantage over m02. |
| m08 | CNN with squeeze-and-excitation | 0.32407 | 0.27656 | 0.29873 | SE alone did not help. |

m05 plus YOLO was 96.50 MB, leaving little capacity under the 100 MB limit.
The evidence favoured better generalization and preprocessing over another large
visual backbone.

## Sep13: depth-led robustness changes

Sep13 reused the leak-checked Sep12 depth-aligned crops and sensor caches. m01
accepted a candidate only when both selected folds did not decline and the
weighted score increased. Visual changes used clip-consistent augmentation plus
method-specific gates or attention.

| Method | Change from Sep12 | Fold 2 | Fold 4 | Weighted | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| m01 | LightGBM narrow regularized search | 0.53889 | 0.46406 | 0.49831 | +0.01356 |
| m02 | Depth-weighted vote + modality/content gate | 0.36852 | 0.35000 | 0.35847 | +0.03203 |
| m03 | Learned IR/Depth fusion gate | 0.34815 | 0.31719 | 0.33136 | +0.03051 |
| m05 | Moderate clip-consistent augmentation | 0.44259 | 0.35938 | 0.39748 | +0.04621 |
| m06 | Temporal attention + modality gate | 0.38148 | 0.32813 | 0.35254 | +0.05593 |
| m07 | Time attention before modality-token Transformer | 0.30000 | 0.25000 | 0.27288 | -0.05085 |
| m08 | SE + spatial attention + modality gate | 0.36481 | 0.31406 | 0.33729 | +0.03856 |
| m04 | Depth-weighted probability sum | Failed | Failed | — | — |

Selected m01 parameters were `num_leaves=32`, `min_data_in_leaf=12`,
`feature_fraction=0.9`, `lambda_l1=0`, `lambda_l2=0.1`, with learning rate
0.035 and 1,000 rounds. Its detector-inclusive deployment size was 78.11 MB.
Sep13 m05 was the best visual screen and established moderate augmentation as
the visual reference.

## Sep14: strong augmentation stress and box features

Sep14 applied shared flip, 15–20% crop/scale, brightness/contrast ±15%, 15%
area erasing at 35%, temporal offset ±1 and speed 0.8–1.2x. This policy was too
strong after YOLO cropping.

| Experiment | Fold 2 | Fold 4 | Weighted | Decision |
| --- | ---: | ---: | ---: | --- |
| m05 Depth_Color-only ResNet-18, strong augmentation | 0.37778 | 0.34219 | 0.35847 | Below Sep13 dual m05; retain IR. |
| m05 dual ResNet-18, IR dropout 35%, strong augmentation | 0.38704 | 0.25156 | 0.31356 | Fold-4 collapse; dropout too strong. |
| m05 dual ResNet-18, reliability gate, strong augmentation | 0.33519 | 0.30469 | 0.31864 | Underfit; reject. |
| m01 Sep13 feature set | 0.53889 | 0.46406 | 0.49831 | Reference. |
| m01 + YOLO-box temporal features, box baseline | 0.55926 | 0.48594 | **0.51949** | Best measured selected-fold result. |
| m01 + YOLO-box temporal features, saved grid candidate | 0.55370 | 0.48906 | 0.51864 | Slightly below box baseline. |

The box baseline used the Sep13 LightGBM parameters plus box features. The
grid runner did not include that baseline in its winner comparison, then refit
a marginally worse candidate. The next m01 full refit should use the measured
box-baseline parameters, rather than the saved grid winner. The detector plus
saved-grid LightGBM was 75.49 MB.

Two public r3 thermal-specialist models were fit on all notebook training rows
and wrote submissions. Plain r3 reached train accuracy 0.7357; strong-aug r3
reached 0.4424. Neither has held-out validation, so neither training accuracy
is a generalization ranking. Each inference model is about 19.69 MB.

## Conclusions through Sep14

1. The most reliable completed result is the five-fold pose-motion blend:
   cross-fitted OOF 0.49736.
2. The strongest selected-fold screen is m01 with YOLO-box temporal features:
   0.51949 weighted accuracy. It needs the correct baseline full refit.
3. Sep13 moderate augmentation improved the visual methods; Sep14 strong
   augmentation degraded every m05 variant. Start future visual work from
   Sep13 with only weak spatial and temporal perturbations.
4. The visual-only Transformer and post-crop reliability gate did not improve
   this dataset. Prioritize fold-safe box features, data quality, weak
   augmentation and probability blending over larger visual encoders.
