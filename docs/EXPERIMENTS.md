# CUHK-X HAR Experiment Ledger

Protocol: frozen subject-held-out CV5, `manifests/cv5/train.csv`, seed `20260719`. Primary metric: pooled OOF clip top-1. `F2` = fold-2 screen; not an OOF result. Historical 3-fold / fold-0 results are not comparable with CV5.

## Accepted / current reference

| Model | Config | Evaluation | Result | Decision |
|---|---|---:|---:|---|
| Synced flip + IMU slot dropout | [config](../configs/synced_flip_imu_dropout.json) | CV5 OOF | 0.47332 | Baseline |
| Pose + motion residual Skeleton | [config](../configs/pose_motion_residual.json) | CV5 OOF | **0.48650** (+0.01318) | Keep |
| Baseline/residual probability blend (0.25/0.75) | Above two configs | CV5 pooled / cross-fitted OOF | **0.50198 / 0.49736** | Current local-CV ensemble; 51.7 MB |
| Same blend, public LB | Above two configs | Public LB | 0.41791 | Below prior 0.42786; do not tune on LB |

CV artifacts: [baseline report](../artifacts/cv5_synced_flip_imu_dropout/cv_report.json), [residual report](../artifacts/cv5_pose_motion_residual/cv_report.json).

## Six-modality model screens

| Change | Config | Evaluation | Result | Decision |
|---|---|---:|---:|---|
| Skeleton ST-GCN | [config](../configs/graph.json) | Historical fold 0 | 0.41152 | Reject |
| Skeleton motion TCN | [config](../configs/motion.json) | Historical fold 0 | 0.51132 | Superseded by residual |
| Visual V2: 192px, 12 frames, directional pooling | [config](../configs/visual_v2.json) | Historical 3-fold OOF | 0.46014 vs synced flip 0.46838 | Reject as replacement |
| Visual V2 regularized | [config](../configs/visual_v2_regularized.json) | F2 | 0.44626 vs baseline 0.48333 | Reject |
| YOLO person crop, all 3 visual streams | [config](../configs/visual_yolo_crop_probe.json) | F2 | 0.51296 vs residual 0.52963 (-0.01667) | Reject; do not run CV5 |
| IMU synchronized device-CNN + relative Transformer | [config](../configs/pose_motion_residual_imu_candidate_probe.json) | F2 | Pending; gate >0.52963, preferred +0.01 | In progress |

## True single-modality baselines

| Modality | Config | CV5 OOF | Correct / available clips | Result |
|---|---|---:|---:|---|
| Skeleton | [config](../configs/modality_base.json) | **0.48520** | — / 2,931 | Strongest independent modality |
| IMU, legacy TCN | [config](../configs/modality_base.json) | 0.27593 | 790 / 2,863 | Legacy reference |
| Depth_Color | [config](../configs/modality_base.json) | 0.23920 | — / 2,931 | Weak alone |
| Thermal | [config](../configs/modality_base.json) | 0.22550 | — / 2,891 | Weak alone |
| IR | [config](../configs/modality_base.json) | 0.19810 | — / 2,933 | Weak alone |
| Radar, legacy statistics TCN | [config](../configs/modality_base.json) | 0.18524 | 261 / 1,409 | Retained Radar path |

Reports: [legacy IMU](../artifacts/modality/IMU/report.json), [legacy Radar](../artifacts/modality/Radar/report.json).

## Visual YOLO crop: useful alone, harmful in current fusion

| Single visual model | Config | F2 no crop | F2 YOLO crop | Delta |
|---|---|---:|---:|---:|
| Depth_Color | [config](../configs/modality_visual_yolo.json) | 0.25612 | **0.40490** | +0.14878 |
| IR | [config](../configs/modality_visual_yolo.json) | 0.18985 | **0.32331** | +0.13346 |
| Thermal | [config](../configs/modality_visual_yolo.json) | 0.23150 | **0.28083** | +0.04934 |

Decision: retain detector/crop artifacts for a future confidence-gated visual fusion experiment; do not apply three-stream crop to the current fusion model.

## IMU / Radar temporal encoders

| Modality | Encoder / preprocessing | Config | CV5 OOF | Delta | Decision |
|---|---|---|---:|---:|---|
| IMU | CNN + BiGRU | Removed config; artifact [report](../artifacts/modality_sequence_v2/IMU/report.json) | 0.24659 | -0.02934 vs legacy TCN | Reject |
| IMU | Regularized CNN + BiGRU rescue | Removed config; artifact [report](../artifacts/modality_sequence_rescue/IMU/report.json) | 0.24869 | -0.02724 vs legacy TCN | Reject |
| IMU | Synchronized TCN | [config](../configs/modality_imu_synced_tcn.json) | 0.25009 | — | New-preprocessing control |
| IMU | Device-aware CNN + relative Transformer | [config](../configs/modality_imu_device_cnn_rel_transformer.json) | **0.33461** | **+0.08453** vs synchronized TCN; all 5 folds up | Keep; fusion F2 gate |
| Radar | CNN + GRU | Removed config; artifact [report](../artifacts/modality_sequence_v2/Radar/report.json) | 0.14904 | -0.03620 vs legacy TCN | Reject |
| Radar | Regularized CNN + GRU rescue | Removed config; artifact [report](../artifacts/modality_sequence_rescue/Radar/report.json) | 0.17175 | -0.01348 vs legacy TCN | Reject |
| Radar | Statistics TCN on point-preserving cache | [config](../configs/modality_radar_statistics_tcn.json) | 0.17033 | — | New-preprocessing control |
| Radar | Raw point cloud PointNet + relative Transformer | Removed after evaluation; artifact retained locally | 0.16395 | -0.00639 vs synchronized TCN | Reject; use legacy Radar TCN in fusion |

Current CV5 reports: [synchronized baselines](../artifacts/modality_sequence_v2/baseline); rejected candidate artifacts are retained locally only.

## Active gates

| Candidate | Fixed baseline | Required result | Next action |
|---|---:|---:|---|
| Six-modality IMU candidate, F2 | Pose-motion residual 0.52963 | >0.52963; preferred >=0.53963 | Pass: CV5. Fail: preserve current ensemble. |
| Radar PointNet (removed) | Radar legacy TCN 0.18524 | Not met | Closed |
| Three-stream YOLO crop fusion | Pose-motion residual 0.52963 | Not met | Closed |
