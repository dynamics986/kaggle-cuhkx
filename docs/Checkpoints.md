# Released checkpoints

This directory contains the compact model files referenced by the technical report. Each file is intended for Git LFS; SHA-256 values are in [SHA256SUMS](SHA256SUMS).

## Detector weights

| File | Purpose | 
| --- | ---  | 
| `yolo_fold2_yolov8n.pt` |Fold-2 visual validation and the released Dual ResNet-18 checkpoint. It excludes Fold-2 validation users. | 
| `yolo_fold4_yolov8n.pt` | Fold-4 visual validation. It excludes Fold-4 validation users. | 
| `yolo_full_yolov8n.pt` |  Final full-data crops for the full LightGBM model. It is never used to score a validation fold. | 

All three are YOLOv8n person detectors fine-tuned for 60 epochs from the
COCO-pretrained base model using only official training frames. Detector
provenance is retained beside the source artifacts.

## LightGBM + YOLO-box features

The released pair is `lightgbm_yolo_box_baseline_full.txt` plus
`yolo_full_yolov8n.pt`. The expected detector-inclusive file size is below
100,000,000 bytes. It uses the Fold-2/Fold-4 selected-fold protocol, with
weighted validation accuracy **0.51949**. Its configuration is
`configs/sep14/m01_yolo_box_grid.json`.

## Dual ResNet-18 vision model

| File | Validation | Required detector |
| --- | --- | --- |
| `dual_resnet18_yolo_crops_fold2.pt` | Fold 2: 0.44259 | `yolo_fold2_yolov8n.pt` |

This is the selected visual-only checkpoint: independent IR and Depth ResNet-18 encoders with moderate clip-consistent augmentation. Its paired detector-inclusive size is 96,475,055 bytes, within 100MB. The Fold-4 score (0.35938) is reported for screening only; the released model is the stronger selected Fold-2 checkpoint. Its configuration is `configs/sep13/m05_dual_resnet18.json`.

## Five-fold fusion ensemble

The 0.50+ fusion result is a probability ensemble, not an individual model. Use the five `fusion_baseline_fold*.pt` checkpoints with weight `0.05` each and the five `fusion_pose_motion_residual_fold*.pt` checkpoints with weight=`0.15` each. 

The baseline has five-fold OOF 0.47332 and the pose-motion residual model has five-fold OOF 0.48650. Their pooled OOF is 0.50198; the cross-fitted OOF, which is the conservative estimate to report, is **0.49736**. The combined checkpoint size is 54,213,984 bytes.

| File |  Config | Weight | 
| --- | --- |  --- |
| `fusion_baseline_fold0.pt` | `configs/synced_flip_imu_dropout.json` | 0.05 |
| `fusion_baseline_fold1.pt` |  same | 0.05 |
| `fusion_baseline_fold2.pt` |same | 0.05 | 
| `fusion_baseline_fold3.pt` | same | 0.05 | 
| `fusion_baseline_fold4.pt` | same | 0.05 | 
| `fusion_pose_motion_residual_fold0.pt` | `configs/pose_motion_residual.json` | 0.15 |
| `fusion_pose_motion_residual_fold1.pt` | same | 0.15 | 
| `fusion_pose_motion_residual_fold2.pt` | same | 0.15 | 
| `fusion_pose_motion_residual_fold3.pt` | same | 0.15 | 
| `fusion_pose_motion_residual_fold4.pt` | same | 0.15 | 

Generate the ensemble prediction with:

```powershell
uv run cuhkx-predict `
  --checkpoints checkpoints\fusion_baseline_fold0.pt checkpoints\fusion_baseline_fold1.pt checkpoints\fusion_baseline_fold2.pt checkpoints\fusion_baseline_fold3.pt checkpoints\fusion_baseline_fold4.pt checkpoints\fusion_pose_motion_residual_fold0.pt checkpoints\fusion_pose_motion_residual_fold1.pt checkpoints\fusion_pose_motion_residual_fold2.pt checkpoints\fusion_pose_motion_residual_fold3.pt checkpoints\fusion_pose_motion_residual_fold4.pt `
  --weights 0.05 0.05 0.05 0.05 0.05 0.15 0.15 0.15 0.15 0.15 `
  --manifest manifests\cv5\test.csv `
  --data-root ..\Small-Model-Track\Testing\data\small_model_track_test `
  --cache-dir cache-64 --views 3 `
  --output artifacts\submission_pose_motion_blend.csv
```

The model architecture, results, and validation protocol are described in
[docs/Architecture.md](../docs/Architecture.md).
