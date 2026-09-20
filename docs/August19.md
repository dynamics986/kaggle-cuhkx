
## Single-Modality YOLO Results: Substantial Visual Improvements

| Single-input model, Fold 2 | No crop | YOLO crop | Delta |
|---|---:|---:|---:|
| Depth_Color | 0.25612 | 0.40490 | +0.14878 |
| IR | 0.18985 | 0.32331 | +0.13346 |
| Thermal | 0.23150 | 0.28083 | +0.04934 |

Therefore, the Fold-2 degradation of the six-modality model cannot be attributed to “failed YOLO detection” or “person cropping having no visual value.” The more accurate conclusion is that the current end-to-end fusion does not convert the improved visual evidence into final accuracy.

Among the 518 validation clips where all three visual streams are present, Skeleton is correct on 259 (50.0%) and YOLO-cropped Depth_Color is correct on 210 (40.5%). Both are correct on 165 clips, only Skeleton is correct on 94, only vision is correct on 45, and both are incorrect on 214. Vision contains real complementary information (45 Skeleton errors corrected by vision), but it is still much weaker than Skeleton and its errors overlap substantially with Skeleton errors. IR and Thermal also have 40 and 32 vision-only correct clips, but have 130 and 144 Skeleton-only correct clips, respectively.

The current fusion model learns the visual encoders, sensor encoders, and Transformer end to end from scratch. Synchronized cropping of the three visual streams changes the distribution of every visual token and removes scene and full-body context. The Transformer is not constrained to use visual evidence only when vision is reliable, so incorrect visual tokens can bias samples that Skeleton already classifies correctly. This is consistent with 42 clips where the baseline is correct and YOLO is incorrect, versus 33 reverse gains. Future fusion work should first test explicit visual-confidence or residual gating on Fold 2, or introduce only the most useful cropped Depth_Color stream. It must not crop all three streams again and run CV5 solely because single-modality scores improved.

## Full CV5 only after gate acceptance

Every fold must train an independent detector; the detector for fold `k` may use annotations only from users with `fold != k`. Run each fold in order to avoid mixing metadata. Fold 2 is complete and is not included in the loop.

```powershell
$remainingFolds = @(0, 1, 3, 4)
$yoloHarCv = "artifacts\cv5_visual_yolo"

foreach ($fold in $remainingFolds) {
  uv run cuhkx-yolo-train `
    --annotation-dir "artifacts\yolo\annotations_800" `
    --manifest $manifest `
    --output-dir "$yoloRoot\detectors" `
    --fold $fold `
    --epochs 60 --image-size 640 --batch 16 --device cuda

  uv run cuhkx-yolo-crops `
    --manifest $manifest `
    --data-root $dataRoot `
    --weights "$yoloRoot\detectors\fold_$fold\yolov8n_fold_$fold.pt" `
    --output-dir "$yoloRoot\crops\fold_$fold" `
    --confidence 0.25 --device cuda

  uv run cuhkx-yolo-inspect `
    --manifest $manifest `
    --data-root $dataRoot `
    --metadata "$yoloRoot\crops\fold_$fold\bboxes.json" `
    --output-dir "$yoloRoot\inspection\fold_$fold" `
    --fold $fold --count 100

  # Manually confirm that the ten overlays for this fold show no systematic misalignment before running HAR.
  uv run cuhkx-train `
    --config configs\visual_yolo_crop_probe.json `
    --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
    --output-dir $yoloHarCv --fold $fold --device cuda
}
```

The Fold-2 HAR outputs are in `$yoloHar\fold_2`; copy them to the complete CV
artifact directory before full OOF to avoid retraining:

```powershell
New-Item -ItemType Directory -Force "$yoloHarCv\fold_2" | Out-Null
Copy-Item "$yoloHar\fold_2\*" "$yoloHarCv\fold_2\" -Recurse -Force
```

Then generate the strict OOF report:

```powershell
uv run cuhkx-cv-report `
  --manifest $manifest `
  --predictions `
    "$yoloHarCv\fold_0\validation_predictions.csv" `
    "$yoloHarCv\fold_1\validation_predictions.csv" `
    "$yoloHarCv\fold_2\validation_predictions.csv" `
    "$yoloHarCv\fold_3\validation_predictions.csv" `
    "$yoloHarCv\fold_4\validation_predictions.csv" `
  --name cv5_visual_yolo `
  --output "$yoloHarCv\cv_report.json"
```

## Final detector and test metadata only after full CV5 acceptance

The final detector may use all 800 **training** annotations, but it may only be
used for final train/test preprocessing. It must never be used to backfill CV
metadata or CV scores.

```powershell
uv run cuhkx-yolo-train `
  --annotation-dir "artifacts\yolo\annotations_800" `
  --manifest $manifest `
  --output-dir "$yoloRoot\detectors" `
  --final `
  --epochs 60 --image-size 640 --batch 16 --device cuda

uv run cuhkx-yolo-crops `
  --manifest manifests\cv5\test.csv `
  --data-root "..\Small-Model-Track\Testing\data\small_model_track_test" `
  --weights "$yoloRoot\detectors\final\yolov8n_final.pt" `
  --output-dir "$yoloRoot\crops\final" `
  --confidence 0.25 --device cuda --allow-test
```

Before final inference or submission, audit the **file** size of the detector
plus the selected HAR ensemble:

```powershell
uv run cuhkx-weight-audit `
  --yolo "$yoloRoot\detectors\final\yolov8n_final.pt" `
  --har <path to the final HAR best.pt file used for inference> `
  --output "$yoloRoot\final_weight_audit.json"
```

Generate final test predictions only when `passes: true` (the total is at most
100 MB).
