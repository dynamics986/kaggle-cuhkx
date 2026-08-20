
### 单模态 YOLO 结果：视觉本身显著提升

| 单输入模型，fold 2 | 无裁剪 | YOLO 裁剪 | 差值 |
|---|---:|---:|---:|
| Depth_Color | 0.25612 | 0.40490 | +0.14878 |
| IR | 0.18985 | 0.32331 | +0.13346 |
| Thermal | 0.23150 | 0.28083 | +0.04934 |

因此 fold-2 六模态下降不能归因于“YOLO 检测失败”或“人体裁剪没有视觉价值”。更准确的
结论是：当前端到端 fusion 没有把已经改善的视觉证据转化为最终准确率。

在三种视觉流都存在的 518 个验证 clips 上，Skeleton 正确 259（50.0%），YOLO
Depth_Color 正确 210（40.5%）；两者同时正确 165，只有 Skeleton 正确 94，只有视觉
正确 45，两者都错 214。视觉有真实的互补信息（45 个 Skeleton 错、视觉对），但仍远弱
于 Skeleton，且视觉错误与 Skeleton 的重叠很大。IR / Thermal 也有 40 / 32 个仅视觉
正确 clip，但分别有 130 / 144 个仅 Skeleton 正确 clip。

当前融合模型是从零端到端学习视觉 encoder、各传感器 encoder 和 Transformer。三路视觉
同步裁剪改变了所有视觉 token 的分布、去掉了场景/完整人体上下文；Transformer 没有被
约束为“仅在视觉可信时采纳视觉证据”，故可能在 Skeleton 已正确的样本上被错误的视觉
token 拉偏。该解释与完整模型的 42 个 baseline 对、YOLO 错，及 33 个反向收益相符。
后续若研究融合，应先在 fold 2 测试显式的视觉可信度/残差门控，或只接入最有益的
Depth_Color 裁剪流；不能仅凭单模态提升就再次同时裁剪三路并运行 CV5。

## 5. 仅在 gate 接受后：完整 CV5

每个 fold 都必须训练独立 detector；fold `k` 的 detector 只可用 `fold != k` 的标注用户。
下面按顺序执行每个 fold，避免混用 metadata。fold 2 已完成，循环不包含它。

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

  # 人工确认该 fold 的 10 张 overlay 没有系统性失配后，再运行本 fold 的 HAR。
  uv run cuhkx-train `
    --config configs\visual_yolo_crop_probe.json `
    --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
    --output-dir $yoloHarCv --fold $fold --device cuda
}
```

fold 2 的 HAR 产物在 `$yoloHar\fold_2`；在完整 OOF 前复制它到完整 CV artifact 目录，
避免重训：

```powershell
New-Item -ItemType Directory -Force "$yoloHarCv\fold_2" | Out-Null
Copy-Item "$yoloHar\fold_2\*" "$yoloHarCv\fold_2\" -Recurse -Force
```

然后生成严格 OOF 报告：

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

## 6. 仅在完整 CV5 决定接受后：final detector 与测试 metadata

最终 detector 可使用全部 800 张**训练**标注，但只能用于最终 train/test 预处理，绝不能
回填任何 CV metadata 或 CV 分数。

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

在最终推理或提交前，审计 detector 加选定 HAR ensemble 的**文件**权重大小：

```powershell
uv run cuhkx-weight-audit `
  --yolo "$yoloRoot\detectors\final\yolov8n_final.pt" `
  --har <填入最终要推理的 HAR best.pt 文件路径> `
  --output "$yoloRoot\final_weight_audit.json"
```

只有 `passes: true`（总和不超过 100 MB）时，才能生成最终测试预测。
