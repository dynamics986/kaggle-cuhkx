
# 单模态实验与 GPU 验证

本实验以冻结的 `manifests/cv5/train.csv` 为唯一训练 manifest。`cuhkx_modality`
会校验其中完整的 `clip_id → fold` 映射，不能替换为旧三折或重新划分的文件。

六个模态均训练五折，因此总共运行 30 个独立模型：

- `Depth_Color`、`IR`、`Thermal`：各自独立的视觉时序模型；
- `Skeleton`、`IMU`、`Radar`：各自独立的传感器时序模型。

一个 clip 若缺少当前目标模态，会从该模态的训练和验证中排除。各模态的有效样本数会记录在其 `summary.json` 和最终的按动作对比表中。

## GPU 预检

先在 Windows PowerShell 运行以下命令。必须看到 `cuda_available: True`，再开始训练：

```powershell
uv run python -c "import torch; print({'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'total_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None})"
```

训练命令明确指定 `--device cuda`。若 CUDA 不可用，训练会立即报错，不会静默退回 CPU。CUDA 训练会将模型和 batch 放到 GPU、启用 `pin_memory`，并在 `amp: true` 时启用自动混合精度。每折启动时会打印实际 device、GPU 名称和 AMP 状态；同样信息会写入 `summary.json`。

## 运行六个模态的五折训练

从 `har-solution` 根目录运行。首次建议把 `configs/modality_base.json` 换成 `configs/modality_smoke.json`，只验证一折流程；正式实验再使用 base 配置。

```powershell
$modalities = @("Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar")
$manifest = "manifests/cv5/train.csv"
$dataRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$cacheDir = "cache-64"
$outputDir = "artifacts\modality"

foreach ($modality in $modalities) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-train `
      --config configs/modality_base.json `
      --modality $modality `
      --manifest $manifest `
      --data-root $dataRoot `
      --cache-dir $cacheDir `
      --output-dir $outputDir `
      --fold $fold `
      --device cuda
  }
}
```

每个训练任务会写入：

```text
artifacts/modality/<modality>/fold_<0..4>/
  best.pt
  history.json
  summary.json
  validation_predictions.csv
```

## 每个模态的 OOF 报告

五折完成后，使用对应的五个验证预测生成一个严格校验的 OOF 文件。该命令会拒绝缺折、重复 clip、跨折预测、标签不一致，以及目标模态实际缺失的 clip。

```powershell
foreach ($modality in $modalities) {
  $predictions = 0..4 | ForEach-Object {
    Join-Path $outputDir "$modality\fold_$($_)\validation_predictions.csv"
  }

  uv run cuhkx-modality-report `
    --manifest $manifest `
    --cache-dir $cacheDir `
    --modality $modality `
    --predictions $predictions `
    --output (Join-Path $outputDir "$modality\report.json")
}
```

每个模态会产生 `artifacts/modality/<modality>/report.json` 和 `report.oof.csv`。

## 按动作比较六个模态

六个 OOF 文件都生成后，运行：

```powershell
$oof = $modalities | ForEach-Object {
  Join-Path $outputDir "$_\report.oof.csv"
}

uv run cuhkx-modality-summary `
  --oof $oof `
  --output (Join-Path $outputDir "per_action_modality_accuracy.csv")
```

输出 `artifacts/modality/per_action_modality_accuracy.csv` 每行对应一个动作，包含六个模态各自的准确率和有效样本数，以及 `best_modality`；准确率并列时会用 `|` 同时列出所有最佳模态。

## 训练曲线图

每个 fold 的 `history.json` 会在每个 epoch 结束后更新。训练完成后（或训练中已有至少一个完整 epoch 时），可生成 loss 和 accuracy 曲线：

```powershell
uv run cuhkx-modality-plot `
  --run-dir artifacts\modality\IMU\fold_0
```

默认生成 `artifacts/modality/IMU/fold_0/training_curves.png`。图中左侧为训练/验证 loss，右侧为训练/验证 accuracy；右图会标注最高验证 accuracy 及其 epoch。

指定输出位置：

```powershell
uv run cuhkx-modality-plot `
  --run-dir artifacts\modality\IMU\fold_0 `
  --output artifacts\modality\IMU\fold_0\curves_final.png
```

## Historical record

Earlier three-fold experiments, rejected architectures, and their scores are
intentionally not presented as the current solution.  They are retained in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for reproducibility and audit.

---

## YOLOv8n 人体裁剪视觉预处理（CV gate）

官网 Small Model Track 要求 CNN/RNN/Transformer 模型总大小不超过 100 MB，且
**No large pretrained backbones**。因此本项目允许可审计的轻量预训练
`yolov8n.pt`，但禁止大型预训练 backbone；最终 YOLO detector 与 HAR 推理
ensemble 的权重总和仍必须不超过 100 MB。YOLO 只用于视觉预处理，不参与动作
分类头。

### 已记录的单模态结果

冻结 CV5 下的 pooled subject-held-out OOF accuracy：Skeleton 0.4852，IMU
0.2759，Depth_Color 0.2392，Thermal 0.2255，IR 0.1981，Radar 0.1852
（Radar 有效预测 1,409）。这说明先在不改变 pose-motion residual 架构和训练
预算的前提下降低视觉背景干扰，是一个可检验的下一步。

### 800 张人工标注协议

以下命令只从 `manifests/cv5/train.csv` 的 `Depth_Color` 训练帧中，以固定 seed
`20260719` 确定性抽取 800 张。抽样先轮转覆盖 action × user，再覆盖 clip 的
15%、50%、85% 时间位置。它生成不可变的 `annotation_manifest.csv`、图片和空的
标签目录；不要改动图片或 manifest。

```powershell
$manifest = "manifests/cv5/train.csv"
$dataRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$annotationDir = "artifacts\yolo\annotations_800"

uv sync
uv run cuhkx-yolo-annotations `
  --manifest $manifest `
  --data-root $dataRoot `
  --output-dir $annotationDir `
  --count 800
```

用 LabelImg 打开 `$annotationDir\images`，设置唯一类别 `person`，选择 **YOLO**
格式，并将 txt 保存到 `$annotationDir\labels`。每张图片至多一个 person bbox；确认
画面中没有人时，不画框即可（没有对应 txt 或空 txt 都是有效的 YOLO 负样本）。绝不能
打开、查看或标注 `Testing` 中的任何图片。完成后先审计；审计失败时先修正
标签，不得开始微调：

```powershell
uv run cuhkx-yolo-audit-labels --annotation-dir $annotationDir
```

审计会检查图片与 manifest 的一一对应、类别恒为 0 (`person`)、bbox 为有限的归一化
值且没有实质性越界，并记录正/负样本数与 manifest SHA-256。边界处 LabelImg 的微小
十进制舍入误差会被接受；明显越界仍会被拒绝。
LabelImg 自动生成的 `labels/classes.txt` 是类别元数据，会被审计忽略；不要将它当作
某张图片的标签。

### Cross-validation detector 与裁剪 metadata

每个 fold 只能用该 fold **训练用户** 的人工标注帧微调 detector；held-out 用户的
帧只可用于 detector 推理、overlay 人工检查和 HAR 验证。`detector_summary.json`
会记录 YOLO 来源 (`COCO pretrained yolov8n.pt`)、Ultralytics 版本、许可证提示、
权重大小、参数、manifest hash、训练用户与 held-out 用户。

```powershell
$yoloRoot = "artifacts\yolo"

# 先只跑 fold 2。训练完成会得到 artifacts\yolo\detectors\fold_2\yolov8n_fold_2.pt
uv run cuhkx-yolo-train `
  --annotation-dir $annotationDir `
  --manifest $manifest `
  --output-dir "$yoloRoot\detectors" `
  --fold 2 `
  --epochs 60 --image-size 640 --batch 16 --device cuda
