# IMU / Radar 时序模型第二轮实验

本轮只训练真正的单模态模型，不修改六模态融合网络。所有训练继续使用冻结的
`manifests\cv5\train.csv`。旧的 `cache-64` 和历史 artifacts 均保留；新实验必须使用
`cache-64-synced-points`，否则 candidate 会明确报错。

## 1. 实验内容

- IMU baseline：修正时间同步后的 80 维输入 + 原 TCN。
- IMU candidate：五设备共享 CNN + device embedding/gated pooling + 两层相对位置 Transformer。
- Radar baseline：同一新缓存中的 13 维逐帧统计量 + 原 TCN。

IMU 会先按 timestamp 排序、合并重复 timestamp，再把五个设备插值到共同的 64 点时间网格；
设备在自身观测范围以外的位置由 mask 排除。

## 2. PowerShell 变量与 GPU 预检

在 `har-solution` 根目录运行：

```powershell
$manifest = "manifests\cv5\train.csv"
$dataRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$cacheDir = "cache-64-synced-points"
$baselineRoot = "artifacts\modality_sequence_v2\baseline"
$candidateRoot = "artifacts\modality_sequence_v2\candidate"

uv run python -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'cuda': torch.version.cuda})"
```

必须看到 `cuda_available: True`。

## 3. 构建新的同步 / Radar 统计缓存

不要把 `$cacheDir` 改回旧的 `cache-64`，也不要给旧缓存执行 `--overwrite`。

```powershell
uv run cuhkx-cache `
  --manifest $manifest `
  --data-root $dataRoot `
  --output-dir $cacheDir `
  --split train `
  --steps 64 `
  --workers 4
```

检查任意缓存文件的字段和形状：

```powershell
uv run python -c "from pathlib import Path; import numpy as np; p=next(Path(r'cache-64-synced-points').glob('train_*.npz')); z=np.load(p); print(p); print({k:z[k].shape for k in z.files})"
```

输出中必须包含：

```text
imu                 (64, 80)
imu_synced          (64, 5, 16)
imu_device_mask     (64, 5)
radar               (64, 13)
```

## 4. 先做 fold 2 smoke/probe

先确认三条路径均能训练、写 checkpoint 和预测。`--max-clips-per-class 2` 只用于流程检查，
这些输出不能作为准确率结论。

```powershell
uv run cuhkx-modality-train `
  --config configs\modality_imu_synced_tcn.json `
  --modality IMU --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
  --output-dir "artifacts\modality_sequence_v2_smoke\baseline" `
  --fold 2 --device cuda --max-clips-per-class 2

uv run cuhkx-modality-train `
  --config configs\modality_imu_device_cnn_rel_transformer.json `
  --modality IMU --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
  --output-dir "artifacts\modality_sequence_v2_smoke\candidate" `
  --fold 2 --device cuda --max-clips-per-class 2

uv run cuhkx-modality-train `
  --config configs\modality_radar_statistics_tcn.json `
  --modality Radar --manifest $manifest --data-root $dataRoot --cache-dir $cacheDir `
  --output-dir "artifacts\modality_sequence_v2_smoke\baseline" `
  --fold 2 --device cuda --max-clips-per-class 2

```

## 5. 完整 CV5：同步后的 TCN baseline

```powershell
$baselineConfigs = @{
  IMU = "configs\modality_imu_synced_tcn.json"
  Radar = "configs\modality_radar_statistics_tcn.json"
}

foreach ($modality in @("IMU", "Radar")) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-train `
      --config $baselineConfigs[$modality] `
      --modality $modality `
      --manifest $manifest `
      --data-root $dataRoot `
      --cache-dir $cacheDir `
      --output-dir $baselineRoot `
      --fold $fold `
      --device cuda
  }
}
```

IMU baseline 必须重跑，因为修复 timestamp 后输入已经改变；不能直接拿旧 0.27593 当作严格对照。

## 6. 完整 CV5：IMU Transformer candidate

```powershell
$candidateConfigs = @{
  IMU = "configs\modality_imu_device_cnn_rel_transformer.json"
}

foreach ($modality in @("IMU")) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-train `
      --config $candidateConfigs[$modality] `
      --modality $modality `
      --manifest $manifest `
      --data-root $dataRoot `
      --cache-dir $cacheDir `
      --output-dir $candidateRoot `
      --fold $fold `
      --device cuda
  }
}
```

Radar PointNet candidate 已在完整 CV5 后拒绝，相关实现和配置已删除；其历史 artifacts 保留在本机审计。

每折产物位于：

```text
artifacts\modality_sequence_v2\<baseline|candidate>\<IMU|Radar>\fold_<0..4>\
  best.pt
  history.json
  summary.json
  validation_predictions.csv
```

## 7. 绘制每折训练曲线

```powershell
foreach ($modality in @("IMU", "Radar")) {
  foreach ($fold in 0..4) {
    uv run cuhkx-modality-plot `
      --run-dir (Join-Path $baselineRoot "$modality\fold_$fold")
  }
}
foreach ($fold in 0..4) {
  uv run cuhkx-modality-plot `
    --run-dir (Join-Path $candidateRoot "IMU\fold_$fold")
}
  }
}
```

每折生成 `training_curves.png`，可用于判断过拟合、欠拟合和最佳 epoch。

## 8. 生成严格 OOF report

```powershell
foreach ($modality in @("IMU", "Radar")) {
  $predictions = 0..4 | ForEach-Object {
    Join-Path $baselineRoot "$modality\fold_$($_)\validation_predictions.csv"
  }
  uv run cuhkx-modality-report `
    --manifest $manifest `
    --cache-dir $cacheDir `
    --modality $modality `
    --predictions $predictions `
    --output (Join-Path $baselineRoot "$modality\report.json")
}
$imuPredictions = 0..4 | ForEach-Object {
  Join-Path $candidateRoot "IMU\fold_$($_)\validation_predictions.csv"
}
uv run cuhkx-modality-report `
  --manifest $manifest `
  --cache-dir $cacheDir `
  --modality IMU `
  --predictions $imuPredictions `
  --output (Join-Path $candidateRoot "IMU\report.json")
```

## 9. 比较 baseline 与 candidate

```powershell
$baseline = Get-Content (Join-Path $baselineRoot "IMU\report.json") | ConvertFrom-Json
$candidate = Get-Content (Join-Path $candidateRoot "IMU\report.json") | ConvertFrom-Json
[pscustomobject]@{
  modality = "IMU"
  examples = $candidate.examples
  baseline_oof = [math]::Round($baseline.oof_accuracy, 5)
  candidate_oof = [math]::Round($candidate.oof_accuracy, 5)
  improvement = [math]::Round($candidate.oof_accuracy - $baseline.oof_accuracy, 5)
}

$comparison | Format-Table -AutoSize
$comparison | Export-Csv `
  "artifacts\modality_sequence_v2\comparison.csv" `
  -NoTypeInformation -Encoding UTF8
```

还应逐折查看，而不只看 pooled OOF：

```powershell
$baseline = Get-Content (Join-Path $baselineRoot "IMU\report.json") | ConvertFrom-Json
$candidate = Get-Content (Join-Path $candidateRoot "IMU\report.json") | ConvertFrom-Json
0..4 | ForEach-Object {
  [pscustomobject]@{
    modality = "IMU"
    fold = $_
    baseline = $baseline.fold_accuracy."$_"
    candidate = $candidate.fold_accuracy."$_"
    delta = $candidate.fold_accuracy."$_" - $baseline.fold_accuracy."$_"
  }
}
```

只有 pooled OOF 提高、且多数 fold 不下降时，才建议把新 encoder 接入六模态融合模型。

## 10. 单模态结论：保留 IMU candidate，拒绝 Radar PointNet

完整五折 OOF 结果如下：

| Modality | Baseline OOF | Candidate OOF | Delta | 决策 |
|---|---:|---:|---:|---|
| IMU | 0.25009 | 0.33461 | +0.08453 | 保留 device-aware CNN + relative Transformer |
| Radar | 0.17033 | 0.16395 | -0.00639 | 拒绝 PointNet + Transformer |

IMU candidate 在五个 held-out fold 都提高。Radar candidate 只在 fold 4 略有提高，整体下降。
融合实验继续使用原 `cache-64` 中的 Radar 13 维统计 TCN；不会使用 PointNet，也不会使用
`cache-64-synced-points` 中改变过采样策略的 Radar 字段。

## 11. 六模态 IMU candidate：固定 fold-2 gate

该 probe 只替换 IMU：它从 `cache-64-synced-points` 读取同步后的五设备张量；视觉、
Skeleton motion residual、融合 Transformer、旧 Radar TCN 都继续从原 `cache-64` 工作。

```powershell
$manifest = "manifests\cv5\train.csv"
$dataRoot = "..\Small-Model-Track\Training\extracted\HAR\data"
$legacyCache = "cache-64"
$probeRoot = "artifacts\cv5_pose_motion_residual_imu_candidate_probe"

uv run cuhkx-train `
  --config configs\pose_motion_residual_imu_candidate_probe.json `
  --manifest $manifest `
  --data-root $dataRoot `
  --cache-dir $legacyCache `
  --output-dir $probeRoot `
  --fold 2 `
  --device cuda
```

训练中可查看进度：

```powershell
uv run cuhkx-monitor `
  --run-dir "$probeRoot\fold_2" `
  --patience 9
```

训练结束后绘制曲线并比较 fold-2 gate：

```powershell
uv run cuhkx-modality-plot `
  --run-dir "$probeRoot\fold_2" `
  --output "$probeRoot\fold_2\training_curves.png"

$baseline = Get-Content "artifacts\cv5_pose_motion_residual\fold_2\summary.json" | ConvertFrom-Json
$candidate = Get-Content "$probeRoot\fold_2\summary.json" | ConvertFrom-Json

[pscustomobject]@{
  baseline = [math]::Round($baseline.best_valid_accuracy, 5)
  candidate = [math]::Round($candidate.best_valid_accuracy, 5)
  delta = [math]::Round($candidate.best_valid_accuracy - $baseline.best_valid_accuracy, 5)
} | Format-List
```

当前基线为 `0.52963`。candidate 必须超过该值，且建议达到至少 `+0.01`，才启动完整 CV5；
否则保留现有最终集成，不训练其余四折。
