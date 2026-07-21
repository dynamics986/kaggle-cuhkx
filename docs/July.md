# CUHK-X HAR：三折结果与后续改进

## 当前结论

Synchronized-flip 的三个 subject-disjoint folds 已训练完成：

| Fold | 验证用户 | 最佳 epoch | 停止 epoch | 验证准确率 |
| --- | --- | ---: | ---: | ---: |
| 0 | user2, user7, user16, user20, user21, user23 | 24 | 32 | 0.50823 |
| 1 | user1, user4, user6, user9, user18, user22 | 37 | 38 | 0.45798 |
| 2 | user3, user5, user8, user17, user19, user24 | 11 | 19 | 0.44080 |

三折共有 3,036 个 held-out 训练 clip，正确 1,422 个：

```text
OOF micro accuracy = 0.46838
三折 accuracy 平均 = 0.46900
```

fold 0 明显比 fold 1/2 容易。后续不能再以 fold 0 单独决定模型，必须报告三折 OOF
或三折平均值。不要为了提高分数而重新划分更容易的验证用户。

## 困难用户

| 用户 | Fold | Clip 数 | Accuracy |
| --- | ---: | ---: | ---: |
| user5 | 2 | 160 | 0.33125 |
| user4 | 1 | 143 | 0.36364 |
| user8 | 2 | 169 | 0.37278 |
| user3 | 2 | 163 | 0.38650 |
| user1 | 1 | 153 | 0.43137 |

这说明主要风险是跨主体泛化，不是模型容量不足。训练准确率较高而新用户准确率较低时，继续加宽
模型通常只会加剧过拟合。

## 困难动作

| 动作 | Clip 数 | 涉及用户数 | OOF Accuracy |
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

主要混淆包括：

```text
Take/use tableware -> Pour drinks
Eat food <-> Drink water
Stir drinks -> Eat food / Pour drinks
Turn pages -> Read documents
Sit down -> Stand up
```

前四类错误需要更强的小物体和精细手部视觉特征；Sit down/Stand up 需要更明确的时间方向。

## 实验解释限制

Synchronized-flip 使用修复后的 IMU 缓存；早期 Base、Clean Base、Balanced、Graph 和 Motion
checkpoint 使用旧缓存。因此它们不是完全严格的单变量对照。当前没有发现
`artifacts/base_fixed_cache`，需要在修复后的缓存上训练 no-flip Base，才能隔离同步翻转的真实
贡献。

旧 IMU 解析器会在设备缺失时压紧剩余设备，导致设备身份错位。现在固定槽位顺序为：

```text
WTC, WTLA, WTLL, WTRA, WTRL
```

在 2,863 个有 IMU 的训练 clip 中，78 个缺少至少一个设备。当前 `cache-64` 已按固定槽位重建。

## 第一优先级：Visual V2

原始视觉分辨率通常是：

```text
Depth_Color: 640 x 480
IR:          640 x 480
Thermal:     320 x 240
```

当前输入为 128 x 128、每模态 8 帧，并使用 `ImageOps.fit` 将 4:3 图像中心裁成正方形。
这可能裁掉左右区域，也容易丢失手机、药物、书页和餐具等小物体。

Visual V2 计划：

1. 用 letterbox/padding 保留完整 4:3 画面，不做中心裁剪。
2. 图像分辨率提高到 192 x 192。
3. 每个视觉模态从 8 帧增加到 12 帧。
4. Depth_Color、IR、Thermal 使用相同的相对时间采样位置，减少跨模态时间错位。
5. 保持训练/验证用户严格分离，所有标准化统计只来自当前 fold 的训练用户。

实测 Visual V2 峰值显存约 2.19 GB，RTX 5060 Laptop 8 GB 可以稳定运行。模型文件大小
基本不受输入分辨率影响；三个 fold 实测分别耗时 76.1、53.7、46.2 分钟。

## 第二优先级：保留动作时间方向

当前 `TemporalEncoder` 最后对时间维直接求平均，容易弱化“坐下”和“站起”的先后区别。
建议把时间汇聚改为：

```text
mean feature
+ max feature
+ last feature - first feature
-> lightweight projection
```

`last - first` 显式保留动作方向，预计能帮助：

```text
Sit down <-> Stand up
Squat <-> Sit down
Pick up <-> Put down 类动作
```

## 第三优先级：原始姿态 + Motion 残差分支

Motion-TCN 单模没有超过 Base，但与 Base 的错误具有互补性。下一版不应完全替换原始骨架分支，
而应使用：

```text
原始姿态 TCN -----------+
                        +-> 可学习门控/残差融合
速度与加速度 TCN -------+
```

门控初始偏向原始姿态，让模型只在验证证据支持时增加 motion 特征权重。

## 类别平衡策略

`class_balance_power=0.5` 已经降低总体 clip accuracy，暂时不要继续使用强平衡采样。
Visual V2 完成后如果少数类仍然很差，可以测试：

```json
"class_balance_power": 0.25
```

Watch TV 只有 12 个 clip、来自 3 个用户。采样权重不能创造新的主体多样性，因此类别平衡不是
当前最优先改进。

## 新实验的评估顺序

1. 先在最困难的 fold 2 筛选新配置。
2. fold 2 相对当前 0.44080 明确提升至少 0.02-0.03，再训练 fold 1。
3. fold 1 也提高，再训练 fold 0。
4. 最终以三折 OOF/平均值决定是否保留，不挑选最好看的单一 fold。
5. 同一阶段只改变一组相关因素，并在 `EXPERIMENTS.md` 记录配置、结果和否决原因。

当前建议的下一次实验是：

```text
Visual V2
+ letterbox
+ 192 x 192
+ 12 frames
+ synchronized relative temporal sampling
+ mean/max/last-first temporal pooling
```

先运行 fold 2，预计约 1-2 小时，不属于 8-12 小时通宵交叉验证。

## Visual V2 已完成结果（2026-07-20）

本轮已经实现并验证：

1. 视觉输入改为 192 x 192 letterbox，保留完整 4:3 画面。
2. 每个视觉模态从 8 帧增加到 12 帧。
3. Depth_Color、IR、Thermal 在一个样本内共享相对时间采样位置。
4. 时序汇聚使用 `mean + max + last - first`，再投影回原特征维度。
5. 训练时仍使用同步多模态水平翻转，并保持固定 IMU 设备槽位。

模型大小为 6.26 MB、参数量 1,630,488，远低于 100 MB 限制。单模型结果为：

| Fold | Synced Flip | Visual V2 | Visual V2 best epoch |
| --- | ---: | ---: | ---: |
| 0 | 0.50823 | 0.50103 | 23 |
| 1 | 0.45798 | 0.42871 | 9 |
| 2 | 0.44080 | 0.45373 | 11 |
| OOF | 0.46838 | 0.46014 | — |

Visual V2 单独没有超过 Synced Flip，但错误具有稳定互补性。使用所有三折共同选择的统一权重：

```text
Synced Flip: 0.575
Visual V2:   0.425
```

得到：

| Fold | Synced Flip | 统一权重融合 | 提升 |
| --- | ---: | ---: | ---: |
| 0 | 0.50823 | 0.53395 | +0.02572 |
| 1 | 0.45798 | 0.48347 | +0.02550 |
| 2 | 0.44080 | 0.47065 | +0.02985 |
| OOF | 0.46838 | **0.49539** | **+0.02701** |

总计从 1,422/3,036 个正确提升到 1,504/3,036 个正确，多 82 个 clip。候选权重
0.40、0.425、0.45、0.475 的 OOF 都在 0.4937-0.4954，说明收益不是只存在于一个尖锐权重点。
选择 0.425 是三折 pooled OOF 的统一结果，没有为每个 fold 分别调权，也没有使用测试标签。

类别层面最明显的改善是 `Sit down`：0.6190 提高到 0.8299，多识别对 31 个样本。这与
`last - first` 保留动作方向的设计目标一致。`Do jumping jacks`、`Wipe bowls`、`Stir drinks`、
`Check the time` 也有提升。当前仍需重点解决 `Make a phone call`、`Write`、`Take medicine`、
`Take and use tableware` 等小物体或细粒度手部动作；融合后这些类别没有改善或略有下降。

复现权重扫描：

```powershell
uv run cuhkx-ensemble-oof `
  --baseline artifacts\synced_flip\fold_0\validation_predictions.csv artifacts\synced_flip\fold_1\validation_predictions.csv artifacts\synced_flip\fold_2\validation_predictions.csv `
  --candidate artifacts\visual_v2\fold_0\validation_predictions.csv artifacts\visual_v2\fold_1\validation_predictions.csv artifacts\visual_v2\fold_2\validation_predictions.csv `
  --steps 40 --output artifacts\visual_v2\oof_ensemble_scan.json
```

### 为什么保留而不是替代旧模型

新视觉设置改善了部分依赖完整画面、小物体或动作方向的类别，但在 fold 1 上更容易过拟合。
因此当前正确用法是把它作为具有不同错误模式的第二模型家族，而不是直接替换 Synced Flip。
统一融合在三个 fold 上全部提升，比只看某一个 fold 的最好结果可靠。

### 下一项建议

下一项低风险实验应是对 Visual V2 加强正则化，而不是继续增加参数：训练准确率后期达到约
90%，跨用户验证仍明显较低，容量不是主要瓶颈。优先只在 fold 2 测试较高 dropout/weight decay，
或实现“原始姿态 + Motion 残差门控”；新实验必须继续与当前 0.49539 OOF 融合基线比较。

## 最终提交训练策略

交叉验证用于选择架构，不能把 fold 0 的最好分数当作测试集预估。架构确定后，应增加“全部训练
用户”模式：

1. 使用全部 18 个有标签训练用户。
2. 不再使用测试集或测试统计量。
3. 根据三个 folds 的最佳 epoch 决定固定训练轮数，而不是查看测试表现。
4. 用多个随机种子训练全数据模型。
5. 对全数据模型和经过验证的 fold 模型做概率集成。
6. 模型总大小必须保持在 100 MB 以内。

这可以让正式模型利用全部标注用户，同时保持无数据泄漏。最终测试推理、CSV 校验和 Kaggle 上传
由参赛者本人执行。
