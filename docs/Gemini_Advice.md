# Next Step Advice

建议接下来优先花 1 天时间把 Clean Control Group 跑出来，修正 Baseline，然后沿着“强化骨骼特征 + 尝试 3D 视频 Backbone”的路线迭代！

## 一、 问题与潜在风险

#### 1. 变量污染（Historical Technical Debt）

* **问题：** 你的核心 Baseline（Base multimodal TCN）使用的是**修正前**的旧 IMU Cache，且带有“仅视觉翻转（Visual-only Flip）”这一破环模态对齐的 Bug；而后续的试验（如 Synchronized flip）使用的是**修正后**的 Cache。
* **风险：** Base 模型的 `0.52675` 高分可能包含了旧 Cache/错位增强带来的随机噪声偏置。你目前的对比（如 Sync flip vs Clean Base）基线不统一，**无法确定某些增益到底来自代码修复还是来自模型本身**。

#### 2. 消融实验的“致命信号”被忽略了

* EXPERIMENTS.md中提到：`Drop-one ablation: Skeleton (-0.21605)`。
* **风险：** 骨骼（Skeleton）模态掉点极其严重（准确率直接崩掉 21%），说明**骨骼是这个任务的绝对核心支撑**。但你的单体骨骼模型（Skeleton ST-GCN）只有 `0.41152`，且被总结为“Reject; not ensemble-complementary”。
* **诊断：** 说明你的多模态模型把骨骼特征用得很好，但**单独的 Graph/ST-GCN 架构可能严重欠拟合，或者跨主体（Cross-Subject）泛化能力极差**，目前没有把骨骼的潜能发挥出来。

#### 3. 模态对齐问题（Modality Fusion Risk）

* 旧代码中“仅翻转图像，未翻转 Skeleton 和 IMU”说明你的数据流水线在**空间对齐**上曾出现偏差。需要检查：图像左翻转时，IMU 的加速度计/陀螺仪 Y/Z 轴是否有进行对应的符号反转（Sign Inversion）？如果没做，物理上依然是对不上的。



## 二、 针对性的改进建议

### 1. 立即清理技术债（P0 级优先）

在做任何新的大改动前，先跑一次 **“Clean Control Group”**：

* 在**修正后的 IMU Cache** 上，跑一个**取消错误翻转**的 Base 重训练（Clean Base Retrain）。
* 以此作为唯一的 **True Baseline**，重新刷新你的单模型与 Ensemble 评估矩阵。

### 2. 拯救骨骼模态（P1 级优先）

既然消融实验证明 Skeleton 权重最高，那么提升骨骼模态的跨主体泛化就是性价比最高的操作：

* **改进数据增强：** 骨骼数据极易过拟合特定 Subject 的体型。加入 `Random Bone Length Scaling`（随机骨骼长度缩放）、`Random Joint Jitter`（关节微小平移）和 `Global 3D Rotation`（空间随机旋转）。
* **架构替换：** ST-GCN 比较老且容易过拟合，尝试更强且更轻量的 **PoseConv1D / PoseConv2D**（把 Keypoints 变成 Heatmap 或 1D 骨骼时序卷积），或者 **2s-AGCN / CTR-GCN**。

### 3. IMU 缺失值的鲁棒性处理（Data Pipeline）

* har-solution\docs\EXPERIMENTS.md 显示有 78 个 Clip 缺失部分 IMU 设备。
* **建议：** 训练时引入 **Device Dropout（设备随机丢弃）**。在 10%-20% 的 Batch 中随机把某个 IMU 设备的通道 Mask 置零。这样可以强迫模型不依赖特定部位的传感器，大幅提升测试集上的鲁棒性。

### 4. 融合策略改进

* 目前你使用的是简单的加权平均（Blend）。
* **建议：** 试试 **Rank Averaging（排名平均）** 或 **Out-Of-Fold Stack（Logistic Regression / Ridge Stacking）**。多模态模型的 Softmax 概率分布往往标定（Calibration）不一致，直接加权容易被预测偏激烈的模型主导。



## 三、 我预测的最强架构（The Winning Architecture）

基于你的日志分析（视频/图像 + IMU + 骨骼 keypoints 的动作识别赛题），能拿金牌的最强架构大概率会是以下形态：

```
                    ┌──────────────────────────────────────────────┐
                    │            Multi-Modal Input Stream          │
                    └──────┬────────────────┬──────────────┬───────┘
                           │                │              │
                   ┌───────▼──────┐  ┌──────▼───────┐  ┌───▼───────────┐
                   │  Video/RGB   │  │ 3D Skeleton  │  │   5-Slot IMU  │
                   └───────┬──────┘  └──────┬───────┘  └───┬───────────┘
                           │                │              │
                           ▼                ▼              ▼
                    [Swin3D / X3D]    [PoseConv2D]    [1D-ResNet/TCN]
                           │                │              │
                           │ (Spatial Temporal Feature Extractor) │
                           └───────┬────────┴──────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Cross-Attention Fusion Block │
                    │ (IMU/Pose Query Video Feats) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    Temporal Pooling & Head   │
                    └──────────────────────────────┘

```

#### 为什么是这个架构？

1. **Backbone 分工：**
* **Visual:** 不要用 2D CNN + Temporal Pooling（你的 Visual V2 只有 `0.46` OOF，太弱了）。切换到专业的 **3D 视频 Backbone**，如 **X3D-M** 或 **Video Swin-Nano/Tiny**，直接提取时空特征。
* **Skeleton:** 弃用普通 GCN，改用 **PoseConv2D**（将骨骼关节点转为 Pseudo-image，用 2D 卷积处理）。这种方法在 Cross-Subject 评估中泛化能力远超 GCN。
* **IMU:** 保持现有的 **1D ResNet / TCN** 结构，但引入 1D Squeeze-and-Excitation (SE) 模块，增强缺失设备时的动态通道注意力调整。


2. **融合机制（Cross-Attention Fusion）：**
* 不要在最后简单地把 Vector concat 起来。
* 使用 **Cross-Attention（交叉注意力）**：用高精度的 Skeleton/IMU 特征作为 Query，去注意力引导（Attend）视频特征。因为 IMU 和骨骼时间分辨率高、噪声低，能帮助视觉特征精准定位到“动作发生的关键帧”。


3. **Cross-Subject 专用 Loss：**
* 引入 **SupCon (Supervised Contrastive Loss)** 或 **ArcFace / CosFace**。在特征层强行拉近“不同 User 做同一动作”的距离，拉远“同一 User 做不同动作”的距离，彻底解决跨主体泛化差的问题。
