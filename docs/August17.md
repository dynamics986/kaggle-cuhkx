
## 数据与训练流程改进

- 修复了 IMU cache 的设备槽位错误：原先缺失某个设备时，其他设备会被压缩并错位；现在固定为 `WTC, WTLA, WTLL, WTRA, WTRL` 五个槽位。
- 修复了不一致的数据增强：早期只翻转视觉、未同步翻转 Skeleton / IMU / Radar，物理语义不一致。后来改为同步翻转：
  - Skeleton 镜像并交换左右关节；
  - IMU 左右设备交换；
  - Radar 的横向坐标符号翻转。
- 加入训练期 IMU device dropout：随机屏蔽一个 IMU 设备槽位，降低对单个穿戴设备的依赖。
- 传感器归一化严格在每个 fold 的训练用户上计算，避免验证集泄漏。
- 从旧的三折协议升级为冻结的五折 subject-held-out CV；后续实验统一使用 `manifests/cv5/train.csv` 和固定 seed `20260719`。
- 增加训练监控、断点恢复一致性检查、OOF 完整性审计与提交 CSV 校验。

## 尝试过的模型/方法

| 方法 | 结果与结论 |
|---|---|
| 基础多模态模型：视觉 CNN + 各传感器 TCN + Transformer 模态融合 | 历史 fold-0 表现较强，但早期 cache/增强条件与后续不完全一致，不能和新五折直接比较。 |
| Inverse-sqrt class balancing | 整体准确率下降；没有继续作为主线。原因是少数类别缺少跨用户多样性，重采样不能创造新用户变化。 |
| Skeleton ST-GCN / 图结构编码器 | 表现明显较差，也没有形成有价值的互补集成，已拒绝。 |
| Skeleton motion TCN（姿态速度、加速度） | 单独不一定超过基础模型，但证明运动信息与原始姿态存在互补性，成为后续残差门控设计的依据。 |
| Visual V2：192px、letterbox、12 帧、三视觉模态共享采样相位、方向池化 `mean + max + last-first` | 单模型旧三折 OOF `0.46014`，低于同期 synced-flip `0.46838`；但旧三折中与 baseline 融合曾到 `0.49539`。后续加强正则化的 fold-2 screen 仅 `0.44626`，未达到门槛，因此没有训练剩余 folds。 |
| 更大 Visual V3 / 扩大 batch 的尝试 | 曾遇 CUDA 内存分配失败；同时 CV 证据并不支持“参数不够”，因此没有把模型盲目扩到 100 MB。 |
| Sync flip + IMU device dropout | 当前严格五折 baseline：OOF `0.47332`。 |
| 原始姿态 + Motion 残差门控 | 当前最佳单模型家族：五折 OOF `0.48650`，比 baseline 高 `+0.01318`。它保留原始 Skeleton 表征，再以可学习门控补充速度/加速度信息。 |
| baseline + pose-motion 概率融合 | 统一权重 `0.25 / 0.75` 的 pooled OOF `0.50198`；更保守的 cross-fitted OOF `0.49736`，且每个 held-out fold 都优于 baseline。当前推荐方案为十个 checkpoint 的加权集成，总体约 `51.7 MB`。 |

## 已补齐的评估能力

- `cuhkx-cv-report`：检查每个 OOF 文件是否覆盖正确 fold、clip 和标签。
- `cuhkx-ensemble-oof`：扫描融合权重，并用 cross-fitted 方式避免直接在同一 OOF 上挑权重造成过拟合。
- `cuhkx-predict`：支持单 checkpoint 或多个 checkpoint 的概率加权、3-view 推理。
- `cuhkx-check-submission`：检查 CSV 路径与预测格式。
- `cuhkx-stress-test`：在 held-out validation folds 上模拟模态缺失与质量退化，不使用 test 标签。

## 已完成的压力验证结论

两类当前模型的共同结论很清楚：

- Thermal 缺失：约下降 `1.3 pt`；
- Radar 缺失：约下降 `0.9 pt`；
- IMU 缺失：下降约 `4.7–5.6 pt`；
- 视觉时间退化：下降约 `3.2–3.5 pt`；
- Skeleton 缺失：下降约 `19.4 pt`。

因此，模型并不特别脆弱于 Thermal/Radar 缺失；最大依赖是 Skeleton。Pose-motion residual 在正常数据和所有压力项下都优于 baseline，但没有消除对 Skeleton 的根本依赖。

## 已知问题与未完成方向

- Public LB：旧提交 `0.42786`，Pose-motion 集成提交 `0.41791`。这与本地 CV 增益不一致，但尚不足以用 Public LB 反向调融合权重。
- 困难仍集中在跨用户泛化，以及细粒度手部/小物体动作：`Write`、`Make a phone call`、`Take medicine`、`Take and use tableware` 等。
- 下一条最有证据支持的训练主线不是扩大参数，而是：
  1. Skeleton 坐标系审计；
  2. root-centered / 尺度归一化；
  3. 小幅 joint jitter、骨长缩放等跨用户增强；
  4. 保留当前 Pose + Motion residual；
  5. 先 fold-2 筛选，再完整五折与 cross-fitted blend 审计。
- 在新训练前，还应让本地验证严格复现最终提交：同样的 3-view TTA、十模型权重与压力条件。当前主 OOF 与最终提交推理设置尚有这层差异。

