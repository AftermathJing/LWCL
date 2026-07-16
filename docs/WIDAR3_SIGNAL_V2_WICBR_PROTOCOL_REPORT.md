# Widar3 Signal-v2 与 Wi-CBR 协议实验汇总

日期：2026-07-16

## 1. 实验目的

本实验将 Signal-v2 放到当前 Wi-CBR 复现所使用的 Widar3 三折 cross-room 协议上，比较两种方法在相同目标环境划分下的表现。

本报告保留原有严格 subject-disjoint 多随机种子实验，不覆盖或替代原主结果。这里的数字只用于 Wi-CBR 协议下的补充比较。

## 2. 数据与划分

Signal-v2 使用 P1 质量控制后的 11,370 个有效样本。Wi-CBR baseline 使用 11,371 个样本；两者唯一差异是 Signal-v2 剔除了所有六个接收器均未通过质量控制的样本：

```text
20181130_user17_g4_p1_o2_i5
```

| 协议 | 训练环境 | 目标环境 | 训练样本 | 目标样本 | 目标受试者 |
| --- | --- | --- | ---: | ---: | --- |
| cr1 | 20181204, 20181209, 20181211 | 20181130 | 4,621 | 6,749 | user5, user10-user17 |
| cr2 | 20181130, 20181211 | 20181204, 20181209 | 9,746 | 1,624 | user1, user2, user6 |
| cr3 | 20181130, 20181204, 20181209 | 20181211 | 8,373 | 2,997 | user3, user7, user8, user9 |

适配后的 Signal-v2 manifest 将目标域同时标记为 `validation` 和 `test`，以复现 Wi-CBR 的目标域选模方式。因此 manifest 中目标样本出现两次，但训练和最终评估分别按 split 读取，不会重复参与梯度更新。

### 协议性质

三个目标环境的受试者均未出现在对应训练环境中。因此当前协议不是纯粹隔离环境因素的 cross-room 实验，而是：

```text
environment shift + subject shift
```

训练过程中没有使用目标域样本更新参数，但目标域标签被用于选择最佳 checkpoint，所以它也不属于严格 source-only validation 下的 zero-shot 评估。

## 3. Signal-v2 实验设置

### 3.1 输入与预处理

| 项目 | 设置 |
| --- | --- |
| 原始接收器数 | 6 |
| 最少有效接收器 | 5 |
| 最大序列长度 | 96 |
| STFT window / hop / NFFT | 251 / 50 / 256 |
| 时间采样间隔 | 50 ms |
| Doppler 范围 | -60 Hz 至 60 Hz |
| Doppler bins | 25 |
| 差分 CSI | 10 个复数分量 |
| 接收器质量特征 | 5 维 |

### 3.2 模型

Signal-v2 共 2,009,186 个参数，全部参与训练。

```text
RSSI / Doppler / Differential CSI 独立 Stem
-> 共享 Receiver Encoder
-> Quality-aware Receiver Fusion
-> Temporal Conv + 一阶差分
-> 单尺度 HSTE
-> Attentive Statistics Pooling
-> Gesture Classifier
```

主要结构配置：

| 模块 | 设置 |
| --- | --- |
| Feature stems | 16 / 64 / 48，融合维度 128 |
| Receiver fusion | 4-head attention + mean/max/std/attention pooling |
| Temporal stem | kernel 3、5，一阶差分，hidden 128 |
| HSTE local | dim 192，window 5，stride 2，1 层，4 heads |
| HSTE global | dim 256，2 层，8 heads |
| 位置建模 | 局部窗口相对 RoPE + 原时间轴中心物理时间 RoPE |
| 输入绝对位置编码 | 关闭 |
| 最终池化 | attentive mean、std、max |
| 激活 | 前端 SiLU，Transformer/分类头 GELU |

### 3.3 训练

| 项目 | Signal-v2 |
| --- | --- |
| Seed | 2025 |
| Precision | bf16 |
| Epochs | 120 |
| Batch | 96 |
| 每 epoch batch 数 | 86 |
| Optimizer | AdamW |
| Backbone / classifier LR | 4e-4 / 8e-4 |
| Weight decay | 0.03 |
| Scheduler | 8% warmup + cosine |
| Loss | CE + 0.1 x cross-subject SupCon |
| Label smoothing | 0.02 |
| Gradient clipping | 1.0 |
| EMA | 关闭 |
| Sampler | 6 gestures x 4 subjects x 4 samples |
| 选模分数 | 0.7 x target macro-F1 + 0.3 x target worst-subject macro-F1 |

训练增强包括幅度缩放、高斯噪声、时间平移、时间遮挡、单接收器 dropout 和 Doppler 频率遮挡。

## 4. Wi-CBR baseline 设置

当前 baseline 复现使用 CSI-ratio phase 与 DFS 两张 224x224 输入图，经两个 ImageNet 预训练 ResNet18 分支、空间门控和 DP fusion 后分类。按当前代码结构统计约 22.36M 参数，约为 Signal-v2 的 11.1 倍。

| 项目 | Wi-CBR baseline |
| --- | --- |
| Seed | 888 |
| Precision | fp32 |
| Epochs | 30 |
| Batch | 10 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Weight decay | 0 |
| Scheduler | StepLR，每 3 epoch 乘 0.5 |
| Loss | CE + 0.1 x proxy contrastive loss |
| Group number / gate threshold | 4 / 0.5 |
| 选模 | 目标域 test accuracy |

两种方法只对齐了数据域划分和使用目标域选模这一总体协议，训练轮数、batch、优化器、损失、输入表示和选模指标并未控制一致。因此本实验是同协议下的方法比较，不是严格单变量架构消融。

## 5. 三折结果

### 5.1 Accuracy

| 协议 | Signal-v2 | Wi-CBR | Signal-v2 差值 |
| --- | ---: | ---: | ---: |
| cr1 | **85.86%** | 75.33% | **+10.53 pp** |
| cr2 | **95.07%** | 93.84% | **+1.23 pp** |
| cr3 | **88.42%** | 87.62% | **+0.80 pp** |
| 三折等权均值 | **89.79%** | 85.60% | **+4.19 pp** |
| 三折离散度 | 3.88% | 7.69% | -3.81 pp |

### 5.2 Macro-F1

| 协议 | Signal-v2 | Wi-CBR | Signal-v2 差值 |
| --- | ---: | ---: | ---: |
| cr1 | **85.83%** | 75.43% | **+10.39 pp** |
| cr2 | **95.13%** | 93.78% | **+1.35 pp** |
| cr3 | **88.37%** | 87.66% | **+0.70 pp** |
| 三折等权均值 | **89.78%** | 85.63% | **+4.15 pp** |
| 三折离散度 | 3.93% | 7.63% | -3.70 pp |

这里的“离散度”是三个环境折之间的总体标准差，不是多随机种子标准差。

### 5.3 合并目标样本结果

将三折混淆矩阵合并后：

| 方法 | 样本数 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Signal-v2 | 11,370 | **87.85%** | **87.85%** |
| Wi-CBR | 11,371 | 81.22% | 81.28% |
| 差值 | - | **+6.64 pp** | **+6.57 pp** |

该合并结果会给予样本量最大的 cr1 更高权重；三折等权均值和合并结果回答的是不同问题，因此应同时报告。

### 5.4 合并后的每类 F1

| Gesture label | Signal-v2 | Wi-CBR | 差值 |
| ---: | ---: | ---: | ---: |
| 0 | **85.86%** | 80.78% | **+5.09 pp** |
| 1 | **89.90%** | 79.94% | **+9.96 pp** |
| 2 | **90.98%** | 87.78% | **+3.21 pp** |
| 3 | **81.61%** | 79.93% | **+1.68 pp** |
| 4 | **87.95%** | 80.69% | **+7.26 pp** |
| 5 | **90.80%** | 78.55% | **+12.25 pp** |

Signal-v2 六类均高于当前 Wi-CBR baseline，但 label 3 仍是两种方法共同的最难类别。

## 6. 最佳 checkpoint

| 协议 | 最佳 epoch | Global step | Signal-v2 Macro-F1 |
| --- | ---: | ---: | ---: |
| cr1 | 57 | 4,988 | 85.83% |
| cr2 | 96 | 8,342 | 95.13% |
| cr3 | 82 | 7,138 | 88.37% |

三折都训练至 120 epoch，正式评估使用各折 `best.pt`，不是最后一个在线模型。

## 7. 结果解释

1. Signal-v2 在三个目标折上都超过当前 Wi-CBR baseline，三折等权 Macro-F1 平均提升 4.15 个百分点。
2. 最大增益来自 cr1，即 20181130。该折中 Signal-v2 将 user17 accuracy 从 51.07% 提升至 61.01%，将 user5 从 67.20% 提升至 82.53%，说明它主要改善了当前最困难的大目标域。
3. cr2 和 cr3 的提升较小，且并非每个受试者都改善。当前证据支持整体协议平均提升，但不能声称对所有目标受试者一致占优。
4. Signal-v2 的跨折波动低于 Wi-CBR，但当前只有每种方法一个 seed。跨折标准差不能替代随机种子稳定性实验。
5. cr2 的 user2 只包含一个 gesture label。其六类 subject Macro-F1 约 15.83% 没有实际可比意义，应使用 90.40% accuracy 描述。该不完整受试者也会压低 Signal-v2 的 worst-subject 选模分数，因此后续应让 worst-subject Macro-F1 只在该受试者实际存在的类别上计算。

## 8. 可以支持与不能支持的结论

当前结果可以支持：

- Signal-v2 在当前 Wi-CBR 数据划分与目标域选模协议下优于已复现的 Wi-CBR baseline；
- 提升主要来自困难的 20181130 目标域；
- 约 2.01M 参数的 Signal-v2 在该协议下达到比约 22.36M 参数双 ResNet18 baseline 更高的平均性能；
- Signal-v2 在联合环境和受试者偏移下具有较好的迁移能力。

当前结果不能直接支持：

- 严格 zero-shot cross-room；
- 纯环境不变性，因为环境与受试者身份混杂；
- 统计显著的随机种子稳定性；
- 在完全一致训练 recipe 下，结构本身独立贡献 4.15 个百分点；
- Signal-v2 对每个受试者和每个环境都稳定优于 Wi-CBR。

## 9. 复现资产与限制

Signal-v2 远端结果：

```text
/home/wj/LWCL-v2.0-staging/outputs/widar3_signal_v2_wicbr_protocol_seed2025/
```

Wi-CBR baseline 远端结果：

```text
/home/wj/LWCL-official-protocol/outputs/widar3_wicbr_official/
```

Signal-v2 运行元数据记录的基础提交为 `6c2ce3c1dbcc84f68fb376fc0d0275b7a84dae03`，运行时 Wi-CBR 适配文件位于远端 dirty worktree。对应适配器主体已在本地提交 `1058a79`，但启动脚本仍有一个未提交修正。正式封存结果前应提交该修正，并记录最终 clean commit；这不影响当前 checkpoint 和指标的有效性，但影响一键复现的版本封存完整性。

