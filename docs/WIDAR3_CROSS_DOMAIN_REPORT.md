# Widar3 跨环境泛化实验报告

实验日期：2026-07-15

## 1. 实验目标

测试不使用 LLM 的纯信号模型在完全未见采集环境上的泛化能力：

```text
Channel Attention -> HSTE -> masked attention pooling -> classification head
```

模型参数量为 2,236,232，配置与此前 subject-disjoint 基线一致。

## 2. 协议

原始数据包含四个采集环境，环境与志愿者完全绑定：同一个志愿者不会跨环境出现。因此本实验同时衡量跨环境和跨志愿者泛化，无法仅归因于环境变化。

采用四折 Leave-One-Environment-Out：

- 每折完整留出一个环境作为 test；
- test 环境及其志愿者不进入 train/validation；
- 其余源环境按 `环境 x 类别` 分层抽取 15% 样本作为 validation；
- validation 与 train 来自相同源环境，允许志愿者重叠；
- 六个类别在每折的 train、validation、test 中均完整覆盖；
- 四折 test 合并后恰好覆盖全部 11,371 个样本一次。

| Target environment | Train | Validation | Test | Held-out subjects |
| --- | ---: | ---: | ---: | --- |
| 20181130 | 3,924 | 697 | 6,750 | user5, user10--user17 |
| 20181204 | 9,024 | 1,597 | 750 | user1 |
| 20181209 | 8,919 | 1,578 | 874 | user2, user6 |
| 20181211 | 7,113 | 1,261 | 2,997 | user3, user7, user8, user9 |

## 3. 四折结果

每折都使用源域 validation macro-F1 选择 `best.pt`，之后只在目标域 test 上评估一次。

| Target | Selected step | Source-val accuracy | Target-test accuracy | Source-val macro-F1 | Target-test macro-F1 | Accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20181130 | 660 | 83.50% | 76.50% | 83.47% | 76.46% | -7.00 pp |
| 20181204 | 1,023 | 90.80% | 85.60% | 90.82% | 85.61% | -5.20 pp |
| 20181209 | 1,155 | 90.75% | 95.31% | 90.75% | 95.35% | +4.56 pp |
| 20181211 | 1,518 | 92.31% | 81.85% | 92.31% | 81.84% | -10.46 pp |

汇总：

| Aggregate | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| 四个环境等权平均 | **84.82%** | **84.81%** |
| 全部 11,371 个 out-of-domain 预测合并 | **79.96%** | **79.92%** |

等权平均把每个环境视为同等重要；合并结果按样本数加权，因此样本最多且较难的 20181130 占主导。四折源域 validation macro-F1 平均为 89.34%，目标域 test 平均为 84.81%，平均下降 4.52 个百分点。

## 4. 合并后的每类结果

| Label | Gesture | Precision | Recall | F1 | Support |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | push/pull | 74.98% | 84.13% | 79.29% | 1,998 |
| 1 | sweep | 80.71% | 84.59% | 82.60% | 1,875 |
| 2 | clap | 86.80% | 85.32% | 86.05% | 1,873 |
| 3 | slide | 76.93% | 72.21% | 74.50% | 1,875 |
| 4 | draw circle | 79.75% | 79.84% | 79.80% | 1,875 |
| 5 | draw zigzag | 81.61% | 73.39% | 77.28% | 1,875 |

合并混淆矩阵，行是真实类别，列是预测类别：

| Actual / Predicted | push/pull | sweep | clap | slide | circle | zigzag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| push/pull | 1681 | 16 | 70 | 118 | 58 | 55 |
| sweep | 46 | 1586 | 17 | 58 | 90 | 78 |
| clap | 87 | 4 | 1598 | 98 | 45 | 41 |
| slide | 167 | 88 | 102 | 1354 | 92 | 72 |
| circle | 146 | 68 | 31 | 69 | 1497 | 64 |
| zigzag | 115 | 203 | 23 | 63 | 95 | 1376 |

跨域下最弱类别是 slide（F1 74.50%）和 zigzag（F1 77.28%）。主要混淆包括 zigzag -> sweep、slide -> push/pull、circle -> push/pull。

## 5. 目标域差异

- `20181209` 最容易，95.31% accuracy；其中 user6 为 96.26%，user2 为 89.60%。
- `20181204` 的单一目标志愿者 user1 为 85.60%。
- `20181211` 为 81.85%，四名目标志愿者在 77.54%--84.93% 之间。
- `20181130` 为 76.50%，但内部差异很大：user17 仅 43.07%，其他多数志愿者约 74%--86%。

这说明当前模型不是在所有新域上稳定下降，而是对特定志愿者/采集分布非常敏感。由于环境与志愿者混杂，不能断言 user17 的下降完全由环境造成。

## 6. 与 subject-disjoint 基线的关系

此前固定 subject-disjoint test 为 84.06% accuracy / 84.15% macro-F1。此次跨环境四折等权平均约 84.82%，但按全部样本合并只有 79.96%。两者协议不同，不能直接把 0.76 个百分点的等权差异解释为提升。

更可靠的结论是：

1. 纯 Channel Attention + HSTE 在部分未见环境上可以保持 85%--95%；
2. 面对 20181130 和 20181211 时存在 7--10 个百分点的源域到目标域下降；
3. 当前跨域总体效果受目标人群组成影响明显，仍需要域归一化、增强或显式域泛化方法。

## 7. 工程验证与产物

- 远程测试：10/10 通过；
- 四折 manifest 均为零缺失路径、六类完整；
- 跨域数据冒烟验证了 train/eval/save、step 4 -> 6 恢复和独立 test；
- 四折正式训练均完成，无 Traceback、OOM、NCCL 或 emergency checkpoint；
- 总训练时间约 7 分 50 秒，GPU 6 观察显存约 1.1 GB；
- 四折输出共约 4.3 GB，已由 `.gitignore` 排除。

远程产物：

```text
/home/wj/LWCL/data/splits/widar3_cross_environment/
/home/wj/LWCL/outputs/widar3_cross_environment/
/home/wj/LWCL/logs/widar3_cross_environment.log
```
