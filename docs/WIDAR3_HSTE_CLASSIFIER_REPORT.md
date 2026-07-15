# Widar3 纯 HSTE 分类基线实验报告

- 实验日期：2026-07-15
- 远程主机：`wj@10.69.216.119`
- Conda 环境：`LWCL`
- GPU：NVIDIA A100-PCIE-40GB（GPU 6）
- 实现提交：`e6035c3`

## 1. 实验目标

本实验暂不使用 Qwen、LLM、LoRA 或 Adapter，只验证论文前端时空编码结构本身的分类能力。实际模型路径为：

```text
Widar3 CSI [T,6,49]
  -> Channel Attention
  -> Hierarchical Spatio-Temporal Encoder (HSTE)
  -> masked attention pooling
  -> MLP classification head
  -> 6 gesture classes
```

所有模块均从头训练，不加载任何语言模型或预训练权重。

## 2. 数据与划分

使用已完成预处理的 Widar3 六接收端动作组数据，共 11,371 个有效样本。输入为变长 `[T,6,49]`，配置最大长度为 64，当前样本通常约 11--13 个时间步。

| Split | Samples | Subjects |
| --- | ---: | --- |
| train | 8,247 | user1, user3, user5, user6, user8, user10, user11, user13, user14, user15, user16 |
| validation | 1,499 | user9, user17 |
| test | 1,625 | user2, user7, user12 |

这是严格的 subject-disjoint 划分。训练集使用类别均衡采样，validation/test 不使用重采样。

## 3. 模型与训练配置

| Component | Configuration |
| --- | --- |
| Channel Attention | output 128, hidden 64, dropout 0.2 |
| HSTE local stage | projection 128, window 4, stride 2, 2 layers, 8 heads |
| HSTE global stage | output 256, 2 layers, 8 heads, FFN factor 4 |
| Classifier | attention pooling, hidden 256, dropout 0.25, label smoothing 0.05 |
| Optimizer | AdamW, encoder LR `3e-4`, classifier LR `1e-3`, weight decay 0.01 |
| Schedule | 66-step warmup + cosine decay |
| Batch | train 256, eval 512, bf16 |
| Stop rule | at most 60 epochs; validation macro-F1 patience 10, min delta 0.001 |

参数量：

| Component | Parameters |
| --- | ---: |
| Channel Attention | 42,689 |
| HSTE | 2,125,440 |
| Classification head | 68,103 |
| **Total/trainable** | **2,236,232** |

## 4. 训练安全验证

正式训练前完成了以下检查：

- 远程 `pytest`：9/9 通过；
- 4-step 前向、反向、验证与检查点保存；
- 从 step 4 的 `last.pt` 恢复并继续到 step 6；
- eval 与 save 在同一步触发；
- 最佳、编号和 last 检查点均可生成；
- 人为在 step 2 抛出异常，成功生成约 2.99 MB 的 emergency checkpoint；
- 恢复后的检查点可执行独立 test evaluation。

正式训练没有出现 Traceback、OOM、RuntimeError、NCCL 或 emergency-checkpoint 事件。

## 5. 最佳验证集结果

最佳检查点位于 global step 1,254，即完成第 38 个 epoch 后：

| Metric | Value |
| --- | ---: |
| accuracy | 67.645% |
| macro precision | 71.440% |
| macro recall | 67.653% |
| macro F1 | **67.036%** |
| loss | 1.3292 |

验证受试者差异很大：user9 为 85.85%，user17 为 49.47%。这表明当前前端的主要问题是跨受试者/跨采集域稳定性，而不是无法拟合训练数据。

训练在 step 1,584（第 48 个 epoch 后）早停，最佳点之后连续 10 次验证没有超过 `0.001` 的有效提升。

## 6. 独立测试集结果

使用 `best.pt`，且不根据测试集选择检查点：

| Metric | Value |
| --- | ---: |
| accuracy | **84.062%** |
| macro precision | 83.926% |
| macro recall | 84.533% |
| macro F1 | **84.152%** |
| loss | 0.6883 |

### 6.1 每类指标

| Label | Gesture | Precision | Recall | F1 | Support |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | push/pull | 86.47% | 78.40% | 82.24% | 375 |
| 1 | sweep | 87.40% | 88.80% | 88.10% | 250 |
| 2 | clap | 85.82% | 94.40% | 89.90% | 250 |
| 3 | slide | 80.82% | 79.20% | 80.00% | 250 |
| 4 | draw circle | 77.57% | 81.60% | 79.53% | 250 |
| 5 | draw zigzag | 85.48% | 84.80% | 85.14% | 250 |

### 6.2 混淆矩阵

行是真实类别，列是预测类别，类别顺序为 `push/pull, sweep, clap, slide, circle, zigzag`。

| Actual \ Predicted | push/pull | sweep | clap | slide | circle | zigzag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| push/pull | 294 | 3 | 14 | 22 | 37 | 5 |
| sweep | 3 | 222 | 2 | 8 | 7 | 8 |
| clap | 7 | 0 | 236 | 5 | 2 | 0 |
| slide | 7 | 9 | 19 | 198 | 6 | 11 |
| circle | 15 | 6 | 4 | 9 | 204 | 12 |
| zigzag | 14 | 14 | 0 | 3 | 7 | 212 |

最明显的单项混淆是 push/pull 被识别为 circle（37/375），其次是 slide 被识别为 clap（19/250）。

### 6.3 分组准确率

| Group | Accuracy |
| --- | ---: |
| subject user12 | 87.47% |
| subject user7 | 83.60% |
| subject user2 | 66.40% |
| position 1--5 | 76.62%, 81.54%, 85.85%, 85.54%, 90.77% |
| orientation 1--5 | 81.23%, 84.00%, 85.85%, 86.46%, 82.77% |

测试集总体分数高于验证集，是因为两个集合包含不同的未见受试者；validation 中 user17 明显更难。这不能解释为测试集参与了模型选择。

## 7. 训练耗时与资源

- 正式训练启动：2026-07-15 16:00:36 +08:00；
- 早停完成：2026-07-15 16:03:30 +08:00；
- 从启动到完成约 2 分 55 秒，首条至末条指标日志跨度 148.6 秒；
- 训练期间抽样观察显存约 0.97 GB，未出现显存压力；
- `best.pt` 和 `last.pt` 各约 26.97 MB；
- 因每个 epoch 保留一个编号检查点，整个未清理输出目录约 1.3 GB，且已由 `.gitignore` 排除。

## 8. 与论文结果的关系

论文记录的 Widar3.0 结果为 `89.5% +/- 1.4%`，本实验 accuracy 为 84.06%，表面差距为 5.44 个百分点。但二者不能作为严格同协议对比：

- 论文主实验使用较窄的数据范围和随机动作组子集；
- 本实验覆盖 11,371 个动作组并使用严格 subject-disjoint 测试；
- 本实验是 Channel Attention + HSTE 的纯信号消融，而论文完整 LWCL 还包括 Adapter、Qwen 和 LoRA；
- 论文完整 LWCL 的 `93.17% +/- 1.93%` 与本实验相差 9.11 个百分点，同样不是同协议消融结果。

因此，本次结果应解释为：**纯时空编码结构已经能在未见受试者上达到约 84% 的可靠六分类性能，但跨受试者方差仍大；它为后续判断 Adapter/LLM 是否带来真实增益提供了必要基线。** 要测量 LLM 的净增益，必须在同一 manifest、同一预处理和同一 subject-disjoint 划分上训练完整模型。

## 9. 复现命令与远程产物

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/remote_hste_classifier_smoke.sh

CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n LWCL \
  python -m lwcl.cli.train \
  --config configs/widar3_hste_classifier.yaml \
  --output-dir outputs/widar3_hste_classifier

CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n LWCL \
  python -m lwcl.cli.evaluate \
  --config configs/widar3_hste_classifier.yaml \
  --checkpoint outputs/widar3_hste_classifier/checkpoints/best.pt \
  --output-dir outputs/widar3_hste_classifier_test \
  --split test
```

远程产物：

- `/home/wj/LWCL/outputs/widar3_hste_classifier/`
- `/home/wj/LWCL/outputs/widar3_hste_classifier_test/test_metrics.json`
- `/home/wj/LWCL/logs/widar3_hste_classifier.log`

数据、日志、模型权重和检查点均不进入 Git 仓库。
