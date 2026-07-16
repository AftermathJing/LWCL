# Signal-v2 受控调优最终报告

日期：2026-07-16  
实验代码提交：`8e408a6`（实验执行时）  
正式结论：**保留原 Signal-v2 / T00，不用 ST01 或 HP07 替换正式基线。**

## 1. 结论摘要

本轮在不引入 LLM、图像化 ResNet 或新依赖的前提下，完成了 19 个单 seed 候选、两个冻结候选测试、T00/ST01 各 5 个随机种子，以及 T00/HP07/ST01 的严格 Wi-CBR 三折复核。

主要证据如下：

| 协议 | T00 | ST01 | ST01 - T00 |
| --- | ---: | ---: | ---: |
| 五 seed validation Macro-F1 | 76.69% ± 0.93% | 75.59% ± 1.75% | -1.09 pp |
| 五 seed user17 validation Macro-F1 | 59.71% ± 2.45% | 56.20% ± 3.84% | -3.51 pp |
| 五 seed test Accuracy | 87.62% ± 1.20% | 88.18% ± 1.55% | +0.57 pp |
| 五 seed test Macro-F1 | 87.34% ± 1.28% | 87.96% ± 1.60% | +0.62 pp |
| 严格 Wi-CBR 三折 Macro-F1 | 88.95% ± 4.68% | 88.26% ± 4.90% | -0.68 pp |

ST01 的五 seed test 均值略高，但没有形成稳定证据：五个配对 seed 中有三个未优于 T00；validation 连续三个后续 seed 下降；user17 只有 seed2025 改善，其余四个 seed 全部下降；严格 Wi-CBR 的 `cr1` 又下降 1.69 pp，超过预设的单折最大下降 1 pp。因此 ST01 不满足“稳定改善”的定型标准。

HP07 的五 seed validation 确实稳定高于 T00，但冻结测试从 89.32% 降到 86.77%；严格 Wi-CBR 三折仅比 T00 高 0.26 pp，未达到 0.5 pp 目标。它也不应替换正式基线。

## 2. 实验边界与选择规则

- 输入仍为 `[B,T,N,F]`，保留 `time_mask`、`receiver_mask`、质量控制和真实时间轴。
- 模型始终是轻量纯信号网络；最大候选 ST04 为 2,669,539 参数，没有候选超过 4M。
- EMA 全部关闭，checkpoint 只按源域 validation Macro-F1 选择。
- 单 seed 筛选固定 seed2025，最多 20 个，实际运行 19 个。
- 只有 validation 排名前两名 HP07、ST01 被允许做冻结 subject-disjoint test。
- 测试结果没有用于恢复搜索或反向修改配置。
- 跨域复核的 validation 只来自 official train，目标域 test 从未用于早停或 checkpoint 选择。

冻结数据标识：

```text
P1 manifest sha256:   6ffecd8c8836120270b6829655149eaca184b3b19782aa4ff5e099347f16697f
subject split sha256: a060defe45f038c20c5f6408b69899e9c2783f7338b733d3785aa5fb309005f0
train/val/test:       8,247 / 1,498 / 1,625
validation subjects: user9, user17
test subjects:       user2, user7, user12
```

## 3. 新输入表示与数据契约

三个新 processed root 均完成全量审计：11,370 个有效样本、0 个重复、0 个读取失败、0 个 NaN/Inf；时间步为 min 11、median 31、p95 41、max 70；73 个样本有 5 个有效接收器，11,297 个样本有 6 个有效接收器。它们与 P1 使用同一受试者划分和标签定义。

| 输入 | 关键张量 | contract SHA256 |
| --- | --- | --- |
| current | RSSI `[T,6,4]`；DFS `[T,6,25]`；差分 CSI `[T,6,10,2]` | 使用冻结 P1 manifest |
| phase_dfs / current_plus_phase | CSI-ratio phase `[T,6,30,2]`；pair `[6,2]`；DFS25 | `8c6c61d53792c0cd86140363d97c8bd62cff0b787d00b037de712cfcc411bc40` |
| current + DFS61 | DFS `[T,6,61]` | `f2d8b09a74d8fc4ecf81d9a640ddb848ac96300c676b6b043034b579118db68e` |
| current + DFS121 | DFS `[T,6,121]` | `fde522852e8e85dea6e8894050388769c4fd22fda2cd8e62aa0b1283fbd78d02` |

CSI-ratio phase 直接以 `sin/cos` 时序张量保存，没有转换为 RGB 图像。每个接收器内部按可验证的幅度均值/方差比选择 numerator 与 denominator 流，避免跨样本或目标域统计。

输入实验均为负结果：

| ID | 输入 | 参数量 | Validation Macro-F1 | user17 |
| --- | --- | ---: | ---: | ---: |
| IN01 | CSI-ratio phase + DFS25 | 2,008,562 | 61.30% | 40.55% |
| IN02 | current + CSI-ratio phase | 2,020,970 | 71.79% | 53.36% |
| IN03 | current + DFS61 | 2,013,794 | 72.64% | 55.03% |
| IN04 | current + DFS121 | 2,021,474 | 72.05% | 51.95% |

结论：当前 CSI-ratio phase 构造不能替代差分 CSI；在当前数据与 Stem 下，提高 DFS 频率分辨率只增加显存和冗余，没有带来泛化收益。IN03/IN04 峰值训练显存分别约 1.53 GiB/2.91 GiB，明显高于 current 输入约 0.77 GiB。

## 4. 19 个单 seed 筛选结果

排序只依据固定 subject-disjoint validation Macro-F1。

| Rank | ID | 改动 | 参数量 | Validation Macro-F1 | user17 |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | HP07 | SupCon 0.10 → 0.05 | 2,009,186 | 79.14% | 66.98% |
| 2 | ST01 | HP07 + HSTE window/stride 3/2 | 2,009,186 | 77.55% | 61.62% |
| 3 | HP06 | 更强 dropout/label smoothing | 2,009,186 | 77.26% | 61.72% |
| 4 | T00 | 受控基线 | 2,009,186 | 77.10% | 57.90% |
| 5 | HP02 | backbone LR 6e-4 | 2,009,186 | 76.78% | 59.87% |
| 6 | HP10 | SupCon temperature 0.15 | 2,009,186 | 76.41% | 60.48% |
| 7 | HP09 | SupCon temperature 0.07 | 2,009,186 | 76.20% | 58.80% |
| 8 | ST04 | 5/2 + 9/4 双尺度 HSTE | 2,669,539 | 76.05% | 58.51% |
| 9 | HP04 | weight decay 0.05 | 2,009,186 | 76.01% | 56.29% |
| 10 | HP05 | 更弱正则化 | 2,009,186 | 75.30% | 55.29% |
| 11 | ST03 | 局部 mean pooling | 1,959,841 | 75.26% | 54.92% |
| 12 | HP08 | SupCon weight 0.20 | 2,009,186 | 74.97% | 55.52% |
| 13 | ST02 | 最终 attention pooling | 1,844,322 | 74.27% | 55.92% |
| 14 | HP03 | weight decay 0.01 | 2,009,186 | 74.25% | 56.34% |
| 15 | IN03 | DFS61 | 2,013,794 | 72.64% | 55.03% |
| 16 | IN04 | DFS121 | 2,021,474 | 72.05% | 51.95% |
| 17 | IN02 | current + phase | 2,020,970 | 71.79% | 53.36% |
| 18 | HP01 | backbone LR 2e-4 | 2,009,186 | 69.47% | 45.01% |
| 19 | IN01 | phase + DFS25 | 2,008,562 | 61.30% | 40.55% |

结构结论：减小窗口到 3 帧在 seed2025 上改善困难动作的局部轨迹；但纯 attention pooling、纯 mean 窗口聚合和双尺度结构都没有超过 T00。ST04 多 32.9% 参数仍下降 1.05 pp，说明继续扩大 HSTE 不是当前瓶颈。

## 5. 冻结前两名测试

| 候选 | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | 相对原 89.32% |
| --- | ---: | ---: | ---: | ---: |
| HP07 | 79.14% | 87.32% | 86.77% | -2.55 pp |
| ST01 | 77.55% | 89.29% | 89.07% | -0.25 pp |

ST01 在 seed2025 达到预先定义的次级门槛：user17 validation 提升 3.72 pp，测试 Macro-F1 下降 0.25 pp。这个结果只授权多 seed 复核，不足以替换正式基线。

## 6. 五随机种子稳定性

| Seed | T00 Val F1 | ST01 Val F1 | T00 user17 | ST01 user17 | T00 Test F1 | ST01 Test F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025 | 77.10% | 77.55% | 57.90% | 61.62% | 89.32% | 89.07% |
| 2026 | 76.86% | 77.17% | 59.13% | 57.83% | 87.03% | 88.86% |
| 2027 | 75.09% | 73.33% | 57.49% | 51.34% | 86.86% | 85.29% |
| 2028 | 76.87% | 74.81% | 60.48% | 54.39% | 87.65% | 87.64% |
| 2029 | 77.52% | 75.11% | 63.57% | 55.81% | 85.84% | 88.94% |

ST01 - T00 的配对 test Macro-F1 差为 `[-0.25, +1.83, -1.57, -0.02, +3.10] pp`，均值 `+0.62 ± 1.84 pp`。其 validation 差为 `[+0.45, +0.31, -1.77, -2.06, -2.41] pp`；user17 差为 `[+3.72, -1.30, -6.15, -6.09, -7.75] pp`。后两项说明 seed2025 的困难受试者改善没有复现。

### 6.1 每类 test F1

类别顺序为 `push/pull, sweep, clap, slide, circle, zigzag`。

| 类别 | T00 | ST01 | ST01 - T00 |
| --- | ---: | ---: | ---: |
| push/pull | 89.50% ± 1.56% | 89.78% ± 2.05% | +0.28 pp |
| sweep | 89.92% ± 1.63% | 90.25% ± 0.96% | +0.32 pp |
| clap | 93.62% ± 0.87% | 93.88% ± 0.49% | +0.25 pp |
| slide | 80.11% ± 4.75% | 81.45% ± 3.32% | +1.34 pp |
| circle | 84.11% ± 2.68% | 84.82% ± 2.35% | +0.71 pp |
| zigzag | 86.79% ± 2.42% | 87.58% ± 2.35% | +0.79 pp |

ST01 对各类的均值都略有改善，slide 增益最大，但受试者与跨域方差抵消了这一收益。五 seed 累计混淆矩阵中，slide 正确数从 954 增至 964，`slide → push/pull` 从 118 降至 104，`slide → clap` 从 73 降至 64；同时 `push/pull → circle` 从 54 增至 73，表明误差被重新分配而不是整体稳定消失。

### 6.2 每个测试受试者

user2 在固定 test 中只有 label0，因此报告实际出现类别 F1，而不是会被五个缺失类别压低的六类 Macro-F1。

| 受试者 | T00 F1 | ST01 F1 | 变化 |
| --- | ---: | ---: | ---: |
| user12 | 87.99% ± 2.11% | 90.01% ± 1.66% | +2.01 pp |
| user2（present label） | 94.29% ± 2.38% | 93.22% ± 2.00% | -1.07 pp |
| user7 | 86.82% ± 1.77% | 86.38% ± 1.70% | -0.44 pp |

ST01 的平均提升主要来自 user12，而不是所有测试受试者共同改善。

## 7. user17 诊断

数据侧审计已经证明 user17 的类别、位置和方向分布近乎均衡；它有最长的中位序列（40 帧）、最低的同环境 packet retention/timestamp health，但 SNR proxy 最高。低分不是由缺失接收器或简单样本不均衡造成。

seed2025 逐样本分析：

| 指标 | T00 | ST01 |
| --- | ---: | ---: |
| Accuracy | 64.22% | 65.82% |
| Macro-F1 | 57.90% | 61.62% |
| ECE | 23.47% | 17.48% |
| NLL | 1.452 | 1.135 |
| 错误样本平均置信度 | 79.04% | 72.74% |

| 类别 | T00 F1 | ST01 F1 | 变化 |
| --- | ---: | ---: | ---: |
| push/pull | 76.87% | 77.99% | +1.12 pp |
| sweep | 76.97% | 70.52% | -6.45 pp |
| clap | 20.14% | 29.93% | +9.79 pp |
| slide | 24.69% | 41.11% | +16.42 pp |
| circle | 64.11% | 68.94% | +4.83 pp |
| zigzag | 84.62% | 81.23% | -3.39 pp |

ST01 seed2025 确实改善了 clap/slide，并降低了错误置信度。其 clap/slide embedding 到训练受试者同类中心的平均余弦距离也从约 `0.56/0.38` 降到约 `0.44/0.33`。但该变化只在 seed2025 稳定出现；五 seed user17 均值反而下降 3.51 pp，所以不能作为正式方法贡献。

T00 的 user17 还呈现以下特征：短/中/长序列 Macro-F1 为 31.88%/41.86%/56.19%，orientation 5 只有 46.14%，clap 大量被判为 push/pull 或 circle，slide 大量被判为 sweep 或 circle。这更像动作阶段/速度与类别中心同时偏移，而不是单纯长度过长。

## 8. 严格 Wi-CBR 三折跨域复核

旧适配器默认把 official test 复制为 validation，导致目标域标签参与 checkpoint 选择。相关运行已封存，不再作为 zero-shot 结果：

```text
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/invalid/wicbr_target_as_validation_20260716T1935
```

修复后的 source validation 按 `environment × label` 分层，从 official train 中通过 sample-id SHA256 确定性选出 10%；validation 与 target test 完全不重叠：

| Fold | Train | Source validation | Target test |
| --- | ---: | ---: | ---: |
| cr1 | 4,154 | 467 | 6,749 |
| cr2 | 8,768 | 978 | 1,624 |
| cr3 | 7,528 | 845 | 2,997 |

结果：

| Fold | T00 Source Val | ST01 Source Val | T00 Test | HP07 Test | ST01 Test | ST01 - T00 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cr1 | 95.07% | 95.52% | 86.65% | 86.71% | 84.97% | -1.69 pp |
| cr2 | 93.85% | 93.33% | 94.33% | 94.40% | 93.90% | -0.43 pp |
| cr3 | 95.74% | 95.05% | 85.86% | 86.52% | 85.93% | +0.07 pp |
| Mean ± std | 94.89% ± 0.96% | 94.63% ± 1.15% | 88.95% ± 4.68% | 89.21% ± 4.49% | 88.26% ± 4.90% | -0.68 pp |

旧报告中的 89.78% 使用了目标域 validation，不能作为严格 zero-shot 基线。修复后的 T00 88.95% 是本轮可比较的严格值。HP07 仅提高 0.26 pp；ST01 在最困难的 cr1 下降 1.69 pp，均未达到成功门槛。

## 9. 效率

五 seed 运行记录如下；训练时间受最佳 epoch/早停影响，`seconds/step` 更适合比较结构开销。推理统计是固定 test 上包含数据加载和指标计算的端到端 batched 评估，不是单样本 CUDA microbenchmark。

| 指标 | T00 | ST01 |
| --- | ---: | ---: |
| 参数量 | 2,009,186 | 2,009,186 |
| 训练时间 | 725.1 ± 267.6 s | 602.7 ± 244.3 s |
| seconds/optimizer step | 0.1167 s | 0.0991 s |
| 训练峰值显存 | 772.7 MiB | 742.0 MiB |
| 评估吞吐 | 952.0 ± 9.4 samples/s | 946.1 ± 43.5 samples/s |
| 平均 batch 延迟 | 189.7 ± 1.9 ms | 191.2 ± 8.7 ms |
| 吞吐折算每样本时间 | 1.050 ms | 1.057 ms |
| 评估峰值显存 | 819.0 MiB | 819.6 MiB |

ST01 的局部窗口更短，训练 step 更快且显存略低，但推理吞吐没有可靠改善。FLOPs 本轮未使用经过验证的工具测量，因此不报告估计值。

## 10. 无效运行与工程修复

两组运行被保留但排除：

1. `structure_pre_hp07_freeze_20260716T1832`：结构实验没有继承冻结后的 HP07 超参，违反阶段依赖，不能进入 19 个合法候选排名。
2. `wicbr_target_as_validation_20260716T1935`：目标 test 被复制为 validation，存在目标域选模，不属于严格 zero-shot。

EMA 正式配置保持 `use_ema: false`。此外，多 seed 汇总器现在同时汇总 `macro_f1_present_labels`，避免 user2 只有一个类别时被错误报告为约 15% 的最差受试者。

## 11. 最终验证

最终候选提交在远端独立 worktree、`LWCL-v2` Conda 环境中完成以下检查：

| 检查 | 结果 |
| --- | --- |
| 完整 pytest | 32 passed |
| 19 候选配置审计 | count 19，errors 0；全部 EMA 关闭、参数 <4M、只按 validation Macro-F1 选模 |
| 单 batch/短训练 | 3 个 optimizer steps，loss/梯度有限 |
| checkpoint 保存 | `best.pt`、`step_00000002.pt`、`last.pt` 均存在且可加载 |
| checkpoint 恢复 | 从 `last.pt` 恢复后正确继续到 global step 5 |
| eval | validation 指标文件成功写出 |
| emergency checkpoint | step 2 主动故障后生成 `emergency_step_00000002.pt` |
| pre-push audit | 139 个 tracked files 通过；数据、模型权重、密钥和 IDE 文件均未跟踪 |

确定性复核重新构建了 phase25 split。由于验证 worktree 通过符号链接访问数据，CSV 中 `feature_path` 的相对前缀不同，因此原始 CSV 文件 SHA 不同；排除 `feature_path` 后，11,370 行在 `sample_id`、split、标签、受试者、环境、位置、方向、时间步和有效接收器等所有字段上有 0 个差异。随后对全量 11,370 个 NPZ 重新审计，得到与首次运行完全相同的内容级 contract hash：

```text
8c6c61d53792c0cd86140363d97c8bd62cff0b787d00b037de712cfcc411bc40
```

这证明划分语义和输入张量内容可重复；路径序列化不应被误解为数据变化。

## 12. 最终推荐

正式模型继续使用：

```text
配置：configs/signal_v2_base.yaml
受控复核配置：configs/tuning/signal_v2_tuning_base.yaml
参数量：2,009,186
正式 seed2025 checkpoint：
/home/wj/LWCL-v2.0-staging/outputs/widar3_signal_v2_raw_selection_seed2025/checkpoints/best.pt
```

ST01 保留为“局部短窗口可能改善 slide/clap，但跨受试者与跨域不稳定”的负结果，不合并到正式配置。HP07 保留为“降低 SupCon 权重可提高源域 validation，但固定 test 不一致”的负结果。

下一步不应继续无界结构搜索。更有依据的工作是：

1. 在固定 T00 上做 grouped subject-disjoint 多折，确认 user17 是否与某些训练受试者组合相关；
2. 针对 clap/slide 引入不增加模型规模的速度归一化或持续时间条件化，并先做单变量消融；
3. 将 proxy-based class-center loss 与当前 SupCon 做受控比较，但仍只使用源域 validation 选模；
4. 增加 receiver/time corruption 压力测试，验证质量感知模块的部署价值。

## 13. 复现命令

以下命令在远端仓库根目录执行：

```bash
cd /home/wj/LWCL-v2.0-tuning-run
export PYTHONPATH="$PWD/src"
CONDA=/home/wj/miniconda3/bin/conda
```

完整测试与 checkpoint smoke：

```bash
$CONDA run --no-capture-output -n LWCL-v2 python -m pytest -q

CUDA_DEVICE=0 \
OUTPUT_ROOT=/home/wj/LWCL-v2.0-staging/smoke/signal_v2_tuning_final \
bash scripts/run_signal_v2_tuning_smoke.sh
```

重建并审计 phase/DFS 输入：

```bash
WORKERS=8 bash scripts/preprocess_signal_v2_tuning_inputs.sh

$CONDA run --no-capture-output -n LWCL-v2 \
  python -m lwcl_v2.cli.audit_tuning_configs \
  configs/tuning/signal_v2_tuning_base.yaml \
  configs/tuning/signal_v2_hp_*.yaml \
  configs/tuning/signal_v2_structure_*.yaml \
  configs/tuning/signal_v2_phase_dfs.yaml \
  configs/tuning/signal_v2_current_plus_phase.yaml \
  configs/tuning/signal_v2_highres_dfs61.yaml \
  configs/tuning/signal_v2_highres_dfs121.yaml \
  --output /home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/summaries/config_audit.json \
  --require-count 19 --strict
```

单 seed validation-only 筛选示例：

```bash
CUDA_DEVICE=0 SEED=2025 \
OUTPUT_ROOT=/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/screening \
bash scripts/run_signal_v2_tuning_validation.sh

CONFIGS="configs/tuning/signal_v2_structure_window3_stride2.yaml \
configs/tuning/signal_v2_structure_attention_pool.yaml \
configs/tuning/signal_v2_structure_local_mean.yaml \
configs/tuning/signal_v2_structure_multiscale.yaml" \
CUDA_DEVICE=0 SEED=2025 \
OUTPUT_ROOT=/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/screening \
bash scripts/run_signal_v2_tuning_validation.sh
```

严格生成排名并冻结前两名：

```bash
$CONDA run --no-capture-output -n LWCL-v2 \
  python -m lwcl_v2.cli.summarize_tuning \
  --run-root /home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/screening \
  --output-json /home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/summaries/all19_validation_ranking.json \
  --output-csv /home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/summaries/all19_validation_ranking.csv \
  --require-count 19 --strict-validation-only \
  --freeze-top 2 \
  --freeze-output /home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/summaries/frozen_top2.json
```

ST01 多随机种子：

```bash
CONFIG_PATH=configs/tuning/signal_v2_structure_window3_stride2.yaml \
RUNS_ROOT=/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/multiseed/st01_repro \
SEEDS="2025 2026 2027 2028 2029" CUDA_DEVICE=0 \
bash scripts/run_widar3_v2_multiseed.sh
```

严格 Wi-CBR source-validation 协议：

```bash
SOURCE_VALIDATION_FRACTION=0.10 \
SOURCE_VALIDATION_SALT=wicbr-source-validation-v1 \
MANIFEST_ROOT=$PWD/data/splits/widar3_signal_v2_wicbr_protocol_sourceval_v1_repro \
OUTPUT_ROOT=/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/crossdomain/t00_repro \
CONFIG_PATH=configs/tuning/widar3_signal_v2_wicbr_t00_sourceval.yaml \
CUDA_DEVICE=0 bash scripts/run_widar3_signal_v2_wicbr_protocol.sh
```

user17 逐样本分析：

```bash
CONFIG_PATH=configs/tuning/signal_v2_tuning_base.yaml \
CHECKPOINT_PATH=/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/screening/signal_v2_tuning_base/train/checkpoints/best.pt \
OUTPUT_DIR=/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/analysis/t00_seed2025_repro \
FOCUS_SUBJECT=user17 CUDA_DEVICE=0 \
bash scripts/export_widar3_subject_analysis.sh
```

## 14. 关键产物位置

```text
19 候选排名：
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/summaries/all19_validation_ranking.json

冻结前两名：
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/summaries/frozen_top2.json

T00/ST01 五 seed：
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/multiseed/
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/multiseed_final/

严格 Wi-CBR：
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/crossdomain/

user17 对比：
/home/wj/LWCL-v2.0-staging/outputs/subject_error_analysis_seed2025/user17_analysis.json
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/analysis/st01_seed2025/user17_analysis.json
```
