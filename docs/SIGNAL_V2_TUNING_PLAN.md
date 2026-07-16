# Signal-v2 受控调优计划

日期：2026-07-16

## 1. 目标与边界

本阶段仅优化 LWCL-v2.0 的纯信号路线。目标是在不引入 LLM、大型视觉骨干或无界搜索的前提下，提高严格 subject-disjoint Macro-F1，并复核联合环境与受试者域偏移下的稳定性。

以下资产视为只读基线，不允许覆盖：

- `configs/signal_v2_base.yaml`；
- `data/processed/widar3_p1_full/`；
- `data/splits/widar3_v2_subject_split.csv`；
- 已封存 Signal-v2 checkpoint、日志和正式结果；
- 现有 Wi-CBR 协议结果。

新增代码必须兼容当前 `[B,T,N,F]`、`time_mask`、`receiver_mask`、标签和 CLI 契约。默认关闭 EMA，不修改 Qwen、Adapter、LoRA 或其他语言模型路线。

## 2. 冻结基线

| 项目 | 冻结值 |
| --- | --- |
| 模型 | Signal-v2 base |
| 参数量 | 2,009,186 |
| 预处理 | P1，STFT 251/50，25-bin Doppler |
| 输入 | RSSI + Doppler + differential CSI |
| subject split | train 8,247 / validation 1,498 / test 1,625 |
| validation subjects | user9, user17 |
| test subjects | user2, user7, user12 |
| 原单 seed 测试 Macro-F1 | 89.323% |
| 原多 seed 测试 Macro-F1 | 86.368% ± 1.214%（seed 2026-2029） |
| 原多 seed validation Macro-F1 | 76.597% ± 1.024% |
| user17 validation Macro-F1 | 61.051% ± 1.858% |

远端不可变标识：

```text
accepted run config hash: be266dcf78d05baf
subject split sha256:      a060defe45f038c20c5f6408b69899e9c2783f7338b733d3785aa5fb309005f0
P1 manifest sha256:        6ffecd8c8836120270b6829655149eaca184b3b19782aa4ff5e099347f16697f
```

调优实验将新增一个结构完全相同、但只按 validation Macro-F1 选模的 `T00` 运行。它是本阶段受控比较基线，不替换上述正式基线。

## 3. 统一评估规则

所有筛选候选均遵守：

1. 使用同一 subject-disjoint 受试者集合、同一指标实现和 seed 2025；
2. checkpoint 只按源域 validation Macro-F1 选择：

```yaml
training:
  selection:
    macro_f1_weight: 1.0
    worst_subject_macro_f1_weight: 0.0
```

3. 筛选阶段禁止加载 test split；
4. 所有 19 个候选完成后，按 validation Macro-F1 冻结排名；
5. 只有前 2 个候选允许评估一次 test split；
6. 不根据 test 结果修改配置或恢复筛选；
7. 每次运行保存 resolved config、commit、manifest hash、seed、参数量、日志、最佳 step、每类指标、每受试者指标和混淆矩阵；
8. user17 Macro-F1、label 3 和 label 5 作为困难项单独报告，但不替代总体 Macro-F1 排名；
9. `user2` 只有一个类别时，不使用其六类 Macro-F1 作为结论，改报 accuracy；
10. fold 标准差与随机种子标准差分开报告。

## 4. 运行上限与实验矩阵

单 seed 筛选最多 20 个，本计划固定为 19 个，不进行完整笛卡尔积。

### 4.1 第一阶段：超参数筛选（11 个）

结构和输入保持 Signal-v2 不变。正则化采用两个小组合以控制总运行量。

| ID | 变化 | 取值 |
| --- | --- | --- |
| T00 | 受控基线 | 当前超参数，validation Macro-F1 选模 |
| HP01 | backbone/classifier LR | 2e-4 / 4e-4 |
| HP02 | backbone/classifier LR | 6e-4 / 1.2e-3 |
| HP03 | weight decay | 0.01 |
| HP04 | weight decay | 0.05 |
| HP05 | 低正则组合 | stem/HSTE dropout 0.05，classifier 0.15，label smoothing 0 |
| HP06 | 高正则组合 | stem/HSTE dropout 0.15，classifier 0.25，label smoothing 0.05 |
| HP07 | SupCon weight | 0.05，temperature 0.10 |
| HP08 | SupCon weight | 0.20，temperature 0.10 |
| HP09 | SupCon temperature | 0.07，weight 0.10 |
| HP10 | SupCon temperature | 0.15，weight 0.10 |

第一阶段结束后只按 validation Macro-F1 选择一个超参数基底。若没有候选超过 T00，则后续继续使用 T00。

### 4.2 第二阶段：轻量结构实验（4 个）

以下候选均继承冻结后的第一阶段最佳超参数：

| ID | 变化 | 动机 |
| --- | --- | --- |
| ST01 | HSTE window 3 / stride 2 | 典型 31 帧产生约 15 个全局 token，保留更短局部轨迹 |
| ST02 | 最终 attention pooling | 与 attentive statistics pooling 做受控比较 |
| ST03 | 局部 mean pooling | 验证 attention+mean 窗口聚合是否确有增益 |
| ST04 | 5/2 + 9/4 轻量双尺度 | 同时建模约 250 ms 与 450 ms 动作片段 |

所有模型在启动前统计参数量。超过 3,013,779 参数时必须明确报告相对 2.01M 基线超过 50% 的容量增量；超过 4M 直接停止该候选。

### 4.3 第三阶段：输入表示实验（4 个）

以下候选继承冻结后的最佳超参数和结构；对应 current 输入使用该冻结基底本身，不重复训练：

| ID | 输入 | 说明 |
| --- | --- | --- |
| IN01 | CSI-ratio phase + 25-bin Doppler | 不使用 RSSI 和 differential CSI |
| IN02 | current + CSI-ratio phase | 在当前三分支基础上增加轻量 phase stem |
| IN03 | current，61-bin DFS | 使用更密集频率网格，保持时域窗口不变 |
| IN04 | current，121-bin DFS | 使用更密集频率网格，保持时域窗口不变 |

CSI-ratio phase 直接保存和处理时序张量，不转换为 RGB 图像。天线流配对沿用已复现 Wi-CBR 代码中可验证的样本内规则：按幅度均值/方差比选择最大和最小流，计算二者相位差。相位以 `sin/cos` 对保存，避免在插值时跨越 `-pi/pi` 产生跳变。

## 5. 新输入数据契约

旧 NPZ 保持可读。新字段仅在对应配置启用时要求存在：

```text
rssi:                 [T,6,4]
doppler:              [T,6,D]       D in {25,61,121}
differential_csi:     [T,6,10,2]
csi_ratio_phase:      [T,6,30,2]    最后一维为 sin/cos
csi_ratio_pair:       [6,2]         每个接收器的 numerator/denominator 流索引
time_mask:            [T]
frame_times_ms:       [T]
receiver_mask:        [6]
receiver_quality:     [6,5]
```

计划生成三个新的 processed roots：

```text
data/processed/widar3_tuning_phase25/
data/processed/widar3_tuning_dfs61/
data/processed/widar3_tuning_dfs121/
```

`current` 直接复用只读 `widar3_p1_full`。每个新数据根必须输出：

- 样本数和类别分布；
- shape 和 dtype；
- 时间长度 min/median/p95/max；
- NaN/Inf 计数；
- 5/6、6/6 接收器分布和有效率；
- phase 配对索引分布；
- 内容级 deterministic contract hash；
- failures manifest。

## 6. 运行目录

所有输出写入独立目录：

```text
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/screening/<candidate>/
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/frozen_test/<candidate>/
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/cross_domain/<candidate>/<fold>/
/home/wj/LWCL-v2.0-staging/outputs/signal_v2_tuning/multiseed/<candidate>/<seed>/
```

不得写入或覆盖现有 `widar3_signal_v2_*` 结果目录。

## 7. Smoke gates

任何正式筛选启动前必须通过：

1. 当前项目测试；
2. phase/DFS shape、有限值、旧 NPZ 兼容测试；
3. 单 batch forward/backward；
4. 每个 feature mode 的模型参数量检查；
5. 2-3 step train/eval/save smoke；
6. 从 `last.pt` 恢复后 global step、optimizer、scheduler 和 RNG 连续；
7. eval/save 同 step；
8. 故障触发 emergency checkpoint；
9. 同配置重复预处理得到相同 manifest 和 contract hash；
10. train/validation/test subject 集合无交集，且不存在样本级重复。

## 8. 冻结测试与最终复核

完成 19 个 validation-only 候选后：

1. 生成包含全部候选 config hash、manifest hash、参数量和 validation 指标的排序表；
2. 将前 2 名配置复制到 frozen candidate 清单并记录冻结时间与 commit；
3. 两个候选各评估一次 subject-disjoint test；
4. 选择最终推荐候选时遵守预先定义的目标门槛，不因单次 test 高分恢复已淘汰配置；
5. 对最终候选和原结构受控基线各运行至少 seed 2025、2026、2027；
6. 对最终候选进行三折跨域复核，checkpoint 只由各折源域 validation 选择；
7. 报告 Accuracy/Macro-F1 mean ± std、每类 F1、每受试者 F1、最差受试者、参数量、峰值显存、训练时间和推理延迟。

## 9. 成功门槛

最佳候选至少满足一项：

- 固定 subject-disjoint test Macro-F1 相比 89.32% 提升至少 0.5 pp；
- user17 validation Macro-F1 提升至少 2.0 pp，且总体 Macro-F1 下降不超过 0.3 pp；
- Wi-CBR 对齐三折平均 Macro-F1 相比 89.78% 提升至少 0.5 pp，且无单折下降超过 1.0 pp。

若多 seed 后没有稳定超过基线，则保留原 Signal-v2 为正式模型，并把新表示或结构作为负结果记录。

## 10. 停止条件

出现以下任一情况立即停止相关训练并先修复或记录：

- 天线映射、时间戳或 CSI-ratio 配对规则无法可靠解析；
- 新特征使用了目标域或全数据统计；
- split 受试者重叠、样本重复或派生特征泄漏；
- NaN/Inf、类别映射改变或接收器对齐异常；
- 需要新增依赖、环境升级、预训练模型或图像化大骨干；
- 候选超过 4M 参数或显著超出当前显存预算；
- 达到 20 个单 seed 筛选仍无稳定 validation 改善；
- 需要查看 test 结果继续调参。

