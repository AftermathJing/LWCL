# CSI-Bench Wi-CBR 早停实验报告

## 运行身份

- 日期：2026-07-19
- 分支：`agent/wicbr-official-protocol`
- 训练代码提交：`b24324a4b890215b30fb16a0f68da8a3cfd56060`
- 官方 Wi-CBR 参考仓库提交：`4f118d2e7105bf486462cbdcac4931b1f8244bc5`，实验前后工作树均为空
- 远程运行目录：`/home/wj/LWCL-wicbr-earlystop-20260719`
- 结果目录：`outputs/csi_bench_har_wicbr_earlystop`
- manifest SHA-256：`ab77192ae868c6d0c3cded7085e49aaac41245d4983fb6473076ec54d39cfea5`
- 训练日志未发现 `Traceback`、OOM、`RuntimeError`、`FAILED` 或 emergency checkpoint

本实验只修改 LWCL 的 Wi-CBR 适配训练器和独立配置，没有修改 `_external/Wi-CBR` 官方代码。

## 协议

模型、图像分辨率、损失、优化器、学习率、batch size、seed 和最大 epoch 与 CSI-Bench Wi-CBR 无早停基线保持一致。新增协议为：

- 选择集：`validation`
- 选择指标：Macro-F1
- 每 86 个训练 step 验证一次
- `patience = 12`
- `min_delta = 0.0005`
- validation 后滚动保存 `last.pt`，只永久保留 `best.pt` 与 `last.pt`

训练在 step 4988（0-based epoch 14，即第 15 个 epoch 内）触发早停。最佳 validation Macro-F1 为 94.5231%，出现在 step 3956；之后连续 12 次验证没有达到最小提升要求。最终五个测试协议均使用 step 3956 的 `best.pt` 评估。

## 完整结果

旧版评估器没有直接输出 Weighted-F1；下表的 Weighted-F1 根据归档的 `per_class_f1` 和 `support` 加权计算。

| 协议 | 样本数 | Accuracy | Weighted-F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| test_id | 4,602 | 96.78 | 96.80 | 95.48 |
| test_cross_device | 9,631 | 46.36 | 45.95 | 39.98 |
| test_cross_env | 6,650 | 44.96 | 44.52 | 33.91 |
| test_cross_user | 12,109 | 60.49 | 58.06 | 48.22 |
| test_cross_user_env | 18,759 | 54.99 | 53.44 | 43.32 |

## 与已冻结结果对比

主指标为 Weighted-F1，括号内为 Macro-F1。

| 方法 | ID | 跨设备 | 跨环境 | 跨用户 |
|---|---:|---:|---:|---:|
| B0 Base | 93.32（90.63） | 57.67（49.87） | 52.26（42.34） | 59.32（52.78） |
| B2 Robust50 | 95.22（93.28） | 57.46（47.70） | 50.12（36.15） | 60.16（49.84） |
| Wi-CBR 无早停 best | 98.54（97.81） | 48.37（41.24） | 42.33（29.41） | 56.71（46.11） |
| Wi-CBR 早停 best | 96.80（95.48） | 45.95（39.98） | 44.52（33.91） | 58.06（48.22） |

相对 Wi-CBR 无早停 best，早停的 Weighted-F1 变化为：ID -1.74 个百分点、跨设备 -2.42、跨环境 +2.19、跨用户 +1.35；Macro-F1 变化分别为 -2.33、-1.26、+4.50、+2.11 个百分点。

## 结论

早停没有形成全面优势。它明显缩短训练（80 epoch 上限提前到 step 4988），并改善跨环境与跨用户泛化，尤其跨环境 Macro-F1 提升 4.50 个百分点；但同域和跨设备均下降。与 B0/B2 相比，Wi-CBR 早停仍保持更高的同域结果，但三个主要跨域协议总体仍弱于 LWCL 基线。

因此，论文中应将它报告为“验证集 Macro-F1 早停对 Wi-CBR 的混合影响”，不能表述为统一提升；无早停结果必须继续作为独立基线保留。
