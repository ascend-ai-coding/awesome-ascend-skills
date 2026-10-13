---
name: profiling-analysis-roofline
description: 对 Ascend profiler 的 kernel_details.csv 做算力瓶颈 Roofline 核对：筛选 aic_mac_ratio≥80% 的算子，比较理论耗时与实际耗时，判断是否需要优化并给出排查方向。当用户提到 Roofline、aic_mac_ratio、理论耗时、算力瓶颈核对、kernel_details 峰值对比时使用。
keywords:
    - profiling
    - Roofline
    - aic_mac_ratio
    - 算力瓶颈
    - 理论耗时
    - kernel_details
---

# Profiling 算力瓶颈 Roofline 核对 Skill

对 `kernel_details.csv` 中 `aic_mac_ratio≥80%` 的算子，按 NPU 峰值算力与带宽计算理论耗时，并与实际耗时对比，给出是否需要优化及初步排查方向。

本 Skill 与 `profiling-analysis-computing` 同级、相互独立：计算瓶颈透视/形状解析仍用 computing；本 Skill 专注 Roofline 理论/实际核对。

## 需要的输入

缺一不可；缺失时先向用户询问，不要默认猜测：

1. `kernel_details.csv` 路径，或含该文件的 profiling 根目录（需含 `aic_mac_ratio` 列）
2. NPU 算力（TFLOPS）：填 **FP16/BF16 Cube 峰值**
3. 内存带宽：默认按 **GB/s** 解释 `--bandwidth`；`1.2 TB/s` 写作 `--bandwidth 1.2 --bw-unit TB/s`，或 `--bandwidth 1200`

可选：`--mac-threshold`（默认 0.80）、`--top`、`--output-path` / `--out`、`--int8-scale`（默认 2）

## 脚本

- `scripts/kernel_roofline_compute_check.py`：分析入口
- `reference/kernel_roofline_compute_check.md`：口径、判据与排查方向

## 使用方式

```bash
python scripts/kernel_roofline_compute_check.py \
  --input-path <profiling根目录> \
  --flops <TFLOPS> --bandwidth 1.2 --bw-unit TB/s \
  --output-path <输出目录>
```

或直接指定 CSV：

```bash
python scripts/kernel_roofline_compute_check.py \
  --csv <path>/kernel_details.csv \
  --flops <TFLOPS> --bandwidth 1200 \
  --out <输出目录>/kernel_roofline_compute_check.csv
```

自检：

```bash
python scripts/kernel_roofline_compute_check.py --self-test
```

## 分析口径（摘要）

- 理论耗时 = `max(FLOPs/算力, Bytes/带宽)`；效率 = 理论/实际
- 默认只分析 `aic_mac_ratio≥80%`
- 可建模：MatMul 系、QuantBatchMatmul、TransposeBatchMatMul、GroupedMatmul/SwigluQuant、Compressor、SparseAttn、HcPre/HcPost 等

## 是否需要优化（摘要）

| 效率 | 判断 |
|------|------|
| 80%～120% | 暂不优先优化 kernel（Cube 已接近峰值） |
| 40%～70% | 建议优化 |
| <40% | 需要优化 |
| >150% | 模型高估，不据此排序 |

详细判据与排查方向见 [算力瓶颈 Roofline 核对](./reference/kernel_roofline_compute_check.md)。

## 输出

- 终端报告：总体概况、按类型汇总、Top 明细、是否需优化摘要、初步排查方向
- CSV（可选）：`kernel_roofline_compute_check.csv`
