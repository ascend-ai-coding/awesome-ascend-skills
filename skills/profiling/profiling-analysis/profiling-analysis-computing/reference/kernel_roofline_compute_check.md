# 算力瓶颈算子 Roofline 核对

对 `kernel_details.csv` 中 `aic_mac_ratio≥80%` 的算子比较理论耗时与实际耗时，判断是否需优化。

## 输入与命令

必填：`--flops`（FP16/BF16 Cube TFLOPS）、`--bandwidth`（配合 `--bw-unit`）。缺失时先询问用户，不要猜测。

```bash
python scripts/kernel_roofline_compute_check.py --input-path <profiling根目录> \
  --flops <TFLOPS> --bandwidth 1.2 --bw-unit TB/s --output-path <输出目录>
# 或 --csv <kernel_details.csv> --bandwidth 1200 --out <明细.csv>
```

## 口径

- 理论耗时 = `max(FLOPs/算力, Bytes/带宽)`；效率 = 理论/实际
- 默认过滤 `aic_mac_ratio≥0.80`（`--mac-threshold` 可调）
- INT8/FP8 Cube 默认 `--int8-scale 2`
- 可建模：MatMul 系、QuantBatchMatmul、TransposeBatchMatMul、GroupedMatmul/SwigluQuant（含 ceiling）、Compressor、SparseAttn、HcPre/HcPost、Scatter 等；Metadata/Indexer 等标未建模

## 是否需要优化

| 效率 | 判断 | 方向 |
|------|------|------|
| 80%～120% | 暂不优先优化 kernel | 图级融合/合批/减调用/算法降算量 |
| 70%～80% | 可选优化 | 核对合批与精度路径 |
| 40%～70% | 建议优化 | M/N/K、合批、库路径、tiling |
| <40% | 需要优化 | 同上，优先浪费最大的 Type |
| >150% | 不据此优化 | 模型高估 |

单次 Duration&lt;10us 的低效率常为 launch 开销。输出含终端报告与可选 `kernel_roofline_compute_check.csv`。
