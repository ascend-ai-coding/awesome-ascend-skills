# Broadcast Shape 专用 Tiling

## 目标与适用条件

用于不同 rank、广播轴和倍率表现差异明显的算子。

## TileLang/PTO 实现

在 Python factory 中按连续性、inner 长度、广播倍率和 dtype 选择 tile、核数、SimdVF/SimtVF 及 stage。保持一个覆盖全部合法 shape 的正确 fallback。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

每个 dispatch 边界两侧都运行精度测试；动态 shape 的 T.assume 只能表达调用方真实保证。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

系统搜索 tile 与 stage，一次只改变一个变量。统计编译变体数量与缓存占用，避免过度特化。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

配置选择方式参考实际 TileLang 源码 `examples/ascend/example_gemm_various_shapes.py` 中的 `bf16_select_config`；使用前核对当前安装版本。
