# Broadcast 总体实现

## 目标与适用条件

用于标量、单轴和多轴广播的 elementwise 算子。将输出空间展平为独立 tile，并从输出索引计算输入索引。

## TileLang/PTO 实现

连续广播轴优先把小输入搬入 UB 一次，然后在 T.SimdVF 中复用；标量使用 SIMD broadcast load。复杂不连续索引使用 T.SimtVF，并通过 Python factory 固化 rank、axis 和 stride。输出任务由 T.Persistent 分配。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

广播前后的 dtype 转换顺序必须与 reference 一致；低精度超越函数与累加提升 fp32。验证 size=1 轴、多个广播轴、零维标量和尾块。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

重点测 GM 重复读取量、索引计算开销和向量利用率。小广播输入常驻 UB，避免每个输出 tile 重复搬运。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

复用 `examples/ascend/example_simdvf_vecadd.py` 的流水骨架和 `examples/ascend/example_simdvf_per_token_cast_to_fp8.py` 的索引模式。
