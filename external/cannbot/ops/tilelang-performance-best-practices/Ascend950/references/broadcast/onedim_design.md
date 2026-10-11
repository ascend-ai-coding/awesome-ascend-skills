# 单轴 Broadcast

## 目标与适用条件

适用于一个输入轴长度为 1，或一段连续数据沿相邻维复制。

## TileLang/PTO 实现

把输出线性索引分解为 outer、broadcast index、inner。inner 连续时，每个 task 搬一次源 slice 到 UB，并在 SIMD 循环中写多个目标位置；标量 inner 使用 broadcast register。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

覆盖 broadcast size 为 1、2、向量 lane±1 和大倍率；检查索引乘法使用足够位宽。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

在广播倍率大时测源 GM 字节是否接近只读一次；倍率小时防止 UB staging 的固定开销超过直接访问。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

基于 examples/ascend/example_simdvf_vecadd.py；动态 stride 按 `tilelang/language/symbolics.py` 与 `src/ascend/op/copy.cc` 中实际接口核对。
