# 多维 Broadcast 与矩形搬运

## 目标与适用条件

适用于可合并为连续矩形的多轴广播。

## TileLang/PTO 实现

先在 host/factory 合并相邻连续维，得到 outer、broadcast、inner 三段。T.copy 只负责连续矩形；不能表达的离散维在 T.SimtVF 中显式索引。禁止假设任意 N 维广播都能 lowering 为单条 DMA。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

验证合并前后逻辑 shape、stride 与 storage offset；非连续 view 使用 T.StridedTensor。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

比较维度合并前后的 copy 数、标量地址计算和总 GM 字节。只有生成代码确认矩形搬运成立时采用 DMA 路径。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

动态维和张量视图分别参考 `testing/ascend/language/test_tilelang_ascend_dynamic_none.py` 与 `tilelang/language/symbolics.py`，搬运限制核对 `src/ascend/op/copy.cc`。
