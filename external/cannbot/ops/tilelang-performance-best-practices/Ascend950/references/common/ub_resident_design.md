# UB/L1 常驻与多版本缓冲

## 目标与适用条件

用于跨多个 tile 重复使用的权重、scale、查找表和 reduction 状态。

## TileLang/PTO 实现

常驻数据在 T.Persistent/T.Pipelined 外分配并只搬一次，不加入 T.annotate_buffer_versions。输入输出 tile 按 num_stages 多版本；reduction carry、online softmax 的 max/sum 和 scan carry 保持单版本。预算为所有 buffer 元素数乘 dtype 字节数与版本数，再加 padding 和安全余量。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

常驻低精度权重可保持原 dtype；统计状态与长链累加保持 fp32。确认每个 task 不会读到上一 task 的残留状态。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

A/B 测试重复 GM 流量、UB 占用、occupancy 和流水重叠。常驻导致 stage 数下降或 spill 时，比较总延迟后再取舍。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

参考 examples/ascend/example_rmsnorm.py 中的权重常驻和 examples/ascend/flash_attention/example_mha.py 中的 fp32 online 状态。
