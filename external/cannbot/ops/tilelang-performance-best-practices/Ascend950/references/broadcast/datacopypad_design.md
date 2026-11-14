# Broadcast 尾部填充

## 目标与适用条件

用于 broadcast 输入或输出的最后一块不满足对齐要求。

## TileLang/PTO 实现

T.copy(..., pad_value=value) 只在目标 PTO 示例支持的形状上使用；max/topk 填负无穷，sum 填 0。输出始终只写 valid 区域。需要区分数据填充值和最终输出值。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

覆盖最后一块有效元素为 1、lane-1、lane+1，并验证负无穷、NaN 和 dtype 转换。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

pad copy 与显式 fill+copy 做 A/B；记录额外 UB 写入和 copy latency。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

直接参考 examples/ascend/example_copy_pad_value.py 与 examples/ascend/example_simdvf_topk_gate.py。
