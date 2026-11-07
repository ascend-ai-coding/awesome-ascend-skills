# T.copy 搬运与访存合并

## 目标与适用条件

用于 GM 到 UB/L1/L0 以及反向写回。目标是减少搬运次数、扩大连续 burst，并让搬运与计算重叠。

## TileLang/PTO 实现

优先复制连续矩形切片；尾部使用精确 valid slice 或 T.copy 的 pad_value。重复读取的权重、scale 和查找表在 Persistent 循环外搬入单版本 buffer。对 strided 输入使用 T.StridedTensor；只有目标示例已经证明的 l2_cache_ctrl 值才能进入正式路径。

相邻短记录或标量应沿外层维合并为矩形/连续搬运，避免每个 token 产生 4B GM 访问；同时核对 nburst、burst_len、无效字节和 UB 占用。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

低精度输入在 UB 中按算子语义提升为 fp32；pad_value 必须是该运算的幺元，例如 max 用负无穷、sum 用 0。禁止读取未初始化 padding lane。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

比较 GM 总字节、copy 指令数、MTE2/MTE3 时间和有效带宽。合并搬运只有在没有增加无效数据量或 UB 压力时保留。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

参考 examples/ascend/example_copy_pad_value.py、examples/ascend/example_simdvf_vecadd.py 和 examples/ascend/example_gemm.py。
