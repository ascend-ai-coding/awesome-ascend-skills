# Elementwise SIMD 与融合

## 目标与适用条件

用于连续逐元素、量化转换、激活和简单复合表达式。

## TileLang/PTO 实现

连续规则路径使用 T.SimdVF 与 T.Parallel；bf16/fp16 超越函数先转 fp32。把只使用一次的中间结果保存在寄存器或同一 UB tile，融合相邻表达式以消除 GM round-trip。复杂索引才使用 T.SimtVF。

仿射链 `a*x+b` 将显式 `S.vmadd` 与 `vmul+vadd` 作为不同 lowering 候选；不假设 fast-math 会自动融合，从生成 PTO 确认指令，并分别验证精度和同口径性能。

短记录可把多个 token 打包到一个向量，用预生成索引执行 `vgather2`/`vscatter`，并将常量与索引模式移出热循环。常见整 tile 可由 Python wrapper 固化 shape 和展开数，其他 shape 保留动态尾块。完整结构见 [索引短记录变换](indexed_short_record.md)。

实现必须使用一维 T.Kernel。纯向量任务的核数不超过 `min(已确认的可用 AIV 核数, 独立任务数)`。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，多版本方式沿用已验证模板并核对 lowering。

## 精度门禁

严格保持 reference 的运算顺序、饱和/舍入模式和输出 cast 边界。量化必须覆盖零 scale、极值与非有限输入契约。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

测向量 lane 利用率、指令数量、GM 字节和 spill。融合导致寄存器压力或代码膨胀时拆分。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

可执行实现见 examples/ascend/example_simdvf_vecadd.py 和 examples/ascend/example_simdvf_per_token_cast_to_fp8.py。
