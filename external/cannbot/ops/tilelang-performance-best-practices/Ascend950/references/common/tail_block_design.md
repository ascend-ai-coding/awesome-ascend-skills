# 非对齐尾块实现

## 目标与适用条件

用于元素数、行宽或矩阵维度不能整除 DMA、SIMD 或 tile 粒度的路径。

## TileLang/PTO 实现

GM 侧只访问 valid 元素；UB 按可能触达的完整寄存器 footprint 分配。Reduction 的无效 lane 写入幺元；transpose/gather 的输出只回写有效矩形。能在 Python factory 中按余数类生成专用 kernel 时，优先消除热循环内的动态分支。

实现必须使用一维 T.Kernel。纯向量任务从 已确认的可用 AIV 核数 取核数，并以 min(核数, 独立任务数) 限制空核。GM 与 UB/L1 之间使用 T.copy；连续规则计算使用 T.SimdVF 和 T.Parallel；跨 tile 任务使用 T.Persistent 或 T.Pipelined，并以 T.annotate_buffer_versions 显式声明多版本缓冲。

## 精度门禁

对每种 dtype 分别验证 DMA 对齐、SIMD lane 数和转换后的 fp32 footprint。用 canary 检查尾部前后没有越界写。

必须覆盖最小 shape、常见 shape、最大 shape、tile-1/tile/tile+1、非 32B 尾块、0、正负极值以及接口规定的 NaN/Inf 语义。不得通过扩大容差、降低 reference 精度或跳过 case 获得通过。

## 性能门禁

分别 benchmark 整 tile 与 tile±1，防止为尾块增加的分支拖慢主路径。若专用 kernel 数量过多导致编译或缓存压力，则合并低频余数类。

先用 `TILELANG_DEFAULT_TARGET=pto` 运行定向精度测试，再运行相关完整测试套；全部通过后，在相同输入、dtype、warmup、repeat、设备和并发条件下测量性能。报告 kernel latency、有效 GM 带宽、UB 占用、stage 数和基线差异。

## 可执行代码与证据

参考 testing/ascend/layout/test_ascend_l0_transpose.py 和 testing/ascend/language/test_tilelang_ascend_simdvf_cast.py。
