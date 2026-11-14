# GEMM 多级流水

## 适用条件

K tile 至少两轮，MTE2、MTE1 和 Cube 之间存在可重叠空间。

## TileLang/PTO 实现流程

A/B L1 buffer 通过 T.annotate_buffer_versions 设置 2 stage，K 循环使用 T.Pipelined。若显式 L0A/L0B 子块，内层同样使用受容量约束的 Pipelined。PTO 调度器负责依赖，不编写手工 event。

基础实现复用 examples/ascend/example_gemm.py：B 的物理布局为 [N,K]，调用 T.gemm(..., transpose_B=True)，L0C 使用 fp32，并且只在第一个 K tile 设置 clear_accum=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

clear_accum 条件必须同时考虑外层 kt 与内层 sk 的首轮。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

先测 2 stage，再测 3 stage；增加 stage 后若 L1 压力降低 occupancy，则保留较浅流水。

对每个候选运行编译、PTO 定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
