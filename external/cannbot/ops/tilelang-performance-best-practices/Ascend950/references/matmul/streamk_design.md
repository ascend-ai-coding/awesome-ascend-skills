# PTO Stream-K 两阶段实现

## 适用条件

普通 M/N tile 数显著少于 Cube 核数且 K 很长，常规 tile 调整仍无法提供并行度。

## TileLang/PTO 实现流程

Kernel A 按 output-tile×K-partition 生成 fp32 partial 到 workspace；Kernel B 让每个输出 tile 由唯一 owner 以确定顺序归约 partial 并转换输出。两个 kernel 由 Python host wrapper 顺序调用，不依赖未验证的 grid barrier 或 atomic 累加。

基础实现复用 examples/ascend/example_gemm.py：B 的物理布局为 [N,K]，调用 T.gemm(..., transpose_B=True)，L0C 使用 fp32，并且只在第一个 K tile 设置 clear_accum=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

workspace 和最终累加均为 fp32；对不同 partition 数单独评估加法顺序误差。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

端到端统计两个 launch 和 workspace GM 流量。只有完整路径胜过普通 GEMM 才 dispatch，不能只比较 Kernel A。

对每个候选运行编译、PTO 定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
