# GEMM 提前发射与深流水

## 适用条件

基础 2-stage 已正确，但生成代码中下一 K tile 的 copy 启动晚于可用时机。

## TileLang/PTO 实现流程

优先通过增加 T.Pipelined stage、调整 buffer versions 和循环嵌套表达预取。只有生成源码和 timeline 证明 copy 前移，才保留配置；不在 DSL 中模拟底层事件。

基础实现复用 examples/ascend/example_gemm.py：B 的物理布局为 [N,K]，调用 T.gemm(..., transpose_B=True)，L0C 使用 fp32，并且只在第一个 K tile 设置 clear_accum=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

预取版本不得覆盖仍被 T.gemm 消费的 L1 tile。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

以生成代码和 timeline 验证实际前移；若仅增加内存没有缩短 gap，则回退。

对每个候选运行编译、PTO 定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
