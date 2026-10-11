# GEMM 一侧驻留

## 适用条件

A 或 B 的完整可复用 panel 连同 scale 可放入 L1，并被多个对侧输出 tile 复用。

## TileLang/PTO 实现流程

把 resident panel 的 T.copy 移到输出 tile 遍历外；流式侧仍按 K tile 搬运。scheduler 的遍历方向应让复用侧在连续任务间保持不变。L1 预算包含 resident panel、scale、流式 buffer versions 和安全余量。

基础实现复用 examples/ascend/example_gemm.py：B 的物理布局为 [N,K]，调用 T.gemm(..., transpose_B=True)，L0C 使用 fp32，并且只在第一个 K tile 设置 clear_accum=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

量化 scale 的索引、block size 和应用顺序必须与 blockscaled reference 一致。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

比较节省的 GM bytes 与 resident 初始化成本；重复次数不足两次不启用。

对每个候选运行编译、PTO 定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
