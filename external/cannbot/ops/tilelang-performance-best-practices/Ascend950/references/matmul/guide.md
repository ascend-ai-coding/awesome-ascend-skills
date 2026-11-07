# PTO GEMM 优化索引

## 适用条件

bf16/fp16/fp32 与 block-scaled GEMM。先建立正确 Cube 基线，再按瓶颈选择 tile、流水、驻留或 shape 专用调度。

## TileLang/PTO 实现流程

Python factory 枚举 BM/BN/BK、MAD tile、核数和 stage；约束 L0A/L0B/L0C/L1 容量与 shape 可整除性。普通输出 tile 使用 T.AscendTileScheduler；动态或尾块落入正确 fallback。

基础实现复用 examples/ascend/example_gemm.py：B 的物理布局为 [N,K]，调用 T.gemm(..., transpose_B=True)，L0C 使用 fp32，并且只在第一个 K tile 设置 clear_accum=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

任何 GEMM 路径只有在当前 TileLang/当前仓库版本完成 lowering、有限值检查和目标容差验证后才能作为优化基线；历史现象若无提交、命令和原始日志，不写入结论。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

依次比较基础、2/3 stage、full-load、L2 控制和 shape specialization；一次只改变一个变量。

对每个候选运行编译、PTO 定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
