# Block-scale 合并载入

## 适用条件

scale/bias/LUT 沿 K 连续且单次片段过小，copy 固定开销明显。

## TileLang/PTO 实现流程

扩大 scale 的 L1 tile，一次搬入多个 K iteration 对应的数据；内层按静态 offset 选择当前 scale 子片。可直接参考 examples/ascend/example_blockscaled_gemm*.py 的 xsf_l1/wsf_l1 和 SF_LOAD_CHUNK_SIZE。

基础实现复用 examples/ascend/example_gemm.py：B 的物理布局为 [N,K]，调用 T.gemm(..., transpose_B=True)，L0C 使用 fp32，并且只在第一个 K tile 设置 clear_accum=True。输出 tile 的核数不超过独立任务数；M/N/K 尾块必须走已验证的 padded-copy 或专用 fallback。

## 精度门禁

每个 K block 的 scale 必须与 operand block 精确对齐，覆盖 K 尾部和不同 scale dtype。

所有输入 dtype 与输出 dtype 分开测试。覆盖 M/N/K 的 tile±1、长 K 消除误差、正负抵消、大小量混合、0、NaN/Inf 契约。任何 K 分片的 partial 和最终归约保持 fp32；不得以放宽容差掩盖累加顺序或输出转换错误。

## 性能门禁

扫描合并倍率，记录 MTE2 指令数、有效带宽和 L1 占用；不使用固定 20KB 阈值代替实测。

对每个候选运行编译、PTO 定向精度测试和统一 benchmark。报告 latency、TFLOPS、Cube/MTE2 时间、GM 字节、L1/L0/UB 占用、核利用率和生成代码体积。只有完整算子端到端更快且没有精度回退时进入 dispatch。
