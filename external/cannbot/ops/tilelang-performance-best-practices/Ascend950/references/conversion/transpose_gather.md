# SIMD Gather Transpose

## 适用场景

dtype 为 16/32 bit，源索引可在 PTO SIMD index vector 范围内表达。

## PTO Kernel 结构

用 T.simd.vci 构造 lane offset，乘 padded stride 后加列偏移，reinterpret 为无符号 index vector，再用 T.simd.vgather2 和 T.simd.vsts。每种 vector_id 在静态 T.Unroll 中生成。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

索引 dtype 必须覆盖最大 UB 偏移；mask 与 dtype lane 数一致。尾部未使用 lane 不得写回 GM。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

检查生成代码中的 gather 数、索引重算和 unroll 体积；索引准备应在列循环外复用。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
