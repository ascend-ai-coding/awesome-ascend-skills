# 双轴切分 Transpose

## 适用场景

两个交换轴都不能完整驻留 UB。

## PTO Kernel 结构

使用二维 tile grid，输入 x_ub 为 block_x×padded_y，输出 out_ub 为 block_y×padded_x。索引向量在 SIMD 区域外按静态 block 大小生成并复用。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

验证四角尾块和两个轴同时非整除。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

在 UB 容量内搜索 block_x/block_y，避免极瘦 tile 降低 DMA 和 SIMD 利用率。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
