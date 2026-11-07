# Transpose 融合

## 适用场景

transpose 前后存在 elementwise、cast、scale 或量化操作，可消除中间 GM round-trip。

## PTO Kernel 结构

在 out_ub 写回前执行融合表达式；规则连续表达式放在 T.SimdVF。保持 transpose 索引与数值操作分层，factory 决定是否生成融合变体。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

严格保持 reference 的运算与 cast 顺序；分别验证仅 transpose、仅 elementwise 和融合结果。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

只有减少总 GM 字节且没有因寄存器/UB 压力造成回退时保留融合。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
