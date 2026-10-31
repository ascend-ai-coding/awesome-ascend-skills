# N-last Layout Conversion

## 适用场景

把指定维移动到最后或从最后维移出，适合先合并其它连续维。

## PTO Kernel 结构

factory 将输入归一化为 outer×move×inner；inner 连续时使用矩形 T.copy，move/inner 交换使用 SIMD gather。无法安全表达的字节索引回退 SIMT。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

验证 rank、负 axis、size=1 维和非连续 stride。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

按 inner 长度选择向量或 SIMT，比较维度合并前后的地址计算与 GM burst。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
