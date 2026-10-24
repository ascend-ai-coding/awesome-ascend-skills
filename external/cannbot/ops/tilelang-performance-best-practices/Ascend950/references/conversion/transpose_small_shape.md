# 小 Shape Transpose

## 适用场景

独立 tile 很少、总字节数较小，启动和索引开销占主导。

## PTO Kernel 结构

使用 min(向量核数, tile 数)；能放入单个 UB tile 时一次 T.copy 搬入、一次向量/SIMT 转置、一次写回，不启用多 stage。按 dtype 和余数类在 factory 中生成少量静态变体。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

覆盖单元素、单行、单列、64/128 边界和 batch=1。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

比较单核与少核；避免启动大量空核和为小 shape 构造复杂索引表。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
