# Transpose 与 Layout Conversion

## 适用场景

根据连续性、dtype 字节数、tile footprint 和目标 layout 选择 DMA、SIMD gather 或 SIMT。

## PTO Kernel 结构

先合并相邻连续维并选择 block_x/block_y。输入可用 T.StridedTensor 表达动态 stride；输出使用连续 T.Tensor。常见二维交换直接复用基线 kernel，复杂 layout 通过 factory 生成静态索引变体。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

转换不能改变元素值或重复/遗漏元素；低精度与整数执行逐元素精确比较，浮点仅在同时发生数值转换时使用项目既有容差。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

优先减少 GM transaction 和地址计算；对小 shape 控制 launch/初始化开销，对大 shape 优化有效带宽。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
