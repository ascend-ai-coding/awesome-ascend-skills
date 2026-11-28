# Transpose 配置选择

## 适用场景

用 Python kernel factory 生成可缓存的静态配置，不引入额外的 host 配置结构。

## PTO Kernel 结构

输入为 shape 余数类、dtype、stride 类别和 layout；输出为 block_x、block_y、num_cores、execution domain 和 stage。动态维只保留在 T.dynamic，硬约束通过 T.assume 表达。

当前仓库的 `testing/ascend/layout/test_ascend_l0_transpose.py` 可用于核对 L0 转置语义，`tilelang/ascend/language/copy_op.py` 与 `src/ascend/op/copy.cc` 用于核对搬运和布局限制；它们不等于通用批量转置 kernel。复用当前算子或新实现时，明确任务分配、stride、dtype、尾块和 UB footprint。技能中的转置适配器需显式传入当前仓库已验证的 kernel factory。

## 精度要求

每个 dispatch 分界两侧都做精度测试，fallback 覆盖所有合法输入。

按当前公开接口覆盖 dtype、batch、stride、允许的对齐与余数类及非连续输入；接口承诺通用尾块时覆盖 tile-1/tile/tile+1。T.assume 只能表达调用方保证；输出与 torch.permute/transpose reference 比较，并使用 canary 检查越界。

## 性能要求

离线枚举候选并在代表 shape 实测；控制变体数量和编译缓存占用。

先通过 PTO 定向精度测试，再以相同条件比较 DMA、SIMD gather 和 SIMT fallback。记录 latency、有效 GM 带宽、标量索引开销、UB 占用和生成代码长度；未实测不得宣称某路径更快。
