# Conversion 数据转换类 — 兜底算法

> Conversion 族（Transpose, Concat, Split）的兜底 Tiling 算法。Transpose 已收录参考策略；Concat/Split 待补充。

## 适用算子

Transpose, Concat, Split

## Transpose 及重排类算子 — 已收录

本目录的 [tiling-flow.md](tiling-flow.md) 与 [tiling-fields.md](tiling-fields.md) 覆盖 **Small-Channel Transpose**（C ≤ 16，建模为 [C, N] → [N, C]）的 Tiling 推导与字段语义。

其余 Transpose 场景（大通道、多轴切分、021、5HD、gather、tensor_move、small_shape 等），以及同族重排算子（BatchToSpace / SpaceToBatch / DepthToSpace）的完整 tiling 策略选择，参考../index.md：

> **性能优化参考流程**：先按 `../index.md` 的策略速查表选定策略；命中 NDDMA 家族（TENSOR_MOVE / SMALL_SHAPE / CUT_ONCE / CUT_TWICE / N_LAST / BIG_DIM）或 small-channel 场景时，回到本目录的 tiling-flow.md 做 TilingData 参数推导。
>
> **BatchToSpace / SpaceToBatch / DepthToSpace 的参考范围**：这类算子本质是带 stride 的多维 DMA 重排，与 transpose 的 NDDMA 家族（CUT_ONCE/CUT_TWICE/BIG_DIM/N_LAST）同构，可原理参考其切分轴选择与多维 DataCopy 地址映射；TENSOR_MOVE 不适用（重排算子必有维度重排），VCONV/GATHER/SMALL_SHAPE/small-channel 需视具体 perm/block_shape 判定。

## Concat / Split — 待补充

- 多核切分策略（Concat 按输入段 / Split 按输出段）
- 单核切分策略（block 为单位搬移，考虑跨 stride 对齐）
- Buffer 规划
- 分支覆盖
