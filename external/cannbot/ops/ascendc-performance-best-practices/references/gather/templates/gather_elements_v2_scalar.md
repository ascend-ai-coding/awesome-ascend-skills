# GatherElementsV2 Scalar 模板

## 1. 模板用途

**Transpose 和 SIMT 之间的候选路径**。index/y 用 DMA 批量搬运到 UB，x 从 GM `GetValue` 逐元素标量读。部分 ops-nn 实现会在 Transpose UB 不够时进入 Scalar，但移植或重新优化时必须把它与 SIMT、small-GD 等路径做 A/B，不能继承为无条件默认选择。

## 2. 适用条件

| 条件 | 要求 |
|------|------|
| 候选条件 | Transpose 模板 UB 不够或成本模型拒绝，且 index/y 连续 DMA 的节省可能覆盖 x 标量随机读成本 |
| gatherMode | PER_ELEMENT（torch.gather / GatherElementsV2） |
| dim | 任意（不限末轴） |
| dtype | 通用（标量读无 dtype 限制，支持 int64） |

## 3. 与 SIMT 的区别（关键）

| 维度 | SIMT 路径 | Scalar 路径 |
|------|----------|------------|
| index 读取 | VEC 流水线逐元素 GM 读 | **DMA 批量搬运到 UB**（连续读，η≈1.0） |
| x 读取 | VEC 流水线逐元素 GM 随机读 | GM `GetValue` 标量随机读（SCALAR 管线） |
| y 写入 | VEC 流水线逐元素 GM 写 | **DMA 批量写回**（连续写，η≈1.0） |
| UB 使用 | 无 | index tile + y tile（不含 x，UB 需求小） |
| 固定开销 | 无 UB/DMA 启动 | 无 workspace/转置，但有 DMA 和标量循环开销 |
| 适用场景 | 大规模散点读（2048 线程并行掩盖延迟） | Transpose UB 不够时的中间路径 |

> **关键**：Scalar 只消除了 index/y 的逐元素 GM 访问，最昂贵的 x 随机 GM 读取仍然存在，而且落在 SCALAR 管线。Transpose UB 不够只能说明 Scalar“可用”，不能说明它“更快”。必须在同一构建、warmup、repeat 口径下与 SIMT A/B；若 gather 轴小且 x 平面可按 inner tile 连续装入 UB，还要加入 [small-GD tiled full-load](gather_elements_small_gd_tiled.md) 候选。

## 4. 执行流程

```
Device (每核):
  对每个 preDim × postDimBlock:
    CopyInIdx: DataCopyPad(连续读 index tile from GM → UB)
    Compute:
      for each idxPos in tile:
        xGatherPos = idxLocal.GetValue(idxPos)    // 标量读 index (from UB)
        xGatherPos = (xGatherPos + xGatherDim) % xGatherDim  // 负索引补偿
        xValue = xGm[xBaseOffset + xGatherPos * xPostDim + idxPostPos].GetValue(0)  // GM 标量读 x
        yLocal.SetValue(idxPos, xValue)           // 标量写 UB
    CopyOutY: DataCopyPad(UB → GM)
```

## 5. 关键技术

| 技术 | 要点 |
|------|------|
| **index tile 驻留 UB** | index 按 `maxIdxDataAlign` 分块加载到 UB，避免逐元素 GM 读 index |
| **标量 GM 读 x** | `xGm[offset].GetValue(0)` 逐元素标量读，SCALAR 管线执行 |
| **负索引补偿** | `(idx + xGatherDim) % xGatherDim`，标量模运算 |
| **批量 CopyOut** | y 先写 UB，再 DataCopyPad 批量写回 GM，减少 MTE3 次数 |
| **核间分组** | 按 preDim 分组 + 组内按 idxData 切分，支持多核并行 |

## 6. TilingData 字段

| 字段 | 含义 |
|------|------|
| `formerGroupFormerData` / `formerGroupTailData` | 大组中大小核的每核数据量 |
| `tailGroupFormerData` / `tailGroupTailData` | 小组中大小核的每核数据量 |
| `maxIdxDataAlign` | index tile 最大对齐元素数（由 UB 容量计算） |

## 7. 关键代码索引

- 模板源码：[dav310/gather_elements_v2_scalar.template](dav310/gather_elements_v2_scalar.template)
  - `CopyInIdx`：index tile GM → UB
  - `Compute`：逐元素 GetValue/SetValue 标量 gather
  - `CopyOutY`：y tile UB → GM
- 公共基类：[dav310/gather_elements_v2_common.h](dav310/gather_elements_v2_common.h)
- Host tiling：[gather_elements_v2_tiling.template](gather_elements_v2_tiling.template)
  - `Tiling4Scalar()`：核间分组 + maxIdxDataAlign 计算
  - `CalcMaxIdxDataAlign()`：`maxIdxDataAlign = availableUb / BLOCK_SIZE * BLOCK_SIZE / (sizeof(int32) + dtypeSize)`

## 8. 不适用场景与替代

| 场景 | 推荐替代 |
|------|---------|
| UB 够走 Transpose | [Transpose + Gather](gather_elements_v2_transpose.md) |
| dim == last + GD 全载 UB | [LastDim](gather_elements_v2_last_dim.md) 或 [SIMD 末轴 Gather](simd_last_gather.md) |
| dim != last + small-GD tile 可装入 UB | [small-GD tiled full-load](gather_elements_small_gd_tiled.md)，优先消除 x 随机 GM 读 |
| 大规模散点读 + SIMT 线程并行更优 | [SIMT](simt_gather.md)（2048 线程并行掩盖延迟） |

## 9. A/B 决策与已知反例

执行以下检查，不要根据单一理论流量直接启用 Scalar：

1. 用 profiling 确认 Scalar 的 SCALAR/SCALARLDST 是否因 x `GetValue` 成为主导。
2. 在同一 case 上至少比较原 SIMT 与 Scalar；目标是完整算子时间，不是仅比较 index/y 搬运段。
3. 对候选路由运行完整用例集，按总体评分选择，避免只优化一个 case 后引入其他 shape 回退。
4. 不写 case ID、固定输出 MiB 阈值或单一 shape 特判。

已观测反例：某 PER_ELEMENT BF16/int32、`Gx=Gi=7`、大 inner 的 case，SIMT 为 16.99 us，Scalar 为 388.69 us；将全部 7 个 x 平面的 inner tile 搬入 UB 后为 7.10 us。这个结果只证明路径选择规则，不应把 `7` 或对应 shape 写成通用门槛。
