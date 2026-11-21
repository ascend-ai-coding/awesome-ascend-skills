# GatherElementsV2 LastDim 模板

## 1. 模板用途

针对 **torch.gather dim==last**（末轴 gather，innerSize==1）场景的高效实现。x 的一行（gatherDimSize 个元素）加载到 UB，用 `Gather` 指令向量化取值，无需 Transpose。

## 2. 适用条件

| 条件 | 要求 |
|------|------|
| dim | == last（末轴 gather，innerSize==1） |
| gatherMode | PER_ELEMENT（torch.gather / GatherElementsV2） |
| dtype | fp16/bf16/fp32/int32（Gather 指令支持） |
| 路由 | x 行、index、y 的 UB 布局合法，且 resident 成本模型筛选通过；最终经 A/B 验证 |

> dim!=last 时把 [Transpose + Gather](gather_elements_v2_transpose.md)、[small-GD tiled full-load](gather_elements_small_gd_tiled.md) 和 Scalar/SIMT 作为候选，不按 dim 单条件决定路径。

## 3. 执行流程

```
Device (每核):
  三种子模式:
    ProcessScalarMode: scalarMode==1 时, 标量逐元素 (小规模兜底)
    ProcessMultRow: eachCalculationLines>1 时, 多行并行 gather
    ProcessSingleRow: 默认, 单行 gather

  ProcessSingleRow / ProcessMultRow:
    对每个 preDim × postDimBlock:
      CopyInX: DataCopyPad(连续读 x 行 from GM → UB, GD 个元素)
      CopyInIdx: DataCopyPad(连续读 index 行 from GM → UB)
      Compute:
        Muls(idxLocal, idxLocal, sizeof(T_X))  // 索引转字节偏移
        Gather(yLocal, xLocal, offsetLocal, 0, count)  // UB 内向量化 gather
      CopyOutY: DataCopyPad(UB → GM)
```

## 4. 关键技术

| 技术 | 要点 |
|------|------|
| **Gather 指令** | `Gather(yLocal, xLocal, offsetLocal, 0, count)` UB 内按索引向量化 gather，一条指令处理 count 个元素 |
| **x 行常驻 UB** | gatherDimSize 个元素一次载入后全程复用 |
| **多行并行** | `eachCalculationLines > 1` 时一次处理多行 index，提升向量核利用率 |
| **负索引补偿** | 与 Transpose 模板相同（ShiftRight + Muls + Add） |

## 5. 关键代码索引

- 模板源码：[dav310/gather_elements_v2_last_dim.template](dav310/gather_elements_v2_last_dim.template)
  - `ProcessSingleRow` / `ProcessMultRow` / `ProcessScalarMode`：三种子模式
  - `Gather` 指令调用：与 Transpose 模板的 `GatherInUb` 相同
- Host tiling：[gather_elements_v2_last_dim_tiling.template](gather_elements_v2_last_dim_tiling.template)

## 6. Host 成本门控与 shape 语义

先按 SoC UB、对齐和 buffer 布局生成可行 tiling，再用下式筛选候选：

```text
resident_cost = output_rows ×
  (x_last_dim × xBytes + index_last_dim × (indexBytes + xBytes))

direct_cost = output_elements ×
  (indexBytes + xBytes + random_read_transaction_penalty)
```

成本模型只用于筛选，不能替代代表 case A/B。不要使用 case ID、64 MiB/128 MiB 等固定输出阈值。

torch.gather 允许非 gather 轴 `index.shape[i] <= x.shape[i]`，因此适配 Host 模板时必须：

- 按 index 的非 gather 轴乘积计算 `output_rows` 和使用核数。
- 分别保存 x/index shape 与 stride，按 x stride 取源、按 index stride 取 index/写 y。
- 仅在对应非 gather 轴确实相等时合轴；不要用 x shape 代替 index shape。

## 7. 不适用场景与替代

| 场景 | 推荐替代 |
|------|---------|
| dim != last | 比较 [Transpose + Gather](gather_elements_v2_transpose.md)、[small-GD tiled full-load](gather_elements_small_gd_tiled.md)、Scalar/SIMT |
| UB 不够 | [Scalar](gather_elements_v2_scalar.md) 或 [SIMT](simt_gather.md) |
| resident 成本不占优 | 保留 SIMT，或在完整用例集 A/B 后选择其他路径 |
