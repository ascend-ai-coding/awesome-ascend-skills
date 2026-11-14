# GatherElements small-GD tiled full-load 实现模式

## 1. 优化目标

优化 `PER_ELEMENT`（torch.gather/GatherElementsV2）中 `dim!=last`、gather 轴平面较少而 inner 很大的场景。按 inner 切片，将同一 tile 的全部 x gather 平面连续搬入 UB，再在 UB 内向量 Gather，把逐元素随机 GM 读转换为连续 DMA。

这是实验性实现模式，当前不提供可直接复制的通用 `.template`。已有实现若把 gatherDim 写成静态常量，只能作为单一 specialization 使用；参数化 `Gx/Gi` 并完成跨 shape 验证前，不得宣称通用。

## 2. 语义边界

保持 PER_ELEMENT 语义：

```text
y[o, g, c] = x[o, index[o, g, c], c]
```

不要直接套用 `gather_ga_all_load.template` 的 PER_SLICE/batch 假设。可以复用其 offset vector、负索引补偿和 UB Gather 范式，但 index 必须逐输出元素读取。

## 3. 候选条件

同时满足以下条件时才生成候选：

- `dim != last` 且 `innerSize > 1`。
- x/index 的非 gather 轴满足 torch.gather 合法 shape 关系。
- dtype、index dtype 和 UB `Gather` offset 类型受目标 SoC 支持。
- `Gx` 个 x 平面、`Gi` 个 index/y 平面的 inner tile 能按对齐要求装入 UB。
- `DataCopyPad` 的 `blockCount/blockLen/srcStride/dstStride` 均在 API 范围内。
- 连续搬运成本模型优于 direct random GM 候选。
- 任务数和每任务 tile 足以使用有效核数；极小输出应保留 SIMT 启动开销优势。

不要使用 case ID、`G==7`、64 MiB/128 MiB 输出大小等作为通用路由条件。

## 4. UB 与成本模型

双缓冲候选的每 inner 元素 UB 字节数：

```text
bytes_per_inner = 2 ×
  (Gx × xBytes + Gi × indexBytes + Gi × xBytes)

innerTile = align_down(
  (ubBytes - reservedBytes) / bytes_per_inner,
  vectorAlignmentElements)
```

必须用目标 SoC 查询得到的 UB 容量、实际 dtype 字节数和 API 对齐推导 tile。`reservedBytes` 包含框架、mask/offset 或额外 TBuf 预算；若实现原地复用 index buffer 作为 uint32 offset，要在文档中明确生命周期。

用流量模型筛选：

```text
direct_cost ≈ outputElements ×
  (indexBytes + xBytes + randomReadTransactionPenalty)

tiled_cost ≈ outerTasks × Gx × innerSize × xBytes
           + outputElements × (indexBytes + xBytes)
```

模型不能准确覆盖 DMA 启动、stride 效率、核间并行和 bank conflict，只负责拒绝显然不划算的候选；最终以 A/B 为准。

## 5. Kernel 数据流

```text
for task in (outer, innerTileId):
  1. DataCopyPad: x 的 Gx 个平面同一 inner tile → xLocal
  2. DataCopyPad: index 的 Gi 个平面同一 inner tile → indexLocal
  3. 对每个输出 gather 平面 g:
       负索引补偿
       byteOffset = (index * alignedXRowStride + column) * xBytes
       Gather(yLocal[g], xLocal, byteOffset)
  4. DataCopyPad: yLocal 的 Gi 个平面 → y
```

Host 和 device 必须分别传递 `Gx` 与 `Gi`：x 的搬入 blockCount 使用 `Gx`，index/y 和计算循环使用 `Gi`，索引合法范围使用 `Gx`。不要假设二者相等。

## 6. 切分与边界

- 按 `(outer, innerTileId)` 形成任务并分核，避免多个核重复加载完全相同的超大 tile。
- 主块和尾块分别计算 32B/VL 对齐 stride；尾块不能沿用主块的物理 row stride。
- offset 使用字节单位，确认最大 offset 不溢出 uint32；超出时回退支持宽地址的 SIMT/Scalar。
- 检查负索引语义和越界行为与框架合约一致。
- 多 batch/outer 场景必须使用 x 的独立 outer stride，不能用 index 输出步长替代。

## 7. A/B 与推广门槛

至少比较 small-GD、原 SIMT 和 Scalar；若 Transpose 能力门控通过，也纳入比较。统一 warmup/repeat，先验证精度，再比较完整算子耗时，最后跑完整用例集评分。

已有观测：一个 BF16/int32、`Gx=Gi=7`、大 inner 的 PER_ELEMENT case，原 SIMT 16.99 us，Scalar 388.69 us，small-GD tiled full-load 7.10 us。该数据证明“消除 x 随机 GM 读”可能产生显著收益，不构成 `G==7` 的通用阈值。

满足以下条件后才把实现升级为正式模板：

1. device kernel 已参数化 `Gx/Gi`，没有 case/shape 常量。
2. 覆盖 `Gx!=Gi`、多 outer、主尾 tile、负索引以及支持的 dtype/index 组合。
3. 代表 shape A/B 不劣于 SIMT/Scalar，完整用例集无不可解释回退。
4. 模板、Host tiling、TilingData 和 usage guide 成套提供。
