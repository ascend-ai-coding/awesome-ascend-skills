# GatherElementsV2 kernel 内 Transpose + Gather 模板

## 1. 范围与关键区别

本模板只描述 **单 kernel 内部** 的 `Transpose → Gather → ReTranspose`。它不是框架层的物理 Transpose 路径：

| 路径 | 组成 | 适用性 |
|------|------|--------|
| 物理 Transpose | Transpose(x) + Transpose(index) + 末轴 Gather + Transpose(y) | dtype 覆盖通常更广；可直接复用高性能 Transpose/Gather 算子 |
| kernel 内 Transpose | 1 个 kernel，workspace 中转三个阶段 | 仅在能力门控通过且 workspace 成本更低时使用 |

`dim != last` 只说明两者可以成为候选，不代表内部 Transpose 一定更快。

## 2. 当前模板能力边界

- 语义：PER_ELEMENT（torch.gather/GatherElementsV2）。
- 数据 dtype：fp16、bf16、fp32、int32，即 `TransDataTo5HD` 可用的 2B/4B 类型。
- index：device 模板按 int32 路径实现；int64 需在外层安全转换并验证范围。
- 不支持 int8 内部 Transpose。把 int8 临时提升为 int16 会增加转换、workspace 和搬运，不能作为无条件替代。
- `postDim > 32`，x/index gatherDim 均至少 16，合轴后必须可连续搬运。
- x 元素数小于 INT32_MAX，且 `xGatherDim * xDtypeSize < 120000`。

UB 容量只是必要条件，不是路径选择的充分条件。

## 3. 正确路由顺序

1. `dim == last`：走 LastDim。
2. `dim != last`：检查内部模板的 dtype、SoC、shape、合轴、postDim、gatherDim 和元素数能力。
3. 能力通过后执行 workspace 搬运成本门控。
4. 内部门控失败后，继续检查物理 Transpose 的平台能力与实测/模型收益；两者均通过才走物理 Transpose×3 + 末轴 Gather。
5. Transpose 不可用或 A/B 不占优：走 Scalar/SIMT。

可复用实现：[gather_elements_v2_route.template](gather_elements_v2_route.template)。它补齐了 device/tiling 模板之外缺失的 API 上游路由。

## 4. 成本门控

参考 ops-nn `aclnn_gather.cpp::MemCheck`，先计算：

```text
x      = [xPreDim,   xGatherDim,   xPostDim]
index  = [idxPreDim, idxGatherDim, idxPostDim]
coresPerGroup = max(1, coreNum / idxPreDim)
workspaceLen  = min(512 / xDtypeSize, ceil(idxPostDim / coresPerGroup))
```

当 `idxPreDim * idxPostDim <= coreNum`：

```text
xElementCount > idxPreDim * idxPostDim * (idxGatherDim + xGatherDim)
```

否则比较搬运代理量：

```text
direct = xElementCount*xDtypeSize + indexElementCount*indexDtypeSize
inside = indexElementCount*xDtypeSize
       + coreNum*workspaceLen*(idxGatherDim*indexDtypeSize + xGatherDim*xDtypeSize)

仅 direct > inside 时进入 kernel 内 Transpose。
```

该模型是路由门控，不代替最终实测。接入新平台后至少对边界 case 做 `physical/internal/scalar/SIMT` A/B。

## 5. Kernel 流程

1. `TransposeProcess`：`DataCopyPad(srcStride)` 跨步加载 x/index，`TransDataTo5HD` 在 UB 内转置，写 workspace。
2. `GatherProcess`：从 workspace 连续加载，使用 `Gather` 指令向量取值，结果写 workspace。
3. `ReTransposeProcess`：读回结果，`TransDataTo5HD` 转回原布局，stride DMA 写 y。

关键实现：

- [dav310/gather_elements_v2_transpose.template](dav310/gather_elements_v2_transpose.template)
- [gather_elements_v2_tiling.template](gather_elements_v2_tiling.template)

这两个文件与当前 ops-nn 内部 GatherElementsV2 Transpose 实现一致；外层路由原先未进入模板集，才是本次缺口。

## 6. case4 为什么没有优化

case4：x=`[8192,8192]` int32，index/output=`[4096,8192]`，dim=0。

- baseline SIMT：3581.81 us。
- 强制进入单-kernel内部 Transpose：3634.21 us，约慢 1.46%。
- 官方 CANNBench：1677.7 us，路径记录为物理 `Transpose×3 + GatherElementsV2`。

以 ops-nn 的 48 核成本代理计算：

```text
direct ≈ 8192*8192*4 + 4096*8192*4 = 402.7 MB
inside ≈ 4096*8192*4 + 48*128*(4096*4 + 8192*4) = 436.2 MB
```

`direct > inside` 不成立，因此内部 Transpose 应被拒绝。此前没有优化不是 device 模板缺少 `carryNumAlign`，而是把官方物理 Transpose 路径误判成了内部模板路径。

## 7. case6 检查结论

case6：x=`[16384,8192]` int8，index/output=`[8192,8192]`，dim=0。

- baseline SIMT：约 5.35 ms。
- 当前内部模板没有 Byte1 `TransDataTo5HD` 实现，实验分支实际未进入 Transpose。
- 官方路径记录为物理 `Transpose×3 + Gather`，约 2.10 ms。

因此 case6 应路由到物理 Transpose（平台可用时），不能把文档中的 2B/4B 内部模板宣称为 int8 优化方案。若要新增 Byte1 内部模板，必须作为新实现单独做精度、编译和全量 A/B，不能只改 dtype 门控。

## 8. 禁止的选型方式

- `dim != last` 就进入内部 Transpose。
- 只检查 128KB transpose buffer/UB 是否放得下。
- 用 cache-line 浪费的理论上界断言 Transpose“恒优”。
- 把物理 Transpose 的官方耗时归因到单-kernel内部模板。
- 在没有 Byte1 device 实现时让 int8 进入内部路径。

## 9. 验证清单

- 路由单测覆盖：last dim、内部可用、成本拒绝、dtype 拒绝、物理 Transpose 不可用。
- case4 必须命中 `PHYSICAL_TRANSPOSE`，不能命中 `INTERNAL_TRANSPOSE`。
- case6 必须因 int8 拒绝内部模板；物理路径能力与成本门控通过时命中 `PHYSICAL_TRANSPOSE`。
- 对所有受影响 case 做精度回归和 CANNBench A/B；路由变化必须记录实际 kernel 序列。
