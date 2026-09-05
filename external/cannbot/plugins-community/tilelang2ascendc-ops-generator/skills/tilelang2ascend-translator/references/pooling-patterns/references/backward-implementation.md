# Backward Implementation: Pooling 反向梯度 AscendC 实现（陷阱与高性能落地）

> **定位**：反向 pooling（AvgPoolGrad / MaxPoolGrad / AdaptivePoolGrad）的 **AscendC 实现篇**，即
> `ops/tilelang-op-design/references/pooling/backward-patterns.md`（设计语义部分，§1-§8）的 §11/§12 拆出内容。
> 阅读前先读 backward-patterns.md 的语义设计（scatter-add、闭式公式、divisor、布局、ArgMax、多核）。
> 反向落地增量踩坑（核数/Axpy/Cast/分派/延迟分桶）另见 [grad-v2-lessons.md](grad-v2-lessons.md)。

## 11. AscendC 实现陷阱（avg_pool3d_grad 实测踩坑）

### 11.1 WAR 竞争：CopyOut DataCopyPad 必须尾随 PipeBarrier<PIPE_ALL>

gather 策略里，每个输入行 `(id,ih)` 的梯度累加复用同一个 `acc_` buffer：循环内先 `Duplicate(acc_, 0.0f)`（VEC 写清零）→ 累加窗口 → `CopyOut` 时用 `DataCopyPad(MTE2 读 acc_)` 把累加结果搬到 GM。

若 `CopyOut` 的 `DataCopyPad`（MTE2 读）之后**没有** `PipeBarrier<PIPE_ALL>`，下一行的 `Duplicate(acc_, 0.0f)`（VEC 写）可能与 MTE2 读 `acc_` 重叠，导致读到的还是上一行残留或清零态。avg_pool3d_grad 实际表现是「每个 block 首个处理行输出被清零」的 grid-stride 模式（如 rows 0-23 全 0、rows 24-63 正确），`matched_ratio` 掉到 0.0006~0.5 但 shape 正确、不崩——典型的 WAR 竞争而非算法错。

**规则**：`CopyOut` 里只要出现 `DataCopyPad` 读 `acc_`，其下一个会写 `acc_` 的指令前必须插入 `PipeBarrier<PIPE_ALL>()`。定位方法：all-ones grads 单 case dump，观察梯度分布是否呈「按 block 步进的网格清零」模式。

### 11.2 性能标杆必须选 ops-nn，不是 torch.autograd

反向算子的标杆若用 `torch.autograd`（`F.avg_pool3d` 求导 + 前向重建 + 图开销）会**严重低估参考延迟**，从而虚高加速比——avg_pool3d_grad 首轮曾据此报出「2.10x」的假象，实测对标 ops-nn（`torch_npu` 的 `F.avg_pool3d` 反向 = `aclnnAvgPool3dBackward` = ops-nn `avg_pool3_d_grad`）仅 0.4~0.6x。反向算子只在 kernel 层比「单前向」或「前向+autograd」都不可比，**标杆必须同为 kernel 级反向实现**（ops-nn 的 `*_grad` 算子）。

### 11.3 大 kernel 反向窗口是结构性性能瓶颈——但 gather 的串行是可消除的

`kernel=8` 时 input-driven gather 的每输入位置需串行 gather 8³=512 个输出窗口（逐元素 GM 读 + PipeBarrier），同步/依赖链主导，实测只有 ops-nn 的 0.4~0.6x。但注意：**瓶颈是 gather 算法本身的逐元素 GM 往返，不是反向 scatter-add 的数学下界**——ops-nn 用 output-driven transpose-scatter（§12）把 512 次串行 GM 读换成一次块读 + UB 内向量化累加，同样的大 kernel 也能快。所以「大 kernel 慢」的结论只在 input-driven gather 实现下成立，换架构可解。

## 12. 高性能落地：output-driven transpose-scatter（对标 ops-nn avg_pool3_d_grad）

官方 ops-nn 的 `avg_pool3_d_grad` 是反向 pooling 的性能范式，其架构与 input-driven gather 截然不同。核心洞察：**把「每个输出位置散到它覆盖的输入位置」这个 scatter 操作，用块读 + transpose + UB 内向量化 Add 完成，把逐元素 GM 往返压成一次块读 + 一次块写**。

### 12.1 算法流程（`AvgPool3DGradNormal::CalcBlock`）

对每个输出分块 `(doBlock, hoBlock, woBlock)`（覆盖 `baseWo` 个 W 输出位置 × `singleCoreNc` 个 NC 通道）：

1. **块读**：一次 `DataCopyPad` 读整块 `[baseWo × singleCoreNc]` grads（`blockCount=ncShape`，`blockLen=baseWo*sizeof(T)`，`srcStride=outDHW-baseWo`），而非逐窗口元素读。
2. **转置**：`TransposeBase16M8`（fp32，row:align16/col:align8）或 `TransposeBase16M16`（fp16/bf16）把 `[W × NC]` 转成 `[NC × W]`，让 NC 维连续可向量化。
3. **UB 内 scatter 累加**：`Duplicate(outputGradUb, 0)` 清零累加 buffer；对每个 `w`：`Muls(TranUb, mulsFactor)`（除 divisor）→ 对 `k ∈ [wStart-blockWStart, wEnd-blockWStart)` 做 `Add(outputGradUb[k*NC], outputGradUb[k*NC], TranUb)`——多个输出位置散到同一输入位置 k 时在 UB 内累加，**零 GM 往返**。
4. **转置回**：`TransposeBase8M16` 把累加结果转回，`CopyOut` 用跨步 `DataCopyPad`（`blockCount=ncShape`，`dstStride=inDHW-window`）一次写回输入梯度。

### 12.2 窗口重叠 → SetAtomicAdd + workspace

当 `stride < kernel`（重叠）时，不同输出 block 会散到同一输入位置，跨 block 无法在 UB 内闭合。官方用 `SetAtomicAdd<float>()` + fp32 workspace 做跨 block 原子累加；fp16/bf16 时先原子累加到 fp32 workspace，最后走独立 cast pass（`ProcessCast`）把 fp32 → fp16/bf16 写回。`SetAtomicAdd` 是硬件加速的，代价远小于逐元素 GM 读。

### 12.3 六路 tiling key 分派（dtype × layout × kernel）

官方 host 侧按 `SetTilingKey` 分 6 路，kernel 侧 `TILING_KEY_IS` 分派：

| key | 类 | 适用 | 说明 |
|---|---|---|---|
| 1000 | Cast | NDHWC + fp16/bf16 | cast 累加 |
| 2000 | NoCast | NDHWC + fp32 | 直接累加 |
| 3000 | OnlyT fp32 | NCDHW + kH==kW==1 | 深度-only 快路径（反向 reduce_d） |
| 4000 | OnlyT bf16 | NCDHW + kH==kW==1 + bf16 | 同上 |
| 5000 | Normal | NCDHW 通用 | transpose-scatter（§12.1） |
| 6000 | Scatter | UB 连单窗口都放不下 | 标量兜底 |

### 12.4 only_t 快路径（反向 reduce_d 对偶）

`kH==kW==1 && dH==dW==1` 时 H/W 是恒等映射，反向退化为纯 D 维累加：`gx[n,c,id,:,:] = Σ grad[n,c,od,:,:]/div(od)`，无需 transpose，直接 `[C×D]` 向量累加（`KernelAvgPool3DGradBaseT`）。这是前向 `reduce_d` 快路径（`ops/tilelang-op-design/references/pooling/reduce-d-fastpath.md`）的**反向对偶**。

### 12.5 gather vs scatter 选择决策（修正版）

| 场景 | 选择 | 理由 |
|---|---|---|
| 窗口无重叠（stride≥kernel）且 kernel 小 | input-driven gather 或 scatter 均可 | gather 代码简单、零原子 |
| 窗口重叠（stride<kernel） | **output-driven transpose-scatter** + SetAtomicAdd | 原子代价 < 串行 GM 往返 |
| kernel 大（如 8³） | **output-driven transpose-scatter** | gather 的 512 次串行 GM 读是灾难 |
| kH==kW==1 | only_t 快路径（无 transpose） | D 维纯累加 |
| UB 放不下单窗口 | 标量 scatter 兜底 | 保底正确性 |

**一句话**：反向 pooling 的性能上限由「GM 往返次数」决定，不由「原子操作」决定。优先把 scatter-add 收进 UB（块读 + transpose + 向量化 Add + 块写），重叠跨 block 才用 SetAtomicAdd。

### 12.6 host 侧 launch 开销：NCDHW 直通 + 内核清零（小 tensor 的最大单一收益）

对小 tensor（单 kernel ~0.3us），wall-clock 由 **host 侧 launch 次数**主导，不是 kernel 计算。avg_pool3d_grad 实测：host 侧 `permute`(NCDHW→NDHWC) + `at::zeros`(输出清零) + kernel + `permute`(回来) ≈ 5 次 launch，把 wall-clock 压在 0.40x；改成 **NCDHW 直通**（kernel 内按 `data_format` 直接读原生 NCDHW/NDHWC）+ **内核清零**（`Duplicate`+`DataCopyPad`+`SyncAll`，输出用 `at::empty` 而非 `at::zeros`）后降到 1 次 launch，wall-clock 0.40x → 0.71x（~1.7x）。**这个收益比 scatter 算法重写本身还大。**

规则：
- 反向 pooling 输出恒 NCDHW，但 kernel 可按 `data_format` 直接读 NCDHW/NDHWC，**不必在 host 强制 permute 成 NDHWC**——C 维连续性用 `DataCopyPad` 的 `blockCount=C, blockLen=sizeof(T)` 广播替代（每 C 值 pad 到 32B，免显式 Broadcast）。
- 输出清零移入 kernel，host 用 `at::empty` 省一次 device fill（`Duplicate`+`DataCopyPad`+`SyncAll`）。
- fp16/bf16 overlap 的 fp32 workspace 只在需要时 `at::empty` 分配，非 overlap 路径零额外 launch。
- `ceil_mode` 在反向是 no-op（已烘焙进前向输出 shape），host 不转发给 kernel。
