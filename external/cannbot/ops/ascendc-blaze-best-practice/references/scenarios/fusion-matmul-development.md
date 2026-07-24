# C+V 融合 MatMul 开发指南

> 适用场景：MatMul + Vector Epilogue 融合，即 `C = epilogue(A x B, extra...)`。
>
> C+V 的开发分两层：先确定 C 部分的 MatMul 主体和 L0C2UB 能力，再设计 V 部分的 Epilogue。不同 MatMul 场景的具体组件不同，但开发方法一致。

---

## §1 C+V 融合模型

C+V 融合是 Cube + Vector 协同模型：AIC 做 MatMul，Fixpipe 将 L0C 写入 UB，AIC 通过 CrossCore flag 通知 AIV；AIV 从 UB 读取 MatMul 结果，加载额外输入，执行 Vector Epilogue 并写回 GM；AIV 完成后再通知 AIC 释放 UB。

融合 kernel 使用 `__mix__(1, 2)`：1 AIC + 2 AIV 协作。AIV 不提前返回，不能沿用纯 MatMul `__cube__` + `if (ASCEND_IS_AIV) return` 的模板。

---

## §2 设计流程总览

| 层次 | 设计目标 | 关键问题 |
|------|----------|----------|
| C 部分 | 选择 MatMul 主体组件 | 当前 MatMul 场景是什么？对应 Block 是否具备 L0C2UB 能力？ |
| V 部分 | 设计 Epilogue | Vector 公式适合 MemBase 还是 RegBase？剩余 UB 如何分配？同步依赖是否完整？ |
| Tiling | 复用 C 部分 tiling | C+V 不新增独立 vector tiling engine，Epilogue 只消费 Cube tiling 后的剩余 UB |
| Launcher | 组织输入和参数 | MatMul 输入、Epilogue 额外输入、TilingData、layout/trans 分发是否一致？ |

---

## §3 C 部分设计

### 3.1 选择 MatMul 场景

| C 部分场景 | Kernel | BlockMmad | Scheduler | Tiling | 说明 |
|------------|--------|-----------|-----------|--------|------|
| 普通 MatMul + V | `Kernel::MatmulKernelFused` | `Block::BlockMmad` | `MatmulSwatScheduler` | `MatmulTilingSwat` | 完整 `blaze_custom` 路径 |
| MXFP8/MXFP4 MatMul + V | `Kernel::MxMatmulKernelFused` | `Blaze::Gemm::Block::BlockMmad` MX | `BlockSchedulerQuantBatchMatmulV3` | `QuantMatmulTilingSwat` | 受控组合态：MX C 部分复用 blaze library，Epilogue 由 custom bridge 承载 |
| Grouped MatMul + V | `Kernel::GroupMatmulKernel<..., Epilogue>` | `Block::BlockMmad` | `GroupMatmulBlockSchedulerSplitM` | 复用对应非 grouped `MatmulTilingSwat` | Epilogue 必须使用 context-based view |

纯 `GemmUniversal` 是纯 MatMul 路径，不承载自定义 Vector Epilogue。需要 Vector Epilogue 时，必须选择能把 L0C 结果写到 UB 并与 AIV 同步的 Kernel / Block 组合。

### 3.2 L0C2UB 能力判断

C+V 不是纯 MatMul 后面硬接一段 Vector。C 部分必须提供以下契约：

| 判断项 | 要求 |
|--------|------|
| TensorC 位置 | BlockMmad 支持 TensorC Location=UB |
| L0C→UB 搬运 | 使用 `CopyL0C2UB` + `CopyL0C2UBSplitMTrait`（`DUAL_DST_SPLIT_M`） |
| Trait 选择 | `__mix__(1,2)` 标准 C+V 必须用 `CopyL0C2UBSplitMTrait`；`CopyL0C2UBNonSplitTrait` 仅用于单 AIV 调试 |
| SplitM | `DUAL_DST_SPLIT_M` 将 L0C 的 M 行对半切分，各 AIV 从各自 UB offset 0 读取半份数据（UB 读取不需要 sub-block 偏移） |
| UB 布局 | Epilogue 必须知道 UB 中 MatMul 结果的 dtype、nAlign、rows 和起始位置 |
| 同步 | Kernel 层负责 AIC/AIV CrossCore set/wait，Epilogue 完成后必须回复 AIC |

### 3.3 最小组装骨架

```cpp
#include "kernel/matmul_kernel_fused.h"
#include "block/matmul_block_mmad.h"
#include "block/block_scheduler_policy.h"
#include "policy/dispatch_policy.h"
#include "epilogue/my_epilogue.h"

template <bool TransA, bool TransB>
__global__ __aicore__ __mix__(1, 2) void fused_matmul_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dExtra, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;
    using LayoutA = AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = MatmulShape;
    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
    using BlockScheduler = MatmulSwatScheduler<NO_FULL_LOAD_MODE>;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC>;
    using EpilogueOp = MyEpilogue;
    using KernelImpl = Kernel::MatmulKernelFused<ProblemShape, BlockMmad, BlockScheduler, EpilogueOp>;

    ProblemShape problemShape{tilingData.m, tilingData.n, tilingData.k, 1L};
    typename BlockMmad::Params mmParams{dA, dB, dC};
    typename KernelImpl::L1Params l1Params{static_cast<uint64_t>(tilingData.kL1)};
    typename KernelImpl::BlockSchedulerParams schParams{
        tilingData.baseM, tilingData.baseN, tilingData.mTailCnt, tilingData.nTailCnt,
        tilingData.mBaseTailSplitCnt, tilingData.nBaseTailSplitCnt,
        tilingData.mTailMain, tilingData.nTailMain};
    typename KernelImpl::QBMMTiling qbmmParams{
        tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.l0cDB};

    // `mTailCnt/nTailCnt` 在普通 SWAT 中的真实语义是尾块切分份数；传给
    // blaze_custom scheduler 时分别对应 `mTailTile/nTailTile`，默认值为 1/1。
    typename EpilogueOp::Params epilogueParams{dExtra, dC};

    typename KernelImpl::Params params{problemShape, mmParams, l1Params, schParams, qbmmParams, epilogueParams};
    KernelImpl kernel;
    kernel(params);
}
```

---

## §4 Epilogue 设计

Epilogue 负责消费 UB 中的 MatMul 结果、加载额外输入、执行 Vector 公式并写回 GM。

### 4.1 MemBase vs RegBase

| Epilogue 类型 | 适用场景 | 特点 |
|---------------|----------|------|
| MemBase | 只有一个简单 vector 操作，且有明确可用的 AscendC LocalTensor API，如 `Mul/Add/Div/Cast` | 中间值通常写 UB，代码简单 |
| RegBase | GELU / SwiGLU / LayerNorm / 多输入 scale / 多中间值公式链 | 中间值在 RegTensor 内流转，减少 UB 占用和读写 |

默认选择 RegBase。只有当公式足够简单、API 明确且 UB 空间满足时，才选择 MemBase。

详细设计：

- 通用入口：`references/modules/blaze-custom/development/epilogue-dev-guide.md`
- MemBase：`references/modules/blaze-custom/development/epilogue-membase-design.md`
- RegBase：`references/modules/blaze-custom/development/epilogue-regbase-design.md`

### 4.2 SplitM 与 row-dependent 输入

`CopyL0C2UBSplitMTrait`（`DUAL_DST_SPLIT_M`）将 L0C 的 M 行对半切分到两个 AIV SubBlock。

**GM offset（需要 sub-block 偏移）**：

```text
origM = blockShapeM
halfM = ceilDiv(origM, GetTaskRation())
localRows = (origM is odd) ? (halfM - GetSubBlockIdx()) : halfM

tileM0 = gmOffset / N
tileN0 = gmOffset % N
subM0 = tileM0 + GetSubBlockIdx() * halfM
stageM0 = subM0 + stageRowOffset

rowDependentInputOffset = stageM0
outputOffset = subM0 * N + tileN0
```

**UB 读取（不需要 sub-block 偏移）**：

`DUAL_DST_SPLIT_M` 硬件自动分片，每个 AIV 从各自 UB offset 0 读取半份数据：

```text
srcAddr = cLocal_.GetPhyAddr()  // 不加 GetSubBlockIdx() * halfM * nAlign
rowSrc = srcAddr + row * nAlign
```

> **关键**：GM 侧 offset 需要加 sub-block 偏移，UB 侧不需要。从 GM 公式推断 UB 也需要偏移是常见错误。

**localRows=0 边界**：当 `curM=1`（奇数，halfM=1）时 V1 的 `localRows=0`，early return 即可，CV 同步由 kernel 层处理。

详见 `references/modules/blaze-custom/development/epilogue-dev-guide.md` §4。

### 4.3 Epilogue 设计输出

每个 Epilogue 应明确：

| 项 | 内容 |
|----|------|
| Params | 所有额外输入和输出的 GM_ADDR |
| UB 分区 | `cLocal_`、extra input、tmp、output staging 的字节范围 |
| 公式伪代码 | 根据用户公式列出 vector API 调用链 |
| 同步伪代码 | CrossCore 之外的 MTE2/V/MTE3 正向与反向依赖 |
| Offset 公式 | SplitM、stage、tail、row-dependent input 和 output offset |
| API 来源 | MemBase 查 `ascendc-api-best-practices`，RegBase 查 `ascendc-regbase-best-practice` |

---

## §5 Tiling 与 UB 剩余空间关系

C+V 融合不新增独立 vector tiling engine。Tiling 选择见 `references/tiling/tiling-selection.md`。

设计关系：

```text
Cube tiling
  -> 决定 baseM/baseN/baseK/kL1 和 L0C2UB 的 matmul result 区域
  -> 计算 matmulAreaBytes
  -> 剩余 UB 交给 Epilogue 设计 extra/tmp/output buffer
  -> 若剩余 UB 不足，回调 C 部分调整 baseM/baseN 或切换 RegBase
```

要点：

- 先保 L0C2UB 的 MatMul 结果完整落 UB。
- V 部分不独立切 tile，只在剩余 UB 中设计 `stageRows/stageSize`。
- 复杂公式链优先用 RegBase 降低 UB 中间值数量。

---

## §6 Launcher 与参数组织

Launcher 需要同时组织 C 部分输入、Epilogue 额外输入和 TilingData。

| 内容 | 要求 |
|------|------|
| GM_ADDR 顺序 | 输入在前、输出在后、tilingData 最后 |
| C 部分输入 | A/B 和可能存在的 MX scale / groupList |
| V 部分输入 | Bias / scale / residual / activation 参数等 |
| gridDim | 使用 `tilingData.usedCoreNum` |
| trans / format 分发 | 与对应 MatMul 场景一致 |

Grouped C+V 使用 `totalM` 调用对应非 grouped SWAT tiling；`groupList/groupNum` 独立传入 grouped kernel，不进入 tiling data。

---

## §7 常见陷阱

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| F1 | AIC hang | AIV epilogue 崩溃或未回复 CrossCore flag | 检查 epilogue 末尾写回与 kernel 层 `CrossCoreSetFlag` |
| F2 | AIV 输出全零 | 未等 AIC 信号就读 UB | 确认 AIV 侧 `CrossCoreWaitFlag` 后才读 UB |
| F3 | 大 shape FAIL | 多 tile flag 轮转错误 | 保持 `count / COUNT_ID_MAX % COUNT_FLAG` 轮转协议 |
| F4 | UB 越界 | epilogue 额外 buffer 未从 UB 预算扣除 | 计算 `remainBytes = UB_SIZE - matmulAreaBytes - epilogueBytes` |
| F5 | tail tile 错乱 | nAlign 用 Init 时 baseN 固定计算 | per-call 从 `blockShapeN` 计算 nAlign |
| F6 | 误用 `__cube__` | 复制纯 MatMul 模板 | C+V 必须使用 `__mix__(1, 2)` |
| F7 | SplitM 精度错位（GM 侧） | Epilogue 未按 SubBlock 计算 rows / gmOffset / extra input offset | 使用 `GetTaskRation()` / `GetSubBlockIdx()` |
| F8 | 纯 `GemmUniversal` 做融合 | 无法接入自定义 Vector Epilogue | 走本文件的 C+V 组装路径 |
| F9 | SplitM 精度错位（UB 侧） | UB 读取 cLocal_ 时加了 sub-block 偏移 | SplitM 已硬件分片，从 UB offset 0 读取 |
| F10 | UB buffer 重叠 | `matmulAreaBytes` 行步长用 16 而非 nAlignL0C | 行步长 = `nAlignL0C`（不是 L0C cube 边长 16） |
| F11 | tail 场景输出错乱 | UB→GM DataCopyPad `srcStride` 传 bytes 而非 32B 单位 | UB 侧 stride 传 0（nAlign 保证 32B 对齐） |
| F12 | 多 stage 数据竞争 | 用 `MTE3_MTE2` 自 Set 自 Wait，无实际同步效果 | 用 `V_MTE2` + `MTE3_V` 分别保护不同 buffer |
| F13 | 首轮 hang 或尾轮 flag 泄漏 | 缺少 Init 预发射或析构排空 | Init 预发射所有反向 SetFlag，析构排空所有 WaitFlag |
