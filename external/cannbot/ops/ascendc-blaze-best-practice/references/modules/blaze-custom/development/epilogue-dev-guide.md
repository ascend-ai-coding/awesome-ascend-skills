# Epilogue 层扩展开发指南

> 适用路径：blaze_custom，C+V 融合场景。
>
> 本文是 Epilogue 设计总入口，负责说明 Epilogue 在 C+V 中的位置、接口契约、MemBase/RegBase 选择、SplitM、UB 分配、同步设计方法和伪代码生成流程。

---

## §1 Epilogue 在 C+V 中的位置

Epilogue 是 C+V 融合算子的 V 部分，负责：

1. 消费 AIC 通过 L0C2UB 写入 UB 的 MatMul 结果。
2. 读取额外输入（Bias、scale、residual、activation 参数等）。
3. 执行 Vector 公式链。
4. 将结果写回 GM，并回复 AIC 可以继续覆盖 UB。

Epilogue 不是独立 kernel。它依附于 C 部分的 Kernel / Block / Scheduler 契约，尤其依赖：

- UB 中 MatMul 结果的布局。
- `DUAL_DST_SPLIT_M` 是否启用。
- CrossCore flag 的轮转策略。

---

## §2 三接口合约

| 接口 | 签名 | 职责 |
|------|------|------|
| `Init` | `void Init(Params, baseM, baseN, ProblemShape)` | UB 布局、GM 绑定、预发反向 event |
| `GetTensor` | `auto GetTensor()` | 返回 UB Tensor（AIC 侧 `CopyL0C2UB` 的 dst） |
| `operator()` | `void operator()(BlockShape, int64_t gmOffset, uint16_t flagId)` | 普通 / MX C+V 逐 tile 后处理 |
| `Params` | `struct Params { ... }` | 额外输入和输出的 GM 地址 |

类型约定：`BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>`。

Grouped C+V 使用 context-based Epilogue，`operator()` 签名为 `void operator()(BlockShape, TileContext, uint16_t flagId)`。普通 linear offset Epilogue 不直接兼容 Grouped C+V。

---

## §3 MemBase vs RegBase 选择

| Epilogue 类型 | 适用场景 | 特点 |
|---------------|----------|------|
| MemBase | 只有一个简单 vector 操作，且有明确可用的 `AscendC::` LocalTensor API，如 `Mul/Add/Div/Cast` | 中间值通常写 UB，代码简单 |
| RegBase | GELU / SwiGLU / LayerNorm / 多输入 scale / 多中间值公式链 | 中间值在 `RegTensor` 内流转，减少 UB 占用和读写 |

默认选择 RegBase。只有当公式足够简单、API 明确且 UB 空间满足时，才选择 MemBase。

详细设计：

- MemBase：`epilogue-membase-design.md`
- RegBase：`epilogue-regbase-design.md`

---

## §4 SplitM 与 offset 原理

普通 C+V 与 MX C+V 的 `CopyL0C2UB` 使用 `DUAL_DST_SPLIT_M` Trait，L0C 按 M 维拆给两个 AIV SubBlock。Epilogue 不能把 UB 当成完整 `blockM × blockN` 整块处理。

### CopyL0C2UB Trait 选择

| Trait | DualDstMode | 适用场景 | UB 数据分布 |
|-------|-------------|---------|------------|
| `CopyL0C2UBSplitMTrait` | `DUAL_DST_SPLIT_M` | `__mix__(1,2)` 标准 C+V | M 对半分片，各 AIV 从各自 UB offset 0 读取 |
| `CopyL0C2UBNonSplitTrait` | `DUAL_DST_DISABLE` | 单 AIV 调试 | 全量数据在 UB offset 0 |

> `__mix__(1,2)` C+V 场景必须使用 `CopyL0C2UBSplitMTrait`。`matmul_block_mmad.h` 参考模板已默认使用 SplitMTrait。

### GM offset（需要 sub-block 偏移）

基础公式：

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

含义：

- `halfM` 是单个 AIV SubBlock 负责的行数上界。
- 奇数 M 时，SubBlock0 处理 `halfM` 行，SubBlock1 处理 `halfM - 1` 行。
- 所有依赖 M 维的额外输入都必须从 `stageM0` 计算 GM offset。

### UB 读取（不需要 sub-block 偏移）

`CopyL0C2UBSplitM` 会**硬件自动**将 L0C 的 M 行对半切分，每个 AIV 从自己的 UB offset 0 开始读取半份数据：

| 操作 | 是否需要 SubBlock 偏移 | 公式 |
|------|----------------------|------|
| UB 读取 cLocal_ | **否** | `cLocal_.GetPhyAddr() + row * nAlign` |
| GM 读取 row-dependent input | 是 | `stageM0 = tileM0 + GetSubBlockIdx() * halfM + stageRowOffset` |
| GM 写回 output | 是 | `gmRowOffset = subM0 * N + tileN0` |

> **常见错误**：从 GM offset 公式推断"UB 读取也需要加 `GetSubBlockIdx() * halfM * nAlign` 偏移"。实际上 UB 数据已被硬件自动分片，加偏移会导致 V1 跳过自己的数据读到后续 buffer 的垃圾值。

### localRows=0 边界场景

当 `curM` 为奇数且 `halfM=1` 时（如 `curM=1`），V1 的 `localRows = 0`。此时：
- V1 应 early return，不需要做任何 Vector 计算或 GM 写回。
- CV 同步由 kernel 层（`MatmulKernelFused`）统一处理，不会因 V1 return 而挂死。

---

## §5 UB 空间分配通用方法

Epilogue 不重新设计独立 tiling engine，而是在 Cube tiling 已确定的前提下消费剩余 UB。

设计顺序：

1. 先保留 L0C2UB 的 MatMul 结果区域。
2. 再根据 MemBase / RegBase 和用户公式决定额外需要几份空间。
3. 最后计算 `stageRows` / `stageSize`。

通用预算：

```text
nAlignL0C = ceil(baseN / (32/sizeof(L0CDataType))) * (32/sizeof(L0CDataType))
splitMRows = ceil(baseM / GetTaskRation())
matmulAreaBytes = splitMRows * nAlignL0C * sizeof(L0CDataType)
remainBytes = TOTAL_UB_SIZE - matmulAreaBytes
epilogueBytes = sum(extra input buffer, tmp buffer, output staging)

epilogueBytes <= remainBytes
```

> **关键**：`matmulAreaBytes` 的行步长是 `nAlignL0C`（UB 中的对齐行宽），**不是** L0C cube 边长 16。L0C cube 边长是 L0C 硬件分型粒度，与 UB buffer 行步长无关。将 `nAlignL0C` 误写为 16 会导致 `matmulAreaBytes` 被低估、后续 buffer 重叠到 cLocal_ 数据区。

设计原则：

- `cLocal_` 是 MatMul 结果区，优先级最高。
- RegBase 尽量把中间值留在 `RegTensor`，只在 UB 中保留输入 staging 和输出 staging。
- MemBase 中间值通常需要额外 UB tmp buffer。
- 若剩余 UB 不足，优先减小 `baseM/baseN` 或切换到 RegBase，而不是给 V 部分另起一套 tiling。

---

## §6 同步设计方法

Epilogue 同时受两层同步约束：

| 层次 | 机制 | 说明 |
|------|------|------|
| AIC ↔ AIV | `CrossCoreSetFlag/WaitFlag` | AIC 写 UB 后通知 AIV；AIV 完成后回复 AIC |
| AIV 内部 | `SetFlag/WaitFlag<HardEvent>` | 组织 MTE2→V→MTE3 和反向依赖 |

同步规则以 `references/fundamentals/blaze-sync-patterns.md` 为准。

最小同步伪代码：

```cpp
WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);
DataCopyPad(extraInputBuf, extraInputGm[offset], ...);

SetFlag<HardEvent::MTE2_V>(ZERO_FLAG);
WaitFlag<HardEvent::MTE2_V>(ZERO_FLAG);

WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);

__VEC_SCOPE__ {
    // 读 cLocal_ / extra input
    // 计算 vector 公式
    // 写 output staging
}

SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);

SetFlag<HardEvent::V_MTE3>(ZERO_FLAG);
WaitFlag<HardEvent::V_MTE3>(ZERO_FLAG);
DataCopyPad(outputGm[outputOffset], outputBuf, ...);
SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);
```

---

## §7 Vector 公式到伪代码的设计流程

设计 Epilogue 时，不应直接写代码，而应先从用户公式生成伪代码。

步骤：

1. 把用户公式分解为可执行的 Vector 操作链。
2. 判断每个中间值是否必须写 UB，还是可以保留在寄存器。
3. 对每个操作到对应 skill 查 API：
   - MemBase / AscendC LocalTensor API：`ascendc-api-best-practices`
   - RegBase API：`ascendc-regbase-best-practice`
4. 根据 `blaze-sync-patterns.md` 安排同步指令。
5. 输出 `Init` / `GetTensor` / `operator()` 所需信息。

输出的伪代码至少应包含：

- 额外输入加载顺序。
- row-dependent offset 公式。
- Vector 计算链。
- 输出写回顺序。
- 同步指令位置。

---

## §8 同步自检清单

| 检查项 | 要求 |
|--------|------|
| MTE2 -> V 正向依赖 | `DataCopyPad` 后必须存在 `MTE2_V` |
| V -> MTE2 反向依赖 | 下一轮覆盖 extra input buffer 前必须存在 `V_MTE2` |
| V -> MTE3 正向依赖 | Vector 计算完成后必须存在 `V_MTE3` |
| MTE3 -> V 反向依赖 | 下一轮 V 或 MTE2 使用相关 buffer 前必须存在 `MTE3_V` |
| CrossCore 配对 | AIC set / AIV wait，AIV set / AIC wait 必须成对 |
| 首轮预发射 | 若首轮要先 Wait 反向依赖，必须在 `Init` 中预发射 |
| 尾轮收尾 | 若最后一轮会遗留反向依赖，必须在析构或收尾阶段排空 |

---

## §9 常见错误

| 错误 | 后果 | 修复 |
|------|------|------|
| 忽略 SplitM 行偏移（GM 侧） | SubBlock1 错位或下半块结果错误 | 用 `GetTaskRation()` / `GetSubBlockIdx()` 计算 GM row offset |
| UB 读取 cLocal_ 时加 sub-block 偏移 | V1 读到垃圾值，精度完全错误 | SplitM 已硬件分片，从 UB offset 0 读取 |
| `matmulAreaBytes` 行步长用 16 而非 nAlignL0C | UB buffer 重叠，bias/scale 覆写 matmul 结果 | 行步长 = `nAlignL0C`（不是 L0C cube 边长 16） |
| UB→GM DataCopyPad `srcStride` 传 bytes | tail 场景行步长错误 | UB 侧 stride 是 32B 单位，nAlign 对齐时传 0 |
| 用 `MTE3_MTE2` 自 Set 自 Wait | 无实际同步效果 | 用 `V_MTE2` + `MTE3_V` 分别保护不同 buffer |
| 缺少 Init 预发射或析构排空 | 首轮 hang 或尾轮 flag 泄漏 | Init 预发射所有反向 SetFlag，析构排空所有 WaitFlag |
| 只设计正向依赖，不设计反向依赖 | 小 shape PASS，大 shape 随机错数或 hang | 按 `blaze-sync-patterns.md` 补全反向依赖 |
| 把 RegBase 中间值写回 UB | UB 压力过大，stageRows 降低 | 中间值优先保留在 `RegTensor` |
| 在 C+V 中重新设计 vector tiling | 与 C 部分 tile / offset 脱节 | 只复用 Cube tiling，V 部分消费剩余 UB |
