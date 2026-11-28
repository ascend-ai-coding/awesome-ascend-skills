# reduction 范式 binary-group Tiling 切分策略

> UB 切分、多核切分、Workspace 分配的完整算法

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)、[reduction-binary-group-dag-buffers.md](reduction-binary-group-dag-buffers.md)、[reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md)、[reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md)

> **本文档只处理 Phase 1 的切分**：UB 切分算法与 Base 完全一致，差异仅在多核切分（A-only → A×R 2D）及 Workspace/TilingData/ScheduleMode 的新增；Phase 2 切分在 kernel 侧现算，不进入 TilingData。

## 0 速查

| 概念 | 一句话 |
|------|--------|
| **UB 双切分** | A、R 各选一个切分轴（`aSplitIdx`/`rSplitIdx`），**互相独立** |
| **Group** | A 用不满核，借 R 补并行度 |
| **2D 分核** | `aOuter × rOuter` 一起分核，对齐到 `aOuter`，`aPerCore=1` 恒成立 |
| **rGroupCnt** | 用 R 开多核的个数 |
| **workspace** | GM 上用于临时缓存中间计算结果， `[rGroupCnt, aTotal]` fp32 dense |
| **ScheduleMode** | 算子在NPU上执行时的调度模式（SyncAll 要求 `SetScheduleMode(1)`） |

## 1 UB 切分

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §1

### 1.1 切分原理

> Group 模板 Phase 1 完全复用 Base 的 UB 切分三步算法。

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §1.1

### 1.2 切分公式

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §1.2（Step 1 ComputeAUbFactor / Step 2 ComputeRUbFactor / Step 3 ExpandAIfRFullyLoaded）

### 1.3 特殊场景

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §1.3

## 2 多核切分

**这里只阐述 phase1 阶段的多核切分**: phase2 的切分策略见 §2.3

### 2.1 切分原理

**Group 触发条件**（`ShouldUseGroup()`）：A 方向外层总 chunk 数 `aOuter = aLoopCntTotal`，R 方向 `rOuter = rLoopCntTotal`。当 `aOuter ≤ coreNum / 2` 且 `rOuter ≥ 2` 时走 group 模板——A 用不满核，R 有额外并行度可借。

```
if aOuter > coreNum / 2                           → base   # A 够分核
else if rOuter <= 1                                → base   # R 无并行度
else                                               → group
```

> 阈值取 `coreNum / 2` 的原因：A×R 2D 分核时，A 切片不能再被拆分（一个 A 外层 chunk 是 UB 切分后不可分的最小调度单元）。若 `aOuter > coreNum / 2`，即使 rGroupCnt 为 2, 也会导致 `usedCoreNum > coreNum`，这在NPU上是不被允许的, SyncAll()会挂死。

**Group 2D 分核**：A_out × R_out 乘起来一起分核。Phase 1 复用 base 模板的 UB 切分，拿到两个外层循环的扁平计数。`aPerCore = 1` 恒成立。

### 2.2 切分公式

**A×R 2D 分核**（`ComputeGroupSplit()`）：

```cpp
totalOuter  = aOuter × rOuter
perCoreNum  = CeilDiv(totalOuter, coreNum)
numBlocks   = CeilDiv(totalOuter, perCoreNum)

# 对齐到 aOuter —— 保 2D 网格整齐、负载均匀
# aOuter < numBlocks 恒成立（group 触发条件 aOuter ≤ coreNum/2 && rOuter ≥ 2 保证）
if (CeilAlign(numBlocks, aOuter) <= coreNum)
    numBlocks = CeilAlign(numBlocks, aOuter)
else
    numBlocks = FloorAlign(numBlocks, aOuter)

usedCoreNum = numBlocks
SetBlockDim(usedCoreNum)
```

拆回 A / R 两维：

```cpp
rGroupCnt = numBlocks / aOuter                            # = rGroups，workspace R 维大小
                                                          # numBlocks 已对齐到 aOuter，整除
```

> **`aPerCore = 1` 恒成立**：`numBlocks` 被对齐到 `aOuter` 的倍数 → `aGroupCnt = aOuter`。每个核只处理 1 个 A chunk。

**约束**：
- 严禁 `SetBlockDim(0)`，空 tensor 场景若使用核数为 0，也需要 `SetBlockDim(1)`, kernel 侧通过 `usedCoreNum=0` 短路处理
- 设置的 BlockDim 严禁超过实际的物理核数
- `aPerCore = 1` 恒成立

### 2.3 特殊场景（Phase 2 切分）

Phase 2 的切分在 **kernel 侧现算**，不进入 TilingData。此处不讨论。

## 3 切分策略微调

### 3.1 微调方式

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §3.1。Group 触发判定本身就是一种微调——A 用不满核时借 R 补并行度。

## 4 整体切分伪码

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §4，Group 模板在步骤 4 触发：

```cpp
// 4) Group 模板判定（条件性）
if (ShouldUseGroup(aLoopCntTotal, rLoopCntTotal)):
    ComputeGroupSplit();        // A×R 2D 分核 → usedCoreNum, rGroupCnt
    SetScheduleMode(1);         // SyncAll 要求
```

## 5 切分后处理

### 5.1 UB 大小计算

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §5.1。Phase 1 UB Buffer 大小公式与 Base 一致。Phase 2 复用 Phase 1 UB 物理槽，不重新计算。

### 5.2 Workspace 分配

**Group 模板 workspace = 用户 workspace + 系统 workspace**：

```cpp
size_t* ws = context->GetWorkspaceSizes(1);
OP_CHECK_NULL_WITH_CONTEXT(context, ws);

// ⚠ GetPlatformInfo() 返回指针必须判空后再传入 PlatformAscendC（构造器仅存裸指针，
//   空指针在 GetLibApiWorkSpaceSize 即解引用崩溃）
fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
size_t usrWorkspaceBytes = rGroupCnt × aTotal × sizeof(fp32);   // [rGroupCnt, aTotal] fp32 dense

if (isGroup) {
    ws[0] = usrWorkspaceBytes + sysWorkspaceSize;
} else {
    ws[0] = sysWorkspaceSize;
}
```

**workspace 布局**：`[rGroupCnt, aTotal]` fp32 dense，行优先。每行 `aTotal` 个 float。

**约束**:
- 必须设置，即使不需要使用 workspace，也需要显式设置
- 如果要使用 workspace，总大小为 `sysWorkspaceSize + 实际使用`（ascendc 要求）
- 如果涉及 workspace 使用，必须使用 INFO 级别日志打印分配大小

### 5.3 TilingKey 设置

```cpp
SetTilingKey({isGroup=1, isEmptyTensor=0});  // Group: isGroup=1
```

**约束**：必须使用 INFO 级别日志打印 tilingkey 各模板参数。

### 5.4 TilingData 设置

同 [reduction-binary-base-tiling.md](../reduction-template-binary-base/reduction-binary-base-tiling.md) §5.4。Group 模板唯一新增字段 `rGroupCnt`。

**Group TilingData 增量**：在 base TilingData struct 末尾追加一个字段：

```cpp
int64_t rGroupCnt;          // Phase 1 分组数 = Phase 2 R 维度（host 端填，kernel 读）
```

> mean 的 `invRTotal` 在 base 已追加，Group 不新增字段。R_total 在 group 和 base 下完全相同（两次 Reduce 合起来消解了全 R）。

### 5.5 ScheduleMode 设置

**Group 模板必须设置 `SetScheduleMode(1)`**（SyncAll 要求，防止多核同步挂死）：

```cpp
if (isGroup) {
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set ScheduleMode!"),
                return ge::GRAPH_FAILED);
}
```

> group 模板使用 SyncAll 做 Phase 1→Phase 2 全核同步。host 端必须调用 `SetScheduleMode(1)`，告诉框架需要等 `usedCoreNum` 个核都空闲时才启动该算子。

## 6 切分策略产出校验

**样例：aOuter=4, rOuter=8, coreNum=8**

- totalOuter = 4 × 8 = 32
- perCoreNum = CeilDiv(32, 8) = 4
- numBlocks = CeilDiv(32, 4) = 8
- CeilAlign(8, 4) = 8 ≤ 8 → numBlocks = 8
- usedCoreNum = 8
- rGroupCnt = 8 / 4 = 2

**Golden**：
- usedCoreNum=8, rGroupCnt=2
- aPerCore=1（恒成立）
- workspace = [2, aTotal] fp32 dense，大小 = 2 × aTotal × 4 字节
- SetScheduleMode(1)

**校验要点**：
1. numBlocks 对齐到 aOuter（CeilAlign(8, 4) = 8 ≤ coreNum=8，取 CeilAlign）
2. rGroupCnt = numBlocks / aOuter（整除）
3. aPerCore = 1 恒成立
4. workspace 布局 [rGroupCnt, aTotal]，每行 aTotal 个 float
5. SetScheduleMode(1) 必须调用
