# reduction 范式 empty Tiling 切分策略

> UB 切分、多核切分、Workspace 分配的完整算法

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)、[reduction-empty-dag-buffers.md](reduction-empty-dag-buffers.md)、[reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md)

## 0 速查

| 概念 | 一句话 |
|------|--------|
| **EMPTY_A** | A 轴含 0 → output 空 tensor，`usedCoreNum=0`，所有核早退 |
| **EMPTY_R** | R 轴含 0 且 A 全非 0 → output = `empty_r_output_value`，按 a 元素数切分 |
| **aUbFactor** | 4 约束取 min：4KB buffer 下界 / 优先多核 / UB 上限 / aTotal 兜底 |
| **UB 切分** | 按 a 元素数切（非 aLoop） |

## 1 UB 切分

### 1.1 切分原理

**EMPTY_A**：无 UB 切分（零计算、零 IO）。

**EMPTY_R**：UB 切分按 **a 元素数** 切。EMPTY_R 下 aTotal 是合所有 A 轴后的线性元素数，按 aUbFactor 切 chunk。

### 1.2 切分公式

**EMPTY_A**：无切分公式。`usedCoreNum = 0`，`SetBlockDim(1)`。

**EMPTY_R**：

```
aTotal = ∏(所有 A 轴 axisShape);   // > 0（EMPTY_R 触发前提）

// 四条约束：
//  maxDtypeSize = max(sizeof(D_T), sizeof(float))   // fp16: 4B, fp32: 4B
//   a) 4KB buffer 下界 → min_a_per_core = CeilDiv(4096, maxDtypeSize)
//   b) 优先多核    → CeilDiv(aTotal, coreNum)
//   c) UB 上限     → maxBufSize = min(ubSize / P_post, 65536); （[reduction-empty-dag-buffers.md](reduction-empty-dag-buffers.md) §2 (2)）
//                  → max_ub_factor = maxBufSize / maxDtypeSize
//   d) aTotal 兜底 → aTotal
aUbFactor = min(max(min_a_per_core, CeilDiv(aTotal, coreNum)), max_ub_factor);
aUbFactor = min(aUbFactor, aTotal);

aLoopCntTotal = CeilDiv(aTotal, aUbFactor);
// 大小核均衡
// ⚠ coreNum/ubSize 来自 GetPlatformInfo()，作除数前必须零值拦截
//   （OP_CHECK_IF coreNum == 0 / ubSize == 0 ... return ge::GRAPH_FAILED，与 example 对齐）
aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
aBigCoreCnt       = aLoopCntTotal % coreNum;
aBigCoreLoopCnt   = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
usedCoreNum       = (aSmallCoreLoopCnt > 0) ? coreNum : aBigCoreCnt;
```

### 1.3 特殊场景

| 情形 | 行为 |
|------|------|
| `aTotal == 0` | 不会走到 EMPTY_R（A 全非 0 时 aTotal ≥ 1；A 有 0 时已被 EMPTY_A 抢走） |
| mean 算子 NaN 输出 | 与 numpy 一致；如果业务要求拒绝空 R，host 端在 tiling 顶部判定时返回 FAIL 而非进 EMPTY_R |

**约束**：
- 严禁 `SetBlockDim(0)`，EMPTY_A 使用核数为 0，也需要 `SetBlockDim(1)`, kernel 侧通过 `usedCoreNum=0` 短路处理
- 设置的 BlockDim 严禁超过实际的物理核数

## 2 多核切分

### 2.1 切分原理

**EMPTY_A**：无多核切分（`usedCoreNum=0`）。

**EMPTY_R**：按 a 元素数切分（`aLoopCntTotal = CeilDiv(aTotal, aUbFactor)`），大小核均衡。——EMPTY_R 下 aTotal 是合所有 A 轴后的线性元素数。

### 2.2 切分公式

**EMPTY_R**：

```cpp
// ⚠ coreNum/ubSize 来自 GetPlatformInfo() 且已零值拦截（OP_CHECK_IF ... return FAIL），作除数前必须保证非 0
aLoopCntTotal = CeilDiv(aTotal, aUbFactor);
aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
aBigCoreCnt       = aLoopCntTotal % coreNum;
aBigCoreLoopCnt   = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
usedCoreNum       = (aSmallCoreLoopCnt > 0) ? coreNum : aBigCoreCnt;
```

**Kernel 端映射**（按 a 元素数切）：

```cpp
// blockIdx → [aStart, aEnd)（按 a 元素数 = chunk 数 × aUbFactor 切）
if (blockIdx < aBigCoreCnt_) {
    aStart = blockIdx * aBigCoreLoopCnt_ * aUbFactor_;
    aEnd   = aStart + aBigCoreLoopCnt_ * aUbFactor_;
} else {
    aStart = aBigCoreCnt_ * aBigCoreLoopCnt_ * aUbFactor_ +
             (blockIdx - aBigCoreCnt_) * aSmallCoreLoopCnt_ * aUbFactor_;
    aEnd   = aStart + aSmallCoreLoopCnt_ * aUbFactor_;
}
aEnd = min(aEnd, aTotal_);                   // 防越界
```

### 2.3 特殊场景

不涉及

## 3 切分策略微调

### 3.1 微调方式

不涉及。Empty 模板的 aUbFactor 由 4 约束（4KB buffer 下界/优先多核/UB 上限/aTotal 兜底）直接算出，无微调。

## 4 整体切分伪码

```cpp
HandleEmptyTensor(context):
    if (∃ A 轴 axisShape == 0):
        // EMPTY_A
        usedCoreNum = 0;
        SetBlockDim(1);                         // 框架要求 ≥ 1
        SetTilingKey({isGroup=0, isEmptyTensor=1});
        FillAndLogTilingData(/* usedCoreNum=0, 其余填 0 */);
        SetWorkspaceSize();                     // 空 tensor workspace = sysWorkspaceSize（无用户 workspace）
        return;

    // EMPTY_R（∃ R 轴 axisShape == 0 && ∀A > 0）
    // ⚠ coreNum/ubSize 经 GetPlatformInfo() 获取并零值拦截（OP_CHECK_IF ... return FAIL）后方可作除数
    aTotal = ∏(所有 A 轴 axisShape);
    maxDtypeSize = max(sizeof(D_T), sizeof(float));
    maxBufSize = min(ubSize / P_post, 65536); // [reduction-empty-dag-buffers.md](reduction-empty-dag-buffers.md) §2 (2)
    max_ub_factor = maxBufSize / maxDtypeSize;  

    aUbFactor = min(max(CeilDiv(4096, maxDtypeSize), CeilDiv(aTotal, coreNum)), max_ub_factor);
    aUbFactor = min(aUbFactor, aTotal);

    aLoopCntTotal = CeilDiv(aTotal, aUbFactor);
    aSmallCoreLoopCnt = aLoopCntTotal / coreNum;
    aBigCoreCnt       = aLoopCntTotal % coreNum;
    usedCoreNum       = (aSmallCoreLoopCnt > 0) ? coreNum : aBigCoreCnt;

    SetBlockDim(usedCoreNum);
    SetTilingKey({isGroup=0, isEmptyTensor=1});
    FillAndLogTilingData(/* aTotal, aUbFactor, 大小核结果, postBufSize, usedCoreNum */);
    SetWorkspaceSize();          // 空 tensor workspace
```

## 5 切分后处理

### 5.1 UB 大小计算

**EMPTY_A**：无 UB buffer，无计算。

**EMPTY_R**：

```
postBufSize    = max(aUbFactor * maxDtypeSize, blockSize)   // 单份至少 1 个 block
postBufSize    = CeilAlign(postBufSize, blockSize)
// 只计算 postBufSize（其他 buffer 不分配）
```

### 5.2 Workspace 分配

```cpp
size_t* ws = context->GetWorkspaceSizes(1);
OP_CHECK_NULL_WITH_CONTEXT(context, ws);
size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
ws[0] = sysWorkspaceSize;
```

**约束**:
- 必须设置，即使不需要使用 workspace，也需要显式设置
- 如果要使用 workspace，总大小为 `sysWorkspaceSize + 实际使用`（ascendc 要求）
- 如果涉及 workspace 使用，必须使用 INFO 级别日志打印分配大小

### 5.3 TilingKey 设置

```cpp
SetTilingKey({isGroup=0, isEmptyTensor=1});  // Empty: isGroup=0, isEmptyTensor=1
```

**约束**：必须使用 INFO 级别日志打印 tilingkey 各模板参数。

### 5.4 TilingData 设置

**EMPTY_A**：`usedCoreNum = 0`，其余字段填 0。

**EMPTY_R** 填写独立的 `ReduceEmptyTilingData`（struct 定义见 [reduction-template-overview.md §4.1.2](../reduction-template-overview.md#412-empty-tilingdata)）：

| 字段 | EMPTY_R 取值 |
|------|------------|
| `usedCoreNum` | 切分结果 |
| `aTotal` | ∏(所有 A 轴 axisShape) |
| `aUbFactor` | 实际单 chunk 大小 |
| `aBigCoreCnt` | `aLoopCntTotal % coreNum` |
| `aBigCoreLoopCnt` | `aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0)` |
| `aSmallCoreLoopCnt` | `aLoopCntTotal / coreNum` (floor) |
| `postBufSize` | §5.1 |

**约束**：TilingData 每个字段必须 `OP_LOGI` 打印。

### 5.5 ScheduleMode 设置

Empty 模板不调用 SyncAll，**不设置** ScheduleMode。

## 6 切分策略产出校验

**样例：EMPTY_R，fp16，aTotal=10000，coreNum=8，ubSize=248KB，P_post=4**

- maxDtypeSize = max(2, 4) = 4
- maxBufSize = min(ubSize / P_post, 65536) = min(253952 / 4, 65536) = min(63488, 65536) = 63488
- max_ub_factor = maxBufSize / maxDtypeSize = 63488 / 4 = 15872
- min_a_per_core = CeilDiv(4096, maxDtypeSize) = CeilDiv(4096, 4) = 1024
- CeilDiv(aTotal, coreNum) = CeilDiv(10000, 8) = 1250
- aUbFactor = min(max(1024, 1250), 15872) = min(1250, 15872) = 1250
- aUbFactor = min(1250, 10000) = 1250
- aLoopCntTotal = CeilDiv(10000, 1250) = 8
- aSmallCoreLoopCnt = 8 / 8 = 1
- aBigCoreCnt = 8 % 8 = 0
- usedCoreNum = 8（aSmallCoreLoopCnt == 1 > 0 → usedCoreNum = coreNum）
- postBufSize = CeilAlign(max(1250 × 4, 32), 32) = CeilAlign(5000, 32) = 5024

**Golden**：
- usedCoreNum=8, aUbFactor=1250, aLoopCntTotal=8
- aBigCoreCnt=0, aBigCoreLoopCnt=1, aSmallCoreLoopCnt=1
- postBufSize=5024
- SetBlockDim(8)
- isEmptyTensor=1
- ws[0]=sysWorkspaceSize

**校验要点**：
1. maxBufSize = min(ubSize / P_post, 65536) = min(63488, 65536) = 63488；max_ub_factor = maxBufSize / maxDtypeSize = 63488 / 4 = 15872
2. aUbFactor 受 4 约束：4KB buffer 下界（1024） < 优先多核（1250），取 max=1250；UB 上限（15872） > 1250；aTotal（10000） > 1250 → aUbFactor=1250
3. aLoopCntTotal=8 = coreNum=8 → 8 核各处理 1 块（aBigCoreCnt=0，无大小核）
4. usedCoreNum=8（aSmallCoreLoopCnt==1>0 时 usedCoreNum=coreNum）
5. postBufSize = CeilAlign(max(aUbFactor × maxDtypeSize, blockSize), blockSize) = CeilAlign(max(1250 × 4, 32), 32) = CeilAlign(5000, 32) = 5024
6. workspace=sysWorkspaceSize（empty 无用户 workspace，只分配系统 workspace）
7. SetBlockDim(8)（≥ 1，严禁 0）
