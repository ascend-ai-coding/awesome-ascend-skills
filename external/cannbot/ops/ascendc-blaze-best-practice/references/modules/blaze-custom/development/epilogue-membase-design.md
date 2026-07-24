# MemBase Epilogue 设计

> 适用场景：只有一个简单 Vector 操作，且有明确可用的 `AscendC::` LocalTensor API，如 `Mul`、`Add`、`Div`、`Cast`。

---

## 1. 适用场景

MemBase 仅适用于：

- 只有一个简单 vector 操作。
- 该操作有明确可用的 `AscendC::` LocalTensor API。
- 不需要多个中间值或复杂公式链。

典型样例：

```text
output = matmul(A, B) * D
output = matmul(A, B) + bias
```

若需要 GELU / SwiGLU / LayerNorm / 多输入 scale 等复杂链路，应切换到 RegBase。

---

## 2. UB 空间分配

MemBase 的 UB 设计通常包含三块：

```text
Offset 0
┌──────────────────────────────────┐
│ cLocal_      matmul 结果区       │ L0C2UB 写入，不可释放
├──────────────────────────────────┤
│ extraBuf_    额外输入 staging    │ 例如 D / bias
├──────────────────────────────────┤
│ tmpBuf_      计算结果 staging    │ 输出前的临时区
└──────────────────────────────────┘
```

预算原则：

```text
nAlignL0C = ceil(baseN / (32/sizeof(L0CDataType))) * (32/sizeof(L0CDataType))
splitMRows = ceil(baseM / GetTaskRation())
matmulAreaBytes = splitMRows * nAlignL0C * sizeof(L0CDataType)
remainBytes = TOTAL_UB_SIZE - matmulAreaBytes

extraBufBytes = stageElems * sizeof(ComputeType)
tmpBufBytes = stageElems * sizeof(OutputType or ComputeType)

extraBufBytes + tmpBufBytes <= remainBytes
```

> **关键**：`matmulAreaBytes` 的行步长是 `nAlignL0C`，**不是** L0C cube 边长 16。详见 `epilogue-regbase-design.md` §3。

---

## 3. stageSize / stageRows 设计

MemBase 常按行分 stage。`stageRows` 或 `stageSize` 的计算必须在 `cLocal_` 之外的剩余 UB 内完成。

```text
usableBytes = TOTAL_UB_SIZE - matmulAreaBytes
stageRows = floor(usableBytes / (nAlign * (extraDtypeBytes + tmpDtypeBytes)))
stageRows >= 1
```

若 `stageRows == 0`，应：

- 优先减小 C 部分 `baseM/baseN`。
- 或切换到 RegBase，减少 UB 中间值。

---

## 4. DataCopyPad 与 stride

MemBase 的额外输入和输出 staging 通常依赖 `DataCopyPad`。

通用处理：

- 行连续输入：按 `blockCount = rowsThisStage`，`blockLen = blockShapeN * sizeof(T)`。
- `gmRowGap` 用原始 N 维计算。
- `ubRowGap` 按 UB 行对齐后的 gap 计算。

输出方向：

```text
tmpBuf_ UB -> GM output
```

注意：

- `nAlign` 必须 per-call 从 `blockShapeN` 计算。
- tail tile 不能复用 `Init` 时的固定 `baseN` 对齐值。

---

## 5. 同步伪代码

MemBase 同步结构与 RegBase 一致，使用 `V_MTE2` + `MTE3_V` 反向依赖保护 buffer 复用：

```cpp
// Init 中预发射：
SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);  // extraBuf_ 可覆盖
SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);  // tmpBuf_ 可覆盖

while (stageOffset < inputSize) {
    // 反向：等 V 读完上一轮 extraBuf_
    WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);
    DataCopyPad(extraBuf_, extraInputGm[offset], ...);

    // 正向：等 MTE2 加载完成
    SetFlag<HardEvent::MTE2_V>(ZERO_FLAG);
    WaitFlag<HardEvent::MTE2_V>(ZERO_FLAG);

    // 反向：等 MTE3 读完上一轮 tmpBuf_
    WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);

    // [USER] Vector 计算
    AscendC::Mul(tmpBuf_, cLocal_[stageOffset], extraBuf_, curStageSize);

    // 反向：通知 MTE2 可覆盖 extraBuf_
    SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);

    // 正向：等 V 计算完成
    SetFlag<HardEvent::V_MTE3>(ZERO_FLAG);
    WaitFlag<HardEvent::V_MTE3>(ZERO_FLAG);

    // MTE3 写回 GM
    DataCopyPad(outputGm[outputOffset], tmpBuf_, ...);

    // 反向：通知 V 可覆盖 tmpBuf_
    SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);

    stageOffset += curStageSize;
}

// 析构中排空：
WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);
WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);
```

需要根据 `blaze-sync-patterns.md` 检查：

- `MTE2_V` 正向依赖是否存在。
- `V_MTE3` 正向依赖是否存在。
- `V_MTE2` 反向依赖保护 extraBuf_ 不被提前覆盖。
- `MTE3_V` 反向依赖保护 tmpBuf_ 不被提前覆盖。
- Init 预发射 / 析构排空是否完整。

---

## 6. SplitM offset

若启用 `DUAL_DST_SPLIT_M`，MemBase 同样必须按 SubBlock 计算 row offset。

### GM offset（需要 sub-block 偏移）

```text
halfM = ceilDiv(blockShapeM, GetTaskRation())
localRows = (oddM) ? (halfM - GetSubBlockIdx()) : halfM

tileM0 = gmOffset / N
tileN0 = gmOffset % N
subM0 = tileM0 + GetSubBlockIdx() * halfM
stageM0 = subM0 + stageRowOffset

extraInputOffset = stageM0
outputOffset = stageM0 * N + tileN0
```

### UB 读取（不需要 sub-block 偏移）

与 RegBase 相同，`CopyL0C2UBSplitM` 硬件自动分片，UB 读取从 offset 0 开始。详见 `epilogue-regbase-design.md` §7。

### localRows=0 边界场景

当 `curM` 为奇数且 `halfM=1` 时，V1 的 `localRows=0`，early return 即可。详见 `epilogue-regbase-design.md` §7。

---

## 7. 常见错误

| 错误 | 后果 | 修复 |
|------|------|------|
| 公式过复杂仍使用 MemBase | UB 放不下中间值，stageRows 过小 | 切换 RegBase |
| `nAlign` 固定用 `baseN` | tail tile 错位 | per-call 从 `blockShapeN` 计算 |
| 忽略 SplitM 的 row-dependent input offset | SubBlock1 精度错位 | 按 `stageM0` 计算额外输入偏移 |
| UB 读取 cLocal_ 时加 sub-block 偏移 | V1 读到垃圾值 | SplitM 已硬件分片，从 offset 0 读取 |
| `matmulAreaBytes` 行步长用 16 而非 nAlign | UB buffer 重叠 | 行步长 = `nAlign`（不是 L0C cube 边长 16） |
| UB→GM DataCopyPad `srcStride` 传 bytes | tail 场景行步长错误 | UB 侧 stride 是 32B 单位，nAlign 对齐时传 0 |
| 用 `MTE3_MTE2` 自 Set 自 Wait | 无实际同步效果，多 stage 数据竞争 | 用 `V_MTE2` + `MTE3_V` 分别保护 extraBuf_ 和 tmpBuf_ |
| 缺少 Init 预发射或析构排空 | 首轮 hang 或尾轮 flag 泄漏 | Init 预发射所有反向 SetFlag，析构排空所有 WaitFlag |
