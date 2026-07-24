# RegBase Epilogue 设计

> 适用场景：GELU / SwiGLU / LayerNorm / 多输入 scale / 多中间值公式链等复杂后融合。

---

## 1. 适用场景

RegBase 适用于：

- 公式链存在多个中间值。
- 需要减少 UB 中间值写回。
- 需要在寄存器内完成多步运算并最终再写回输出。
- 存在多路额外输入，如 per-channel scale、per-token scale、residual 等。

典型样例：

```text
GELU
SwiGLU
LayerNorm
matmul * scaleN * scaleM + activation
```

---

## 2. 从公式提取 RegBase API

设计流程：

1. 先把用户公式拆成有序的计算链。
2. 判断每个步骤需要的输入、输出和中间值类型。
3. 到 `ascendc-regbase-best-practice` 查找对应 `Reg::` API。
4. 对不能直接确认的 API，再去 `ascendc-api-best-practices` 和本地头文件核验。

以 GELU_tanh 为例：

```text
acc -> cast float
x = acc * scaleN * scaleM
u = sqrt(2/pi) * x * (1 + 0.044715 * x * x)
u_clip = clamp(u, -10, 10)
tanh = (1 - exp(-2*u_clip)) / (1 + exp(-2*u_clip))
y = 0.5 * x * (1 + tanh)
bf16_out = cast(y)
```

常见 API 对应：

| 步骤 | RegBase API |
|------|-------------|
| load | `Reg::LoadAlign` |
| cast | `Reg::Cast` |
| add/sub | `Reg::Add` / `Reg::Sub` / `Reg::Adds` |
| mul | `Reg::Mul` / `Reg::Muls` |
| div | `Reg::Div` |
| exp | `Reg::Exp` |
| clamp | `Reg::Maxs` / `Reg::Mins` |
| store | `Reg::StoreAlign` |

---

## 3. UB 空间分配

RegBase 的核心原则：尽量让中间值停留在 `RegTensor`，UB 只保留输入 staging 和输出 staging。

典型布局：

```text
Offset 0
┌──────────────────────────────────────────┐
│ cLocal_       MatMul 结果区              │ L0C2UB 写入，不可释放
├──────────────────────────────────────────┤
│ extraBufA_    额外输入 A（如 per-channel）│
├──────────────────────────────────────────┤
│ extraBufB_    额外输入 B（如 per-token）  │
├──────────────────────────────────────────┤
│ outBuf_       输出 staging               │
└──────────────────────────────────────────┘
```

预算：

```text
nAlignL0C = ceil(baseN / (32/sizeof(L0CDataType))) * (32/sizeof(L0CDataType))
splitMRows = ceil(baseM / GetTaskRation())
matmulAreaBytes = splitMRows * nAlignL0C * sizeof(L0CDataType)
remainBytes = TOTAL_UB_SIZE - matmulAreaBytes

extraBytes = sum(each extra input staging bytes)
outBytes = stageRows * nAlignOut * sizeof(OutputType)

extraBytes + outBytes <= remainBytes
```

> **关键**：`matmulAreaBytes` 的行步长是 `nAlignL0C`（UB 中的对齐行宽），**不是** L0C cube 边长 16。L0C cube 边长是 L0C 硬件分型粒度，与 UB buffer 行步长无关。常见错误是将 `nAlignL0C` 写成固定值 16，导致 `matmulAreaBytes` 被低估、后续 buffer 重叠到 cLocal_ 数据区。

若 buffer 不够：

- 优先减少 `stageRows`。
- 再考虑调整 `baseM/baseN`。

---

## 4. 多 extra input 设计

复杂公式链经常有多路额外输入。每一路都要回答三个问题：

1. 它依赖 N 维还是 M 维？
2. 它是 tile 级加载还是 stage 级加载？
3. 它是否可以跨 stage 复用？

典型模式：

| 输入类型 | 例子 | 加载方式 |
|----------|------|----------|
| 列依赖输入 | per-channel scale / bias | tile 级加载，可跨 stage 复用 |
| 行依赖输入 | per-token scale / residual row scalar | stage 级加载，offset 依赖 `stageM0` |
| 全局标量配置 | epsilon / alpha / beta | 可直接常量化或通过标量参数传入 |

多 extra input 的 buffer 设计要避免互相覆盖，且要和同步顺序匹配。

### tile 级与 stage 级 input 的同步分离

tile 级 input（加载一次、跨 stage 只读复用）和 stage 级 input（每 stage 覆盖）必须使用**不同 eventID** 的 `V_MTE2` 反向依赖，避免 tile 级 Wait 阻塞 stage 级 Set 造成假依赖：

```text
tile 级 extraBufA_:  V_MTE2(eventID=0) / MTE2_V(eventID=0)
stage 级 extraBufB_: V_MTE2(eventID=1) / MTE2_V(eventID=1)
outBuf_:             MTE3_V(eventID=0) / V_MTE3(eventID=0)
```

> eventID 分离原则详见 `references/fundamentals/blaze-sync-patterns.md` §5.3。

---

## 5. Vector 计算伪代码

RegBase 伪代码模板：

```cpp
for (stageRowOffset = 0; stageRowOffset < localRows; stageRowOffset += stageRows) {
    curStageRows = min(stageRows, localRows - stageRowOffset);

    // 加载额外输入
    DataCopyPad(extraBufA_, extraInputA[row/col offset], ...);
    DataCopyPad(extraBufB_, extraInputB[row/col offset], ...);

    __VEC_SCOPE__ {
        for each row in curStageRows:
            for each vector lane group:
                LoadAlign(cLocal_ -> vregL0C)
                Cast(vregL0C -> vregCompute)
                LoadAlign(extraBufA_ -> vregExtraA)
                LoadAlign(extraBufB_ -> vregExtraB)

                // 公式链
                // [USER] Reg::Mul/Add/Exp/Div/Maxs/Mins

                Cast(vregCompute -> vregOut)
                StoreAlign(vregOut -> outBuf_)
    }

    DataCopyPad(outBuf_ -> output GM)
}
```

要求：

- `nAlign` 必须 per-call 从 `blockShapeN` 计算。
- row-dependent 输入 offset 必须按 SplitM 修正。
- Golden 与 device 最好同式，避免公式差异干扰定位。

---

## 6. MTE2/V/MTE3 同步伪代码

以下伪代码展示 tile 级与 stage 级 input 的完整同步结构（eventID 分离）：

```cpp
// ---- tile 级 extra input：加载一次，跨 stage 只读复用 ----
WaitFlag<V_MTE2>(TILE_EVENT_ID);
DataCopyPad(extraBufA_, extraInputAGm[tileN0], ...);
SetFlag<MTE2_V>(TILE_EVENT_ID);
WaitFlag<MTE2_V>(TILE_EVENT_ID);

// ---- stage 循环 ----
for (stageOffset = 0; stageOffset < localRows; stageOffset += stageRows) {
    // stage 级 extra input：每 stage 覆盖
    WaitFlag<V_MTE2>(STAGE_EVENT_ID);
    DataCopyPad(extraBufB_, extraInputBGm[stageM0], ...);
    SetFlag<MTE2_V>(STAGE_EVENT_ID);
    WaitFlag<MTE2_V>(STAGE_EVENT_ID);

    // 等上一轮 MTE3 读完 outBuf_
    WaitFlag<MTE3_V>(ZERO_FLAG);

    __VEC_SCOPE__ {
        // RegTensor load / cast / compute / store
    }

    // 通知 MTE2 可覆盖 extraBufB_
    SetFlag<V_MTE2>(STAGE_EVENT_ID);

    SetFlag<V_MTE3>(ZERO_FLAG);
    WaitFlag<V_MTE3>(ZERO_FLAG);
    DataCopyPad(outputGm[outputOffset], outBuf_, ...);
    SetFlag<MTE3_V>(ZERO_FLAG);
}

// 通知 MTE2 可覆盖 extraBufA_（tile 级只读复用完毕）
SetFlag<V_MTE2>(TILE_EVENT_ID);
```

### DataCopyPad stride 单位

`DataCopyExtParams` 的 `srcStride` / `dstStride` 单位因搬运方向不同：

| 方向 | srcStride | dstStride |
|------|-----------|-----------|
| GM → UB | bytes | 32 字节单位 |
| UB → GM | 32 字节单位 | bytes |

> 当 UB 行按 `nAlign` 对齐排布时（`nAlign = ceil(N / (32/sizeof(T))) * (32/sizeof(T))`，保证 `nAlign * sizeof(T)` 是 32 的倍数），UB 侧 stride 恒等于 0，直接传 `0` 即可。详见 `references/fundamentals/blaze-sync-patterns.md` §9.5。

同步规则以 `references/fundamentals/blaze-sync-patterns.md` 为准，重点检查：

- 正向依赖：`MTE2_V`、`V_MTE3`
- 反向依赖：`V_MTE2`、`MTE3_V`
- 首轮等待是否需要在 `Init` 中预发射
- 尾轮是否需要收尾等待

---

## 7. SplitM offset

若启用 `DUAL_DST_SPLIT_M`，RegBase 必须按 SubBlock 计算每一路 row-dependent 输入和输出 offset。

### GM offset（需要 sub-block 偏移）

```text
halfM = ceilDiv(blockShapeM, GetTaskRation())
localRows = (oddM) ? (halfM - GetSubBlockIdx()) : halfM

tileM0 = gmOffset / N
tileN0 = gmOffset % N
subM0 = tileM0 + GetSubBlockIdx() * halfM
stageM0 = subM0 + stageRowOffset

rowDependentInputOffset = stageM0
outputOffset = stageM0 * N + tileN0
```

所有依赖 M 维的输入都必须使用 `stageM0` 计算偏移。

### UB 读取（不需要 sub-block 偏移）

`CopyL0C2UBSplitM`（`DUAL_DST_SPLIT_M`）会**硬件自动**将 L0C 的 M 行对半切分，每个 AIV 从自己的 UB offset 0 开始读取半份数据。因此：

```text
srcAddr = cLocal_.GetPhyAddr()  // 从 offset 0 开始
rowSrc = srcAddr + row * nAlign  // 不加 GetSubBlockIdx() * halfM * nAlign
```

| 操作 | 是否需要 SubBlock 偏移 | 公式 |
|------|----------------------|------|
| UB 读取 cLocal_ | **否** | `cLocal_.GetPhyAddr() + row * nAlign` |
| GM 读取 row-dependent input | 是 | `stageM0 = tileM0 + GetSubBlockIdx() * halfM + stageRowOffset` |
| GM 写回 output | 是 | `gmRowOffset = subM0 * N + tileN0` |

> **常见错误**：从 GM offset 公式推断"UB 读取也需要加 `GetSubBlockIdx() * halfM * nAlign` 偏移"，导致 V1 跳过自己的数据读到后续 buffer 的垃圾值。

### localRows=0 边界场景

当 `curM` 为奇数且 `halfM=1` 时（如 `curM=1`），V1 的 `localRows = halfM - 1 = 0`。此时 V1 应 early return，不需要做任何 Vector 计算或 GM 写回。CV 同步由 kernel 层（`MatmulKernelFused`）统一处理，不会因 V1 return 而挂死。

---

## 8. Golden / Device 公式一致性提示

复杂公式链容易出现 “device 公式”和 “golden 公式” 不一致。

建议：

- 设计阶段就明确 golden 与 device 是否同式。
- 若 device 使用 `clip + exp-ratio tanh`，golden 最好也使用同式。
- 若不能同式，必须在设计中写清误差来源和阈值依据。

---

## 9. 常见错误

| 错误 | 后果 | 修复 |
|------|------|------|
| 把中间值写回 UB | UB 压力大，stageRows 降低 | 中间值尽量保留在 `RegTensor` |
| 多路 extra input 共用同一 buffer 无同步保护 | 偶发错数或覆盖 | 分离 buffer 或严格配对事件 |
| 只修 output offset，不修 row-dependent input offset | 下半块 scale/bias 错位 | 所有依赖 M 的输入都按 `stageM0` 修正 |
| UB 读取 cLocal_ 时加 sub-block 偏移 | V1 读到垃圾值，精度完全错误 | SplitM 已硬件分片，从 offset 0 读取 |
| `matmulAreaBytes` 行步长用 16 而非 nAlignL0C | UB buffer 重叠，bias/scale 覆写 matmul 结果 | 行步长 = `nAlignL0C`（不是 L0C cube 边长 16） |
| UB→GM DataCopyPad `srcStride` 传 bytes | tail 场景行步长错误 | UB 侧 stride 是 32B 单位，nAlign 对齐时传 0 |
| tile 级与 stage 级 input 共用同一 eventID | tile 级 Wait 阻塞 stage 级 Set，假依赖 | 使用不同 eventID（如 0 和 1） |
| 缺少 Init 预发射或析构排空 | 首轮 hang 或尾轮 flag 泄漏 | Init 预发射所有反向 SetFlag，析构排空所有 WaitFlag |
| 未检查 Golden 与 device 同式 | 精度问题定位困难 | 设计阶段明确公式一致性 |
