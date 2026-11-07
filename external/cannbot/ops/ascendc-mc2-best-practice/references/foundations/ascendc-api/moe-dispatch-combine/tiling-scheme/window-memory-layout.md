# MoE Dispatch/Combine 窗口内存布局

本文只记录 dispatch 数据区、dispatch 状态区、combine 数据区、combine 状态区的布局公式、地址规则和状态区相关的 UB 大小计算。

进入本页前，默认已经明确：

- dispatch / combine 的阶段划分和中间量语义
- 通信 token 与 `expandIdx` / `epRecvCounts` / `epSendCounts` 的闭环关系
- 是否采用 window + 状态协议，以及 shared expert、量化、mask 等路径是否参与主布局

本页不重复控制流、接口契约和 API 一致性规则，只把这些语义落实成字节级布局。

## Dispatch 数据区布局

### 布局规则

```
排布维度：sourceRank → localExpert → tokenSlot
```

每个 tokenSlot 存放一个完整通信 token = payload（32B 对齐）+ triple（3 × int32）。

### 基础常量

```c++
payloadBytes         = AlignUp(H * sizeof(dtype), 32)   // 32B 对齐
tripleBytes          = 3 * sizeof(int32_t)               // = 12B，[srcRank, srcTokenIdx, srcTopKIdx]
commTokenBytes       = payloadBytes + tripleBytes
expertRegionBytes    = BS * commTokenBytes               // 单个 (srcRank, localExpert) 区域
sourceRankRegionBytes = localExpertNum * expertRegionBytes
rankDataBytes        = epWorldSize * sourceRankRegionBytes
```

### 地址公式

```c++
GM_ADDR dstDataAddr =
    GetBaseWindAddrByRankId(ctx, dstRank, curRankId)
    + (uint64_t)srcRank    * sourceRankRegionBytes
    + (uint64_t)localExpert * expertRegionBytes
    + (uint64_t)dstSlot    * commTokenBytes
```

其中 `dstSlot = curExpertCnt`，即该 token 在目标 expert 已接收 token 中的顺序序号。

### Triple 写入位置

```c++
// payload 写完后，triple 紧随其后
uint32_t tripleOffset = payloadBytes / sizeof(dtype);  // dtype 单位偏移
// triple[0] = srcRankId, triple[1] = srcTokenIdx, triple[2] = srcTopKIdx
```

---

## Dispatch 状态区布局

### 布局规则

```
排布维度：localExpert → sourceRank → stateSlot
```

每个 stateSlot = 32B（8 × int32_t）。

### Slot 格式

```
slot[0] = flag         (0=未就绪, 1=就绪)
slot[1] = tokenCount   (该 srcRank → localExpert 的实际发送 token 数)
slot[2..7] = padding   (仅用于 32B 对齐)
```

### 相关常量

```c++
STATE_SLOT_INT32   = 8                   // 每 slot 8 个 int32_t
STATE_SLOT_BYTES   = 32                  // = 8 * 4
STATE_READY_OFFSET = 0                   // flag 在 slot 中的下标
STATE_TOKEN_OFFSET = 1                   // tokenCount 在 slot 中的下标

totalStateElems    = localExpertNum * rankDim * STATE_SLOT_INT32
totalStateBytes    = localExpertNum * rankDim * STATE_SLOT_BYTES
```

### 地址公式

```c++
GM_ADDR dstStateAddr =
    GetBaseWindStateAddrByRankId(ctx, dstRank, curRankId)
    + ((uint64_t)localExpert * rankDim + (uint64_t)srcRank) * STATE_SLOT_BYTES
```

等待阶段读取本地状态区：

```c++
GM_ADDR localStateBase = GetStatusDataSpaceGm(ctx)
// slot[localExpert][srcRank] 地址：
localStateBase + (localExpert * rankDim + srcRank) * STATE_SLOT_BYTES
```

---

## Combine 数据区布局

### 布局规则

```
排布维度：srcTokenIdx → slotIdx (K + sharedExpertNum slots per token)
前 K 个 slot：MoE expert 结果（按 topK 顺序）
后 sharedExpertNum 个 slot：共享专家结果
```

### 基础常量

```c++
slotCountPerToken    = K + sharedExpertNum
payloadBytes         = H_expand * sizeof(ExpandXType)
slotBytes            = AlignUp(payloadBytes, winAlignBytes)
tokenRegionBytes     = slotCountPerToken * slotBytes
rankCombineBytes     = BS * tokenRegionBytes
```

### 地址公式

```c++
GM_ADDR dataAddr =
    GetBaseWindAddrByRankId(ctx, dstRank, curRankId)
    + (uint64_t)srcTokenIdx * tokenRegionBytes
    + (uint64_t)srcSlotIdx  * slotBytes
```

其中 `srcSlotIdx = srcTopKIdx`（MoE slot）或 `K + sharedExpertSlotIdx`（共享专家 slot）。

---

## Combine 状态区布局

### 布局规则

```
排布维度：srcTokenIdx → slotIdx (同数据区)
```

每个状态槽大小由实现决定，最小为 32B。

### 基础常量

```c++
stateStrideBytes      = 32    // 单个状态槽
tokenStateRegionBytes = slotCountPerToken * stateStrideBytes
```

### Slot 格式

```
slot[0] = readyFlag   (0=未就绪, 1=就绪)
slot[1..7] = padding
```

等待条件：

```c++
// 每个 token 等待自己需要的所有 slot 就绪
targetReadyCount(token) = K + sharedExpertNum   // 无 mask 时
// 有 xActiveMask 或 zero expert 时需相应减少
```

### 地址公式

```c++
GM_ADDR stateAddr =
    GetBaseWindStateAddrByRankId(ctx, dstRank, curRankId)
    + (uint64_t)srcTokenIdx * tokenStateRegionBytes
    + (uint64_t)srcSlotIdx  * stateStrideBytes
```

---

## 状态区相关 UB 大小

```c++
// 写单个状态 slot（SetRemoteStatus 使用）
stateWriteBuf  = STATE_SLOT_INT32 * sizeof(int32_t)  = 32B

// 读全量本地状态区（WaitRemoteStatus 使用）
stateReadBuf   = localExpertNum * rankDim * STATE_SLOT_INT32 * sizeof(int32_t)

// 重置全量本地状态区（ClearLocalStatus 使用）
stateResetBuf  = localExpertNum * rankDim * STATE_SLOT_INT32 * sizeof(int32_t)
```

多核后等待阶段只读自己负责的状态段时，`stateReadBuf` 大小可相应缩减为"本核负责段大小"。

## 下一跳

- 分核数量公式：`multi-core-formulas.md`
- 获取 window 基址的接口：`../api-rules/mte-address-access.md`
- 双缓冲四区协议：`double-buffer-protocol.md`
