# MoE Dispatch/Combine 分核数量计算公式

本文只记录各阶段"发多少工作量、每核分多少"的原子公式，不包含阶段划分原则（见 `index.md`）和地址规则（见 `window-memory-layout.md`）。

## 通用均分公式

所有分核均采用"均分 + 余数前移"策略：

```
basePerCore = totalCnt // aivNum
remain      = totalCnt % aivNum

// 核 i 的工作量
countForCore(i) = basePerCore + (i < remain ? 1 : 0)

// 核 i 的起始偏移
startForCore(i) = i * basePerCore + min(i, remain)
```

## Dispatch 各阶段分核

### 阶段一：Token 发送

按线性发送次数（linearIndex）分核。

```
// MoE 专家路径
sendCntMoe    = BS * K

// 共享专家路径
sendCntShared = BS * sharedExpertNum

// 混合路径（先 MoE 后 Shared，两段各自独立分核）
// 或合并为一段 linearIndex，index < sendCntMoe 属于 MoE，否则属于 Shared
```

**linearIndex 的语义**：`linearIndex = tokenIdx * K + topkIdx`（MoE），单调有序，唯一标识一次发送任务。

### 阶段二：状态发布（Status 写入）

按目标 rank × expert 的"状态槽"数量分核。

```
// MoE 专家卡
statusCntMoe    = epWorldSize * moeExpertNumPerRank

// 共享专家卡（每张共享卡只负责 1 个 expert，每卡给所有 rank 发状态）
statusCntShared = epWorldSize * 1

// 合并参考：sharedExpertRankNum + moeExpertNum （与 reading/design-overview.md 中保持一致）
statusCntTotal  = sharedExpertRankNum + moeExpertNum
```

每个状态槽由唯一的一个核负责写入，不允许多核写同一槽位。

### 阶段三：接收等待

按本卡需要等待的状态槽总数分核。

```
// MoE 专家卡：等待来自所有 srcRank 对每个 localExpert 的状态槽
waitSlotCntMoe    = epWorldSize * moeExpertNumPerRank

// 共享专家卡：等待来自所有 srcRank 对本卡唯一 expert 的状态槽
waitSlotCntShared = epWorldSize * 1
```

每核轮询自己负责的状态段（连续区间），不跨段轮询。

### 阶段四：回搬输出

与等待阶段共用同一组 expert/状态段边界，不另行分核。

```
// 回搬范围与阶段三完全对应
// 核 i 负责的状态段 = [startSlot_i, startSlot_i + slotCount_i)
// 从该状态段对应的 window 区域回搬 expandX / expandIdx
```

## Combine 各阶段分核

### 阶段一：回传发送

按 `epSendCounts` 总发送次数分核。

```
// epSendCounts 来自 dispatch 输出的 epRecvCounts（完全同构）
// 若 epSendCounts 是前缀和形式：
sendCntTotal = epSendCounts[epWorldSize - 1]   // 最后一个元素即总数

// 若 epSendCounts 是数量形式：
sendCntTotal = sum(epSendCounts[0..epWorldSize))
```

每核处理 `countForCore(i)` 次回传，linearIndex 从 0 到 `sendCntTotal - 1`。

### 阶段二：接收等待 + 本地聚合

按 BS（本卡原始 token 数）分核。

```
waitCntPerCore = BS // aivNum
waitRemain     = BS % aivNum

// 核 i 等待并聚合 token 区间 [startForCore(i), startForCore(i) + countForCore(i))
// 等待条件：该 token 需要的所有 expert slot 状态全部就绪
// targetReadyCount(token) = K + sharedExpertNum（或按 mask 调整）
```

## 参数说明

| 符号 | 含义 |
| --- | --- |
| `BS` | 本卡原始 batch token 数（等同于 `bs`） |
| `K` | 每个 token 路由的 MoE expert 数（`topK`） |
| `aivNum` | 当前可用 AIV 核数 |
| `epWorldSize` | EP 域内 rank 总数 |
| `moeExpertNumPerRank` | 本卡负责的 MoE expert 数量 |
| `sharedExpertNum` | 共享专家数量 |
| `sharedExpertRankNum` | 承载共享专家的 rank 数 |

## 与其他文档的关系

- 地址计算：`window-memory-layout.md`
- 分核原则与阶段规则：`index.md` / `split-core-design.md`
- UB 大小参考：`window-memory-layout.md` 中的 UB 节
