# apace 通信原理与接口

> 本文档覆盖 apace 算子的通信侧：通信原理（URMA/Win 区模型、AIV 驱动通信）、通信框架接口（CollectiveComm 四段式 + GET/PUT 钩子）、同步接口（TeamBarrier/CrossCore flag/SyncAll）、host 侧建链机制。

## 目录

1. [通信原理：URMA 与 Win 区模型](#1-通信原理urma-与-win-区模型)
2. [通信框架接口：CollectiveComm 四段式](#2-通信框架接口collectivecomm-四段式)
3. [GET/PUT 钩子职责](#3-getput-钩子职责)
4. [同步接口](#4-同步接口)
   - 4.1 [TeamBarrier（跨卡）](#41-teambarrier跨卡)
   - 4.2 [CrossCoreSetFlag/WaitFlag（跨核）](#42-crosscoresetflagwaitflag跨核)
   - 4.3 [SyncAll（块间）](#43-syncall块间)
5. [通信上下文 CommContext](#5-通信上下文-commcontext)
6. [Host 侧建链机制](#6-host-侧建链机制)
7. [扩展通信原语指南](#7-扩展通信原语指南)
8. [官方 master 漂移登记（pin 之后演进，核对前必读）](#8-官方-master-漂移登记pin-之后演进核对前必读)
- [常见陷阱](#常见陷阱)
- [后续阅读](#后续阅读)

---

## 1. 通信原理：URMA 与 Win 区模型

```
kernel/<op>/<op>_impl.h
    │
    ├── CollectiveComm<Op, Mode, T, Barrier>   ← 统一类型别名（编译期分发）
    │       │
    │       ├── CollectiveCommBase<Impl,...>    ← CRTP 基类（公共逻辑）
    │       │       │
    │       │       └── 4 钩子: PostInit / DoCommit / DoWait / DoFinalize
    │       │
    │       ├── AllToAllCommGetImpl             ← GET 模式实现（ReadNbi，官网暂无算子使用）
    │       ├── AllToAllCommPutImpl             ← PUT 模式实现（WriteNbi）
    │       └── AllGatherCommPutImpl            ← AllGather PUT 实现
    │
    ├── TeamBarrier                             ← 跨卡同步原语（UBMEM 协议）
    │
    └── Hcomm<COMM_PROTOCOL_UBC_CTP>            ← 底层通信对象（ReadNbi/WriteNbi/Drain）
```

### 机制说明

- **Win 区共享 GM**：Win 区是各 rank 共享的 GM 区域。每 rank 的 Win 区基地址存于 `commBufferAddrs[]`，由 Host 侧 `CommChannelBuilder` 建链后填充（见 §5、§6）；kernel 侧按地址公式直接读写远端 Win 区（见 §3）。
- **URMA channel**：每 peer rank 的 URMA channel 句柄存于 `channelHandles[]`（self 不填充，见 §5），经 UDMA 引擎下发通信。
- **AIV 核驱动通信**：UDMA 引擎下通信由 AIV 核发起（ReadNbi/WriteNbi），AIC 专注 Matmul 计算，两侧经 CrossCore flag 同步（见 §4.2）。
- **Hcomm 基础原语**：底层通信对象为 `Hcomm<COMM_PROTOCOL_UBC_CTP>`（ReadNbi/WriteNbi/Drain）；完整签名与约束见 `ascendc-api-best-practices` skill `references/api-hcomm.md`。

### 文件位置（官网仓 `apace/` 相对路径，逻辑引用）

> ⚠️ 下表路径为官网快照的逻辑引用。CANN 内置树中物理子路径随版本漂移（如某版本为 `apace/core/aiv_comm/`、实现文件为 `..._urma_impl.h`），**引用核对时以 Step 1 实测登记的实际路径为准**，禁止按下表写死。

| 组件 | 文件 |
|:---|:---|
| 统一 API + 编译期分发 | `apace/block/aiv_comm/collective_comm_api.h` |
| CRTP 基类 | `apace/block/aiv_comm/collective_comm_base.h` |
| 通信上下文结构体 | `apace/block/aiv_comm/collective_comm_context.h` |
| AllToAll GET | `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` |
| AllToAll PUT | `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` |
| AllGather PUT | `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` |
| TeamBarrier | `apace/block/aiv_comm/barrier/barrier_ubmem.h` |
| Host 建链 builder | `apace/utils/comm_channel_builder.h` |
| CommTilingData 定义 | `apace/tiling/comm_tiling_data.h` |
| PUT 算子样例（AllToAll） | `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` |
| PUT 算子样例（AllGather） | `apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_impl.h` |

### winOffset 多对象复用

当 data 和 scale 两个通信对象复用同一 Win 区时，通过 `winOffset` 区分段（官网 `AllToAllMxQuantMatmulUdmaImpl::Init` 中 `allToAllScaleA_.Init(..., baseParams_.rankSize * baseParams_.rankDataBytes)`）：

```
rankDataBytes = axisM × axisKa × sizeof(AType)   // axisM = 单个通信 chunk 的 M 行数（= 每卡输出 M 段行数 m，本卡 A 总 M 为 rankSize × axisM），axisKa = 本卡 A 的 K 轴大小（官方 udma_impl.h InitBaseParams）
winOffset_scale = rankSize × rankDataBytes
```

**Win 区布局（winOffset 复用）**：

```
commBufferAddrs[rankId] → ┌──────────────────────────────┐
                          │  data chunk[0]                │  ← winOffset=0
                          │  data chunk[1]                │
                          │  ...                          │
                          │  data chunk[rankSize-1]       │
                          │  scale chunk[0]               │  ← winOffset=rankSize×rankDataBytes
                          │  scale chunk[1]               │
                          │  ...                          │
                          │  scale chunk[rankSize-1]      │
                          └──────────────────────────────┘
```

> **注意**：每个通信对象需要独立的 UB commBuf（COMM_WORKSPACE_SIZE = 512B），两个对象共需双对象翻倍 + barrier UB（见 §4.1 UB 预算）。

### UDMA vs HCCL windows 双引擎

| 引擎 | 底层 API | CommContext | 使用场景 | 直调支持 |
|:---|:---|:---|:---|:---|
| **UDMA** | `Hcomm::ReadNbi/WriteNbi/Drain` | 需要 `CommContext{udmaCtx, ubmemCtx}` | AIV 核驱动通信 | **是** |
| **HCCL windows** | `GetHcclContext<0>()` | 不需要 `CommContext` | 组合模式（CCU hcomm 变体） | 否（官网存在 CCU hcomm 变体 `all_to_all_mx_quant_matmul_hcomm_impl.h`，但无 `__global__` 直调入口） |

#### 验收条件

| 模式 | 入口签名特征 | tiling_data.h 特征（启发式判据） |
|:---|:---|:---|
| UDMA | 含 `__gm__ CommContext*` 参数 | 该模式使用的 tiling 结构含 `CommContext` 聚合体 |
| HCCL windows | 含 `GetHcclContext`，无 `CommContext` 参数 | 该模式使用的 tiling 结构不含 `CommContext` |

> tiling_data.h 判据是启发式：同一头文件可共存多种 tiling 结构（如 `all_to_all_matmul_tiling_data.h` 同时定义 `CommContext` 和 CCU 变体用的 `ccuAllToAllMatmulTilingData`），判据应针对具体使用的结构而非整个文件。

> **注意**：HCCL windows（`GetHcclContext`）是 kernel 级 API，与 blaze-shmem 路线禁止的 HCCL 高阶 API（`Hccl::AllReduce` 等服务端调度 API）不同。apace 路线允许 HCCL windows。

---

## 2. 通信框架接口：CollectiveComm 四段式

### 2.1 四段式语义

| 阶段 | 语义 | 阻塞性 |
|:---|:---|:---|
| `Init()` | 初始化通信对象，分配 targetRank，计算偏移 | 同步 |
| `Commit()` | 发起当前 tile 的通信（GET 拉 / PUT 推），非阻塞返回 | **非阻塞** |
| `Wait()` | 等待 Commit 发起的通信完成 | 阻塞 |
| `Finalize()` | 收尾（最终 barrier 等） | 同步 |

Commit 的非阻塞特性是通算流水的关键——AIV 可以在 Commit 后、Wait 前插入其他指令。

### 2.2 CRTP 基类 CollectiveCommBase

`CollectiveCommBase`（`apace/block/aiv_comm/collective_comm_base.h`）使用 CRTP 模式，提供公共逻辑，派生类只需实现 4 个钩子（`PostInit` / `DoCommit` / `DoWait` / `DoFinalize`）。

#### Init 完整签名（`CollectiveCommBase::Init`）

```cpp
template<uint8_t BarrierMode = BARRIER_BOTH>
__aicore__ inline void Init(
    __gm__ CommUdmaContext* udmaCtx,   // UDMA 通信上下文
    Barrier& barrier,                   // TeamBarrier 实例
    const CommTilingData& tilingData,   // 通信切分参数（5 字段，见 apace/tiling/comm_tiling_data.h）
    GM_ADDR localAddr,                  // 本地 GM 地址（GET=目标 cGM，PUT=源 aGM）
    __ubuf__ uint8_t* commbuf,          // UB 通信 workspace（COMM_WORKSPACE_SIZE = 512B）
    uint32_t totalJobs,                 // 参与通信的核数（通常=rankSize）
    uint32_t jobIndex,                  // 当前核索引（GetBlockIdx()）
    uint64_t winOffset = 0);            // Win 区偏移（多对象复用用）
// 返回值：void。totalTiles 通过 GetCommTurn() 获取，另有 GetCommByteSize() 访问器
```

BarrierMode 常量（同文件定义）：`BARRIER_NONE=0`、`BARRIER_DEVICE=1`、`BARRIER_CORE=2`、`BARRIER_BOTH=3`。

> **官方调用形态锚点（winOffset 缺省 = 0）**：官方数据对象 Init 为 **7 参调用**（省略第 8 参 `winOffset`，走默认值 0）——`all_to_all_mx_quant_matmul_udma_impl.h:198-199` 的 `allToAllA_.Init(...)` 无 winOffset 实参。官网布局（barrier 在独立 BARRIER_BUF）下 Win 数据区从 0 可用，winOffset=0 即正确；仅多对象复用同一 Win 区时显式传第 8 参分段（见 §"winOffset 多对象复用"）。另注意：`Commit()` 不带模板参时默认 `BARRIER_BOTH`，与 `Init<BARRIER_NONE>` 不同源——PUT 路径 DoCommit 不消费 BarrierMode 故两者功能等价，但风格上应与 Init 对齐显式写 `Commit<BARRIER_NONE>()`（GET 路径不等价，见 §2 BarrierMode 选择规则）。

**BarrierMode 选择规则（官网锚点）**：

| 场景 | 规则 | 锚点 |
|------|------|------|
| 多通信对象共享同一 TeamBarrier | **仅一个对象使能 barrier，其余 `Init<BARRIER_NONE>` 去重**，避免重复同步 | AllToAll：`allToAllA_.Init<BARRIER_NONE>` + scale 默认（`all_to_all_mx_quant_matmul_udma_impl.h` Init）；AllGather：data `BARRIER_NONE` + scale `BARRIER_DEVICE`（`all_gather_mx_matmul_udma_impl.h` Init） |
| 同 channel 多对象（data + scale） | **只 Wait/Drain 一次**（"scale 和 a 矩阵的通信使用同一 channel，因此只需要 wait 一次"） | `RunAllToAll()` 只对 `allToAllA_.Wait<BARRIER_DEVICE>()` |

#### Init 不变量

| 不变量 | 说明 |
|:---|:---|
| 保存上下文 | udmaCtx、barrier、localAddr、tilingData、commBuf、winOffset |
| 底层 comm_ 初始化 | Hcomm 对象使用 commBuf 的 COMM_WORKSPACE_SIZE（512B）workspace |
| jobIndex → targetRank 自动分核映射 | `targetRankPerCore = ceil(rankSize / totalJobs)`；`targetRankStart = jobIndex * targetRankPerCore`；`targetRankCnt` 三分支：`targetRankStart + targetRankPerCore <= rankSize` 取 `targetRankPerCore`，`targetRankStart < rankSize` 取 `rankSize - targetRankStart`，否则钳到 0 |
| **早退语义** | `jobIndex >= totalJobs` 时 Init 提前 return，字段未初始化——调用方必须用分核守卫保护后续 Commit/Wait/Finalize（见本节末尾「AIV 分核惯例」） |
| chunk 大小计算 | 从 CommTilingData 的 5 字段推导：`chunkSize = splitAxisTileSize*splitAxisTileCnt + splitAxisTailSize*splitAxisTailCnt`；`chunkBytes_ = chunkSize * nonSplitAxisSize * sizeof(Dtype)`；`tileMaxByteSize_ = max(splitAxisTileSize, splitAxisTailSize) * nonSplitAxisSize * sizeof(Dtype)` |
| PostInit 钩子调用 | Init 末尾调用 `PostInit<BarrierMode>()`，派生类可在此插入前置逻辑（如 PUT 的 barrier） |
| 返回值 | **void**（totalTiles 经 `GetCommTurn()` 获取） |

#### 访问器

| 方法 | 语义 |
|:---|:---|
| `GetCommTurn()` | 返回 `splitAxisTileCnt + splitAxisTailCnt`（总通信轮次/tile 数） |
| `GetCommByteSize()` | 返回 `chunkBytes_ * rankSize`（本 rank 通信总字节数） |

#### Commit/Wait 不变量

| 阶段 | 不变量 |
|:---|:---|
| Commit | `remainingChunkSize_ <= 0` 则直接返回；计算当前 tile 大小（`currentTileIdx_ < splitAxisTileCnt` 取头块 `splitAxisTileSize`，否则取尾块 `splitAxisTailSize`，再钳到 `remainingChunkSize_`）；遍历 `targetRankCnt_` 个 targetRank 调用 `DoCommit<BarrierMode>(targetRankId, currentTileByteSize)`；更新 `currentTileIdx_++`、`slotByteOffset_ += rankSize * tileMaxByteSize_`、`tileByteOffset_`、`chunkByteOffset_ += chunkBytes_`、`remainingChunkSize_ -= currentTileSize` |
| Wait | 签名 `Wait<BarrierMode = BARRIER_BOTH>(bool waitLast = false)`（`collective_comm_base.h:132-142`）。`Wait(false)`（默认）每个 tile 都执行 `DoWait`；`waitLast=true` 是调用方**可选**的早退语义：`if (waitLast && currentTileIdx_ != totalTiles - 1) return;`——在典型的 `Commit(); Wait(true);` 逐 tile 循环中（Commit 末尾 `currentTileIdx_++`），`DoWait` 仅在 `currentTileIdx_ == totalTiles - 1` 时执行一次（即倒数第二轮），**最后一轮通信不被 Drain**。循环体：遍历 `targetRankCnt_` 个 targetRank 调用 `DoWait<BarrierMode>(targetRankId)`。⚠️ 使用 waitLast 模式（GET 语义）时需自行评估该行为是否满足时序要求 |

#### 受保护字段（基类提供，钩子可访问）

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `udmaCtx_` | `__gm__ CommUdmaContext*` | UDMA 通信上下文（rankId、rankSize、channelHandles、commBufferAddrs） |
| `tilingData_` | `const CommTilingData*` | 通信切分参数（5 字段） |
| `barrier_` | `Barrier` | 跨卡同步原语 |
| `comm_` | `Hcomm<COMM_PROTOCOL_UBC_CTP>` | 底层通信对象 |
| `localAddr_` | `GM_ADDR` | 本地 GM 地址（GET=目标，PUT=源） |
| `commBuf_` | `__ubuf__ uint8_t*` | UB 通信 workspace |
| `winOffset_` | `uint64_t` | Win 区偏移 |
| `chunkBytes_` | `uint64_t` | 一个完整 chunk 的字节数 |
| `currentTileIdx_` | `uint64_t` | 当前 tile 索引 |
| `tileByteOffset_` | `uint64_t` | 当前 tile 在 chunk 内的字节偏移 |
| `tileMaxByteSize_` | `uint64_t` | 最大 tile 字节数 |
| `slotByteOffset_` | `uint64_t` | 当前 slot 偏移（环形） |
| `targetRankStart_` | `uint32_t` | 本核负责的起始 targetRank |
| `targetRankCnt_` | `uint32_t` | 本核负责的 targetRank 数量 |
| `remainingChunkSize_` | `uint64_t` | 当前 chunk 剩余未通信字节数 |
| `chunkByteOffset_` | `uint64_t` | 当前 chunk 内字节偏移 |

#### AIV 分核惯例（按场景二选一，禁止混用）

| 场景 | 分核映射 | 适用 |
|:---|:---|:---|
| **通信在前**（官方 A2A/AG PUT 算子） | **前 R 核通信**：`if (GetBlockIdx() < rankSize)` 守卫包裹 Commit/Wait（配合 Init 早退语义——超出 rankSize 的 block 未初始化通信字段）；`SyncAll<true>()` 与 `CrossCoreSetFlag` 在守卫外由所有 AIV block 执行，Finalize 无守卫（全 AIV 执行） | `all_to_all_quant_matmul`（RunAllToAll）、`all_gather_quant_matmul`（AllGatherProcess）官方惯例，移植时以官网 kernel 实现为准 |
| **compute-first 严格分离**（默认生产形态） | **后 R 核通信**：`jobIndex = GetBlockNum() - 1 - GetBlockIdx()`，`isCommBlock = (jobIndex < rankSize)`；**前 (核数-R) 核归约**：`isComputeBlock = (blockIdx < usedCoreNum - rankSize)` | 计算在前算子（如 ReduceScatter）严格分离编排，AllToAll(t) ∥ ReduceSum(t-1) 错位流水（[`fusion.md`](fusion.md) §6.2.1） |

两个惯例的**物理核映射不同**，totalJobs 语义也随之不同：通信在前（官方）惯例下通信对象与 TeamBarrier **均为 totalJobs=rankSize**（`teamBarrier_.Init(buf, ctx, rankSize, GetBlockIdx())`，CrossDevice 由 `Wait<BARRIER_DEVICE>` 内建触发，前 R 核分布式轮询覆盖全部 rank）；compute-first 惯例下通信对象 totalJobs=rankSize、TeamBarrier 可采用 totalJobs=1 + 显式 CrossDevice（后 R 核映射下 blockIdx≥rankSize 会被 TeamBarrier 守卫早退，须由指定核显式同步）。混用两种映射（如前 R 核通信 + 后段核归约但 rsCoreNum 按前段算）会导致归约核与通信核重叠或空转，同步计数失衡。

#### 通信并行度：totalJobs 配置（红线）

通信对象与 TeamBarrier 的 `totalJobs` 是**两个独立配置**，按分核惯例取值，禁止跨惯例混搭：

| 对象 | 通信在前（官方 A2A/AG） | compute-first 严格分离（自研） |
|:---|:---|:---|
| AllToAll/AllGather 通信对象 | **rankSize**：前 R 核各负责 1 个 targetRank 并行 PUT（`GetBlockIdx() < rankSize` 守卫），通信时间降为串行的 1/rankSize | **rankSize**：后 R 核各负责 1 个 targetRank |
| TeamBarrier | **rankSize**（官方两算子均为 `teamBarrier_.Init(buf, ctx, rankSize, GetBlockIdx())`）：前 R 核各作 jobIndex 参与，`Wait<BARRIER_DEVICE>` 内建触发 CrossDevice，分布式轮询覆盖全部 rank | **1**（自研方案）：仅 jobIndex=0 的核显式执行 CrossDevice（step=1 轮询所有远端 rank）——因后 R 核映射下 blockIdx≥rankSize 会被 TeamBarrier 的 jobIndex 守卫早退，沿用 totalJobs=rankSize 将失效，故改用 1 + 显式调用 |

> ⚠️ **已证伪的臆造约束**："多核同时写同一 UBMEM flag 存在竞态，因此通信必须 totalJobs=1（仅 blockIdx==0 执行 Commit/Wait）"——**该约束不存在**。TeamBarrier 自身保证每个 job 只触碰自己的 per-job flag 槽位（CrossDevice 计数区 `base + 32 + jobIndex*32`），多核 PUT 各核写各自 targetRank 的 Win 槽位与各自的 channel，不写同一 flag。把**通信对象**退化为 totalJobs=1 会让 R 个 target 串行 PUT，通信时间放大 R 倍，是生产实测过的重大性能回退（见 optimization-playbook.md）。

---

## 3. GET/PUT 钩子职责

GET/PUT 的算子级编排模式见 `fusion.md`，本节只描述钩子契约。

### 3.1 GET 钩子（AllToAllCommGetImpl）

> **官网暂无 GET 算子样例**：GET 钩子基础设施（`apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` 的 `AllToAllCommGetImpl`）已就绪并注册进 `CollectiveCommHelper<AllToAll, GET, ...>` 分发（`apace/block/aiv_comm/collective_comm_api.h`），但官网 kernel/ 下两个算子均为 PUT 模式，无 GET 使用方。本节为钩子契约级描述，地址公式与 self 跳过规则均可在该头文件中直接验证。

GET 模式 = 计算→通信：AIC 先算 C 写到 Win 区，AIV 从远端 Win 区拉回本 rank 的 C 段。

#### GET 钩子不变量表

| 钩子 | 不变量 | 违反后果 |
|:---|:---|:---|
| `PostInit()` | 空实现（GET 不需要前置 barrier） | 若加入 barrier，产生不必要的同步开销 |
| `DoCommit(targetRankId, tileByteSize)` | ① 按 `BarrierMode` 按需调用 `CrossCore()`/`CrossDevice()`（确保对端 AIC 已写入 Win 区） ② 跳过 `targetRankId == rankId`（self rank，直接 return） ③ `srcAddr = commBufferAddrs[targetRankId] + winOffset_ + slotByteOffset_ + rankId * tileMaxByteSize_` ④ `dstAddr = localAddr_ + targetRankId * chunkBytes_ + tileByteOffset_` ⑤ `ReadNbi` 返回值经 `ascendc_assert(ret == 0, ...)` 检查 | ① 遗漏 barrier → 读到未初始化数据 ② 未跳过 self → 自身 Win 区读写错误 |
| `DoWait(targetRankId)` | ① 跳过 `targetRankId == rankId`（self 直接 return） ② `Drain` 返回值经 `ascendc_assert(ret == 0, ...)` 检查 | ① 未跳过 self → 无谓 Drain |
| `DoFinalize()` | 按 `BarrierMode` 按需调用 `CrossCore()`/`CrossDevice()` | 遗漏 → 后续操作可能访问未完成通信的 buffer |

#### Barrier 机制

模板参数 `BarrierMode` 控制 barrier 行为（编译期 `if constexpr (BarrierMode & BARRIER_*)`，常量定义于 `apace/block/aiv_comm/collective_comm_base.h`）：
- `BARRIER_CORE`：跨核 barrier（`barrier_.CrossCore()`）
- `BARRIER_DEVICE`：跨设备 barrier（`barrier_.CrossDevice()`）

GET 的 `DoCommit()` 和 `DoFinalize()` 中 barrier 调用顺序为**先 Core 后 Device**。

> ⚠️ **GET 模式 Commit 前 barrier 不可省**：`DoCommit` 内的 `CrossCore()+CrossDevice()` 是生产者（对端 AIC）槽位就绪的回压保证，删除会读到未写入数据；对应 `DoFinalize()` 的 barrier 同样不可省（防止尾部越界覆盖）。

#### GET 地址语义

```
远端 rank 的 Win 区布局:
┌─────────────────────────────────────────┐
│  rank 0 的槽位     rank 1 的槽位  ...   │  ← commBufferAddrs[targetRankId]
├─────────┬─────────┬─────────┬──────────┤
│ tileMax │ tileMax │ tileMax │  ...     │
│ ByteSize│ ByteSize│ ByteSize│          │
└─────────┴─────────┴─────────┴──────────┘
     ↑
     本 rank 的槽位 = slotByteOffset_ + rankId * tileMaxByteSize_

本地 cGM 布局:
┌─────────────────────────────────────────┐
│  rank 0 的 C 段   rank 1 的 C 段  ...   │  ← localAddr_
├─────────┬─────────┬────────────────────┤
│ chunkBytes │ chunkBytes │  ...          │
└─────────┴─────────┴────────────────────┘
     ↑
     本 tile 偏移 = targetRankId * chunkBytes_ + tileByteOffset_
```

#### 自跳过规则

GET 的 DoCommit/DoWait 和 PUT 的 DoCommit 都会跳过 `targetRankId == rankId`（self rank，直接 return），因为本 rank 的数据已在本地，不需要跨卡通信。注意 PUT 的 `DoWait` 对 self 仅跳过 `Drain`，**仍会执行** CrossDevice/CrossCore barrier（`apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 的 `AllToAllCommPutImpl::DoWait`：Drain 在 `if (targetRankId != rankId)` 内，barrier 在其外）。

### 3.2 PUT 钩子（AllToAllCommPutImpl）与 GET 差异

PUT 模式 = 通信→计算：AIV 先推数据到远端 Win 区，AIC 从 Win 区读取计算。实现见 `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 的 `AllToAllCommPutImpl`（AllGather PUT 变体见 `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` 的 `AllGatherCommPutImpl`，钩子结构相同，仅地址公式不同）。

#### 与 GET 的钩子差异

| 钩子 | GET | PUT |
|:---|:---|:---|
| `PostInit()` | 空 | `CrossDevice()+CrossCore()`（通知对端即将写入） |
| `DoCommit()` | `CrossCore()+CrossDevice()` → `ReadNbi`（拉） | `WriteNbi`（推），**无**前置 barrier；self rank 直接 return |
| `DoWait()` | `Drain`（self 直接 return） | `Drain`（仅非 self）→ `CrossDevice()+CrossCore()`（含 self） |
| `DoFinalize()` | `CrossCore()+CrossDevice()` | 空 |

> **顺序差异**：GET 钩子内 barrier 顺序是先 Core 后 Device；PUT 的 PostInit/DoWait 是先 Device 后 Core。以 `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` / `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 实际代码为准。

PUT 的 src/dst 与 GET 完全镜像：GET 从远端读，PUT 往远端写。PUT 地址公式（`AllToAllCommPutImpl::DoCommit`）：`srcAddr = localAddr_ + targetRankId * chunkBytes_ + currentTileIdx_ * tileMaxByteSize_`；`dstAddr = commBufferAddrs[targetRankId] + winOffset_ + rankId * chunkBytes_ + tileByteOffset_`。

> ⚠️ **PUT/GET 数据区与 Win 区元数据区必须分离（布局验证原则）**：Win 区内若存在元数据/barrier 区，通信数据写入偏移必须跳过该区域——0 偏移覆盖元数据会造成"假通过"（精度碰巧正确但同步机制已被破坏，大 shape/多轮时紊乱），精度验证无法发现，必须靠设计红线拦截。注意两种布局并存：① apace 官网布局下 TeamBarrier flag 位于 `CreateDeviceContext` 独立分配的 2MB BARRIER_BUF（见 §4.1），**不在 Win 数据区内**，Win 数据区从偏移 0 可用；② 共享 Win 区布局的实现（部分生产算子将 barrier counter 置于 Win 区头部）必须按约定偏移跳过（示例：128B）。**偏移由 host 建链布局决定，host 侧预留与 kernel 侧读写偏移必须同源**。

> GET/PUT 的算子级编排模式见 `fusion.md`。

---

## 4. 同步接口

### 4.1 TeamBarrier（跨卡）

`TeamBarrier`（`apace/block/aiv_comm/barrier/barrier_ubmem.h`）是基于 UBMEM 协议的跨卡同步原语，替代 blaze-shmem 路线的 `aclshmemx_barrier_all_vec`。

#### 关键特性

- 基于 GM flag counter 递增 + 轮询远端 flag
- **支持部分核参与**（`totalJobs` / `jobIndex` 参数），非全核 barrier
- 仅 AIV 核执行（内部 `if ASCEND_IS_AIV` 保护；AIC 调用是空操作，不会报错，但应避免）
- UB 需求量固定 32 字节（`UB_SIZE = BARRIER_FLAG_SIZE = 32`）

#### 常量

```cpp
constexpr uint32_t BARRIER_FLAG_SIZE = 32;              // 每个同步槽 32B
constexpr uint32_t UB_SIZE = BARRIER_FLAG_SIZE;          // kernel 侧 ubOffset 累加用
constexpr uint32_t BARRIER_FLAG_ELEMS = 8;               // 32B / sizeof(int32_t)
```

#### Init

```cpp
__aicore__ inline void Init(
    __ubuf__ uint8_t* syncBuf,           // UB 同步缓冲（32B 对齐）
    __gm__ CommUbmemContext* ctx,        // barrier 通道上下文（含远端 flag 地址）
    uint32_t totalJobs,                   // 总 job 数（参与同步的核数）
    uint32_t jobIndex);                   // 当前核索引（GetBlockIdx()）
```

#### CrossDevice 机制（跨卡，仅 AIV）

`jobIndex_ >= totalJobs_` 时提前 return，否则：

1. 读本 rank per-job 槽（`commBufferAddrs[rankId] + 32 + jobIndex*32`），count+1
2. 先把 count 写到**基址 flag**（`commBufferAddrs[rankId]`，偏移 0），随后轮询**其他 rank 的基址 flag**（偏移 0）直到 ≥ count——注意轮询的是**跨步子集**：`step = min(totalJobs, rankSize)`，`for (i = jobIndex; i < nranks; i += step)`，跳过本 rank，并非轮询所有其他 rank 的 per-job counter（`barrier_ubmem.h:149-189`，`CrossDeviceExecute`）
3. 轮询通过后才把 count 写回 per-job 槽
4. **无超时保护**：远端 rank 未就绪将无限等待挂死（不会 assert）。规避：确保所有 rank kernel 已 launch，且 `CreateDeviceContext` 后做了跨 rank host barrier（见 §6）

> ⚠️ **框架限制：totalJobs=rankSize 时跨设备同步静默失效（compute-first 后 R 核映射场景）**。失效机制为复合根因：**主因——per-rank 单槽基址 flag 多核共写竞态 + 进度合并**（每 rank 的基址 flag 是偏移 0 处的**单值槽**，totalJobs=rankSize 时同 rank 的多个核各自 count+1 后写同一基址槽，任一核写入即可能让对端轮询提前通过——同步语义退化为"任一核到达"而非"全部核到达"）；**次因——守卫早退下的等效零轮询**（后 R 核映射时 `jobIndex = GetBlockNum()-1-GetBlockIdx()` 可能 ≥ totalJobs 或映射错位，被 Init/CrossDevice 守卫跳过或轮询不到有效远端子集）。注意：前 R 核映射（官方惯例，jobIndex=GetBlockIdx()<rankSize）下 totalJobs=rankSize 是**官方验证可用**的形态——块 j（j≠rankId）各自轮询远端 rank j 恰一个（`step=rankSize` 时 `for (i=jobIndex; i<nranks; i+=step)` 首轮即命中 i=jobIndex），分布式覆盖全部 rank；官方 A2A/AG 均用此形态且 ST 全量通过。现象（compute-first 场景实测）：Rank0 精度 PASS、Rank1 NaN/大面积元素错误（约 80% 数据错误）。**解法（compute-first 后 R 核映射专用）**：TeamBarrier `totalJobs=1`（单核写基址槽 + `step=1` 正确轮询所有 remote rank）+ 通信对象 `totalJobs=rankSize`（工作分片，后 R 核各负责 1 个 target 并行 PUT），手动 `teamBarrier_.CrossDevice()` 完成跨设备 fence。完整失败链（5 次迭代）见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.1a。

#### CrossCore 机制（跨核，仅 AIV）

`jobIndex_ >= totalJobs_` 时提前 return，否则：

1. 读本 job 的 localFlag（`commBufferAddrs[rankId] + 32 + totalJobs*32 + jobIndex*32`）
2. count+1 写回
3. 轮询其他 job 的 flag 直到 ≥ count（同样无超时，无限 do-while）

#### GM flag 区来源与预算

TeamBarrier 轮询的 GM flag 位于 `CreateDeviceContext` 内部分配的 2MB `BARRIER_BUF_SIZE` 区域（device ctx 之后），容量需求为 `(1 + 2×totalJobs) × 32B`/rank（基址 flag + CrossDevice 槽 + CrossCore 槽）。注意：`aclrtMemset` 清的是 HCCL **数据** buffer；barrier flag 区无显式 memset，零初值依赖 `HcclEngineCtxCreate` 的分配语义——新算子建议显式 memset 或验证该假设。

#### UB 预算

单通信对象：`COMM_WORKSPACE_SIZE`(512B) + `UB_SIZE`(32B) = **544B**。
data+scale 双通信对象：512×2 + 32 = **1056B**。

> **AIV 归约侧 UB 总预算**：DAV_3510 硬件 UB = 248KB 框架可用（`GetCoreMemSize(UB)` 运行时获取，示例值 253952；芯片规格以 npu-arch skill 为唯一知识源，数值仅工程参考）。MC2 通算融合算子中 AIV 归约模块推荐 `TOTAL_UB = 192KB`、可分配上限 `MAX_UB_BYTES = 180KB`（扣除 guard 通信区后）。6-slot 归约布局下每元素 18B，`maxElements = MAX_UB_BYTES / 18`。详见 [`architecture.md`](architecture.md) §6 UB 容量说明、[`fusion.md`](fusion.md) §6.2.6 归约 UB 布局。

#### 常见错误

| 错误 | 后果 | 正确做法 |
|:---|:---|:---|
| `BARRIER_NONE` 但无外部 CrossCore flag 保证时序 | GET 读到 AIC 未写完的数据 | `Init<BARRIER_NONE>` 时必须靠 `CrossCoreWaitFlag` 保证 AIC 已写完 |
| barrier UB 按 64B 分配 | 多分配 32B | 按 `UB_SIZE = 32B` 分配 |
| 远端 rank 未启动 | CrossDevice **无限等待挂死**（无超时保护） | 确保所有 rank 已 launch + host 侧跨 rank barrier |
| AIC 侧调用 TeamBarrier | 空操作（内部 ASCEND_IS_AIV 保护），但语义混乱 | 仅在 AIV 侧调用 |

#### 与 SHMEM barrier 的区别

| 特性 | SHMEM `barrier_all_vec` | apace `TeamBarrier` |
|:---|:---|:---|
| 协议 | SHMEM/URMA | UBMEM |
| 参与者 | 全部核 | 支持部分核（jobIndex/totalJobs） |
| 实现机制 | SHMEM 库内部 | GM flag 轮询（用户可见） |
| 执行核 | AIV | AIV |
| 超时保护 | 依赖库实现 | **无**（无限等待） |

> 注：SHMEM 列以 SHMEM 文档为准，非 apace 仓实证。

### 4.2 CrossCoreSetFlag/WaitFlag（跨核）

CrossCore Flag 是 AIC↔AIV 跨核同步的核心机制。每个 flag 由 `<MODE, PIPE, flagId>` 三元组标识。完整签名、flagId 硬件规则与平台生效性见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`；GET/PUT 的 flag 编排不变量见 `fusion.md`。

#### MODE 常量

| MODE 值 | 常量名 | 含义 |
|:---|:---|:---|
| `0x2` | `CROSS_CORE_INNER_CUBE_VEC_SYNC` | Cube↔Vector 同核同步（apace 唯一使用） |

> 注：apace 代码中使用字面量 `0x2`（如 `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)`）；常量名 `CROSS_CORE_INNER_CUBE_VEC_SYNC` 是 MC2 框架惯例命名，非 apace 仓符号。

#### PIPE 选项

| PIPE | 含义 | 典型场景 |
|:---|:---|:---|
| `PIPE_FIX` | FixPipe（AIC 侧） | AIC 完成 fixpipe 输出后 SetFlag |
| `PIPE_M` | M（MAD/Cube 主流水，AIC 侧） | AIC 等待回压（WaitFlag） |
| `PIPE_S` | Scalar（AIV 侧） | AIV 等待 AIC 通知（WaitFlag） |
| `PIPE_MTE3` | Mte3（AIV 侧） | AIV 完成通信后 SetFlag 回压 |

#### flagId 选择规则

flagId 选择规则（16 通道截断机制、计数器 0-15 衡量未消费积压、轮次索引避开 SyncAll 保留区）统一维护在 [`fusion.md`](fusion.md) §3.3，本节不重复。

### 4.3 SyncAll（块间）

`SyncAll<true>()` 是 AIV 块间硬同步原语，其内部占用保留 flagId **14**（`SYNC_AIV_ONLY_ALL`；`<false>` 变体占 11/12/13——保守避让整个 [11,14]，见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §3）。PUT 模式的逐轮编排用法（每轮 `SyncAll<true>()` 保证 WriteNbi 对端可见性）见 `fusion.md`；完整签名与平台生效性见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`。

> ⚠️ `SyncAll<false>()`（非 isAIVOnly 变体）需要 **AIC + AIV 双方参与**——只在单侧调用会永久等待（调试实测踩坑）。MC2 场景块间同步一律用 `SyncAll<true>()`（仅 AIV），除非确认 AIC 侧也有对齐的调用点。

---

## 5. 通信上下文 CommContext

聚合体 `CommContext{udmaCtx, ubmemCtx}` **不**在 `apace/block/aiv_comm/collective_comm_context.h` 中定义——该头文件仅定义 `CommUdmaContext` / `CommUbmemContext` 两个子结构及常量（`COMM_MAX_RANK_NUM`、`COMM_WORKSPACE_SIZE`）。聚合体 `CommContext` 由各算子在自己的 tiling_data.h 中定义：

| 算子 | 定义位置 | 命名空间 |
|:---|:---|:---|
| PUT（all_to_all_quant_matmul） | `apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h` | 全局命名空间 |
| AG（all_gather_quant_matmul） | `apace/kernel/all_gather_quant_matmul/all_gather_mx_matmul_udma_tiling_data.h` | `Apace::AivComm`（文件内 `using Apace::AivComm::CommContext;` 导出到全局） |

### 结构

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `udmaCtx` | `CommUdmaContext` | UDMA 通信通道 |
| `ubmemCtx` | `CommUbmemContext` | Barrier 通道 |

### CommUdmaContext

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `rankId` | `uint32_t` | 本 rank ID |
| `rankSize` | `uint32_t` | 总 rank 数 |
| `channelHandles[]` | `uint64_t[COMM_MAX_RANK_NUM]` | 每 rank 的通信 channel 句柄 |
| `commBufferAddrs[]` | `uint64_t[COMM_MAX_RANK_NUM]` | 每 rank 的 Win 区基地址 |

### CommUbmemContext

| 字段 | 类型 | 含义 |
|:---|:---|:---|
| `rankId` | `uint32_t` | 本 rank ID |
| `rankSize` | `uint32_t` | 总 rank 数 |
| `commBufferAddrs[]` | `uint64_t[COMM_MAX_RANK_NUM]` | Barrier flag 的 GM 地址 |

> `COMM_MAX_RANK_NUM`（=64）和 `COMM_WORKSPACE_SIZE`（=512B）定义在 `apace/block/aiv_comm/collective_comm_context.h`。

### 填充细节

| 细节 | 说明 |
|:---|:---|
| `channelHandles[self]` 不填充 | builder 建链循环跳过 `peer == rankId`，本 rank 条目保持 0；kernel 侧 DoCommit/DoWait 必须跳过 self（见 §3 自跳过规则） |
| `commBufferAddrs[self]` | = 本地 HCCL buffer 地址（有效，用于本地 Win 区读写） |
| ctxTag 复用语义 | 同 tag 命中 `HcclEngineCtxGet` 会**直接复用已有 context 并跳过字段填充与建链**——这是特性；不同通信域/不同 group 必须用不同 tag |
| rankSize 上限 | `rankSize <= COMM_MAX_RANK_NUM`（64 卡），数组定长越界即静默错位 |

### CommContext 不变量

| 不变量 | 说明 |
|:---|:---|
| 传递方式 | 通过 `__gm__` 指针传递（`__global__` 入口的第一参数），不按值传递 |
| Host 构造 | Host 侧构造后写入 GM，kernel 通过指针读取 |
| tiling 按值 | tilingData 作为 `__global__` 入口参数按值传递 |

---

## 6. Host 侧建链机制

CommContext 的 `CommUdmaContext` 和 `CommUbmemContext` 不能手动赋值，必须通过 `CommChannelBuilder`（`apace/utils/comm_channel_builder.h`）创建 HCCL channel 后自动填充。

### Host 侧验收条件

| 验收条件 | 说明 |
|:---|:---|
| TCP 交换 RootInfo | rank0 生成 `HcclRootInfo` 并通过 TCP 广播给其他 rank（官网 ST 用 `apace/tests/st/utils/root_info_exchanger.h` 的 `RootInfoExchanger`） |
| 创建 HCCL comm | `HcclCommInitRootInfoConfig` 创建 `HcclComm` |
| HCCL 数据 buffer 清零 | builder 的 `AllocRegAndBuildChannels` 内部对 HCCL 内置 buffer 做 `aclrtMemset(buf, hcclBufSize, 0, hcclBufSize)`（清的是数据 buffer；barrier flag 区零初值依赖 engine 分配语义，见 §4.1） |
| CommChannelBuilder 填充 | 通过 `builder.CreateDeviceContext` 自动填充 `udmaCtx` 和 `ubmemCtx` |
| **跨 rank host barrier（强制）** | `CreateDeviceContext` 返回后必须做一次 rank 间 barrier（官网 ST 用 `RootInfoExchanger::Barrier()`），确保所有 channel 握手完成，再 launch kernel（`CreateDeviceContext` 头注释明确要求："调用方应在本函数返回后对 rank 间做一次 barrier"） |
| engine 一致性 | `HcclChannelAcquire` 与 `HcclEngineCtxCreate/Get/Copy` 必须使用同一 engine（apace 用 `BUILDER_COMM_ENGINE_AIV = 4`，`apace/utils/comm_channel_builder.h:27`），否则 `HcclEngineCtxGet` 复用失效 |
| ctxTag 唯一性 | 不同通信域用不同 ctxTag；同 tag 命中 `HcclEngineCtxGet` 会直接复用并跳过填充（见 `CommChannelBuilder::CreateDeviceContext` 头注释） |
| 资源生命周期 | devContext 由 HCCL engine 管理：随 `HcclCommDestroy` 释放，或显式 `HcclEngineCtxDestroy`（推断：释放路径未经官网验证；实证：AG ST 有 aclrtFree(devContext)，all_to_all ST 无——两份 ST 处置不一致）；builder 无清理接口。⚠️ 官网两份 ST 处置不一致：all_gather ST（`apace/tests/st/all_gather_quant_matmul/src/main.cpp`）有 `aclrtFree(devContext)`，all_to_all ST（`apace/tests/st/all_to_all_quant_matmul/src/main.cpp`）不释放——推荐范式：不单独释放，随 HcclCommDestroy 连带释放 |
| 禁止手动填充 | `channelHandles` 和 `commBufferAddrs` 必须由 `CommChannelBuilder` 通过 HCCL API 获取 |

### 禁止行为

| 禁止 | 原因 |
|:---|:---|
| 手动赋值 `channelHandles` / `commBufferAddrs` | 必须由 HCCL API 获取，手动赋值导致通信失败 |
| 依赖 `GetRankId()` / `GetRankSize()` 的时机 | builder 有独立 `Init()` 方法（内部调 `HcclGetRankId/HcclGetRankSize`）；`CreateDeviceContext` 内部也会获取。确保在 HCCL comm 创建之后调用 |

### CreateDeviceContext 不变量

| 步骤 | 不变量 |
|:---|:---|
| ctxTag 复用检查 | 先 `HcclEngineCtxGet(comm, ctxTag, engine, ...)`；命中已存在 context 直接返回复用，跳过建链与字段填充 |
| 创建 device context | `HcclEngineCtxCreate(comm, ctxTag, engine, totalSize, &devCtx)`；`totalSize = ctxSize + BARRIER_BUF_SIZE`（有 barrierCtx 时），`BARRIER_BUF_SIZE = 2MB`（`comm_channel_builder.h:117` constexpr） |
| 获取 rank 信息 | `HcclGetRankId` / `HcclGetRankSize`（自动获取，无需手动调用） |
| 填充 CommUdmaContext | `rankId`/`rankSize` + `AllocRegAndBuildChannels(URMA)` → `channelHandles[peer]` + `commBufferAddrs[peer]`（建链循环跳过 `peer == rankId`，self 的 channelHandle 保持 0；`commBufferAddrs[self]` 填本地 HCCL buffer 地址） |
| 填充 CommUbmemContext | `rankId`/`rankSize` + barrier buffer 取 `devCtx + ctxSize`（2MB 区域）+ `HcclCommMemReg` 注册 + `BuildChannels(UBMEM)` → `commBufferAddrs[peer]` |
| 拷贝到 device | `HcclEngineCtxCopy(comm, engine, ctxTag, hostCtx, ctxSize, 0)` 把 hostCtx 拷到 device GM |
| 返回 device 指针 | kernel 通过此指针访问 CommContext |

### 填充后的 CommContext 结构

```
CommContext (device GM)
├── CommUdmaContext udmaCtx
│   ├── rankId                                ← 本 rank ID
│   ├── rankSize                              ← 总 rank 数
│   ├── channelHandles[COMM_MAX_RANK_NUM]     ← 每 peer rank 的 URMA channel 句柄（self 不填充）
│   └── commBufferAddrs[COMM_MAX_RANK_NUM]    ← 每 rank 的 Win 区基地址
└── CommUbmemContext ubmemCtx
    ├── rankId
    ├── rankSize
    └── commBufferAddrs[COMM_MAX_RANK_NUM]    ← 每 rank 的 barrier flag GM 地址
```

> 字段顺序以 `apace/block/aiv_comm/collective_comm_context.h` 为准：`rankId`、`rankSize` 在前，数组在后。host 侧聚合初始化/布局推算必须按此顺序。

> 完整实现见官网 `apace/utils/comm_channel_builder.h` 的 `CommChannelBuilder::CreateDeviceContext()`。

---

## 7. 扩展通信原语指南

### 当前支持的通信原语

| 原语 | 模式 | 文件 | 状态 |
|:---|:---|:---|:---|
| AllToAll | GET | `apace/block/aiv_comm/all_to_all/all_to_all_udma_get.h` | ✅ 钩子已实现（已注册分发；官网暂无 GET 算子使用方） |
| AllToAll | PUT | `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` | ✅ 已实现（all_to_all_quant_matmul 使用） |
| AllGather | PUT | `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` | ✅ 已实现（all_gather_quant_matmul 使用） |
| AllGather | GET | — | ❌ 未实现 |
| AllReduce | — | — | ❌ 未实现（**连 `CommCollectiveOp` 枚举值都没有**——`collective_comm_api.h` 枚举仅 AllToAll/AllGather/ReduceScatter；扩展须先补枚举，见下方"新增集合操作三步法"） |
| ReduceScatter | — | — | ❌ 未实现（`CommCollectiveOp::ReduceScatter` 枚举值已在 `apace/block/aiv_comm/collective_comm_api.h` 预留但无分发实现；已有自研编排用 AllToAll PUT + 3 级流水 workspace 架构实现，见 `fusion.md` §6.2） |

### hcomm 可用未封装原语（设计新通信原语时的备选能力）

底层 `adv_api/hcomm/hcomm.h`（`Hcomm<COMM_PROTOCOL_UBC_CTP>`，即 CollectiveCommBase 内 `comm_` 对象的类型）还提供 apace 尚未封装的原语——设计 ReduceScatter/AllReduce 变体时应作为候选路径评估（哪怕结论是"不推荐"）：

| 原语 | 能力 | 适用 |
|:---|:---|:---|
| `WriteReduceNbi` | 远端原位规约（`dst += src`，reduceOp 支持 SUM/MAX/MIN；dtype 支持 int8/16/32、uint32、half、float、bfloat16） | ReduceScatter/AllReduce 直接规约路线（AllReduce 两条路线的路线 A，见 [`paradigm-mapping.md`](../operator-design/paradigm-mapping.md) §4） |
| `WriteValueNbi` | 远端写常量 | 状态区初始化类场景 |
| `AtomicFAA` / `AtomicCAS` | 远端原子 fetch-add / compare-swap | 完成计数、轻量跨卡握手 |
| `WriteNbi` / `ReadNbi` / `Drain` | 单向写 / 读 / 完成等待 | 现有三实现即用这三个 |

> 注意：规约在**远端内存**完成——`Drain` 只保证本端发出，完成语义须配合 CrossDevice barrier（PUT 型 Wait 模式），与 §3.2 PUT DoWait 的 barrier 语义设计同源。

### 新增集合操作三步法（新增 AllReduce/ReduceScatter 实现的标准流程）

**Step 0（仅缺枚举的原语需要）**：在 `collective_comm_api.h` 的 `CommCollectiveOp` 补枚举值。当前枚举仅 `{ AllToAll, AllGather, ReduceScatter }`——AllReduce 须先加 `AllReduce` 枚举；ReduceScatter 已预留可跳过。

**Step 1**：新建 `apace/block/aiv_comm/<op>/<op>_udma_put.h`（或 `_get.h`），CRTP 继承四钩子：

```cpp
template<typename Dtype, typename Barrier = TeamBarrier>
class <Op>CommPutImpl : public CollectiveCommBase<<Op>CommPutImpl<Dtype, Barrier>, Dtype, Barrier> {
    friend class CollectiveCommBase<<Op>CommPutImpl<Dtype, Barrier>, Dtype, Barrier>;
    template<uint8_t BarrierMode> __aicore__ inline void PostInit();
    template<uint8_t BarrierMode> __aicore__ inline void DoCommit(uint32_t targetRankId, uint64_t tileByteSize);
    template<uint8_t BarrierMode> __aicore__ inline void DoWait(uint32_t targetRankId);
    template<uint8_t BarrierMode> __aicore__ inline void DoFinalize();
};
```

语义约束（对齐 §3 已有实现）：
- `DoCommit`：一次 tile 到一个 targetRank 的搬移；地址基于基类受保护成员（`udmaCtx_/localAddr_/winOffset_/chunkBytes_/tileByteOffset_/slotByteOffset_/currentTileIdx_`，见 §2 受保护字段表）；返回值经 `ascendc_assert(ret == 0, ...)` 校验
- PUT 型：`DoWait` = Drain + CrossDevice barrier（self 跳过 Drain 但 barrier 保留）；GET 型：`DoCommit` 前 barrier（先 Core 后 Device）、`DoWait` 仅 Drain、`DoFinalize` 补 barrier
- 规约型（WriteReduceNbi）：额外设计**首轮覆盖/清零语义**——每 tile 独立 slot 布局天然免清零，否则 PostInit 清零或首轮 WriteNbi 覆盖
- 自 rank：DoCommit 跳过 `targetRankId == rankId`

**Step 2**：`collective_comm_api.h` 注册特化：

```cpp
template<typename T, typename Barrier>
struct CollectiveCommHelper<CommCollectiveOp::<Op>, CommMode::PUT, T, Barrier> {
    using type = <Op>CommPutImpl<T, Barrier>;
};
```

**Step 3**：配套约定——沿用 `CommTilingData` 5 字段（chunk 语义按新原语重述并在算子 tiling 头注释，参考 compute-first 场景"chunk=一个 dest 分片"的写法）；窗口布局头文件注释画明；可仿 `kernel/all_*_quant_matmul/` 增加参考 kernel。

> 共享层纪律（R4）：以上改动落在 `block/` 共享层，超出单算子范围——按扩展决策树走"独立分支验证 → 评审合入"，算子工程内不得夹带共享层副本。

### 扩展边界

- **禁止修改现有文件**：`collective_comm_api.h`、`collective_comm_base.h`、`all_to_all_udma_get.h` 等已实现的文件不能改
- **允许新增文件**：可在 `apace/block/aiv_comm/` 下**新增**目录和文件（如 `apace/block/aiv_comm/all_reduce/all_reduce_udma.h`）
- 新增 block 文件属于"创建新通信原语"，超出常规开发范围

### 扩展决策树

```
需要新通信原语？
├── 是 AllToAll/AllGather 的变体（如 GET→PUT）
│   └── 参考现有实现，新增 block 文件
├── 是完全新原语（AllReduce/ReduceScatter/Broadcast）
│   ├── 评估是否超出常规开发范围
│   ├── 如继续：实现 4 个钩子 + 注册到分发器 + tiling 适配
│   └── 建议先在独立分支验证，再合入共享层
└── 只是使用方式不同（如换 dtype/shape）
    └── 不需要新增 block 文件，只改 kernel/<op>/ 下的文件
```

### 常见误区

| 误区 | 正确做法 |
|:---|:---|
| 修改 `all_to_all_udma_get.h` 适应新场景 | 新增 `all_to_all_udma_get_v2.h` 或在 kernel 层适配 |
| 在 kernel 中直接调用 `Hcomm::ReadNbi` | 使用 `CollectiveComm` 四段式 API，保持抽象一致性 |
| 跳过 `CollectiveCommHelper` 直接实例化实现类 | 通过 `CollectiveComm<Op, Mode, T, Barrier>` 编译期分发，保持类型安全 |

> ReduceScatter 的自研编排实现（AllToAll PUT + 3 级流水 + workspace 槽位独占）见 `fusion.md` §6.2。

---

## 8. 官方 master 漂移登记（pin 之后演进，核对前必读）

> 本 skill 全部路径锚点以 pin 快照（ops-transformer `7e6cf8bba`，2026-08-10）为基准。pin 之后 origin/master 又有 10 个涉及 `mc2/common/op_kernel/apace` 的提交；拉取 master 后**必须先对照本表 diff 校验**，文档引用失效时更新文档。登记时间：2026-08-30。

| 演进 | commit | 影响 | 对本 skill 的处置 |
|:---|:---|:---|:---|
| **目录迁移 block/ → core/** | `9ef00cfd1` | `block/aiv_comm/` → `core/aiv_comm/`；`block/blaze_ext/gemm/block/qmm_mx_block_mmad_fragment.h` → `block/mmad/`；kernel 重组为 `kernel/fusions/<op>/`（impl）+ `kernel/matmul/quant_batch_matmul/`（mm kernel） | 本 skill 文档**仍用 pin 路径**（`block/aiv_comm/...`）；用 master 核对时按此映射换算 |
| **ReduceScatter UBMEM 官方模板** | `8fbd92b1d` | 新增 `kernel/fusions/quant_matmul_reduce_scatter/`（`quant_matmul_reduce_scatter_impl.h`、`block_epilogue_all_to_all.h`、`utils/comm_resource_builder.h` + ST 全套，UBMEM 通信路径） | pin 快照内 ReduceScatter 仍仅枚举占位（§7 支持表口径不变）；**设计 ReduceScatter 类算子前须先核对 master 是否已直接覆盖需求**（[`scenarios/compute-first-reduce-scatter/design.md`](../scenarios/compute-first-reduce-scatter/design.md) §1 已同步标注） |
| **hcomm GM cache flush（dcci）** | `33444e461` | AG/A2A urma impl 在两个 Finalize 后新增 `dcci()` 刷新通信上下文区（修多轮 launch 精度问题） | 官方 dcci 刷的是 **CommContext 通信上下文区**（跨 kernel launch 一致性），与陷阱 #14 禁止对 **staging 数据**加 dcci 不矛盾——两个语义不同场景，勿混淆 |
| CCU/AICPU 通信多选 | `6c4b1b358` | `Using_Apace_Impl` 模板参数、纯 C 通信计算、all2all_out 与 worldsize 分发 | 影响 hcomm/CCU 路线选型；本 skill hcomm 契约按 pin 口径，使用 master 前须重核 |
| barrier/测试修复若干 | `32de19efd` `2eb6e2b89` `42f69c50f` `6ee5f3620` `5b5efabac` `0f2296ba3` | barrier ubmem 修复 ×2、AG 用例 bug 修复、all_gather ubmem decode、allto_all MX UT 看护、安装重名头文件整改 | 低影响；barrier 相关修复与 TeamBarrier 契约核对时留意 |

> 漂移核对方法：`git -C <ops-transformer> log --oneline <pin>..origin/master -- mc2/common/op_kernel/apace`；样例代码经 `scripts/fetch_apace.sh`（`--ref`/`APACE_PIN_REF` 锚定）现取现读，拉取 master 后与 pin 快照 `diff -rq` 校验结构。

---

## 常见陷阱

| # | 陷阱 | 后果 | 规避 |
|:---|:---|:---|:---|
| 1 | flagId 冲突 | 同步紊乱 | flagId ∈ [0, FLAG_ID_MAX)；多组 Set/Wait 并存时 flagId 互不冲突 |
| 2 | UB 分配溢出 | 超出 UB 容量 | 单对象基线（COMM_WORKSPACE_SIZE + barrier UB_SIZE）+ kernel 专属 UB 需求，双对象翻倍；总量不能超过 UB 容量 |
| 3 | GET 模式 barrier 时序遗漏 | 读到未初始化数据 | `DoCommit` 中先 `CrossCore()+CrossDevice()` 再 `ReadNbi` |
| 4 | 自跳过规则遗漏 | 自身 Win 区读写错误 | DoCommit/DoWait 跳过 `targetRankId == rankId` |
| 5 | HCCL windows 模式误加 CommContext | 编译错误或内存浪费 | 使用 `GetHcclContext` 的 kernel 不需要 `CommContext` 结构 |
| 6 | TeamBarrier 远端 rank 未就绪 | CrossDevice **无限等待挂死**（无超时保护，不会 assert） | 确保所有 rank kernel 已 launch；`CreateDeviceContext` 后做跨 rank host barrier 再 launch |
| 7 | hcomm 调用返回值未检查 | 通信失败静默扩散 | 所有 hcomm 调用（WriteNbi/ReadNbi/Drain）返回值必须 `ascendc_assert(ret == 0, ...)`（官网 PUT/GET/AG 全部实现均如此） |
| 8 | TeamBarrier flag 区复用或未清零 | epoch 计数错乱 → 同步提前放行或挂死 | TeamBarrier flag 为单调递增 epoch：HCCL **数据** buffer 的清零由 builder `AllocRegAndBuildChannels` 内 `aclrtMemset` 完成（§6）；barrier flag 区（BARRIER_BUF）零初值依赖 `HcclEngineCtxCreate` 分配语义（§4.1，未显式 memset）——新算子若新增自管理 flag/计数区必须显式清零并验证，kernel 生命周期内禁止复用/重置 |
| 9 | AIV UB 静态偏移与 TPipe 混用 | buffer 重叠踩踏 | 通信对象的 UB 用静态偏移（`MakeMemPtr<UB>` 顺序排布 commBuf 512B×2 + barrierBuf），不与 TPipe 管理的 buffer 区域混用 |
| 10 | 多对象共享 TeamBarrier 时重复 barrier | 多余同步开销甚至死锁 | 仅一个对象使能 barrier，其余 `Init<BARRIER_NONE>`；同 channel 多对象只 Wait 一次（见 §2 BarrierMode 选择规则） |
| 11 | GET 模式 `Drain` 返回非 0（未实现/兼容性）（GET 场景） | assert 失败中断 | URMA Win 区是共享内存，可绕过 Drain：`TeamBarrier.CrossDevice()`（跨 rank 就绪）+ `SyncAll<true>`（核间可见）后直接 `DataCopyPad` 读远端 Win 区（GET 算子开发实测绕行方案）。⚠️ 该绕行绕开 §3.1 的 Drain assert 纪律：仅在确认 Drain 兼容性问题时作为兜底使用，正常 GET 实现仍以钩子契约为准 |
| 12 | PUT/GET 数据覆盖 Win 区内元数据/barrier 区 | "假通过"（精度碰巧对、同步已破坏），大 shape/多轮时紊乱 | 原则与两种布局见 §3.2 ⚠️注（唯一事实源）；共享布局须按约定偏移跳过头部（具体实现形态见 [`scenarios/compute-first-reduce-scatter/design.md`](../scenarios/compute-first-reduce-scatter/design.md) §3.6；失败链见 [`failure-navigation.md`](../troubleshooting/failure-navigation.md)） |
| 13 | 单轮 PUT 数据量偏大（PUT 大数据量场景） | ⚠️ **风险提示（非硬红线，边界未定）**：bring-up 期过大单轮曾见间歇失败，官方代码无显式约束；亦有同平台更大单轮稳定运行的工程实例 | 大单轮 PUT **按 case 复核**（连续多轮精度 + 重复 launch 一致性验证），不作 host 强制拒绝/shape 拒绝依据；bring-up 期可先用小单轮 PUT 跑通再放大复核 |
| 14 | compute-first 归约读 staging 得旧值/0 | 归约结果错误或全 0 | **先查三处，勿先加 dcci**：① AIC `SetFlag<PIPE_FIX>` 与 AIV `WaitFlag<PIPE_MTE2>` 是否逐轮配对（配对即内存序保证，staging 可见性由此而来，参考实现不依赖 dcci）；② 多核归约是否写竞争（须按行块多核分治，见 `fusion.md` §6.2.6 纪律 5）；③ staging 写/读地址是否同源。对 staging 加 dcci 属误诊，掩盖真根因 |

---

## 后续阅读

- `fusion.md` — GET/PUT 编排模式、flag 编排、环形回压、localMatmul
- `operator-anatomy.md` — 算子完整骨架中的通信对象使用
- `host-and-testing.md` — host launcher 序列（建链调用时机）
- `ascendc-api-best-practices` skill `references/api-hcomm.md`、`ascendc-api-best-practices` skill `references/api-crosscore-sync.md`、`ascendc-api-best-practices` skill `references/api-hccl-host.md`
