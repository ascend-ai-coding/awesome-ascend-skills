# apace 通算融合组合模式

> 本文档覆盖通算融合的组合层：通信与计算如何重叠、GET/PUT 模式选型、flag 编排模式、localMatmul 模式、多对象复用与扩展语义推导。前置阅读：communication.md（通信接口）、compute.md（计算接口）。

## 目录

1. [通算重叠原理](#1-通算重叠原理)
2. [GET vs PUT 模式选型](#2-get-vs-put-模式选型)
3. [Flag 编排模式](#3-flag-编排模式)
4. [PUT 算子级编排验收](#4-put-算子级编排验收)
5. [localMatmul 模式（0/1/2）](#5-localmatmul-模式012)
6. [多对象复用与扩展语义推导](#6-多对象复用与扩展语义推导)

---

## 1. 通算重叠原理

AIC（计算）与 AIV（通信）在同一 kernel 内并行执行，两侧通过 CrossCore flag 做逐 tile 精细同步（见 §3），通信延迟被计算掩盖。`Commit()` 的非阻塞特性是通算流水的关键——AIV 可以在 Commit 后、Wait 前插入其他指令（机制详见 communication.md）。

掩盖条件：每个 tile 的计算耗时 ≥ 通信耗时，且流水深度足够。

- **PUT 模式**：AIV 先推数据、AIC 后算（通信→计算）；localMatmul=1 时本地 A×B 前置计算与 AIV PUT 并行（见 §5）。
- **GET 模式**：AIC 先算、AIV 后拉（计算→通信），Win 槽位环形复用 + 回压（见 §3.4）。
- 重叠粒度受 comm tile 限制（CalcDependTileIdx + waitedMask 去重，见 §4.2）。

---

## 2. GET vs PUT 模式选型

**方向语义**：

- **GET 模式 = 计算→通信**：AIC 先算 C 写到 Win 区，AIV 从远端 Win 区拉回本 rank 的 C 段。
- **PUT 模式 = 通信→计算**：AIV 先推数据到远端 Win 区，AIC 从 Win 区读取计算。实现见 `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` 的 `AllToAllCommPutImpl`（AllGather PUT 变体见 `apace/block/aiv_comm/all_gather/all_gather_udma_put.h` 的 `AllGatherCommPutImpl`，钩子结构相同，仅地址公式不同）。

> **官网暂无 GET 算子样例**：GET 钩子基础设施已就绪并注册分发，但官网 kernel/ 下两个算子均为 PUT 模式，无 GET 使用方（详见 [`communication.md`](communication.md) §3 GET 钩子契约）。本文 GET 内容为钩子契约级描述。

### 2.1 钩子差异

GET/PUT 钩子差异表（`PostInit`/`DoCommit`/`DoWait`/`DoFinalize` 四钩子职责与 barrier 顺序）、PUT 地址公式、以及数据区/元数据区分离原则——以 [`communication.md`](communication.md) §3.2 为唯一事实源，本节不重复。

### 2.2 flag 编排对比

| 维度 | GET 模式（契约，官网无样例） | PUT 模式（官网实现） |
|:---|:---|:---|
| 发起方 | AIC 先 SetFlag → AIV WaitFlag | AIV 先通信 → SetFlag → AIC WaitFlag |
| Wait 参数 | Wait(true) 按 waitLast 早退语义：DoWait 仅在 currentTileIdx_ == totalTiles - 1 时执行一次（典型 Commit→Wait 循环中为倒数第二轮），末轮不 Drain（详见 communication.md） | `Wait<BARRIER_DEVICE>()`（BarrierMode 模板参数） |
| SyncAll | 不需要（CrossCore flag 已保证时序） | **每轮必须** `SyncAll<true>()`（WriteNbi 对端可见性） |
| flagId | `tid`（tile 索引） | `tid`（round 索引） |
| 回压 | AIC WaitFlag<0x2, PIPE_M>(tid-bufCnt) | 无（PUT 不需要环形回压） |

### 2.3 Wait 参数维度差异

GET 约定的 `Wait(true)` 中 `true` 是 `waitLast` 位置参数——Wait(true) 按 waitLast 早退语义：DoWait 仅在 currentTileIdx_ == totalTiles - 1 时执行一次（典型 Commit→Wait 循环中为倒数第二轮），末轮不 Drain（详见 communication.md）；PUT 的 `Wait<BARRIER_DEVICE>()` 是 `BarrierMode` 模板参数（控制 DoWait 中是否插入 CrossDevice barrier）。两者是完全不同的维度。

---

## 3. Flag 编排模式

CrossCore Flag 是 AIC↔AIV 跨核同步的核心机制。每个 flag 由 `<MODE, PIPE, flagId>` 三元组标识；MODE 常量与 PIPE 选项机制表见 communication.md（本文不复制）。

### 3.1 GET 编排不变量

> **官网暂无 GET 算子样例**：以下为 GET 模式的契约级编排模式（与 communication.md 的 GET 钩子契约配套），官网 kernel/ 下无可验证的 GET 使用方。

```
AIC (MatmulProcess)                            AIV (AllToAllProcess)
─────────────────────────────                  ─────────────────────────────
tile tid:
  [if ring: WaitFlag<0x2, PIPE_M>(tid-bufCnt)] ←── 回压
  RunMatmul(C → Win[tid % bufCnt])
  SetFlag<0x2, PIPE_FIX>(tid)          ──→    WaitFlag(tid)   // 无模板：阻塞标量流
                                                Commit(GET C[tid])
                                                Wait(true)
  [if ring: ...]                        ←──   SetFlag<0x2, PIPE_MTE3>(tid)
```

> GET 模式 `AllToAllProcess` 中不需要 `SyncAll`。同步完全由 CrossCore flag 负责。
>
> 以上为**契约级推导**（官网无 GET 样例）：AIV 侧 Wait 后紧随标量流 `Commit`（ReadNbi 下发），故用无模板形态（形态语义见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §4.1）；AIC 侧回压 Wait gate 计算主流水，用 `PIPE_M` 模板形态。实现前须对照钩子源码逐条验证（见 [`scenarios/get-all-gather-quant-matmul/development.md`](../scenarios/get-all-gather-quant-matmul/development.md)）。

### 3.2 PUT 编排不变量

PUT 编排不变量（官方实现见 `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h` 的 `RunAllToAll` / `RunMatmul`）：

```
AIV (RunAllToAll)                              AIC (RunMatmul)
─────────────────────────────                  ─────────────────────────────
round tid:
  Commit(scale A[tid])
  Commit(data A[tid])
  Wait<BARRIER_DEVICE>()
  SyncAll<true>()
  SetFlag<0x2, PIPE_MTE3>(tid)          ──→    WaitFlag<0x2, PIPE_MTE2>(tid)
                                                从 Win 读 A[tid] → Matmul
```

> PUT 模式 AIC 侧 `UdmaCommWaitPolicy::WaitTile()` 使用 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)`——PIPE_MTE2 对应数据搬入流水，等待 AIV WriteNbi 数据到达。AllGather PUT 算子同一模式：循环外先预触发 `CrossCoreSetFlag<0x2, PIPE_MTE3>(0)`（自身数据始终就绪，AIC 可直接消费），循环内 `SyncAll<true>()` 后 `CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1)`（flagId 用 round+1 与预触发的 0 错开）。

### 3.3 flagId 选择规则

flagId 的硬件规则（16 通道截断、计数器 0-15 衡量未消费积压、发射顺序不保证）以 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` 为唯一事实源，本节只保留编排层推导：

- 官网两个 PUT 算子均使用**轮次索引 `tid`/`round`** 作为 flagId（`CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)`），硬件基础是 16 通道截断（Set/Wait 双方截断到同一 flag，配对保持）。实践约束：轮次索引式 flagId 时轮次索引 ∈ [0, FLAG_ID_MAX)（apace 官方定义于 `utils/constant.h:59`），**且须避开 SyncAll 保留区**——官方 PUT 算子每轮 `SyncAll<true>`（占用 flagId 14，见 `api-crosscore-sync.md` §3），轮次索引实际安全上界为 13；超限必须放大 tileM 降低轮次。**更优做法（compute-first 工程验证）**：改用**计数式固定 flagId（如 0/1）**——Set/Wait 各 T 次、计数自然平衡，不受轮次数影响，也不与保留区冲突
- **计算在前 per-tile 流水算子** T>1 时为逐轮计数式配对，紧邻配对下积压小；flagId 数量 16 是硬事实；计数器 0-15 衡量**未消费积压**（紧邻配对下 ≈ 1-2，官方有据，详见 §6.2.3），不构成 T 上限，T 上限按 case 实测确定
- 另有 `waitedMask` 为 uint32 位掩码 → 通信 tile 总数 ≤ 32（§4.2），与上述 flagId 约束取更严者

**compute-first 算子的两种 flag 模式（均合法，按场景选型）**：

| 模式 | 形式 | 优点 | 注意点 |
|:---|:---|:---|:---|
| **计数式 flag（推荐）** | flagId 恒定（如 0），AIC T 次 Set ↔ AIV T 次 Wait，计数器递推天然支持流水 | 简单、无需按轮次管理 flagId、不与 SyncAll 保留区冲突 | Set/Wait 必须严格逐轮配对；计数器 0-15 衡量未消费积压，紧邻配对下不触顶（不构成 T 上限，R9 口径不设数值拒绝） |
| **per-turn flag** | 每轮独立 flagId（`flagId + t`），AIC Set(flagId+t) ↔ AIV Wait(flagId+t) | 各轮 flag 独立可观测，调试时可按轮次定位配对缺失 | `flagId + T - 1` < 16 且避开 SyncAll 保留区 14（每轮 `SyncAll<true>` 时实际上界 13） |

> 两模式受同一硬件 flagId 数量上限（16 个通道）约束；选型不影响正确性，**推荐计数式**（实现更简）。

### 3.4 环形回压与 bufferCount

> **GET 模式契约级描述**：环形回压是 GET 模式（计算→通信）Win 槽位复用的配套机制，官网 kernel/ 下暂无 GET 算子样例可验证；PUT 模式（官网两个算子）不需要环形回压。

#### 机制

`bufferCount` 个槽位环形复用 Win 区。当 `totalTileCnt > bufferCount` 时，需要回压机制防止 AIC 覆盖未消费的数据。

#### 两种模式

| 模式 | 条件 | 行为 |
|:---|:---|:---|
| **全量缓冲** | `bufferCount >= totalTileCnt` | 无回压，所有 tile 的 Win 槽位独立 |
| **环形复用** | `bufferCount < totalTileCnt` | AIC 等 AIV 释放旧槽位（回压 flag） |

#### 环形回压不变量

| 侧 | 不变量 |
|:---|:---|
| AIC | 当 `bufferCount < totalTileCnt && tid >= bufferCount` 时，`WaitFlag<0x2, PIPE_M>(tid - bufferCount)` 等待 AIV 释放 |
| AIV | 当 `bufferCount < totalTileCnt && tid < totalTileCnt - bufferCount` 时，`SetFlag<0x2, PIPE_MTE3>(tid)` 通知 AIC 可复用 |

### 3.5 flag 计数平衡规则

CrossCore flag 是**计数式**同步：每个 flagId 维护一个计数器，Set 递增、Wait 等待达到预期值。以下三条平衡规则违反即挂死（来源：ops-transformer 注册版 MC2 实现实证）：

1. **未参与计算的核必须补做同样次数的 SetFlag/WaitFlag**。`block_idx >= usedCoreNum` 的核跳过 mm 计算，但必须执行与计算核相同次数的 flag Set/Wait，否则全局 flag 计数不一致（注册版 `matmul_reduce_scatter_fp16_bf16.h`）。
2. **多级 flag 的顺序不可颠倒**：AIC 先完成全 AIC 自同步（如 flagId=4，SYNC_MODE0），再通知 AIV（如 flagId=8，SYNC_MODE2/PIPE_FIX 发、PIPE_S 等）——先 4 后 8 顺序不可交换，否则 AIV 读到未写完的数据（注册版 `matmul_a2a_vec_reduce_fp16_bf16.h`）。
3. **跨核信号改造必须先画等待依赖图、确认无环**。把 per-core SetFlag 改成"AIV 轮询所有核进度（min(coreProgress)）"这类全局协调时，极易成环（AIV 等所有 AIC 推进 ↔ AIC 等 AIV 发数据 ↔ SyncAll 永远不齐 → 循环死锁）。任何同步结构改动，先列出 AIC↔AIV↔跨 rank 三方的等待边，确认无环再动手（卡死定位实测教训）。

---

## 4. PUT 算子级编排验收

### 4.1 AIV 通信流水线（PUT）验收

`RunAllToAll()` 是 AIV 侧主通信循环（官方 AllToAll impl `apace/kernel/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_udma_impl.h`）。

#### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | Commit 顺序 | scale 先、data 后（`allToAllScaleA_.Commit()` → `allToAllA_.Commit()`，同 channel 串行） |
| 2 | Wait 方式 | `allToAllA_.Wait<BARRIER_DEVICE>()`；scale 与 data 同 channel，只 Wait 一次 |
| 3 | SyncAll | 每轮 `SyncAll<true>()` |
| 4 | AIC 通知 | 每轮 `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` 通知 AIC 第 tid 个通信 tile 就绪 |
| 5 | 分核保护 | Commit/Wait 仅在 `GetBlockIdx() < rankSize` 的 block 执行；`SyncAll` 与 `CrossCoreSetFlag` 在守卫**之外**，全 AIV 执行 |
| 6 | Finalize | 循环结束后 `allToAllScaleA_.Finalize()` + `allToAllA_.Finalize()`（无分核 guard，全 AIV 执行） |
| 7 | 无环形回压 | 每轮独立，无 bufferCount 槽位复用（与 GET 的环形回压不同，见 §3.4） |

#### 流水线要素

| 要素 | 代码 | 含义 |
|:---|:---|:---|
| 通信 | `Commit()` × 2 + `Wait<BARRIER_DEVICE>()` × 1 | PUT 模式：把本 rank A/scaleA 推到远端 Win |
| 同步 | `SyncAll<true>()` | 保证所有 AIV block 完成本轮通信 |
| 通知 AIC | `CrossCoreSetFlag<0x2, PIPE_MTE3>(tid)` | AIC 侧 `WaitTile(tid)` 的配对端 |
| 分核保护 | `GetBlockIdx() < rankSize` | 只有前 rankSize 个 block 下发通信（要求 rankSize ≤ BlockNum） |

### 4.2 AIC 等待机制

AIC 在 kernel 内经 `CommPolicy::WaitTile` 等待通信数据，策略类由模板注入。

#### 等待机制契约

| 要素 | 机制 | 锚点 |
|:---|:---|:---|
| 策略注入 | `UdmaCommWaitPolicy::WaitTile(tileIdx)` 底层为 `CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx)` | 官方 AllToAll impl `struct UdmaCommWaitPolicy` |
| 依赖计算 | `CalcDependTileIdx(mPos + blockM - 1, headTileSize, totalTiles)` 由 block 末行 mPos 推导依赖的通信 tile，越界钳到 `totalTiles - 1` | 官方 AllToAll qbmm kernel `CalcDependTileIdx` |
| 按位去重 | `waitedMask`（`uint32_t`）按位记录已 wait 的 tile，同一 tile 只 wait 一次 | 官方 AllToAll qbmm kernel `ProcessSingleBatch` |
| 尾部兜底 | 循环结束后遍历全部 tile，对未 wait 的位补 `WaitTile`（drain 兜底，非 LOCAL 模式） | 官方 AllToAll qbmm kernel `ProcessSingleBatch` 末尾 |

> ⚠️ **≤32 硬约束**：`waitedMask` 是 `uint32_t` → 通信 tile 总数 ≤ 32，超出会静默出错（高位 tile 的等待位溢出丢失）。

#### Flag 配对关系

| AIV 操作 | AIC 操作 | flagId | 含义 |
|:---|:---|:---|:---|
| `SetFlag<0x2, PIPE_MTE3>(tid)` | `WaitFlag<0x2, PIPE_MTE2>(tid)` | tid | 通信 tile tid 就绪，AIC 可消费 |

**配对规则**：AIV SetFlag 的 flagId 必须 == AIC WaitFlag 的 flagId；`SyncAll` 在 SetFlag 之前，保证 flag 可见性。

> Flag 机制（`<MODE, PIPE, flagId>` 三元组、完整编排图、flagId 选择规则）详见本文档 §3；上表为 AIC 消费者视角的锚点对照。

---

## 5. localMatmul 模式（0/1/2）

> **范围说明**：localMatmul 是 AllToAll PUT 算子的专有概念（字段定义于 `allToAllMatmulTilingData`）。compute-first 算子不走 localMatmul 决策，见 §6.2。
>
> 本节是 PUT 模式下 `localMatmul` 参数（0/1/2）的**完整决策参考**，覆盖模式选择、L0C 容量约束、PipeBarrier 修复方案。适用于 AllToAll 语义算子（AIC 直写 yGm）。ReduceScatter 语义算子采用 §6.2 架构，不走 localMatmul 决策。

### 5.1 三模式定义

`localMatmul` 是 Host 侧 tiling 中的 `uint32_t` 字段（官方 AllToAll tiling `allToAllMatmulTilingData::localMatmul`，注释："0：不使能；1：使能atomiadd"），控制 PUT 模式下 AIC 的计算编排。分支逻辑在官方 AllToAll impl（`Run`/`SetupParams`）与 qbmm kernel（`QuantMatmulMxKernel::Init`/`ProcessSingleBatch`）：

| localMatmul | MatmulMode | AIC 执行顺序 | AtomicAdd | splitKNum | 适用场景 |
|:---:|:---|:---|:---:|:---:|:---|
| **0** | REMOTE-only | 遍历全部 rank 的 A；self rank 数据本地 GM 直读（`gmALocal`，`ProcessSingleBatch` REMOTE 分支 `rank==rankId && localMatmul!=1` 时改切 local 地址），但在 per-tile wait 之后才计算 → L0C 累加 → 单次 fixpipe → C | 不需要 | rankSize | 融合基线模式（官网 ST main.cpp 固定 `localMatmul = 0`） |
| **1** | LOCAL+REMOTE | **RunLocalMatmul**（本 rank A × B → C，直读 GM，首次写入）→ **SetAtomicAdd** → **RunMatmul**（远端 A × B → C，AtomicAdd 累加）→ **SetAtomicNone** | REMOTE 阶段开启 | rankSize-1 | **推荐**：本地 A 可前置计算，与 AIV PUT 并行 |
| **2** | DEFERRED_SYNC | per-tile 内：本 rank A × B → **L0C**（不 fixpipe）→ wait_flag → 远端 A × B → L0C 累加 → 最后一次触发单次 fixpipe → C | 不需要 | rankSize | L0C 容量足够时 |

> - `waitedMask` 为 `uint32_t`（qbmm kernel `ProcessSingleBatch` 内局部变量）→ 通信 tile 总数 ≤32 是硬约束。
> - LOCAL 阶段 `splitKNum=1`：本 rank 部分和单次 fixpipe 直写 C；仅 REMOTE 阶段逐 rank 在 L0C/GM 上累加。
> - `splitKNum` 赋值见官方 AllToAll impl `SetupParams`：localMatmul==1 → `rankSize - 1`；localMatmul==0/2 → `rankSize`。
> - mode 0 **不开启 AtomicAdd**：`isAtomicAdd_` 仅在 `matmulMode==REMOTE && localMatmul==1` 时置位（qbmm kernel `Init`）。
> - ⚠️ 官网 ST 仅覆盖 `localMatmul = 0`（官方 AllToAll ST main.cpp 显式设置）；mode 1/2 在 kernel 分支中存在但无 ST 样例。
> - ⚠️ **mode 2 仅 UDMA impl 可达**：`localMatmul==2` → `DEFERRED_SYNC` 的选择分支只存在于官方 AllToAll UDMA impl `RunMatmul()`。hcomm 变体（CCU 引擎）对任何 `localMatmul != 0` 都做 LOCAL 前置 + 恒 REMOTE（`splitKNum = rankDim - 1`）；注意 hcomm 把 `tilingData_->localMatmul` 原样透传给共享 kernel，kernel 在 `matmulMode==REMOTE && localMatmul==1` 时置 `isAtomicAdd_` 并由 `Run` 包裹 `SetAtomicAdd/SetAtomicNone`——即 **hcomm mode 1 经共享 kernel 开启 AtomicAdd（且正是 splitKNum=rankDim-1 路径正确的前提）**；但 mode 2 时 kernel REMOTE 分支仅 `localMatmul==1` 才跳过 self → hcomm 下 `localMatmul==2` 的 self 被 LOCAL 前置与 REMOTE 各算一次、mmad 计数与 `splitKNum` 错配，**静默产生错误结果（无任何报错）**。使用 mode 2 必须确认走 UDMA impl。

### 5.2 模式选择决策树

```
算子语义需要 ReduceScatter（M 轴输出切分 + 跨 rank 聚合）？
├── 是 → 采用 §6.2 计算在前架构（staging 即通信源 + per-tile 流水 + 增量归约）
│   ├── 无 LOCAL/REMOTE 双阶段，无 AtomicAdd，无 localMatmul 字段
│   ├── AIC vendor 复用官方 mm kernel（LOCAL 模式 + rank 退化）→ staging
│   ├── AIV 逐轮 AllToAll PUT + 增量归约（src 双缓冲 + FP32 累加）
│   └── 不适用本节 localMatmul 0/1/2 决策
│
└── 否（AllToAll 语义，AIC 直写 yGm）→ 按以下决策：
    ├── 默认 → localMatmul=0（REMOTE-only，官网 ST 默认）
    │   ├── 收益：最简单，无需 RunLocalMatmul
    │   └── 注意：self/remote 均在依赖 tile 的 wait 之后才计算
    │
    ├── 通算并行优化 → localMatmul=1（LOCAL+REMOTE+AtomicAdd）
    │   ├── 风险：MTE 异常（PipeBarrier 修复，见 §5.4）
    │   └── 收益：AIC local matmul 与 AIV PUT 并行
    │
    └── 精度优先 → localMatmul=2（DEFERRED_SYNC）
        ├── 前提：L0C 容量足够（见 §5.3）
        └── 收益：L0C 全 FP32 累加，单次 fixpipe，精度最好
```

**选择原则**：
1. **ReduceScatter 语义算子** → 直接采用 §6.2 架构，不走 localMatmul 决策
2. **AllToAll 语义算子** → 默认 `localMatmul=0`（官网 ST 固定下发 0）；通算并行优化用 `localMatmul=1`（需 PipeBarrier 修复，见 §5.4）；精度优先用 `localMatmul=2`
3. **localMatmul=0 是融合基线模式** — 官网 ST 固定下发 0，不是纯 AllToAll 专用

> **适用范围注记**：本节 localMatmul 0/1/2 决策仅适用于「AIC 直写 yGm」架构的 AllToAll 语义算子。ReduceScatter 语义算子采用 §6.2 的计算在前 + per-tile 流水架构，无 localMatmul 字段。

### 5.3 L0C 容量约束（DAV_3510）

DAV_3510 的 L0C 容量为 **256 KB**（芯片规格以 npu-arch skill 为唯一知识源，此处数值仅为约束推导参考，生产代码按目标芯片实际容量核算）。所有模式下，单个 tile 内各 rank 的部分和都**累加进同一块 L0C FP32 累加器**（mmad 第 8 参 `0` = reset、递增 = 累加、计满 `splitKNum` 触发单次 fixpipe；REMOTE 与 DEFERRED_SYNC 分支均如此，见官方 AllToAll qbmm kernel `ProcessSingleBatch` 及 Blaze `block_mmad_qbmm_mx.h`——Blaze 头文件来自 ops-tensor 仓，由 `cmake/third_party/ops-tensor.cmake` 按 pin 拉取，asc-devkit 安装内无 blaze 头文件），因此 L0C 需求**与 rankSize 无关、dtype 固定为 FP32，且对三种 localMatmul 模式完全相同**：

```
L0C 需求 = baseM × baseN × 4B（FP32）

约束：L0C 需求 ≤ 128KB（dbL0C ping-pong 半区）或 ≤ 256KB（单缓冲）

示例：
  baseM=128, baseN=128 → 128×128×4 = 64KB  ≤ 128KB ✅ 双缓冲可用 localMatmul=2
  baseM=256, baseN=256 → 256×256×4 = 256KB > 128KB ❌ 双缓冲不可用；=256KB ⚠️ 单缓冲临界
  baseM=128, baseN=256 → 128×256×4 = 128KB = 128KB ⚠️ 双缓冲临界
```

> `baseM`/`baseN` 由 `QuantMatmulTilingSwat::GetTilingData` 推导，Developer 可在 Host 侧打印确认（`QuantMatmulTilingBase::PrintTilingData` 会自动打印）。
> `dbL0c`（L0C 双缓冲）> 1 时实际可用容量减半（128KB ping-pong 半区），约束更严格。

**该约束对三种模式一视同仁，mode 2 相对 mode 1 没有额外 L0C 负担** — REMOTE 阶段（mode 0/1）与 DEFERRED_SYNC（mode 2）都是把各 rank 部分和累加进同一块 L0C 累加器（首份 reset、末份触发 fixpipe）；mode 1 的 AtomicAdd 发生在 fixpipe 写 GM 时，并不改变 L0C 占用；mode 1 的 LOCAL 阶段单发 mmad（`splitKNum=1`）同样只占这一块累加器。L0C 容量不足时三种模式同样不可用，只能调小 `baseM`/`baseN`。

### 5.4 localMatmul=1 的流水屏障修复

无屏障时，LOCAL 阶段 fixpipe（MTE3 管线）尚未排空，REMOTE 阶段的 `SetAtomicAdd` + AtomicAdd fixpipe 在同一 MTE3 管线上做 read-modify-write 会读到未完全写入的旧值 → MTE 硬件异常/超时类错误。**修复**：在 `RunLocalMatmul()` 与 `RunMatmul()` 之间加 `PipeBarrier<PIPE_ALL>()`（强制等待所有 pipe 完成后再继续，API 详见 `ascendc-api-best-practices` skill `references/api-pipeline.md`）。

> ⚠️ 官方 AllToAll UDMA impl `Run()` 当前**无**此屏障（`RunLocalMatmul()` 直连 `RunMatmul()`），使用 localMatmul=1 时 Developer 须自行添加。*工程提示：PipeBarrier 开销远小于通算并行收益，勿因加屏障直接回退 localMatmul=2。*

### 5.5 Host 侧 localMatmul 配置

`localMatmul` 在 `main.cpp` 中显式赋值（`all_to_all_matmul_tiling_data.h` 的默认值 `uint32_t localMatmul{0}` 与实际使用值可能不一致，禁止依赖默认值）。取值选择见 §5.2 决策树；官网 ST 固定下发 0。

### 5.6 速查表

| 问题 | 答案 | 详见 |
|:---|:---|:---|
| AllToAll 算子默认选哪个模式？ | localMatmul=0（官网 ST 默认） | §5.2 |
| localMatmul=1 报 507015？ | 加 `PipeBarrier<PIPE_ALL>()` | §5.4 |
| localMatmul=2 什么时候能用？ | L0C 容量足够（见公式），仅 UDMA impl | §5.3 |
| localMatmul=0 什么时候用？ | 融合基线模式（官网 ST 默认） | §5.2 |
| ReduceScatter 算子用什么架构？ | 计算在前 + per-tile 流水 + 增量归约（§6.2） | §6.2 |

### 5.7 ReduceScatter 语义算子

> ReduceScatter 语义（M 轴输出切分 + 跨 rank 聚合）是**计算在前模式**的典型示例，生产实现见本文档 §6.2：per-tile 流水 + 增量归约。

| 条件 | 推荐方案 |
|:---|:---|
| ReduceScatter 语义（计算在前 + 跨 rank 聚合） | §6.2 计算在前模式 |
| AllToAll 语义 + 通算并行 | localMatmul=1（§5.2） |
| AllToAll 语义 + 精度优先 | localMatmul=2（§5.2） |
| AllToAll 语义 + 融合基线 | localMatmul=0（§5.2） |

---

## 6. 多对象复用与扩展语义推导

### 6.1 winOffset 多对象复用

多通信对象（data + scale）复用同一 Win 区的 `winOffset` 分段布局、rankDataBytes 计算公式、每对象独立 UB commBuf 预算、以及"同 channel 多对象只 Wait/Drain 一次"的通道级不变量——以 [`communication.md`](communication.md) §1（winOffset 多对象复用）与 §2（Commit/Wait 不变量表）为唯一事实源，本节不重复。

> PUT 算子级编排验收标准见本文档 §4。

### 6.2 计算在前（compute-first）算子模式：per-tile 流水 + 增量归约

> **compute-first 方法论（从生产实践提炼）**：6 项关键约束——
> ① localLast 强制（移除 → 峰值 2T）；② UB 管理机制互斥（混用 → 507015）；
> ③ T>1 多套 tiling（单套 → baseM/headMSize 不匹配）；④ winOffset 按布局设置（共享布局 0 → 数据错误）；
> ⑤ TransA/TransB 参数化（固定 → 不支持转置）；⑥ 批量归约（逐行 → flag 爆炸）。
> 详细设计合同见各节。

> **本节主体是通用模式**：适用于一切**计算在前**的算子（AIC 先算输出、AIV 再通信输出并聚合），ReduceScatter 语义算子（M 轴输出切分 + 跨 rank 求和）仅作为已生产验证的示例引用——其中出现的具体数值均为示例实现的佐证，不是模式的一部分。
>
> **验证状态**：「FragmentTensor kernel + 严格分离编排 + 手动 UB 批量归约」与「vendor 复用 mm kernel + 时分复用 + TPipe 归约」两条路线均已工程验证；选型判据见 §6.2.1/§6.2.2/§6.2.6。

#### 6.2.1 架构总览（计算在前 + per-tile 流水）

适用场景：**计算在前**（AIC 先算 C），**通信在后**（AIV PUT C）。与 §4 PUT 编排（通信在前）方向相反。

计算在前算子的同步面天然最小：AIC 不消费任何通信数据，只存在单向"mm 完成"通道（AIC→AIV），无 AIV→AIC 回压通道，无跨卡部分和 ⇒ 无 `SetAtomicAdd/SetAtomicNone`。死锁论证随之简化：AIC 必然完成 → AIV Wait 必然解除。

三级流水：mm → aivComm → reduce。AIC 计算 C 写入 staging（workspaceGM）并按轮 SetFlag 通知 AIV；AIV 逐轮 WaitFlag 门控后执行通信与归约（默认严格分离编排：通信核「Commit → Wait」与归约核「Reduce(上一轮)」并行，见本节末尾 AIV 组织方式）。

**AIV 两种组织方式**：

| 方式 | 结构 | 适用 |
|:---|:---|:---|
| **严格分离（专职化，默认）** | 后 R 核专职通信、前 (核数-R) 核专职归约，`AllToAll(t) ∥ Reduce(t-1)` 错位流水 | 默认基线，R 越大收益越显著。通信对象 `totalJobs=rankSize`（每核 1 个 target 并行 PUT）；TeamBarrier `totalJobs=1`（仅 jobIndex=0 执行 CrossDevice）；**配 `Wait<BARRIER_NONE>`（仅 Drain）+ 手动 `teamBarrier_.CrossDevice()`**——严格分离下禁止 `Wait<BARRIER_DEVICE>`：其内建 CrossDevice 与分核守卫/手动 CrossDevice 序列叠加会打乱 rendezvous 配对，跨设备同步失效（曾致大面积元素错误，机制见 communication.md §2.2） |
| **时分复用（兜底）** | 全部 AIV 既通信又归约，每轮「WaitFlag 门控 → Commit → Wait → SyncAll → 归约本轮」串行推进、跨轮重叠 | 核数极少（核数-R 不足以摊薄归约）、归约工作量轻或调试定位期 |

**严格分离逐轮操作序列**（compute-first 通用编排模式；T=1 自然退化为单轮）：

| 阶段 | 通信核（后 R 核） | 归约核（前 核数-R 核） |
|:---|:---|:---|
| 门控 | `CrossCoreWaitFlag(flagId)`（无模板形态，见 §6.2.3） | 同左（全 AIV 同序同次数，计数平衡） |
| 核间同步 | `SyncAll<true>()` | 同左 |
| 通信/归约 | `Commit<BARRIER_NONE>()` → `Wait<BARRIER_NONE>()`（仅 Drain） | tile 0 无归约；tile t≥1 执行 `ReduceSum(t-1)` |
| 核间同步 | `SyncAll<true>()` | 同左 |
| 跨设备 fence | `teamBarrier_.CrossDevice()`（仅 jobIndex=0） | — |
| 核间同步 | `SyncAll<true>()` | 同左 |
| 尾轮 | `Finalize()` | `ReduceSum(T-1)` |

> SyncAll 一律放在分核守卫**外**（全 AIV 执行，计数平衡）；退化路径（commTurn≤1）合并为单轮。完整伪代码骨架（含 isCommBlock/isComputeBlock 守卫与退化路径）见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.1。

**分核映射红线（compute-first 严格分离）**：通信核为**后 R 核**（`jobIndex = GetBlockNum() - 1 - GetBlockIdx()`），归约核为**前 (核数-R) 核**（`blockIdx < rsCoreNum`）——与通信在前官方算子的"前 R 核"惯例正交，禁止混用（见 [`communication.md`](communication.md) §2 AIV 分核惯例）。

**基线边界声明**：compute-first 下通信依赖完整 mm 输出，mm 段天然暴露（无法被通信掩盖）；跨方案性能对比时须显式声明"mm 段暴露是架构边界而非回归"。

流水重叠时序：

```
AIC: [轮0 R个子mm] SetFlag [轮1 R个子mm] SetFlag [轮2 R个子mm] SetFlag ...
AIV:                 WaitFlag [comm0+reduce0] WaitFlag [comm1+reduce1] WaitFlag [comm2+reduce2]
```

AIC 计算不依赖通信，第 t+1 轮的 mm 与第 t 轮的通信+归约并行（WriteNbi 非阻塞，UDMA 后台搬运）。T=1 时自然退化为单次全量 mm + 单轮通信归约，无额外开销。

#### 6.2.2 mm 内核选型：FragmentTensor 默认推荐

**默认推荐：FragmentTensor 自研 kernel（消 R 循环）**。当 A 数据全在本卡 GM 连续 `[m, k]`、各 rank 段按 `chunkM` 间距排列时（compute-first 算子的典型形态），用 FragmentTensor 打包 R 个 rank 段地址，**一次 matmul 调用覆盖 `R × curTileM` 行**：

- Params 构建一次、调用开销最小；per-fragment L1 缓存隔离（tile 跨越 rank 边界时自动切换 fragment 地址，`QmmMxBlockMmadFragment`，详见 [`compute.md`](compute.md) §7）
- 约束：`R ≤ 32`（`MAX_FRAGMENT_COUNT=32` 是 **FragmentTensor 单次构建的 fragment 数上限**，即 rank 数上限；与通信轮次 T 无关——每轮构建 R 个 fragment，T 轮各自独立构建。工程验证通过）

**例外：vendor 复用官方 mm kernel（R×T 子调用）**。仅当 **R×T 很小且为大 shape**（CUBE 时间足够长、SCALAR 被淹没）时可选——选择时必须在 DESIGN.md 中论证 SCALAR 占比可接受，否则按 FragmentTensor 实现。优势：逻辑零修改、官方 kernel 演进可随共享层同步吸收。vendor 复用要点：LOCAL 模式 + rank 退化参数（`rankId=0 / rankSize=1 / splitKNum=1`）实例化，`isAtomicAdd` 恒 false。

> ⚠️ **R×T 子调用的 SCALAR 风险（选型红线）**：R×T 循环每轮重建完整 Params（tiling 字段换算 + 地址偏移）+ BlockScheduler 重新初始化，SCALAR 指令占比随 R×T 线性放大。特征：R×T 较大时即使 CUBE/MTE2 流水占比接近饱和，cube_utilization 仍被 SCALAR 压到极低——pipe 高占比但利用率极低是 SCALAR 主 bound 的特征信号。规避：默认 FragmentTensor 一次调用；vendor 路径下或减少 T（增大 tileM）、或对 Params 做增量更新（只改地址字段）。

与 AllGather FragmentTensor 的对比（数据来源/打包/跨核同步/C 输出/dependId 预触发）以 [`compute.md`](compute.md) §7.2 为唯一事实源，本节不重复。

**mm/comm 深度重叠的正确拆法**：T>1 时 AIC 把全量 mm 拆为按轮驱动的子区间（覆盖全 M），**tile 大小不变、只缩 problem M**——不增加 K-window 总数与 MTE2 次数，反而因子问题 L2 footprint 缩小提升 cache 命中与 cube 利用率（大 shape 实测显著收益）。FragmentTensor 路径下该拆分由 fragment 编排天然完成、无 R 循环；vendor 路径下才是 R×T 子调用（受上述 SCALAR 风险约束）。注意与"拆小 tile"路线区分：后者 MTE2 次数翻倍，已证伪（见 [`optimization-playbook.md`](../troubleshooting/optimization-playbook.md) §3）。

**localLast 编排（compute-first 强制）**：FragmentTensor 打包时把本 rank 段排到 fragment 末尾（`[remote..., local]`），AIC 在 remote→local 边界处提前 `SetFlag(flagA)` 通知 AIV 启动通信、全部算完再 `SetFlag(flagB)`——**remote 段算完即启动 AllToAll，无需等本卡 local 段**，通信提前一拍进入流水。落地三条纪律（缺一不可）：① `cFragAddrs_` 必须保持**原始 rank 顺序**（mmadFrag 内部 L1 cache 管理依赖原始顺序，重排只作用于 A/ScaleA 的 addrList）；② 边界预计算 `localFragBoundary = headMainRows − fragM`，调度循环内 `mPos >= localFragBoundary` 首次满足时 Set flagA；③ 未跨边界核（无本卡 tile）必须**兜底补 Set flagA**，否则 AIV `WaitFlag(flagA)` 挂死。**移除 localLast = 阻塞级错误**：若改用"每轮末 Set 双 flag"替代，flagId 用量翻倍且丧失通信提前启动能力；Reviewer 不得以"A/C 片段错位"为由建议移除（错位根因是 cFragAddrs_ 顺序写错）。**收益边界（实测）**：通信提前启动的收益与瓶颈归属相关——AIC MTE2 主导（AIV 82% 时间在 WaitFlag）时重排收益 ≈ 0 但无害（低成本保险；静态分析曾误判"收益≈0 即无价值"，实测推翻的是"无价值"而非"无收益"）；通信暴露场景（小 shape/浅流水）收益显著。因 flagA/flagB 配对契约依赖此编排，**无论收益场景均不得移除**。实现要点见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.8。

FragmentTensor 详见 [`compute.md`](compute.md) §7。

#### 6.2.3 per-tile 流水编排

**flag 编排**：AIC `CrossCoreSetFlag<0x2, PIPE_FIX>(flagId)` ⇔ AIV **无模板** `CrossCoreWaitFlag(flagId)` 配对——官方配对惯例即"模板 Set + 无模板 Wait"（注册版 `chunk_gated_delta_rule_stage3.h`、`matmul_reduce_scatter_fp16_bf16.h` 均此形态，见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §4.1），flagId 按 §3.3 规则选取。工程验证形态：计数式 flag，event ID 0（flagA，remote 段算完提前通知）和 1（flagB，本轮全算完）。T=1 时循环外单次配对；T>1 时逐轮计数式配对（T 次 Set ⇔ T 次 Wait）。Set 用 `PIPE_FIX`（fixpipe 排空语义）；**AIV 侧 Wait 必须用无模板形态**：Wait 之后 AIV 紧随标量流操作（通信对象 `Commit`/WriteNbi 下发）——模板形态（如 `<0x2, PIPE_MTE2>`）只把等待插入指定 pipe 队列、**不阻塞标量流**，Commit 会在 flag 生效前执行 → PUT 读到未写完的 staging → 对端 Win 槽数据缺失（生产实测间歇精度失败，竞态反例见 `api-crosscore-sync.md` §4.1）。模板 Wait 之后若紧跟 `SyncAll<true>()`/`PipeBarrier<PIPE_ALL>()` 会**恰好掩盖**该缺陷——掩盖不等于正确，编排顺序一变竞态即暴露。**计数深度口径**：硬件事实为"每核 16 个 flagId（0-15）"与"Set/Wait 必须严格配对"；每个 flagId 计数器范围 0-15（官方有据），衡量**未消费积压**——紧邻配对编排下积压 ≈ 1-2 不触顶，不构成 T 上限。设计时以"flagId 数量 16 + Set/Wait 严格配对"为硬约束，T 上限按 case 实测确定而非套用固定数值。

**双 flag 计数式（compute-first 默认，配合 §6.2.2 localLast）**：用两个 flagId 分工——`flagA`（AIC 在 remote→local 边界 Set，AIV 通信核 Wait 后启动 AllToAll）与 `flagB`（AIC 全 fragment 算完 Set，AIV 归约核 Wait 后启动 reduceSum）。每轮每核两个 flagId 各 Set 一次，紧邻配对下两 flagId 计数各自平衡；两 flagId 互不冲突。

**T=1 退化路径**：`T=1` 时循环执行 1 次，自然退化为单次全量 matmul + 单次通信 + 单次归约，无额外开销。统一 `for t` 循环路径，无 if/else 分支。

**SetFlag 位置**：在每轮 mm 子区间完成之后。所有 AIC 核都执行（含零 tile 核），保证逐核 Set/Wait 计数平衡。

**同步机制**：

| 机制 | 用途 | 说明 |
|:---|:---|:---|
| `CrossCoreSetFlag<0x2, PIPE_FIX>(flagId)` | AIC → AIV 通知 | 本轮 mm 完成（全核含零 tile 核） |
| `CrossCoreWaitFlag(flagId)`（AIV 侧无模板形态） | AIV 等 AIC | 逐轮门控，T=1 单次 / T>1 计数式；Wait 后有标量流通信下发，**禁止模板形态**（不阻塞标量流 → 竞态，见 §6.2.3 与 `api-crosscore-sync.md` §4.1） |
| `SyncAll<true>()` | AIV 核间同步 | 门控后 + 归约前各一次；**放在分核守卫外**（全 AIV 同序同次数，计数平衡） |
| `Wait<BARRIER_DEVICE>()` | 通信完成等待（入门基线） | 内建 Drain + CrossDevice 每轮 rendezvous（仅 `GetBlockIdx() < rankSize` 的核执行 Commit/Wait；self-target 核跳过 Drain 但仍执行 CrossDevice，计数天然平衡）。正确性优先、写法最简单 |
| `Wait<BARRIER_NONE>()` + 手动 `teamBarrier_.CrossDevice()` | 通信完成等待（多核并行 + 严格分离时的默认） | Wait 仅 Drain 本 block channel，不做框架内建 CrossDevice；改由 `SyncAll<true>()` 后由 block 0（jobIndex=0）显式调用 `teamBarrier_.CrossDevice()` 完成跨设备 fence，再一次 `SyncAll<true>()` 放行。收益：通信等待与跨设备 rendezvous 解耦，同步点显式可控（该变体已工程验证） |

> **通信轮次计数同源**：外部需要轮次计数时，不得调用早退核（`jobIndex >= totalJobs`，CollectiveCommBase::Init 早退）的通信对象访问器——其成员未初始化，读到 UB 会挂死。应读 impl 侧全核已初始化的 tilingData 字段（同一结构体按址传入，同源性等价）。

> localLast + 双 flag 计数式（compute-first 默认编排）的完整落地机制见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.8。

#### 6.2.4 staging 与 win 区布局

**staging 布局**（AIC 写入的 mm 输出，= workspaceGM）：完整 `[M, N]` 连续 BF16。M 轴第 targetRank 段恰好就是 PUT 要发给 targetRank 的 chunk——chunk 连续布局与完整 mm 输出天然一致，**零重排、零额外拷贝**（compute-first 场景的关键红利：mm 输出即通信源）。

**win 区布局**（AllToAll PUT 后接收的数据）：按**源 rank 分槽**，槽 i 存放 rank i 推来的本卡结果段。

**win 区头部元数据区（布局相关，条件性规则）**：Win 区内是否存在 barrier 元数据区取决于 host 建链布局，两种布局与"假通过"风险以 [`communication.md`](communication.md) 陷阱 #12 为唯一事实源；host 侧预留、PUT 写入偏移、归约读取偏移三处必须同源（示例实现形态见场景文档 design.md §3.6）。

**归约来源二选一寻址**（第 i 个来源）：
- `i != rankId`：`winGm + i×chunkBytes`（远端 rank 推来的数据）
- `i == rankId`：`stagingGm + rankId×chunkBytes`（**本卡段本地合并**，不经通信链路——PUT self 槽无写入者，本卡结果段直接从 staging 读，省一次拷贝；代价是 Win self 槽闲置 R 分之一，共享层零修改红线下为已知取舍）

**Win 区容量**：需求 = `M × N × sizeof(CType)`，host 侧必须用 `HcclGetHcclBuffer` 实测值校验（见 §6.2.10）。

#### 6.2.5 通信模式选型：PUT 优先

GET 模式 4+ rank 稳定性**未验证**（官网无 GET 算子样例；已知一次工程迭代 PUT→GET 回退但根因未定位——弱证据，非结论性否定），计算在前场景优先使用 PUT 模式。

| 维度 | PUT（推荐） | GET |
|:---|:---|:---|
| 数据搬运 | 1×（推送） | 2×（拷贝 + 拉取，多 tile 场景） |
| 稳定性 | 4+ rank 验证通过 | 4+ rank 未验证（无样例 + 一次未定位根因的回退） |
| 官网样例 | 有（all_to_all / all_gather） | 无 |
| 数据方向 | AIV 推 → 远端 win 区 | AIV 拉 ← 远端 win 区 |

GET 钩子基础设施已就绪（`AllToAllCommGetImpl`），但无算子使用方。GET 地址公式与 self 跳过规则见 [`communication.md`](communication.md) §3。

#### 6.2.6 增量归约（reduceSum）实现

**增量语义**：每轮处理输出行区间 `[t×tileM, (t+1)×tileM)`，各轮行区间不相交、无跨轮累加状态——收齐一轮立即归约一轮，与"全部通信完成后一次性归约"精确等价，但通信延迟被向量归约掩盖。

**UB 管理两种策略（性能耦合，非纯并列——见下方耦合说明）**：

| 策略 | 要点 | 性能耦合判据 |
|:---|:---|:---|
| 手动 UB 静态偏移（**推荐，与官方算子一致**） | `MakeMemPtr<Location::UB>(offset)` / `LocalTensor` 字节偏移划分，无 TPipe 状态。布局组成要素：**FP32 累加器（单份，语义依赖）+ src 双缓冲（BF16 搬入 + FP32 Cast 目标，pingpong 交替）+ 输出 BF16 buffer**，按 float 大小对齐；多行批量归约（一次处理多行，行数按 UB 容量自适应）摊薄 flag 次数 | **批量归约的必要前提**——只有手动 UB 能用多行 LocalTensor 布局 + 2D DataCopyPad(blockCount=多行） 实现批量摊薄（显著摊薄同步开销） |
| TPipe + guard TBuf | 先 `InitBuffer` 一个 guard TBuf（544B = commBuf 512B + barrierBuf 32B）占位静态通信区，TPipe 管理的归约 buffer 从其后分配，**物理消除与静态通信区的重叠**（[`communication.md`](communication.md) 陷阱 #9）；buffer 声明式分配 | ⚠️ **TQue 单 buffer 搬运（AllocTensor/EnQue/DeQue）天然倾向逐行**：每次搬运一行，无法满足纪律 5 的批量摊薄要求，flag/同步次数爆炸。仅归约逻辑极简单、单轮数据量小（flag 次数可忽略）时可接受；否则选手动 UB |

> **互斥性红线**：`TPipe::InitBuffer` 与 `Te::MakeMemPtr<Te::Location::UB>` **必须二选一**，禁止混用——两套机制偏移空间不共享，混用导致地址重叠 → MTE2 UB out of bounds（507015）。通信区与归约区必须使用同一机制。官方 AllGather/AllToAll 算子均不建 TPipe，UB 用 `MakeMemPtr` 静态分配（`operator-anatomy.md` §4.3）。

> **耦合结论**：批量归约（纪律 5）⇒ 必须手动 UB；选 TPipe/TQue ⇒ 默认接受逐行低性能。不要用"TQue ⭐推荐"推导归约实现——TQue 适用边界见 `ascendc-api-best-practices` skill `references/api-pipeline.md`。

**归约事件 4 类 HardEvent**（BF16→FP32→Add→BF16 归约的通用配对模式，pingpong slot = i%2；Set 无配对 Wait = 挂死 507014）：

| 事件 | 语义 | Set 时机 | Wait 时机 |
|:---|:---|:---|:---|
| `MTE2_V` | MTE2 搬入完成 → V 可读 UB | DataCopyPad 后立即 Set | Cast 前 Wait |
| `V_MTE2` | V 读完 src → MTE2 可复用该 slot | Cast+Add 完成后 Set | **下下次**复用同 slot 搬入前 Wait（`i≥2` 时 `WaitFlag<V_MTE2>(slot)`） |
| `V_MTE3` | V 写完 dst → MTE3 可搬出 | Cast（输出）后 Set | DataCopyPad 搬出前 Wait |
| `MTE3_V` | MTE3 搬出完成 → V 可复用 dst | DataCopyPad 搬出后 Set | 下一批 Cast（输出）前 Wait |

> 完整 C++ 代码模板（含 Init 预置 MTE3_V、R=1 边界、循环结束残留事件消费）与批量归约循环骨架、6-slot UB 布局、host 预算公式——见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.2-§5.4。

**同步次数量化判据**（reviewer 据此判 FAIL）：
- 逐行归约：flag/同步次数 = `行数 × N段数 × R`（随三维度线性爆炸）——**逐行归约（blockCount=1）= 性能 FAIL**
- 批量归约：flag/同步次数 = `ceil(行数/batchRows) × N段数 × R`（除以 batchRows 摊薄）

**共同纪律**（与策略无关，违反即出 bug 或性能劣化）：

1. **事件链同迭代配对**：所有 Set/WaitFlag（HardEvent）必须在同一迭代内配对——跨迭代 Set-Set 无中间 Wait 在 Ascend 950（dav-3510）实测挂死；V-V 依赖加 `PipeBarrier<PIPE_V>` 防御
2. **src 双缓冲**：srcBF16 双份 + 事件 ID 双份，使 MTE2(source i+1) 与 V(Cast+Add source i) 跨 pipe 重叠；**累加器等有语义依赖的 buffer 保持单份**，不能盲目复制。**禁止 in-place BF16→FP32 Cast**（API 级禁令见 `ascendc-api-best-practices` skill `references/api-precision.md`）：FP32 占 2× 空间，in-place Cast 的 FP32 输出会覆盖同 buffer 内尚未读完的 BF16 数据 → 精度系统性错误；必须使用独立的 srcFP32 双缓冲
3. **DataCopyPad 2D + 64B pitch + 多行 blockCount**：归约搬入/输出必须用 2D DataCopyPad 且 `blockCount = 本批行数`（多行一次搬运）；**禁止 `blockCount=1` 逐行搬运**（性能反模式）；行距按 64B pitch 公式换算，UB 预算按 pitch 后元素数核算。**⚠️ strided 场景硬件隐式上限**（`srcStride > 0`，即 N > 单次搬运列宽 redUbN 时 blockCount 存在隐式上限、超出行静默丢零——API 级行为与防御三法见 `ascendc-api-best-practices` skill `references/api-datacopy.md`）：本场景的两条合法规避路径为**方案 A**——host 侧 UB 预算推导时限制 `redUbM ≤ 32`（N > redUbN 时）；**方案 B**——`gmStride > 0` 时退化 1D 逐行 DataCopyPad（blockCount=1 无 stride，绕开限制；仅在 strided 场景触发，非 strided 场景保留 2D 批量，flag 次数由批量事件管理摊薄不爆炸）
4. **N 超限时按列分段**：单轮 `tileM × N` 超出 UB 预算时，按 N 维分段循环处理（`maxNPerSeg = UB预算 / 每元素字节数`，如 18B/elem——6-slot 全布局精确口径：2×2B src BF16 + 2×4B cast FP32 + 4B acc + 2B out（§5.4 同口径）；保守口径 24B（生产实现含 guard 预留）——双缓冲布局下 180KB 预算 → 精确口径 N=7680 一段）；分段数由 host 按 UB 预算推导，不要为固定 N 写死 UB 布局
5. **多核并行归约**：归约核集合按 AIV 组织方式确定（时分复用 = 全 AIV；严格分离 = 前 核数-R 核），集合内按连续行块均分本轮 `tileM` 行（余数前置核 +1，每核只处理自己的 `[startRowId, endRowId)` 行区间；行块均分属算子内辅助逻辑，自行实现即可）；**禁止单核归约**（如 `GetBlockIdx()==0` 独立承担全量行：写竞争或归约耗时独占，= 性能 FAIL）；**单 tile 退化路径（T≤1）同样用批量归约**——退化路径 flag 次数占比更高
6. **staging 可见性由 flag 配对保证，禁止加 dcci**：AIC 经 fixpipe（MTE3）写 staging、AIV 归约核经 MTE2（DataCopyPad）读 staging，其跨核可见性由 `CrossCoreSetFlag<PIPE_FIX>` ⇔ `CrossCoreWaitFlag<PIPE_MTE2>` 配对（含 SyncAll）提供内存序保证——**参考实现不依赖 dcci**。归约读到 staging 旧值/0 时，根因是 flag 配对缺失、多核写竞争或地址不同源，**禁止以"cache 一致性"为由对 staging 加 dcci**（属误诊，掩盖真根因）
7. **TPipe 与静态通信区绝不混用**：通信对象 UB（commBuf/barrierBuf）用静态偏移顺序排布，归约 buffer 经 guard TBuf 或静态偏移与其物理隔离——混用重叠 = 通信数据被踩踏 → 死锁/精度错（[`communication.md`](communication.md) 陷阱 #9）

**累加精度策略（生产基线：FP32 中间累加）**：BF16 → Cast float32 → Add → Cast 回 BF16（`CAST_RINT`）是工程验证基线——BF16 直接 Add 在 rank 数多时累加误差放大超容差，FP32 中间累加可稳定收敛到容差内。

**归约降精度原则**：**先升精度保底；只有收益显著（未被 mm/通信流水掩盖）且全量配置精度实测达标时，才考虑降精度，且保留一键回退**——被掩盖的模块做激进降精度，收益≈0 而风险实在（rank 数增大时累加误差放大）。实证案例：开发期曾提议"Ascend 950 支持 BF16 直加、省去两次 Cast"，最终保留 FP32 中间累加——降精度类提议未经全量精度实测（randn + 多 rank + 重复 launch）不得采纳。详见 [`optimization-playbook.md`](../troubleshooting/optimization-playbook.md) §4。

#### 6.2.7 通信轮次 T 派生与 tail tile 处理

**策略 B（默认）：host 派生 T 无尾块 + 单份 tiling 复用**

PUT 钩子的源地址公式为 `src = localAddr + target×chunkBytes + tileIdx×tileMaxBytes`。**尾块偏移的精确条件**：源地址按 `tileIdx × tileMaxByteSize_`（头块尺寸）推进（地址计算在 `all_to_all_udma_put.h` DoCommit；`tileMaxByteSize_` 等游标推进由基类 `CollectiveCommBase::Commit()` 完成），而 host 侧 chunk 实际尺寸按"头块数×头块 + 尾块数×尾块"累加——**单尾块且尾块 ≤ 头块时**（`headTileCnt×tileMaxBytes` 恰等于尾块起点，尾块起点偏移精确），**偏移不错位，PUT 直传合法**（工程验证形态）；**多尾块或尾块 > 头块时**才发生 `tileIdx×tileMaxBytes` 偏移错位。因此：
- **单尾块（≤头块、16 对齐）是合法形态**：`tailMSize = chunkM % headMSize` 如实填 `commTilingData.splitAxisTailSize`、`splitAxisTailCnt=1`，tail 轮用 tail tiling（`GetTilingData(rankSize×tailMSize, ...)`）
- host 派生无尾块优先（策略 B，实现最简）；shape 天然带尾块时单尾块直传（工程验证形态）优于强切 padding
- 仅当无法保证"单尾块 ≤ 头块"时才必须走策略 A（padding + realFragmentSize）

host 派生（策略 B）：

```
T0 = max(1, CeilDiv(mSeg, 目标 tile 行数))
在 [T0, CeilDiv(mSeg, 16)] 内取最小满足 T | mSeg 的 T（tileM = mSeg / T）
找不到（如 mSeg 为质数）→ 回退 T=1
```

由于 T>1 按轮拆子区间（**tile 大小不变、只缩 problem M**），tiling 只需 **1 份**（完整 `{M, N, K}`）——FragmentTensor 路径由 fragment 编排天然完成（消 R 循环），vendor 例外路径才显式拆 R×T 个子调用（复用同一份 tiling，受 §6.2.2 SCALAR 约束）——无需为子问题/tail 单独生成 tiling。

**策略 A（合法替代路径）：tail padding + 多套 tiling**

当算子语义无法保证无尾块时（如 tile 行数由外部契约指定，或需要在 R≥4 时用小 headMSize 增强流水掩盖而产生 tail）：

1. `paddedCurTileM = (curTileM + 31) & ~31`（32 对齐），`headRows = R × paddedCurTileM` 使 BlockScheduler 无 unaligned tail
2. `realFragmentSize = curTileM`（实际行数），用 `MakeFragParam(fragSize=padded, realFragSize=real, ...)` 限制实际读取范围
3. 为不同子问题生成专用 tiling（全量 / head / tail 多套），`blockDim` 固定用全量 tiling 的 `usedCoreNum`——选择逻辑（per-tile 流水 + tail 处理的通用组织模式）：`T=1` 恒用全量 tiling；`T>1` 时 head tile 用 head 子问题 tiling（`GetTilingData(headMSize, n, k)`），tail tile 用 tail tiling（`GetTilingData(rankSize×tailMSize, n, k)`）；字段契约示例（C++ struct）见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.5

**headMSize 自适应决策**（通用原则）：决策方向是"分片小→减小 tile 让通信尽早启动；分片大→增大 tile 减少同步次数"，受 16 对齐约束（flagId 上限约束见 §6.2.10）。具体分档标定值随算子 shape/dtype 与流水深度目标 per-case 搜索确定，**不由本通用文档给出固定数值**——已验证实现的分档规律（含 N 维度模型：tileMaxBytes = tileM×N×2 主导分档，小 N 大 tileM / 大 N 小 tileM 双向实测锚点）见 [`scenarios/compute-first-reduce-scatter/development.md`](../scenarios/compute-first-reduce-scatter/development.md) §5.6（属示例实现，勿直接照搬）。

**tail 路径特征信号**：tail 行数非 32 对齐时 Blaze matmul 产出垃圾数据（根因：逻辑 M 不能被 baseM 整除产生非对齐 tail tile）——"仅非对齐 shape 精度 FAIL、其余全 PASS" 即指向 tail 路径。

**host 侧内存分配必须与 kernel 实际读取范围一致**（详见 [`optimization-playbook.md`](../troubleshooting/optimization-playbook.md) §4）。

#### 6.2.8 SWAT baseM 减半优化

SWAT tiling 在 `blockNum ≥ aicNum` 时 `baseM` 保持 256、tiles/core 过低时的手动减半后处理（触发条件、L1 安全验证、尾块参数重置）——以 [`optimization-playbook.md`](../troubleshooting/optimization-playbook.md) §2「tiles/core ≈ 1 且 CUBE bound」行为唯一事实源，本节不重复。

#### 6.2.9 优化方向索引

优化路径、现象→手法速查、已证伪方向（dbL0c=2 / 增大 stepK / 拆小 tile / totalJobs=1 臆造约束等）、实验纪律与"分析判定须被最小实验复核"原则——统一以 [`optimization-playbook.md`](../troubleshooting/optimization-playbook.md) 为唯一事实源，本节不重复。

#### 6.2.10 硬限制与边界

| 限制项 | 上界 | 说明 |
|:---|:---|:---|
| flagId 数量 | 16 个（0-15） | 硬件事实；Set/Wait 必须严格配对（不配对 = 未定义行为/挂死）。计数器 0-15 衡量**未消费积压**（紧邻配对下 ≈ 1-2 不触顶，官方有据）——不构成 T 上限；计数式固定 flagId 配对已在多 tile 场景工程验证（T=16）。T 上限按 case 实测确定，不作 host 拒绝依据 |
| 通信轮次 T | `T \| mSeg`（策略 B）；单尾块合法（见 §6.2.7） | **单尾块且尾块 ≤ 头块时 PUT 直传合法**（偏移精确）；多尾块或尾块>头块才错位（走策略 A）。轮次索引型 flagId 时轮次索引 < FLAG_ID_MAX(16) 且避开 SyncAll 保留区（`SyncAll<true>` 占 14 → 实际安全上界 13）；计数式固定 flagId 不受轮次数限制 |
| `tailMSize` | 单尾块：16 对齐且 ≤ headMSize；策略 A：16 对齐，padding 后 32 对齐 | Blaze matmul 最小行粒度 16；单尾块直传形态见 §6.2.7，padding 形态见策略 A |
| `CeilDiv(K_mm, 64)` | 必须为偶数 | MXFP8 scale 对齐（每组 64 元素，2 字节）；`K_mm` 指**单次 mm 调用的 K 长度**（K 被 rank 切分的语义下为 per-rank K；官方 ST 强制校验：`tests/st/all_to_all_quant_matmul/src/main.cpp`、`tests/st/all_gather_quant_matmul/src/main.cpp`） |
| `m` | 必须被 `rankSize` 整除 | M 轴按 rankSize 切分 |
| `usedCoreNum` | ≥ `rankSize` | 否则 TeamBarrier rendezvous 永不齐 → 无超时挂死；host 强制校验 |
| Win 区容量 | `M×N×sizeof(CType)` ≤ `HcclGetHcclBuffer` 实测值 | host 前置校验，非法即拒绝 launch |
| Win 区数据/元数据分离 | PUT/GET 数据不得覆盖 Win 区内元数据/barrier 区 | 官网布局 barrier flag 在独立 BARRIER_BUF（Win 数据区从 0 可用）；共享布局按约定偏移跳过，host/kernel 偏移同源（[`communication.md`](communication.md) 陷阱 #12） |
| 单轮 PUT 数据量 | ⚠️ 风险提示（非硬红线） | 无官方上限（bring-up 期过大单轮曾见间歇失败，边界未定）；大单轮 PUT 按 case 复核（连续多轮精度+重复 launch 一致性），不作 host 强制拒绝依据（[`communication.md`](communication.md) 陷阱 #13） |
| mm 段暴露 | 架构边界 | compute-first 下通信依赖完整 mm 输出，mm 段无法被通信掩盖；跨方案性能对比时须显式声明 |

> 通用上限（commTurn ≤16、waitedMask tile 总数 ≤32、rankSize ≤64、AG 变体 ≤8、FragmentTensor ≤32 仅自研路径、每通信对象 UB 512B）见 [`operator-anatomy.md`](../operator-design/operator-anatomy.md) §7.6，本表不重复。

---

## 后续阅读

- communication.md — 通信接口契约（四段式/钩子/TeamBarrier/CrossCore flag 机制）
- compute.md — 计算接口（Blaze/MatmulMode/L0C）
- operator-anatomy.md — 完整算子骨架
- `ascendc-api-best-practices` skill `references/api-atomic.md` — SetAtomicAdd 约束
