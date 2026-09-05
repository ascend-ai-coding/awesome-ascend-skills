# 跨核同步 API 使用指南

> **适用场景**：AIC（Cube 核）与 AIV（Vector 核）协同的 Mix 算子中的跨核同步，包括 CrossCoreSetFlag/CrossCoreWaitFlag 流水线通知和 SyncAll 块间同步。本文档只覆盖 asc-devkit 标准 API 的签名、约束与平台差异；具体算子的编排模式请参考对应框架文档。

---

## 目录

1. [概述](#1-概述)
2. [API 签名与参数](#2-api-签名与参数)
3. [flagId 硬件规则](#3-flagid-硬件规则)
4. [平台生效性差异](#4-平台生效性差异)
5. [最小通用示例](#5-最小通用示例)
6. [常见错误](#6-常见错误)
7. [检查清单](#检查清单)

---

## 1. 概述

Mix 算子（如 `KERNEL_TYPE_MIX_AIC_1_1` 核配比）中同一份 kernel 二进制同时运行在 AIC 和 AIV 上，靠编译期分支 `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` 隔离职责，两侧通过跨核同步原语协调：

- `CrossCoreSetFlag<modeId, pipe>(flagId)` — 跨核通知（AIC↔AIV）
- `CrossCoreWaitFlag<modeId, pipe>(flagId)` — 等待对端通知
- `SyncAll<isAIVOnly>()` — 块间同步（所有 block 到齐）
- `SetFlag<HardEvent>` / `WaitFlag<HardEvent>` — 核内 pipe 间硬件事件同步（与 CrossCore 不同层）

---

## 2. API 签名与参数

> 头文件：`basic_api/kernel_operator_block_sync_intf.h`

### 2.1 CrossCoreSetFlag

```cpp
template<uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);
```

| 参数 | 类型 | 含义 |
|:---|:---|:---|
| `modeId` | `uint8_t` | 同步模式（见 §2.5） |
| `pipe` | `pipe_t` | 发起 flag 的流水线阶段 |
| `flagId` | `uint16_t` | flag 标识符（硬件规则见 §3） |

### 2.2 CrossCoreWaitFlag

```cpp
template<uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

| 参数 | 类型 | 含义 |
|:---|:---|:---|
| `modeId` | `uint8_t` | 建议与 SetFlag 的 modeId 一致（A3/910b 上该模板参数不生效，实践中存在 Set 0x2 / Wait 默认 0 的配对） |
| `pipe` | `pipe_t` | 等待 flag 的流水线阶段（平台约束见 §4） |
| `flagId` | `uint16_t` | 必须与 SetFlag 的 flagId 一致（截断后相同即可，见 §3） |

### 2.3 SyncAll

```cpp
// 3510/5102 架构（支持 SyncAllConfig 指定 trigger/wait 流水）
template<bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
__aicore__ inline void SyncAll();

// 其他架构
template<bool isAIVOnly = true>
__aicore__ inline void SyncAll();
```

| 参数 | 类型 | 含义 |
|:---|:---|:---|
| `isAIVOnly` | `bool` | `true`：仅 AIV（Vector）核参与同步；`false`：AIC+AIV 全部参与 |
| `config` | `SyncAllConfig` | 仅 3510/5102：指定 triggerPipe/waitPipe（仅支持 MTE2/MTE3/PIPE_ALL），且仅在 `isAIVOnly=true` 时有效 |

**前置条件（官方）**：
- 纯 Vector 算子必须 `isAIVOnly=true`，否则卡死
- Mix 算子 `isAIVOnly=true` 只同步 Vector 核
- block 数不得超过物理核数
- 多流并发场景需 batchmode，否则死锁
- SyncAll 硬同步内部占用 flagId [11-14]（见 §3），官方不建议与 CrossCoreSetFlag 混用

**性能代价（多核流水场景必须知晓）**：SyncAll 是全 block 硬栅栏——所有参与核必须全部到达才能放行，会**打散不同角色核之间的时间线重叠、阻断流水**。使用纪律：

1. **禁止出现在 tile 内层循环的热路径上**（每轮通信后一次 SyncAll 是常见的性能杀手）；每轮必须块间同步时，同步次数即流水轮次的固定开销，轮次 T 的设计要把 SyncAll 次数计入成本
2. 能用**计数式 CrossCore flag 握手**（生产者-消费者配对）解决的时序，不用 SyncAll
3. 必须使用时，放在分核守卫**外**由所有参与核同序同次数调用（计数平衡），且确认没有"部分核多调一次"的路径
4. 同步频率与通信粒度解耦：需要降低 SyncAll 开销时，增大 tile 粒度减少轮次，而不是删同步点（同步点是数据可见性的正确性保障，次数不可裁剪——见 §2.4 与 api-hcomm.md）

### 2.4 SetFlag / WaitFlag（核内硬件事件）

```cpp
template<HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID);

template<HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID);
```

> 用于核内任意 pipe 间硬件事件同步（如 `HardEvent::MTE1_MTE2`、`HardEvent::M_MTE1`、`HardEvent::V_MTE3` 等），与 CrossCore flag 不同层——CrossCore 是跨核（AIC↔AIV），HardEvent 是核内 pipe 间。

### 2.5 modeId 取值

| modeId | 含义 | 平台 |
|:---|:---|:---|
| `0` | AI Core 核间同步（AIC 调用时同步所有 AIC；AIV 调用时同步所有 AIV） | 950/A3/A2 |
| `1` | AI Core 内部两个 AIV 之间的同步 | 950/A3/A2 |
| `2`（`0x2`） | 同核 AIC 与所有 AIV 之间的同步 | 950/A3/A2 |
| `4`（`INTRA_MODE`） | AscendC Matmul 高阶 API 内部使用 | 仅 950，且要求 `KERNEL_TYPE_MIX_AIC_1_2` |

> `CROSS_CORE_INNER_CUBE_VEC_SYNC`（=0x2）常量名常见于通算融合框架代码；asc-devkit 文档以数值 0x2 表述。

---

## 3. flagId 硬件规则

| 规则 | 说明 |
|:---|:---|
| 数量上限 | 模式 0/1/2 每核仅 **16 个 flagId（0-15）**，超出**截断低 4bit** |
| 截断风险 | 超出 15 的 flagId 会被截断低 4bit——**禁止**直接拿无界索引（如 tile id）当 flagId：截断后可能撞入保留区 [11,14] 与 SyncAll 冲突，且复用节奏不可控。正确做法：显式分配少量固定 flagId 做 ping-pong 轮转（见下方分配策略） |
| 计数器语义 | 每个 flagId 对应一个计数器（0-15），Wait 消耗一次计数；Set/Wait 必须配对，否则未定义行为/异常中断；逐 tile 计数式配对时**峰值（Set 未被 Wait 消耗的次数）必须 ≤ 15**，host 侧强制校验 |
| 保留区间 | **SyncAll 硬同步内部占用 flagId [11-14]**；Matmul 高阶 API 占用 flagId [0, 2N-1]（N = 高阶 API 内部使用的 flag 通道数，最多 4 个即 [0,7]；Blaze 模板 matmul 同样占用此区间） |
| 混用风险 | 官方不建议同时使用 CrossCoreSetFlag 与 SyncAll 硬同步；组合使用时，自定义 flagId 落入 [11,14]（截断后）存在冲突风险 |
| 发射顺序 | 同一核连续发出的 CrossCoreSetFlag，硬件**不保证执行顺序**——不要依赖 per-flag 的先后次序 |

**flagId 分配策略（通道式流水场景）**：可用集合 = [0,15] − Matmul 保留区 [0, 2N-1] − SyncAll 保留区 [11,14]；按最保守估计（Matmul 占满 [0,7]）自定义可用仅剩 **[8, 9, 10, 15] 共 4 个**。设计原则：

1. 从空闲区显式挑选固定 ID——**选值前必须确认同 kernel 内 matmul 实现的实际保留范围**（实际占用可能少于最保守的 [0,7]，空闲区会更大），不能仅凭"最多 [0,7]"假设选值
2. 一条生产者→消费者通道用一个固定 ID 做计数式配对（T 次 Set ⇔ T 次 Wait，峰值 ≤ 15）
3. 需要多条通道（如计算完成通知 + 回压）时各用一个固定 ID
4. 与 SyncAll 同 kernel 使用时，再次确认自定义 ID 不在 [11,14]

---

## 4. 平台生效性差异

| 平台 | modeId/pipe 模板参数 | pipe 约束 |
|:---|:---|:---|
| **A3 / 910b（DAV_2201）** | **不生效**——CrossCoreWaitFlag 阻塞全部流水 | 参数无实际作用 |
| **950（DAV_3510）** | 生效 | 模式 0/1/2 **不支持 `PIPE_S`/`PIPE_ALL`**；`PIPE_S` 仅模式 4 支持 |

**移植要点**：在 A3 上能运行的 `CrossCoreWaitFlag<0x2, PIPE_S>(id)` 写法依赖"A3 参数不生效"的硬件行为；迁到 950 时模式 0/1/2 下 `PIPE_S` 违反官方约束，需改为合法 pipe。

**pipe 选择原则（数据可见性）**：pipe 参数的本质是"flag 挂在哪条流水线上生效"，选择规则：

| 方向 | 规则 | 典型搭配 |
|:---|:---|:---|
| Set（通知方） | pipe 必须**覆盖数据产出路径**——数据经哪条 pipe 写出，Set 就挂哪条，保证 Set 生效时数据已物理落盘 | AIC fixpipe 写出计算结果 → `PIPE_FIX`；AIV MTE3 搬出/通信写出 → `PIPE_MTE3` |
| Wait（消费方） | pipe 必须**覆盖数据消费路径**——消费方第一条触碰该数据的 pipe | AIV 随后用 MTE2 搬入数据 → `PIPE_MTE2`；AIC 复用 buffer 继续算 → `PIPE_M` |

配错 pipe 的后果：Set 挂的 pipe 先于数据写出完成 → 消费者读到脏数据（偶发、难复现）；A3 上因参数不生效不会暴露，迁 950 才发作。

---

## 5. 最小通用示例

### 5.1 CrossCoreSetFlag/WaitFlag 配对

```cpp
// AIC 侧：计算完成后通知 AIV（modeId=0x2 同核 Cube↔Vec 同步）
if ASCEND_IS_AIC {
    // ... 计算写出 ...
    CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
}

// AIV 侧：等待 AIC 通知后消费数据
if ASCEND_IS_AIV {
    CrossCoreWaitFlag<0x2, PIPE_S>(flagId);   // A3 上 pipe 参数不生效
    // ... 消费数据 ...
    CrossCoreSetFlag<0x2, PIPE_MTE3>(flagId); // 回压：通知 AIC 可复用
}

// AIC 侧：复用前等待回压
if ASCEND_IS_AIC {
    CrossCoreWaitFlag<0x2, PIPE_M>(flagId);   // PIPE_M = MAD/Cube 主流水
}
```

要点：Set/Wait 的 `flagId` 必须配对（截断后相同）；AIC/AIV 两侧都 Set 也都 Wait，构成双向握手。

### 5.2 SyncAll 基础用法

```cpp
// Mix 算子：仅同步 AIV 核（如所有 AIV 都完成跨卡写入后再统一通知 AIC）
if ASCEND_IS_AIV {
    // ... 各 block 完成自己的工作 ...
    SyncAll<true>();          // 阻塞直到所有 AIV block 到达
    CrossCoreSetFlag<0x2, PIPE_MTE3>(flagId);
}
```

要点：`isAIVOnly=true` 时仅 AIV block 参与；所有参与的 block 必须都执行到 SyncAll，否则挂死。

---

## 6. 常见错误

| 错误 | 后果 | 正确做法 |
|:---|:---|:---|
| Set/Wait 的 flagId 不配对 | 未定义行为/异常中断 | 双方 flagId 截断后必须相同；Set 几次就 Wait 几次 |
| 自定义 flagId 落入 [11,14] | 与 SyncAll 内部 flag 冲突 | 组合使用 SyncAll 时避开保留区间 |
| 950 上模式 0/1/2 使用 `PIPE_S` | 违反官方约束 | 950 上改用合法 pipe（PIPE_S 仅模式 4） |
| 依赖连续 SetFlag 的执行顺序 | 偶发同步紊乱 | 硬件不保证顺序，用计数器配对语义而非次序假设 |
| 纯 Vector 算子 `SyncAll<false>()` | 卡死 | 纯 Vector 必须 `isAIVOnly=true` |
| 混用 HardEvent SetFlag 与 CrossCoreSetFlag 概念 | 同步层级错误 | HardEvent 是核内 pipe 间；CrossCore 是跨核 |

---

## 检查清单

- [ ] Set/Wait flagId 配对（截断后相同，次数相等）
- [ ] 自定义 flagId 避开 SyncAll 保留区间 [11,14] 与 Matmul 高阶 API 区间 [0, 2N-1]
- [ ] 目标平台已确认 modeId/pipe 生效性与 pipe 约束（§4）
- [ ] SyncAll：纯 Vector 用 `isAIVOnly=true`；所有参与 block 都能到达
- [ ] 未依赖连续 SetFlag 的执行顺序
- [ ] 核内 pipe 同步用 `SetFlag<HardEvent>`，跨核用 CrossCoreSetFlag，未混用

---

## 相关文档

- [api-hcomm.md](api-hcomm.md) — Hcomm 跨卡通信原语（常与 CrossCore flag 配合编排）
- [api-atomic.md](api-atomic.md) — DMA 原子操作（多核写同一 GM 地址场景）
