# Hcomm 通信原语使用指南

> **适用场景**：AIV 核发起的跨卡点对点数据搬运（URMA），包括 ReadNbi/WriteNbi/Drain/Commit 的正确用法。本文档只覆盖 asc-devkit 标准 API（`adv_api/hcomm/hcomm.h`）的签名与约束；channel 的创建与具体算子的编排模式请参考对应框架文档。

---

## 目录

1. [概述](#1-概述)
2. [API 签名与参数](#2-api-签名与参数)
3. [通信协议](#3-通信协议)
4. [最小通用示例](#4-最小通用示例)
5. [常见错误](#5-常见错误)
6. [检查清单](#检查清单)

---

## 1. 概述

`Hcomm` 是 asc-devkit 提供的 AIV 侧跨卡点对点通信模板类，基于 URMA 队列（SQ 投递 WQE、按序执行）实现：

| 场景 | API | 数据方向 |
|:---|:---|:---|
| 拉远端数据到本地 | `ReadNbi` | 远端 GM → 本地 GM |
| 推本地数据到远端 | `WriteNbi` | 本地 GM → 远端 GM |
| 推数据并远端通知 | `WriteWithNotifyNbi` | 本地 → 远端 + 写通知值 |
| 原子计数 | `AtomicFAA` | 远端原子加 |
| 原子比较交换 | `AtomicCAS` | 远端 CAS |

**关键语义**：
- 同一 channel 上的多个任务在 URMA 队列中**按序执行**，不会互相覆盖；可以连续发起多个搬运后只 Drain 一次
- `Drain` 保证该 channel 所有已投递任务完成（远端写已生效）
- `channel` 句柄（`ChannelHandle` = uint64_t）由 host 侧建链获取并经上下文传入 kernel——Hcomm 本身不负责建链

---

## 2. API 签名与参数

> 头文件：`adv_api/hcomm/hcomm.h`，模板类 `Hcomm<CommProtocol commProtocol = COMM_PROTOCOL_UBC_CTP>`

### 2.1 Init

```cpp
// 方式一：UB 指针（内部自动对齐到 32B）
__aicore__ inline int32_t Init(__ubuf__ uint8_t* buff, uint32_t len);

// 方式二：LocalTensor（起始地址必须 32B 对齐）
template<typename T>
__aicore__ inline int32_t Init(const LocalTensor<T>& buff, uint32_t len);
```

| 参数 | 类型 | 含义 |
|:---|:---|:---|
| `buff` | `__ubuf__ uint8_t*` / `LocalTensor<T>` | UB 工作区（指针重载内部自动 32B 对齐；LocalTensor 重载要求调用方保证 32B 对齐） |
| `len` | `uint32_t` | 工作区字节数，URMA 实现最小 512B（不足返回失败） |
| 返回值 | `int32_t` | 0=成功，-1=失败 |

### 2.2 ReadNbi

```cpp
template<bool commit = true,
         pipe_t commitPipe = PIPE_S,
         pipe_t reqPipe = PIPE_MTE3,
         auto const& config = URMA_DEFAULT_CFG>
__aicore__ inline int32_t ReadNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);
```

| 参数 | 类型 | 含义 |
|:---|:---|:---|
| `channel` | `ChannelHandle` (uint64_t) | 通信通道句柄（host 建链获取） |
| `dst` | `GM_ADDR` | 本地 GM 目标地址 |
| `src` | `GM_ADDR` | 远端 GM 源地址 |
| `len` | `uint64_t` | 搬运字节数 |
| `commit` | `bool` | `true`（默认）：内部自动提交；`false`：需手动调 `Commit` |
| 返回值 | `int32_t` | 0=成功，-1=失败 |

### 2.3 WriteNbi

```cpp
template<bool commit = true,
         pipe_t commitPipe = PIPE_S,
         pipe_t reqPipe = PIPE_MTE3,
         auto const& config = URMA_DEFAULT_CFG>
__aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);
```

参数同 ReadNbi，方向相反（`dst`=远端，`src`=本地）。

### 2.4 Drain（阻塞等待）

```cpp
template<pipe_t pipe = PIPE_MTE3>
__aicore__ inline int32_t Drain(ChannelHandle channel);
```

阻塞直到该 channel 所有已投递任务完成。同 channel 多个串行任务可只 Drain 最后一次。

### 2.5 Commit（手动提交）

```cpp
template<pipe_t pipe = PIPE_S>
__aicore__ inline int32_t Commit(ChannelHandle channel);
```

> 仅当 ReadNbi/WriteNbi 的 `commit=false` 时需要手动调用。默认 `commit=true` 时无需调用。

### 2.6 WriteWithNotifyNbi

```cpp
template<bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
         auto const& config = URMA_DEFAULT_CFG>
__aicore__ inline int32_t WriteWithNotifyNbi(
    ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len,
    GM_ADDR notifyAddr, uint64_t notifyVal);
```

| 参数 | 含义 |
|:---|:---|
| `notifyAddr` | 远端通知地址（写入后远端可轮询此地址判断数据就绪） |
| `notifyVal` | 通知写入的值 |

### 2.7 AtomicFAA / AtomicCAS

```cpp
// Fetch-and-Add（T ∈ {int32_t, uint32_t, int64_t, uint64_t}）
template<typename T, bool commit = true, pipe_t commitPipe = PIPE_S,
         pipe_t reqPipe = PIPE_MTE3, auto const& config = URMA_DEFAULT_CFG>
__aicore__ inline int32_t AtomicFAA(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T addVal);

// Compare-and-Swap（T 同上）
template<typename T, bool commit = true, pipe_t commitPipe = PIPE_S,
         pipe_t reqPipe = PIPE_MTE3, auto const& config = URMA_DEFAULT_CFG>
__aicore__ inline int32_t AtomicCAS(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T compareVal, T swapVal);
```

> 需 channel 初始化后调用；仅支持 32/64 位整型。浮点累加场景用 `SetAtomicAdd`（见 [api-atomic.md](api-atomic.md)），两者作用层不同。

---

## 3. 通信协议

| CommProtocol | 值 | 用途 |
|:---|:---|:---|
| `COMM_PROTOCOL_UBC_CTP` | 4 | URMA 点对点数据通道（Hcomm 默认值，device 侧数据搬运使用） |
| `COMM_PROTOCOL_UB_MEM` | 6 | UBMEM 协议（host 侧建链时用于 barrier 类通道） |
| `COMM_PROTOCOL_HCCS` | 0 | HCCS 片间互连协议 |

---

## 4. 最小通用示例

```cpp
// --- 初始化（必须先 Init，workspace ≥ 512B）---
AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> comm;
__ubuf__ uint8_t* commBuf = /* UB 中分配的 512B 工作区 */;
int32_t ret = comm.Init(commBuf, 512);
ascendc_assert(ret == 0, "Hcomm Init failed, ret=%d", ret);

// --- GET：从远端拉数据到本地 ---
ChannelHandle ch = /* host 建链获取的 channel 句柄 */;
ret = comm.ReadNbi(ch, localDst, remoteSrc, byteLen);
ascendc_assert(ret == 0, "ReadNbi failed, ret=%d", ret);

// --- PUT：推本地数据到远端 ---
ret = comm.WriteNbi(ch, remoteDst, localSrc, byteLen);
ascendc_assert(ret == 0, "WriteNbi failed, ret=%d", ret);

// --- 等待完成（同 channel 的 Read+Write 按序执行，可只 Drain 一次）---
ret = comm.Drain(ch);
ascendc_assert(ret == 0, "Drain failed, ret=%d", ret);
```

要点：
- `Init` 必须先于任何搬运调用
- 返回值必须检查（0=成功，-1=失败）
- 同 channel 任务 URMA 队列按序执行，连续发起后只需 Drain 一次

---

## 5. 常见错误

### 错误1：Init 未调用或 workspace 不足

```cpp
// ❌ 错误：未 Init 直接 ReadNbi
comm.ReadNbi(channel, dst, src, len);  // 行为未定义

// ✅ 正确：先 Init，workspace ≥ 512B（URMA 实现强制检查，不足返回 -1）
comm.Init(commBuf, 512);
```

### 错误2：误以为同 channel 并发会互相覆盖

URMA 实现是向 SQ 投递 WQE 排队执行，同 channel 多任务**按序执行不会覆盖**。正确认知：同 channel 的多个串行搬运（如 data + scale）可连续 Commit 后只 Drain 一次，无需逐个 Drain。

### 错误3：返回值未检查

```cpp
// ❌ 错误：忽略返回值
comm.ReadNbi(channel, dst, src, len);

// ✅ 正确：检查返回值
int32_t ret = comm.ReadNbi(channel, dst, src, len);
ascendc_assert(ret == 0, "ReadNbi failed, ret=%d", ret);
```

### 错误4：跨块场景漏块间同步

`Drain` 只保证**本 block 的** channel 任务完成。若后续动作依赖**所有 block** 都完成搬运（如统一通知计算核数据就绪），`Drain` 后还需 `SyncAll<true>()` 做块间同步再通知（SyncAll 用法见 [api-crosscore-sync.md](api-crosscore-sync.md)）。

> ⚠️ **SyncAll 不是免费的**：它是全 block 硬栅栏，会打散通信/计算流水重叠。逐 tile 热路径上"Drain + SyncAll + 通知"三连是流水性能差的典型根因。优化方向不是删同步点（数据可见性不可裁剪），而是：① 增大 tile 粒度减少轮次从而降低 SyncAll 频率；② 能改用计数式 CrossCore flag 生产者-消费者握手的时序就不用 SyncAll；③ 提高通信并行度（多核分摊 target）缩短 Drain 本身。详见 [api-crosscore-sync.md](api-crosscore-sync.md) 性能代价小节。

---

## 检查清单

- [ ] `Init` 在搬运之前调用（UB workspace ≥ 512B；LocalTensor 重载需 32B 对齐）
- [ ] ReadNbi/WriteNbi/Drain 返回值检查为 0
- [ ] 同 channel 多任务利用 URMA 按序语义，合并 Drain（而非逐个 Drain）
- [ ] 依赖"所有 block 完成"时已加 `SyncAll` 块间同步
- [ ] 浮点原子累加用 `SetAtomicAdd`（DMA 层），整型跨卡原子用 `AtomicFAA/CAS`（UDMA 层），未混用

---

## 相关文档

- [api-crosscore-sync.md](api-crosscore-sync.md) — CrossCore flag 与 SyncAll（块间/跨核同步）
- [api-atomic.md](api-atomic.md) — SetAtomicAdd（DMA 层原子累加）
- [api-hccl-host.md](api-hccl-host.md) — HCCL Host 侧 API（channel 建链）
