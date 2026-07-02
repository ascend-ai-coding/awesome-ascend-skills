# Ascend C 同步指令编码指导

> **适用架构**：DAV_3510（Ascend 950）
>
> 本文档覆盖核内流水事件同步（SetFlag/WaitFlag）、核间同步（CrossCoreSetFlag/CrossCoreWaitFlag）、全流水栅栏（PipeBarrier）的编码规范与常见陷阱。

---

## 1. 同步机制总览

### 1.1 三种同步机制对比

| 机制 | 粒度 | 适用场景 | 性能开销 |
|------|------|---------|---------|
| `SetFlag/WaitFlag<HardEvent>` | 精确到两个 pipe 之间 | 双缓冲流水、buffer 生命周期管理 | 低（硬件事件寄存器） |
| `CrossCoreSetFlag/WaitFlag` | AIC↔AIV 核间 | MIX 模板中 Cube 与 Vector 的 tile 级同步 | 中（跨核信号） |
| `PipeBarrier<pipe_t>` | 排空单个 pipe 或全部 pipe | 调试、核间同步前强制排空 | 高（全流水停顿） |

### 1.2 硬件流水线架构

**AIV（Vector 核）**：
```
GM ──MTE2──▶ UB ──V──▶ UB ──MTE3──▶ GM
              │          │
         (额外输入加载)  (计算)
```

**AIC（Cube 核）**：
```
GM ──MTE2──▶ L1 ──MTE1──▶ L0A/L0B ──M──▶ L0C ──FIX──▶ UB/GM
              │             │            │        │
          (A/B 加载)    (L1→L0)      (MMAD)   (量化/Cast)
```

**关键原则**：DataCopy/DataCopyPad 是**异步 DMA**，立即返回。必须通过同步机制确保消费者在读数据前，生产者的搬运已完成。

---

## 2. HardEvent 事件全表与选取指南

### 2.1 事件枚举（按核类型分组）

**AIC 专用事件**（在 AIV 上调用会被编译器静默跳过）：

| HardEvent | 生产者 pipe | 消费者 pipe | 用途 |
|-----------|------------|------------|------|
| `MTE1_MTE2` | MTE1 | MTE2 | MTE1 用完 L1 buffer，MTE2 可覆盖 |
| `MTE2_MTE1` | MTE2 | MTE1 | MTE2 加载 L1 完成，MTE1 可读取 |
| `MTE1_M` | MTE1 | M | MTE1 加载 L0 完成，Cube 可计算 |
| `M_MTE1` | M | MTE1 | Cube 用完 L0，MTE1 可重新加载 |
| `M_FIX` | M | FIX | Cube 累加完成，FixPipe 可读取 L0C |
| `FIX_M` | FIX | M | FixPipe 读完 L0C，Cube 可重新累加 |
| `MTE2_M` | MTE2 | M | MTE2 写 UB 完成，Cube 可读取 |
| `M_MTE2` | M | MTE2 | Cube 用完 UB，MTE2 可覆盖 |

**AIV 专用事件**（在 AIC 上调用会被编译器静默跳过）：

| HardEvent | 生产者 pipe | 消费者 pipe | 用途 |
|-----------|------------|------------|------|
| `MTE2_V` | MTE2 | V | MTE2 加载 UB 完成，Vector 可计算 |
| `V_MTE2` | V | MTE2 | Vector 用完 UB，MTE2 可覆盖 |
| `V_MTE3` | V | MTE3 | Vector 计算完成，MTE3 可写回 GM |
| `MTE3_V` | MTE3 | V | MTE3 写完，Vector 可读取 |
| `MTE3_MTE2` | MTE3 | MTE2 | MTE3 写回完成，MTE2 可覆盖 UB |
| `MTE2_MTE3` | MTE2 | MTE3 | MTE2 加载完成，MTE3 可读取 |
| `V_V` | V | V | 等同 `PipeBarrier<PIPE_V>` |

**通用事件**：

| HardEvent | 说明 |
|-----------|------|
| `V_V` | 内部实现为 `PipeBarrier<PIPE_V>`，排空 Vector pipe |
| `FIX_FIX` | 内部实现为 `PipeBarrier<PIPE_FIX>`，排空 FixPipe |
| `S_V` / `V_S` | Scalar ↔ Vector 同步 |
| `S_MTE2` / `MTE2_S` | Scalar ↔ MTE2 同步 |
| `S_MTE3` / `MTE3_S` | Scalar ↔ MTE3 同步 |

### 2.2 事件选取决策表

给定"生产者操作完成后，消费者才能开始"的需求，查表选取事件：

| 生产者 ↓ \ 消费者 → | MTE1 | MTE2 | M (Cube) | V (Vector) | MTE3 | FIX |
|---------------------|------|------|----------|-----------|------|-----|
| **MTE1** | — | `MTE1_MTE2` | `MTE1_M` | — | `MTE1_MTE3` | `MTE1_FIX` |
| **MTE2** | `MTE2_MTE1` | — | `MTE2_M` | `MTE2_V` | `MTE2_MTE3` | `MTE2_FIX` |
| **M (Cube)** | `M_MTE1` | `M_MTE2` | — | `M_V` | — | `M_FIX` |
| **V (Vector)** | `V_MTE1` | `V_MTE2` | — | `V_V` | `V_MTE3` | — |
| **MTE3** | `MTE3_MTE1` | `MTE3_MTE2` | — | `MTE3_V` | — | `FIX_MTE3` |
| **FIX** | `FIX_MTE1` | `FIX_MTE2` | `FIX_M` | — | `FIX_MTE3` | `FIX_FIX` |

> **注**：表中"—"表示该组合在常规流水中不出现。并非所有组合都有实际硬件支持。

---

## 3. SetFlag/WaitFlag 配对规则

### 3.1 核心原则

```
生产者操作（如 DataCopyPad）
    ↓
SetFlag<Event>(eventID)     ← 标记"我完成了"
    ↓
    ... （硬件异步推进）...
    ↓
WaitFlag<Event>(eventID)    ← 阻塞等待"生产者完成"
    ↓
消费者操作（如 Vector 计算）
```

**规则**：
1. **Set 在生产者之后，Wait 在消费者之前**
2. **同一 eventID 的完整生命周期**：`Set → Wait` 为一个周期。Wait 返回后该 eventID 才可复用
3. **Set 和 Wait 必须使用相同的 HardEvent 类型和 eventID**

### 3.2 构造预发 / 析构排空（Init-Drain 模式）

双缓冲流水或单缓冲但多轮生产/消费中，第一轮迭代的 Wait 没有"上一轮"可等。解决方案：**构造函数预发所有 slot 的 SetFlag，析构函数排空所有 slot 的 WaitFlag**。

```cpp
class BlockMmad {
    BlockMmad() {
        // 预发：标记所有 L1 slot 为"空闲可写"
        SetFlag<HardEvent::MTE1_MTE2>(0);  // slot 0
        SetFlag<HardEvent::MTE1_MTE2>(1);  // slot 1
        // 预发：标记所有 L0 slot 为"空闲可写"
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
    }

    ~BlockMmad() {
        // 排空：等待所有 in-flight DMA 完成
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
    }
};
```

### 3.3 双缓冲循环中的配对

```cpp
for (uint64_t iter = 0; iter < totalIter; ++iter) {
    uint64_t bufId = iter & 0x1;  // 交替 0, 1

    // ── 生产者侧（MTE2 加载 L1）──
    WaitFlag<HardEvent::MTE1_MTE2>(bufId);   // 等 MTE1 用完此 slot
    CopyGM2L1(l1Buffer[bufId], ...);          // MTE2 写入 L1
    SetFlag<HardEvent::MTE2_MTE1>(bufId);    // 通知 MTE1 可读取

    // ── 消费者侧（MTE1 读取 L1 → L0）──
    WaitFlag<HardEvent::MTE2_MTE1>(bufId);   // 等 MTE2 加载完成
    CopyL12L0(l0Buffer[...], l1Buffer[bufId]);
    SetFlag<HardEvent::MTE1_MTE2>(bufId);    // 通知 MTE2 可覆盖
}
```

---

## 4. 正向依赖与反向依赖

### 4.1 概念

| 类型 | 方向 | 含义 | 示例 |
|------|------|------|------|
| **正向依赖** | 数据流方向 | "数据准备好了，你可以用" | `MTE2_V`：MTE2 搬完 UB，V 可计算 |
| **反向依赖** | buffer 回收方向 | "我用完了，你可以覆盖" | `V_MTE3`：V 算完，MTE3 可写回 |
| | | | `MTE3_MTE2`：MTE3 写回完，MTE2 可覆盖 UB |
| | | | `MTE1_MTE2`：MTE1 读完 L1，MTE2 可覆盖 |
| | | | `FIX_M`：FixPipe 读完 L0C，Cube 可重新累加 |

### 4.2 多轮计算中反向依赖的必要性

单轮计算中，buffer 只使用一次，不存在覆盖问题。**多轮计算**（每核处理多个 tile）中，buffer 被循环复用，必须通过反向依赖保护：

```
轮次 N:   MTE2 写 UB[0] → V 读 UB[0] → MTE3 写回 GM
轮次 N+1: MTE2 写 UB[0] → V 读 UB[0] → MTE3 写回 GM
                ↑
    如果轮次 N 的 MTE3 尚未完成，MTE2 覆盖 UB[0] → 数据损坏！
    需要 MTE3_MTE2 反向 barrier 保护。
```

### 4.3 遗漏反向依赖的症状

| 场景 | 现象 | 原因 |
|------|------|------|
| 小 shape（单轮） | PASS | buffer 不复用，无冲突 |
| 大 shape（多轮） | crash 或数据错乱 | 上一轮 V 未计算完，本轮 MTE2 覆盖同一 buffer |
| 确定性测试 | 两次运行结果不同 | 时序竞争：V 完成时间随调度波动 |

### 4.4 epilogue 循环中的正确 barrier 位置

```cpp
for (int64_t mOff = 0; mOff < halfM; mOff += stageRows) {
    // ── 反向 barrier：等上一轮 V 计算完成,
    // 注意：首轮SetFlag需要在Init中预发射，此处伪代码虽不作展示，但不能遗漏！
    WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);
    
    // ── 加载 (M,1)/(M,N) 额外输入（每 stage 不同行）──
    DataCopyPad(pertokenBuf, pertokenGM[start + mOff], ...);

    // ── 正向 barrier：等 MTE2 加载完成 ──
    SetFlag<HardEvent::MTE2_V>(ZERO_FLAG);
    WaitFlag<HardEvent::MTE2_V>(ZERO_FLAG);

    // ── 反向 barrier：V等上一轮 MTE3 完成后才能开始计算,
    // 注意：首轮SetFlag需要在Init中预发射，此处伪代码虽不作展示，但不能遗漏！
    WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);

    // ── V 计算 ──
    __VEC_SCOPE__ { ... }

    // ── 反向 barrier：通知下一轮 MTE2 可以开始 ──
    // 注意：尾轮WaitFlag需要在析构函数中完成，此处伪代码虽不作展示，但不能遗漏！
    SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);

    // ── 正向 barrier：等 V 计算完成 ──
    SetFlag<HardEvent::V_MTE3>(ZERO_FLAG);
    WaitFlag<HardEvent::V_MTE3>(ZERO_FLAG);

    // ── MTE3 写回 GM ──
    DataCopyPad(outputGM[offset], bf16Out, ...);

    // ── 反向 barrier：通知下一轮 V 可以开始 ──
    // 注意：尾轮WaitFlag需要在析构函数中完成，此处伪代码虽不作展示，但不能遗漏！
    SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);
}
```

> **关键**：`V_MTE2` barrier 放在循环内、MTE2 操作之前WaitFlag, V 完成后SetFlag，注意首轮尾轮和中间轮次的区别。

---

## 5. EventID 管理

### 5.1 取值范围

| 架构 | eventID 数量 | 可用值 |
|------|-------------|--------|
| DAV_3510 / V200+ | 8 | 0~7（`EVENT_ID0` ~ `EVENT_ID7`） |
| 旧架构 | 4 | 0~3 |

```cpp
// 编译器内置枚举
typedef enum {
    EVENT_ID0 = 0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
    EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7,  // V200+
} event_t;
```

### 5.2 双缓冲 eventID 分配

**`& 0x1` 模式**（最常用）：

```cpp
uint64_t pingPong = 0;
for (...) {
    uint64_t bufId = pingPong & 0x1;  // 交替 0, 1
    WaitFlag<Event>(bufId);
    // ... 操作 buffer[bufId] ...
    SetFlag<Event>(bufId);
    pingPong++;
}
```

**`& (N-1)` 模式**（N 缓冲 ring，N 必须为 2 的幂）：

```cpp
uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);  // l1BufNum=4 → & 3
```

**XOR toggle**（等价于 `& 0x1`）：

```cpp
aL1BufferID_ = aL1BufferID_ ^ 1;  // 0→1, 1→0
```

### 5.3 不同事件类型共享 eventID 空间

**关键陷阱**：同一核上，不同 HardEvent 类型共享同一组硬件 eventID 寄存器。`MTE1_MTE2` 的 ID0 和 `M_MTE1` 的 ID0 是**同一个物理寄存器**。

**解决方案**：用偏移量隔离不同事件类型的 eventID。

```cpp
// L1 双缓冲：MTE1_MTE2 使用 ID 0,1
SetFlag<HardEvent::MTE1_MTE2>(0);
SetFlag<HardEvent::MTE1_MTE2>(1);

// L0 双缓冲：M_MTE1 使用 ID 6,7（偏移 6 避免与 L1 冲突）
constexpr uint16_t L0_FLAG_OFFSET = 6;
SetFlag<HardEvent::M_MTE1>(0 + L0_FLAG_OFFSET);
SetFlag<HardEvent::M_MTE1>(1 + L0_FLAG_OFFSET);
```

### 5.4 L1_EVENT_ID_OFFSET：A/B 矩阵独立 eventID

A 矩阵和 B 矩阵的 L1 加载可以独立进行。使用偏移量分离 eventID 避免假依赖：

```cpp
constexpr uint16_t L1_EVENT_ID_OFFSET = 2;

// A 矩阵使用 ID 0,1
WaitFlag<HardEvent::MTE1_MTE2>(abL1EventID_ & 0x1);
// B 矩阵使用 ID 2,3
WaitFlag<HardEvent::MTE1_MTE2>((abL1EventID_ & 0x1) + L1_EVENT_ID_OFFSET);
```

### 5.5 TPipe 动态分配

当使用 TPipe 管理事件时，框架自动分配/释放：

```cpp
auto eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();  // 分配
// ... 使用 eventId ...
GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventId);        // 释放
```

内部实现使用位图（`sff0` 找首个空闲位，`sbitset1` 标记占用，`sbitset0` 释放）。

---

## 6. PipeBarrier 使用指南

### 6.1 语义

`PipeBarrier<pipe_t>()` 排空指定 pipe 的所有 in-flight 操作，阻塞直到该 pipe 完全空闲。

### 6.2 pipe_t 枚举

| pipe_t | 含义 | 说明 |
|--------|------|------|
| `PIPE_S` | Scalar pipe | 标量计算 |
| `PIPE_V` | Vector pipe | 向量计算 |
| `PIPE_M` | Matrix pipe | Cube 矩阵乘 |
| `PIPE_MTE1` | L1→L0 搬运 | |
| `PIPE_MTE2` | GM→L1/UB 搬运 | |
| `PIPE_MTE3` | UB→GM 搬运 | |
| `PIPE_FIX` | FixPipe（L0C→UB/GM） | dav-3510+ |
| `PIPE_ALL` | 所有 pipe | 全流水停顿 |

### 6.3 架构差异

| 架构 | PIPE_MTE3 on AIC | PIPE_S | PIPE_V on AIC |
|------|-----------------|--------|---------------|
| DAV_3510 | no-op | 不支持 | no-op |
| DAV_5102 | no-op | 支持 | no-op |

### 6.4 使用场景

| 场景 | 推荐 | 原因 |
|------|------|------|
| 调试同步问题 | `PipeBarrier<PIPE_ALL>()` | 快速验证是否为同步问题 |
| 高性能流水 | **不用** PipeBarrier | 用 SetFlag/WaitFlag 精确同步 |

### 6.5 与 SetFlag/WaitFlag 的选择

```cpp
// ❌ 性能差：PipeBarrier 全流水停顿
DataCopyPad(x, gm, size);
PipeBarrier<PIPE_ALL>();
Compute(x);

// ✅ 性能好：SetFlag/WaitFlag 精确同步
DataCopyPad(x, gm, size);
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
Compute(x);
```

---

## 7. 核间同步（CrossCoreSetFlag / CrossCoreWaitFlag）

### 7.1 API 签名

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

### 7.2 MODE 4 语义

`modeId = 4` 为 intra-block 模式，用于同一 block 内 AIC↔AIV 配对核间同步。底层调用 `set_intra_block(pipe, flagId)` / `wait_intra_block(pipe, flagId)`。

### 7.3 PIPE 参数选择

| 操作 | 推荐 PIPE | 原因 |
|------|----------|------|
| AIC SetFlag（通知 AIV 数据就绪） | `PIPE_FIX` | FixPipe 写完 UB 后才可通知 |
| AIV WaitFlag（等待 AIC 数据） | `PIPE_V` | Vector pipe 等待，阻塞后续 V 计算 |
| AIV SetFlag（通知 AIC 消费完成） | `PIPE_MTE3` | MTE3 写回 GM 后才可通知 |
| AIC WaitFlag（等待 AIV 消费完） | `PIPE_FIX` | FixPipe pipe 等待，阻塞下一 tile 的 L0C→UB |

### 7.4 FLAG_ID_MAX 与双 AIV

1:2 CV 比例下，1 个 AIC 配 2 个 AIV。使用 `FLAG_ID_MAX = 16` 偏移分离两个 AIV 的 flag 空间：

```cpp
// AIC 通知 AIV0
CrossCoreSetFlag<MODE4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + countId);
// AIC 通知 AIV1
CrossCoreSetFlag<MODE4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + countId + FLAG_ID_MAX);

// AIC 等待 AIV0
CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId);
// AIC 等待 AIV1
CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
```

### 7.5 count 轮换机制

硬件 flag 寄存器有限（16 个），多 tile 场景需要轮换：

```cpp
constexpr int16_t COUNT_ID_MAX = 15;  // 每个 flag slot 服务 15 个 tile
constexpr int16_t COUNT_FLAG = 3;     // 3 个 flag slot 循环

countId = count / COUNT_ID_MAX % COUNT_FLAG;
// 产生序列：0,0,...,0(15次), 1,1,...,1(15次), 2,2,...,2(15次), 0,0,...
```

flag 地址分布：

| 方向 | AIV0 | AIV1 |
|------|------|------|
| AIV→AIC | 5, 6, 7（轮换） | 21, 22, 23（轮换） |
| AIC→AIV | 8, 9, 10（轮换） | 24, 25, 26（轮换） |

### 7.6 构造/析构预发排空模式

```cpp
class KernelMatmulMixFixpipeOpti {
    KernelMatmulMixFixpipeOpti() {
        if ASCEND_IS_AIV {
            // 预发：让 AIC 第一轮 Wait 直接通过
            CrossCoreSetFlag<MODE4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);      // ping
            CrossCoreSetFlag<MODE4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + 1);  // pong
        }
    }

    ~KernelMatmulMixFixpipeOpti() {
        if ASCEND_IS_AIC {
            // 排空：等所有 AIV 完成最后一轮
            CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
            CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
            CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1);
            CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + 1 + FLAG_ID_MAX);
        }
    }
};
```

### 7.7 空闲核处理

空闲核必须在 return 前发送 flag，否则对端永久等待：

```cpp
if ASCEND_IS_AIC {
    if (curBlockIdx >= realBlockNum) {
        // 空闲 AIC 必须通知 AIV
        CrossCoreSetFlag<MODE4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        CrossCoreSetFlag<MODE4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
        return;
    }
}
```

---

## 8. 常见挂死问题与排查

### 8.1 挂死分类表

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| D1 | AIC hang（循环结束后） | 未 drain 最后一轮 AIV→AIC WaitFlag | 循环后/析构中补 WaitFlag |
| D2 | AIV hang | 空闲 AIC 未发 flag 就 return | return 前补 CrossCoreSetFlag |
| D3 | 大 shape crash，小 shape PASS | 缺少 MTE3_MTE2 反向 barrier | 循环内 MTE2 操作前加 MTE3_MTE2 |
| D4 | 全 shape hang | SetFlag/WaitFlag eventID 不匹配 | 检查两侧 flagId 计算一致 |
| D5 | 大 shape 数据错乱（非确定性） | 反向同步设置错误 | 检查 V_MTE2 ， MTE3_V 等反向同步是否设置正确 |
| D6 | L1 双缓冲 hang | 构造时未预发 MTE1_MTE2 | 构造函数 SetFlag 所有 slot |
| D7 | L0C ping-pong deadlock | l0cDB==1 和 l0cDB==2 同步模式混用 | 统一 L0C 同步模式 |
| D8 | EnQue/DeQue hang | 队列空等或满等 | 检查 Alloc/Free/EnQue/DeQue 配对 |

### 8.2 排查流程

```
程序卡死/超时？
│
├─ [1] 检查 plog 定位卡死位置
│   └─ grep "timeout\|hang\|deadlock" plog
│
├─ [2] 判断确定性
│   └─ 同输入跑两次比对
│       ├─ diff ≠ 0 → 竞争（查 D5：PipeBarrier / CV 同步）
│       └─ diff = 0 → 确定性错误（查 D3/D4/D6/D7）
│
├─ [3] 隔离触发维度
│   └─ 单独放大 M/N/K
│       ├─ 仅大 shape 触发 → D3（反向 barrier）或 D5（WAR 竞争）
│       └─ 全 shape 触发 → D4（eventID 不匹配）或 D6（预发缺失）
│
└─ [4] 检查 CV 同步完整性
    └─ Set/Wait 数量是否匹配
        ├─ AIC Set 数 = AIV Wait 数
        ├─ AIV Set 数 = AIC Wait 数
        └─ 空闲核是否发送了 flag
```

---

## 9. 速查表


### 9.1 Matmul 双缓冲同步模板

```cpp
// ── L1 双缓冲（GM→L1→L0）──
// 构造：预发
SetFlag<MTE1_MTE2>(0); SetFlag<MTE1_MTE2>(1);
// 循环：
WaitFlag<MTE1_MTE2>(bufId);   // 等 MTE1 释放 L1 slot
CopyGM2L1(...);                // MTE2 写 L1
SetFlag<MTE2_MTE1>(bufId);    // 通知 MTE1 可读取
WaitFlag<MTE2_MTE1>(bufId);   // 等 MTE2 写完
CopyL12L0(...);                // MTE1 读 L1→L0
SetFlag<MTE1_MTE2>(bufId);    // 通知 MTE2 可覆盖

// ── L0 双缓冲（L1→L0→MMAD）──
// 构造：预发（注意 eventID 偏移避免与 L1 冲突）
SetFlag<M_MTE1>(OFFSET + 0); SetFlag<M_MTE1>(OFFSET + 1);
// 循环：
WaitFlag<M_MTE1>(OFFSET + bufId);  // 等 Cube 用完 L0
CopyL12L0(...);                     // MTE1 写 L0
SetFlag<MTE1_M>(OFFSET + bufId);   // 通知 Cube 可计算
WaitFlag<MTE1_M>(OFFSET + bufId);  // 等 MTE1 写完
Mmad(...);                          // Cube 计算
SetFlag<M_MTE1>(OFFSET + bufId);   // 通知 MTE1 可覆盖

// ── 析构：排空所有 slot ──
WaitFlag<MTE1_MTE2>(0); WaitFlag<MTE1_MTE2>(1);
WaitFlag<M_MTE1>(OFFSET + 0); WaitFlag<M_MTE1>(OFFSET + 1);
```

### 9.2 AIV Epilogue 同步模板

```cpp
for (int64_t mOff = 0; mOff < halfM; mOff += stageRows) {
    // ── 反向 barrier：等上一轮 V 计算完成,
    // 注意：首轮SetFlag需要在Init中预发射，此处伪代码虽不作展示，但不能遗漏！
    WaitFlag<HardEvent::V_MTE2>(ZERO_FLAG);
    
    // ── 加载 (M,1)/(M,N) 额外输入（每 stage 不同行）──
    DataCopyPad(pertokenBuf, pertokenGM[start + mOff], ...);

    // ── 正向 barrier：等 MTE2 加载完成 ──
    SetFlag<HardEvent::MTE2_V>(ZERO_FLAG);
    WaitFlag<HardEvent::MTE2_V>(ZERO_FLAG);

    // ── 反向 barrier：V等上一轮 MTE3 完成后才能开始计算,
    // 注意：首轮SetFlag需要在Init中预发射，此处伪代码虽不作展示，但不能遗漏！
    WaitFlag<HardEvent::MTE3_V>(ZERO_FLAG);

    // ── V 计算 ──
    __VEC_SCOPE__ { ... }

    // ── 反向 barrier：通知下一轮 MTE2 可以开始 ──
    // 注意：尾轮WaitFlag需要在析构函数中完成，此处伪代码虽不作展示，但不能遗漏！
    SetFlag<HardEvent::V_MTE2>(ZERO_FLAG);

    // ── 正向 barrier：等 V 计算完成 ──
    SetFlag<HardEvent::V_MTE3>(ZERO_FLAG);
    WaitFlag<HardEvent::V_MTE3>(ZERO_FLAG);

    // ── MTE3 写回 GM ──
    DataCopyPad(outputGM[offset], bf16Out, ...);

    // ── 反向 barrier：通知下一轮 V 可以开始 ──
    // 注意：尾轮WaitFlag需要在析构函数中完成，此处伪代码虽不作展示，但不能遗漏！
    SetFlag<HardEvent::MTE3_V>(ZERO_FLAG);
}
```

### 9.3 CV 核间同步模板

```cpp
// ── AIC 侧 ──
if (enableCVSync) {
    countId = count / COUNT_ID_MAX % COUNT_FLAG;
    CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_FLAG + countId);           // 等 AIV0
    CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_FLAG + countId + 16);      // 等 AIV1
}
blockMmadOp(...);  // 含 CopyL0C2UB
count++;
countId = count / COUNT_ID_MAX % COUNT_FLAG;
CrossCoreSetFlag<MODE4, PIPE_FIX>(AIC_FLAG + countId);                // 通知 AIV0
CrossCoreSetFlag<MODE4, PIPE_FIX>(AIC_FLAG + countId + 16);           // 通知 AIV1

// ── AIV 侧 ──
count++;
countId = count / COUNT_ID_MAX % COUNT_FLAG;
CrossCoreWaitFlag<MODE4, PIPE_V>(AIC_FLAG + countId);                // 等 AIC
epilogueOp(...);
CrossCoreSetFlag<MODE4, PIPE_MTE3>(AIV_FLAG + countId);              // 通知 AIC

// ── AIC drain（循环后）──
if (enableCVSync) {
    countId = count / COUNT_ID_MAX % COUNT_FLAG;
    CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_FLAG + countId);
    CrossCoreWaitFlag<MODE4, PIPE_FIX>(AIV_FLAG + countId + 16);
}
```

### 9.4 核内生产-消费流水同步的基本原则：
任何流水，如果产出的数据被其他流水使用，**必须**同时设置正向同步和反向流水，正向流水设置在于"生产流水"和"反向流水"之间；反向同步的WaitFlag在生产流水之前，SetFlag在消费流水之后。反向同步设置时，首轮SetFlag在Init中预发射，尾轮WaitFlag在析构函数中完成。伪代码如下：

```cpp
class SomeClass {
    __aicore__ void Init () {
        // ... 其他Init函数逻辑

        SetFlag<HardEvent::消费流水_生产流水>(id); // 预发射首轮SetFlag
    }

    __aicore__ void operator() ()
    {
        // ... 其他逻辑代码

        // 反向等待消费流水完成,必须设置, 否则会导致多tile轮次计算间有读写竞争
        WaitFlag<HardEvent::消费流水_生产流水>(id);

        {
            // ... 生产流水代码
        }

        // 正向同步
        SetFlag<HardEvent::生产流水_消费流水>(id);
        WaitFlag<HardEvent::生产流水_消费流水>(id);

        {
            // ... 消费流水代码
        }
        
        // 反向通知下一轮生产流水可以开始,必须设置，不设置会导致多tile轮次计算间有读写竞争
        SetFlag<HardEvent::消费流水_生产流水>(id);

        // ... 其他逻辑代码
    }

    __aicore__ ~SomeClass()
    {
        // ... 其他析构函数逻辑代码

        // 析构函数中设置最后一轮反向等待的WaitFlag
        WaitFlag<HardEvent::消费流水_生产流水>(id);
    }
};
```

**重要：**
1. 同步代码和流水代码的位置必须绑定，**不允许**同步代码在循环外，但流水代码在循环内。当修改流水代码位置时，必须将对应的同步代码位置一起修改.
2. SetFlag和WaitFlag的数量必须匹配，先Set后Wait。尤其是涉及循环的情况，需要仔细验证SetFlag和WaitFlag的数量是否匹配。