# 同步与数据一致性

> Vector 核同步机制。仅讨论 TBuf（手动同步），基于 TPipe-TQue 框架编程范式。

## 1 核间同步

`SyncAll()` 用于多核间的全局同步。所有启动核必须都调用，否则挂死。

**约束**：
- 调用 `SyncAll()` 的算子，host 端 Tiling 必须 `SetScheduleMode(1)`，告诉框架需要等 `usedCoreNum` 个核都空闲时才启动该算子
- `SyncAll()` 自身包含 fence 语义，无需额外跨核同步原语

**典型场景**：Reduce Group 模板 Phase 1 各核写 workspace，Phase 2 各核读 workspace。必须在 Phase 1 末尾、Phase 2 之前调用 `SyncAll()`：

```cpp
void ProcessGroup() {
    Phase1Process();        // 各核写 workspace
    SyncAll();              // 全核同步，保证 workspace 写完
    Phase2Process();        // 各核读 workspace
}
```

**host 端配合**：

```cpp
if (isGroup) {
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set ScheduleMode!"),
                return ge::GRAPH_FAILED);
}
```

## 2 核内同步

### 2.1 概览

Vector 核内部有 4 个异步并行执行的单元，共享同一片 UB（Local Memory）：

| 流水 | 含义 |
|------|------|
| PIPE_S | 标量流水线，GetValue/SetValue 等标量操作 |
| PIPE_V | 矢量计算流水线，Add/Mul/Reduce 等 |
| PIPE_MTE2 | GM→UB 数据搬运流水线 |
| PIPE_MTE3 | UB→GM 数据搬运流水线 |

4 条流水异步并行执行，当访问同一片 UB 时可能存在数据依赖，需通过同步机制协调时序。典型数据流：MTE2 搬入 → V 计算 → MTE3 搬出。

**同步分两类**：
- **多流水同步**：不同类型流水间的数据依赖。使用 SetFlag/WaitFlag 接口。
- **单流水同步**：同一类型流水内的数据依赖。使用 PipeBarrier 接口，保证前序指令所有读写完成后后续指令才开始。不同的 VF 间硬件保证串行，不需要 PipeBarrier。注意 PipeBarrier 不支持 PIPE_S。

**TBuf 需手动同步**：TBuf 不像 TQue 那样通过 EnQue/DeQue 隐式插入 fence。使用 TBuf 时，所有跨流水同步必须开发者手动插入。

### 2.2 核内同步方法

#### 2.2.1 SetFlag / WaitFlag

同一核内不同流水线之间的同步。具有数据依赖的不同流水线指令之间需要插此同步。

**同步 = 跨流水线的数据依赖。** 同一个 buffer，流水线 A 写、流水线 B 读（RAW），或流水线 A 读、流水线 B 写（WAR）——就必须同步。同流水线内自动串行（RAR/WAW 同流水线不需同步）。不同 buffer 互不干扰。

**接口**：

```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID);

template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID);
```

- `event`：模板参数，HardEvent 枚举值，命名规则 `源流水_目标流水`（如 `MTE2_V` = MTE2 是源，V 是目标——V 等 MTE2）
- `eventID`：事件 ID。通过 `GetTPipePtr()->FetchEventID(HardEvent::XXX)` 获取，不可硬编码
- **SetFlag/WaitFlag 必须成对出现**

**Vector 算子常用事件**：

| HardEvent | 含义 | 场景 |
|-----------|------|------|
| `MTE2_V` | MTE2→V，V 等 MTE2 | MTE2 写完 UB → V 要读该 UB |
| `V_MTE2` | V→MTE2，MTE2 等 V | V 正在读 UB → MTE2 要覆盖该 UB |
| `V_MTE3` | V→MTE3，MTE3 等 V | V 写完 UB → MTE3 要读该 UB |
| `MTE3_V` | MTE3→V，V 等 MTE3 | 上轮 MTE3 读 UB → 下轮 V 要写该 UB（跨迭代 WAR） |
| `MTE3_MTE2` | MTE3→MTE2，MTE2 等 MTE3 | 上轮 MTE3 读 UB → 下轮 MTE2 要覆盖该 UB |

**正向同步**（循环内）：生产者→消费者。如 MTE2→V（搬完才能算）、V→MTE3（算完才能搬出）。

**反向同步**（循环间）：下一轮迭代的生产者等上一轮迭代的消费者完成。如 V→MTE2（V 读完 buffer 后 MTE2 才能覆盖）、MTE3→MTE2（MTE3 读完 buffer 后下轮 MTE2 才能覆盖）、MTE3→V（MTE3 读完 buffer 后下轮 V 才能覆写）。首轮跳过 Wait，末轮跳过 Notify。

**Event ID 管理**：
- arch35 上 `QUE_MAX_EVENT = 8`，即每种 HardEvent 类型有独立的 8 个 eventID slot
- eventID 池按 HardEvent 类型独立分配：`FetchEventID(HardEvent::MTE2_V)` 在 `MTE2_V` 类型的池中分配，`FetchEventID(HardEvent::V_MTE3)` 在 `V_MTE3` 类型的池中独立分配
- `FetchEventID` 返回该类型池中第一个空闲 slot（0–7），同类型只用 1 个时每次返回同一个值
- 不同类型事件各自独立池，不会冲突
- 通过 `FetchEventID()` 获取，在 `Init()` 或 `Process()` 开头统一获取，避免运行时重复调用

**调用模式**：

**场景 1**：单 buffer 贯穿 CopyIn → Compute(in-place) → CopyOut，跨迭代 MTE3→MTE2 / MTE3→V WAR：

```cpp
// Init 或 Process 开头获取 eventID
int32_t evMTE2toV = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
int32_t evVtoMTE3 = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
int32_t evMte3toV = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
int32_t evMTE3toMTE2 = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

// 典型循环：每轮 CopyIn → Compute → CopyOut，inBuf 贯穿三阶段
for (int64_t flat = start; flat < end; flat++) {
    // ── 跨迭代反向：上轮 CopyOut(MTE3 读 inBuf) → 本轮 CopyIn(MTE2 覆写) / Compute(V 覆写) ──
    // 首轮跳过 WaitFlag（没有上一轮）
    if (flat != start) WaitFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    if (flat != start) WaitFlag<HardEvent::MTE3_V>(evMte3toV);    // MTE3 读完 inBuf，V 才可覆写

    // ── 正向同步：CopyIn(MTE2 写 inBuf) → Compute(V 读 inBuf) ──
    DataCopy(inBuf, gm[flat * TILE], TILE);       // MTE2 写 inBuf
    SetFlag<HardEvent::MTE2_V>(evMTE2toV);        // MTE2→V
    WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
    AscendC::Add(inBuf, inBuf, inBuf, TILE);      // V 读写 inBuf（in-place）

    // ── 正向同步：Compute(V 写 inBuf) → CopyOut(MTE3 读 inBuf) ──
    SetFlag<HardEvent::V_MTE3>(evVtoMTE3);        // V→MTE3
    WaitFlag<HardEvent::V_MTE3>(evVtoMTE3);
    DataCopy(gm[flat * TILE], inBuf, TILE);       // MTE3 读 inBuf

    // ── 跨迭代反向：本轮 CopyOut(MTE3 读 inBuf) → 下轮 CopyIn(MTE2 覆写) / Compute(V 覆写) ──
    // 末轮跳过 SetFlag（没有下一轮）
    if (flat != end - 1) SetFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    if (flat != end - 1) SetFlag<HardEvent::MTE3_V>(evMte3toV);    // MTE3 流标记读完 inBuf
}
```

**场景 2**：内层循环中 buffer 被 V 读后，下一轮迭代被 MTE2 覆写（V→MTE2 WAR）：

```cpp
// 多见于 reduction 等式的 R chunk 循环：每轮 CopyIn → CastSquare(V) → Reduce(V)，
// inBuf 被下一轮 CopyIn 覆写
int32_t evVtoMTE2 = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

for (int64_t rIdx = 0; rIdx < rEnd; ++rIdx) {
    // ── 跨迭代反向：上轮 V 读 inBuf → 本轮 CopyIn 覆写 inBuf ──
    // 首轮跳过 WaitFlag
    if (rIdx != 0) WaitFlag<HardEvent::V_MTE2>(evVtoMTE2);

    // CopyIn → V 链（MTE2_V 正向同步略）
    DataCopy(inBuf, ...);
    SetFlag<HardEvent::MTE2_V>(evMTE2toV);
    WaitFlag<HardEvent::MTE2_V>(evMTE2toV);
    Vf1(inBuf, ...);    // V 读 inBuf
    Vf2(...);            // V，不访问 inBuf -> tmpBuf

    // ── 跨迭代反向：本轮 V 读 inBuf → 下轮 CopyIn 覆写 inBuf ──
    // 末轮跳过 SetFlag
    if (rIdx != rEnd - 1) SetFlag<HardEvent::V_MTE2>(evVtoMTE2);
}
```

> **反向同步要点**：
> - `MTE3_MTE2`：SetFlag 在本轮 CopyOut 之后（MTE3 读完），WaitFlag 在下轮 CopyIn 之前（MTE2 等覆写）
> - `V_MTE2`：SetFlag 在本轮 V 链最后一步之后（V 读完 buffer），WaitFlag 在下轮 CopyIn 之前（MTE2 等覆写）
> - 首轮跳过 WaitFlag，末轮跳过 SetFlag
> - 同类型事件可共用 1 个 eventID——前一个 WaitFlag 清零后后一个 SetFlag 可复用
>
> ⚠ **同步点必须由持有法则 trace 推导得出**，禁止跳过 trace 直接编写同步代码。trace 中不存在的 RAW/WAR crossing 不得插入 SetFlag/WaitFlag。详见 §2.2.2。

#### 2.2.2 持有法则

> SetFlag/WaitFlag 的配套分析工具。通过逐行标注 buffer 持有三态，推导每个 buffer 的跨流水 RAW/WAR 依赖，精准定位同步点。

**执行动作与读写**：

| 动作 | 流水线 | 对 UB 的行为 |
|------|--------|-------------|
| CopyIn | MTE2 | **写** buffer — 从 GM 搬入新数据 |
| Compute | V | **读** src buffer + **写** dst buffer |
| CopyOut | MTE3 | **读** buffer — 搬出到 GM，对 UB 是最后一次读 |

- **写**：向 buffer 写入新数据。写入后 buffer 是否保留在持有中，取决于新数据有无未来消费者。
- **读**：从 buffer 读取数据。读本身不改变持有，但读完后该 buffer 少了一个未来消费者。如果是最后一个消费者，执行后 buffer 从持有消失。
- **原地读写**（如 MulAddDst、Muls）：buffer 被读的同时也被写，新数据覆盖旧数据。按写处理。

**持有** = buffer 的当前数据会被 Kernel 全生命周期中未来某步读，不可释放。判定依据是全局视角：扫描当前动作之后的所有步骤，如果任何一个步骤会读这个 buffer，该 buffer 就在持有列表中。无未来消费者的 buffer 执行后自然从持有列表消失。有未来消费者的保留。

**三态标注**：每个执行动作标注在代码行上方：

```
// 执行前: 持有=[...]
// 执行中: 持有=[...]
// 执行后: 持有=[...]
代码;
```

| 态 | 含义 |
|----|------|
| 执行前持有 | 上一动作执行后留下的持有列表 |
| 执行中持有 | 执行前 + 当前动作新占用的 buffer（dst 写入 + src 正在被读尚未释放）。buffer 占用峰值时刻 |
| 执行后持有 | 执行中 − 当前动作后无未来消费者的 buffer |

前一动作的执行后 = 下一动作的执行前。第一步执行前为 `[]`。

**从持有推导同步点**：遍历每个 buffer 的生命周期，标记每一次跨流水线的 RAW/WAR，即为同步点。持有法则的每行动作注释直接暴露依赖。

**RAW 示例**（生产者写→消费者读，正向）：

```
// 执行后: 持有=[UB0]                          ← MTE2 写完了 UB0
CopyOneInput(coord, IN0, UB0);
// 执行前: 持有=[UB0]
// 执行中: 持有=[UB0, UB2]                     ← V 要读 UB0 (RAW)
AscendC::Mul(buf_[UB2].Get<T>(), buf_[UB0].Get<T>(), buf_[UB0].Get<T>(), count);
```
→ 插 `MTE2_V`：SetFlag 在 Mul 之前，WaitFlag 等 MTE2 写完 UB0。

**WAR 示例**（消费者在读→生产者要覆盖，反向）：

```
// 执行后: 持有=[UB0, UB2]                     ← V 还在持有 UB0(in0)
AscendC::Mul(buf_[UB2]..., buf_[UB0]..., buf_[UB0]..., count);
//                   ↓ V 读 UB0
// 执行中: 持有=[UB0, UB2, UB3]               ← MTE2 要写 UB0(data_m) 覆盖 in0 (WAR)
CopyOneInput(coord, IN2, UB0);
```
→ 插 `V_MTE2`：SetFlag 在 Mul 之后（V 读完 UB0），WaitFlag 在 CopyIn 之前（MTE2 等 V）。

**验证**：
- 任意执行中持有数不超过该算子分配的 buffer 总数
- 执行后持有与释放时机一致

### 2.3 单流水同步 PipeBarrier

同一流水线内具有数据依赖的指令之间的同步。作用是保证前序指令中所有数据读写全部完成后，后序指令才开始执行。

**接口**：

```cpp
template <pipe_t pipe>
__aicore__ inline void PipeBarrier();
```

**模板参数 pipe**：指定阻塞的流水，支持 PIPE_V / PIPE_MTE2 / PIPE_MTE3 / PIPE_ALL，不支持 PIPE_S（Scalar 流水由硬件自动保证）。

**约束**：
- 连续 Vector 计算之间、连续 VF 之间硬件保证顺序，无需手动 PipeBarrier<PIPE_V>()
- `PipeBarrier<PIPE_ALL>()` 会等待所有流水线完成，对性能影响大，能用单流水 PipeBarrier 解决时不要用 PIPE_ALL
- **MTE2/MTE3 搬运地址重叠时需手动插入**：硬件不保证搬运地址重叠时的串行化，需手动 `PipeBarrier<PIPE_MTE2>()`（UB 目的地址重叠）或 `PipeBarrier<PIPE_MTE3>()`（GM 目的地址重叠）

**示例**：

```cpp
// MTE3 搬运目的地址重叠，需手动插入
DataCopy(dstGm[0], srcLocal, count);
PipeBarrier<PIPE_MTE3>();              // 保证前一个 DataCopy 完成
DataCopy(dstGm[overlap], srcLocal2, count);  // 目的地址与前一个重叠
```

### 2.4 VF LocalMemBar

LocalMemBar 是 VF 函数（`__simd_vf__`）内部的同步指令，用于 Reg 矢量计算中 UB 读写流水线间的保序。

**接口**：

```cpp
template <MemType src, MemType dst>
__simd_callee__ inline void LocalMemBar();
```

**MemType 取值**：

| MemType | 含义 |
|---------|------|
| VEC_STORE | VF 内矢量写 UB（StoreAlign/StoreUnAlign/Store） |
| VEC_LOAD | VF 内矢量读 UB（LoadAlign/LoadUnAlign/Load） |
| SCALAR_STORE | VF 内标量写 UB |
| SCALAR_LOAD | VF 内标量读 UB |
| VEC_ALL | VF 内所有矢量读写 UB |
| SCALAR_ALL | VF 内所有标量读写 UB |

**原理**：dst 流水线等待 src 流水线上所有指令完成后才执行。当读指令和写指令使用**不同寄存器**访问**同一 UB 地址**时，硬件不保证顺序，需插入 LocalMemBar。

**不需要 LocalMemBar 的情况**：读写指令使用**相同寄存器**时，硬件自动触发寄存器保序，指令按代码顺序执行。

**典型场景**：VF 循环内 in-place 操作（同一 UB 地址跨迭代读写）：

```cpp
__simd_vf__ inline void InPlaceAddVF(__ubuf__ T* addr, ..., uint16_t repeatTimes) {
    for (uint16_t i = 0; i < repeatTimes; i++) {
        // 第二次迭代读 addr 需等第一次迭代写 addr 完成
        LocalMemBar<VEC_STORE, VEC_LOAD>();  // VEC_LOAD 等 VEC_STORE
        LoadAlign(srcReg, addr);
        Add(dstReg, srcReg, srcReg, mask);
        StoreAlign(addr, dstReg, mask);       // in-place 写回同一地址
    }
}
```

**常见错误**：在标准 elewise 循环（每次迭代目标地址不同）末尾加 LocalMemBar 是多余的——地址不同无依赖。

### 2.5 不需要同步的情况

| 场景 | 是否需要同步 | 原因 |
|------|-------------|------|
| 同一流水线内连续 Vector/VF 计算 | 不需要 | 连续 Vector/VF 计算之间硬件保证顺序 |
| MTE2/MTE3 搬运地址重叠 | **需要 PipeBarrier** | 硬件不保证搬运地址重叠时的串行化，需手动 `PipeBarrier<PIPE_MTE2>()` 或 `PipeBarrier<PIPE_MTE3>()` |
| 不同 buffer 互不干扰 | 不需要 | 无数据依赖，各流水独立执行 |
| 无跨流水线数据依赖 | 不需要 | 如 V 写 buffer0、MTE2 写 buffer1，互不干扰 |
| MTE2 写 UB → V 读同一 UB | **需要 SetFlag/WaitFlag** | 跨流水线 RAW，插 `MTE2_V` |
| V 写 UB → MTE3 读同一 UB | **需要 SetFlag/WaitFlag** | 跨流水线 RAW，插 `V_MTE3` |
| V 读 UB → MTE2 覆写同一 UB | **需要 SetFlag/WaitFlag** | 跨流水线 WAR（反向），插 `V_MTE2` |
| MTE3 读 UB → MTE2 覆写同一 UB | **需要 SetFlag/WaitFlag** | 跨流水线 WAR（反向，跨迭代），插 `MTE3_MTE2` |
