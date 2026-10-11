---
name: ascendc-sync-guide
description: |
  AscendC 同步机制指南
  Use when: 不理解AscendC同步机制；遇到 AscendC 相关的同步问题；算子生成或者定位过程中需要在不同管线、不同核之间协调数据依赖时。算子生成过程中涉及同步，需了解 PipeBarrier / SetFlag / CrossCore / TQue 的用法和边界。
---

# AscendC 同步机制指南

AscendC 有四种同步机制，分别解决不同层面的数据依赖问题。本文档按使用场景介绍每种机制的语义、正确用法和误用模式。

---

## 1. 同步机制总览

| 机制 | 层级 | 解决的问题 |
|:---|:---|:---|
| `PipeBarrier` | 同一管线内 | 同一条硬件管线中前后指令的顺序保障 |
| `SetFlag/WaitFlag` | 同一核内跨管线 | 不同硬件管线之间的数据依赖（如 MTE2 搬完 → Vector 用） |
| `CrossCoreSetFlag/WaitFlag` | 核间 | 不同 AI Core（AIC↔AIV）之间通过 GM 的数据交换同步 |
| `TQue EnQue/DeQue` | 队列内 | 流水线各阶段之间的数据传递，内部自动插入管线同步 |

### 四种机制的关系与职责边界

```cpp
// ③ CrossCore：等 AIC 数据准备完毕
CrossCoreWaitFlag<0x2, PIPE_MTE2>(CUBE_NOTIFY_VECTOR_ID);

// ④ TQue：AllocTensor → DataCopy → EnQue（内部自动 MTE2 同步）
inQueue_.AllocTensor<float>(inLocal_);
DataCopy(inLocal_, slot, tileSize);
inQueue_.EnQue(inLocal_);

// ④ TQue：DeQue → 计算 -> FreeTensor
inQueue_.DeQue<float>(inLocal_);

// ① PipeBarrier：同一 V 管线内，先 Cast 再 Mul
Cast(fp32Local, inLocal_, RoundMode::CAST_NONE, tileSize);
PipeBarrier<PIPE_V>();
Mul(fp32Local, fp32Local, scaleLocal, tileSize);

// ② SetFlag/WaitFlag：Vector 完成后 MTE3 才能搬出
int32_t ev = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
SetFlag<HardEvent::V_MTE3>(ev);
WaitFlag<HardEvent::V_MTE3>(ev);
DataCopy(dstGm, fp32Local, tileSize);
inQueue_.FreeTensor(fp32Local);

// ③ CrossCore：通知 AIC 数据处理完毕 
CrossCoreSetFlag<0x2, PIPE_MTE2>(VECTOR_NOTIFY_CUBE_ID);
```

| 机制 | 职责 | 不要用它做 |
|:---|:---|:---|
| `PipeBarrier` | 核内同一管线内前后顺序 | 跨管线同步、核间同步 |
| `SetFlag/WaitFlag` | 同核内跨管线同步 | 核间同步、管线内顺序 |
| `CrossCoreSetFlag/WaitFlag` | AIC↔AIV 核间同步 | 同核内管线同步 |
| `TQue EnQue/DeQue` | 流水线阶段数据传递 | 临时计算缓冲（应用 TBuf） |

---

## 2. PipeBarrier：同一管线内顺序屏障

### 语义
`PipeBarrier<pipe>()` 阻塞**同一条硬件管线**内的后续指令，直到该管线之前的所有指令完成。

`<pipe>` 参数指定管线类型：

| 参数 | 管线 | 说明 |
|:---|:---|:---|
| `PIPE_MTE2` | 数据搬入管线 | GM→UB、GM→L1 等 |
| `PIPE_MTE3` | 数据搬出管线 | UB→GM 等 |
| `PIPE_V` | 向量计算管线 | 向量运算 |
| `PIPE_S` | 标量管线 | 由硬件保证顺序，**禁止显式使用** |
| `PIPE_ALL` | 所有管线 | 保守兜底，性能差，只在无法精确判断时使用 |

### 正确用法

```cpp
// 同一 V 管线内：先 Abs 再 Mul，第二步依赖第一步的结果
Abs(tmpLocal, srcLocal, count);
PipeBarrier<PIPE_V>();       // 等 Abs 写完成
Mul(dstLocal, tmpLocal, scale, count);

// 同一 MTE2 管线内：分两次 DataCopy 到同一 UB 冲突的地址
DataCopy(ubLocal, gmSrc1, count1);
// some compute using ubLocal
PipeBarrier<PIPE_MTE2>();    // 必须等第一次搬完
DataCopy(ubLocal + count1, gmSrc2, count2);
```

### 误用模式

**❌ 误用 1：PipeBarrier 不能替代跨管线同步**
```cpp
DataCopy(inLocal, srcGm, count);   // MTE2 管线
PipeBarrier<PIPE_MTE2>();          // 仅保证 inLocal 对后续 MTE2 管线搬运肯定完成
VectorCompute(outLocal, inLocal);  // V 管线可能读到未就绪数据
```
**正确理解：✅ 跨管线应使用 SetFlag/WaitFlag<HardEvent::MTE2_V>**

**❌ 误用 2：PIPE_S 不可用于 PipeBarrier**
```cpp
PipeBarrier<PIPE_S>();  // 标量管线硬件保证顺序，显式 barrier 会引发硬件错误
```

---

## 3. SetFlag/WaitFlag：同一核内跨管线同步

### 语义
`SetFlag/WaitFlag` 用于同一 AI Core 内**不同硬件管线**之间的数据依赖。`SetFlag` 在源管线设置完成标志，`WaitFlag` 阻塞目标管线直到标志被设置。

HardEvent 命名规则为 `源管线_目标管线`：

| HardEvent | 语义 |
|:---|:---|
| `MTE2_V` | MTE2 搬入UB完成后，Vector 才能读 UB |
| `MTE2_MTE1` | MTE2 搬入L1完成后，MTE1才能搬运至L0 |
| `V_MTE3` | Vector 计算完成后，MTE3 才能搬出 |
| `MTE3_MTE2` | MTE3 写出完成后，MTE2 才能读取相关 GM 地址 |

### 正确用法

```cpp
// MTE3→MTE2：同一 GM 地址先 MTE3 写、后 MTE2 读
// 场景：Vec2 在第 N 轮写 wsAccOGm_，第 N+1 轮从同一地址读
if (!isFirst) {
    int32_t ev = static_cast<int32_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(ev);
    WaitFlag<HardEvent::MTE3_MTE2>(ev);
    DataCopy(oPrevUb, wsAccOGm_[accOffset], chunkSize); // MTE2 管线
}

// V→MTE3：Vector 计算完成后才能搬出
int32_t ev = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
VectorCompute(outLocal, inLocal); // V 管线
SetFlag<HardEvent::V_MTE3>(ev);
WaitFlag<HardEvent::V_MTE3>(ev);
DataCopy(dstGm, outLocal, count);  // MTE3 管线
```

### 关键规则
- `SetFlag` 和 `WaitFlag` 必须使用**相同 HardEvent、相同 eventID**
- eventID **必须通过 `FetchEventID` 获取**，禁止手写 0/1/2

### 误用模式

**❌ 误用 1：HardEvent 方向不匹配**
```cpp
SetFlag<HardEvent::MTE2_V>(ev);
WaitFlag<HardEvent::V_MTE3>(ev);  // 一个等 MTE2→V，一个等 V→MTE3，方向不同步
```

**❌ 误用 2：eventID 手写**
```cpp
SetFlag<HardEvent::V_MTE3>(0);      // ❌ 固定 ID
WaitFlag<HardEvent::V_MTE3>(0);
// 正确：GetTPipePtr()->FetchEventID(HardEvent::V_MTE3)
```

---

## 4. CrossCoreSetFlag/WaitFlag：核间同步

### 语义
用于**不同 AI Core** 之间的同步。最常见的场景是融合算子中 AIC（Cube 核）与 AIV（Vector 核）通过 GM workspace 交换数据。

### 三种 mode

| mode | 参与者 | 语义 |
|:---|:---|:---|
| `0x0` | 同类核 barrier | 所有 AIC 间 / 所有 AIV 间的阶段屏障 |
| `0x1` | 同 AI Core 内两个 AIV 间 | 两个 AIV 子核成对推进 |
| `0x2` | **AIC↔AIV 交叉同步** | 最常用；AIC 与 AIV 之间双向同步 |

### mode2 使用模式

mode2是AI Core内部，AIC与AIV之间的同步控制。**在AIC核执行CrossCoreSetFlag之后，两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。**
mode2 的第二个模板参数 `PIPE` 表示发信号时的管线上下文，**两个方向使用不同的 PIPE**。

```
AIC → AIV（生产者→消费者）:
  AIC: CrossCoreSetFlag<0x2, PIPE_FIX>(flagId)     // Fixpipe 写完 workspace 后发信号
  AIV: CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagId)   // MTE2 读之前等信号

AIV → AIC（消费者→生产者，释放槽位）:
  AIV: CrossCoreSetFlag<0x2, PIPE_MTE2>(flagId)     // 读完 workspace 后发信号
  AIC: CrossCoreWaitFlag<0x2, PIPE_FIX>(flagId)     // Fixpipe 写之前等槽位空闲
```

### 误用模式

**❌ 误用1：CV1:2模式下仅AIV0 Set或Wait信号计数不匹配**
**✅ 正确理解：AIC的Set对应其相应的两个AIV的Wait，AIC的Wait对应其相应的两个AIV的Set**
```cpp
if (ASCEND_IS_AIC) {
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag);      // wait 1 次
}
if (ASCEND_IS_AIV) {
    if (vecBlockIdx == 0) {
        CrossCoreWaitFlag<0x2, PIPE_MTE3>(flag);    // 仅vec 0 set 1次，错误
    }
}
```

**❌ 误用2：CV1:2模式下仅AIC Set两次**
**✅ 正确理解：AIC的Set对应其相应的两个AIV的Wait，AIC的Wait对应其相应的两个AIV的Set**
```cpp
if (ASCEND_IS_AIC) {
    CrossCoreSetFlag<0x2, PIPE_FIX>(flag);
    CrossCoreSetFlag<0x2, PIPE_FIX>(flag); // 错误，AIC Set1次其对应的两个AIV
}
if (ASCEND_IS_AIV) {
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag);
}
```

**❌ 误用3：提前return，跳过了部分信号**
```cpp
CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag1);
if (someCondition) {
    // do something
    return;  // ❌ 提前返回，跳过了后续的 Set 
}
CrossCoreSetFlag<0x2, PIPE_MTE3>(flag2);    // 在某些条件下可能永远无法执行，导致对方死等
}
```

**❌ 误用4：CrossCoreWaitFlag 漏写 PIPE 模板参数**
```cpp
// ❌ Wait 侧缺少 PIPE
CrossCoreWaitFlag<0x2>(flagId);

// ✅ Set 和 Wait 两侧都必须写 PIPE，且与对侧配对
// AIC 侧 Set/Wait → PIPE_FIX
// AIV 侧 Set/Wait → PIPE_MTE2（MTE3 写操作用 PIPE_MTE3）
CrossCoreWaitFlag<0x2, PIPE_FIX>(flagId);    // AIC 侧 Wait
CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagId);   // AIV 侧 Wait
```

> **⚠️ 验证规则**：封装 WorkspaceQueue 等工具类时，若 `ProducerAcquire`/`ConsumerAcquire` 等泛型方法被 AIC 和 AIV 两侧复用，需通过模板参数将 PIPE 传入，**禁止**依赖默认 PIPE。AIC 侧始终传入 `PIPE_FIX`，AIV 侧始终传入 `PIPE_MTE2`。

---

## 5. TQue EnQue/DeQue：流水线队列同步

### 语义
`TQue` 是 AscendC 流水线编程模型的核心，提供 `AllocTensor → DataCopy → EnQue → DeQue → 使用 → FreeTensor` 的数据流。`EnQue` 内部自动插入管线同步屏障，保证数据就绪。

### TPosition 选择

| TPosition | 物理存储 | 用途 | depth | API 形式 |
|:---|:---|:---|:---|:---|
| `VECIN` | Unified Buffer | 向量输入（GM→UB） | 0 | **引用形式** |
| `VECOUT` | Unified Buffer | 向量输出（UB→GM） | 0 | **引用形式** |
| `VECCALC` | Unified Buffer | 临时计算缓冲 | — | `TBuf`（非 TQue） |
| `A1` | L1 Buffer | Cube 左矩阵 | 1 | 返回值形式 |
| `B1` | L1 Buffer | Cube 右矩阵 | 1 | 返回值形式 |
| `A2` | L0A Buffer | MMA 输入 A | 1 | 返回值形式 |
| `B2` | L0B Buffer | MMA 输入 B | 1 | 返回值形式 |
| `CO1` | L0C Buffer | MMA 输出 C | 1 | 返回值形式 |

### 正确用法

```cpp
// A1/B1：GM→L1（ND→NZ）
auto a1Local = inQueueA1.AllocTensor<half>();
DataCopy(a1Local, aGlobal, nd2nzParams);
inQueueA1.EnQue(a1Local);

// A2/B2：L1→L0A/L0B（NZ→Zz）
auto a1Local = inQueueA1.DeQue<half>();
auto a2Local = inQueueA2.AllocTensor<half>();
LoadNzL1ToZzL0A(a2Local, a1Local, baseM, baseK);
inQueueA2.EnQue(a2Local);
inQueueA1.FreeTensor(a1Local);

// CO1→GM：L0C→GM（NZ→ND）
auto c1Local = outQueueCO1.DeQue<float>();
FixpipeNzL0cToNdGm(cBlock, c1Local, baseM, baseN);
outQueueCO1.FreeTensor(c1Local);
```