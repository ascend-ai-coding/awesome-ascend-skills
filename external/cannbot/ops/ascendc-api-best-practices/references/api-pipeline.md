# 流水线同步机制指南

MTE 与 Vector 同步的核心机制。

---

## 目录

1. [核心问题](#核心问题)
2. [解决方案](#解决方案)
3. [三种方案对比](#三种方案对比)
4. [完整流水线模板](#完整流水线模板)
5. [调试技巧](#调试技巧)

---

## 核心问题

**DataCopy/DataCopyPad 是异步 DMA 操作，直接在搬运后的数据上做 Vector 计算可能读到未完成的数据！**

### 硬件架构

```
GM → MTE2 (异步) → UB → Vector (同步) → MTE3 (异步) → GM
```

### 问题场景

```cpp
// ❌ 错误：DataCopyPad 后直接使用数据
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);  // 可能读到未完成搬运的数据！
```

**现象**：输出数据随机、错误

---

## 解决方案

### 方案一：EnQue/DeQue 队列同步（推荐）

**原理**：TQue 的 EnQue/DeQue 机制自动提供硬件同步点。

```cpp
// ✅ 正确：使用 EnQue/DeQue 同步
// Step 1: CopyIn - MTE2 搬运
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
inQueueX.EnQue(xLocal);                    // 标记"就绪"

// Step 2: Compute - Vector 计算
AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 阻塞等待 MTE2 完成
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
AscendC::Adds<float>(yLocal, xIn, 1.0f, count);
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xIn);

// Step 3: CopyOut - MTE3 搬运
AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 阻塞等待 Vector 完成
AscendC::DataCopyPad(yGm[gmOffset], yOut, copyOutParams);
outQueueY.FreeTensor(yOut);
```

**关键点**：
- `EnQue(xLocal)` 标记 buffer 数据就绪
- `DeQue<float>()` 阻塞等待数据就绪
- DeQue 返回后，数据一定已经搬运完成

### 方案二：PipeBarrier 手动同步

```cpp
// ✅ 可用：PipeBarrier 手动同步（替代队列同步）
AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyInParams, padParams);
AscendC::PipeBarrier<PIPE_MTE2>();  // 等待 MTE2（GM→UB）完成

AscendC::Adds<float>(yLocal, xLocal, 1.0f, count);
AscendC::PipeBarrier<PIPE_V>();     // 等待 Vector 完成

AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyOutParams);
AscendC::PipeBarrier<PIPE_MTE3>();  // 等待 MTE3（UB→GM）完成
```

---

## 三种方案对比

| 特性 | EnQue/DeQue | SetFlag/WaitFlag（HardEvent 事件式） | PipeBarrier |
|-----|-------------|--------------------------------------|-------------|
| 同步粒度 | buffer 级别 | pipe 对级别（如 MTE2→V、V→MTE3） | 单 pipe 或全 pipe 栅栏 |
| 性能 | 高（支持并行） | 高（精细配对，开销最小） | `PipeBarrier<PIPE_V>` 等单 pipe 栅栏廉价；`PipeBarrier<PIPE_ALL>` 全流水线停顿，仅调试用 |
| 代码复杂度 | 需要队列管理 | 需手动管理 eventID 配对/复用 | 简单直接 |
| 适用 | TPipe/TQue 管理的标准搬运流水 | 手动 UB 管理、精细 pipe 间流水编排 | 防御性顺序依赖（如 V 内 Cast→Add）、调试定位 |

> 三条路径不是二选一：TQue 场景用 EnQue/DeQue；手动 UB（无 TPipe）场景用 SetFlag/WaitFlag 事件流水；单 pipe 内的顺序依赖用细粒度 PipeBarrier 防御。

### 事件式流水（SetFlag/WaitFlag HardEvent）要点

```cpp
// 核内 pipe 间事件同步：DataCopy(MTE2) 与 Cast/Add(V) 跨 pipe 重叠
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);   // 标记 MTE2 搬运完成
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);  // V 侧等待后消费
```

使用纪律：

1. **Set/Wait 必须同 eventID 配对**，且**在同一迭代内配对**——跨迭代 Set-Set 无中间 Wait 在 950 实测挂死（`aclError:507014`）。**循环结束后必须消费残留事件**：最后一次/几次 Set 无对应 Wait 时需补 Wait；注意按实际 Set 次数守卫（如 pingpong 双 eventID 时，第二个 eventID 仅在迭代数 ≥ 2 时才被 Set 过，不可无条件 Wait——Wait 多于 Set 同样是未定义行为）
2. **eventID 是有限共享资源**（TQue 与手动 SetFlag/WaitFlag 共用配额，典型 8 个/核）——手动事件与 TQue 混用时预算要合并计算
3. V 内部的顺序依赖（Cast→Add）优先用 `PipeBarrier<PIPE_V>` 防御，省去 V→V 的 eventID 开销
4. **跨 pipe 依赖必须用事件同步**：`PipeBarrier<PIPE_X>` 只保证同一 pipe 内的顺序，不保证跨 pipe 可见性——如 V 侧 Cast 写完的数据要经 MTE3 搬出时，必须 `SetFlag/WaitFlag<V_MTE3>`，仅 `PipeBarrier<PIPE_V>` 不足（生产踩坑：跨 pipe 缺事件同步导致搬出读到未写完的数据）

### EnQue/DeQue 的双重作用

1. **队列管理**：Double Buffer 场景下管理多 buffer 轮转
2. **硬件同步**：提供 MTE ↔ Vector 之间的同步点

```cpp
// EnQue/DeQue 不仅仅是"队列"，更重要的是同步机制
inQueueX.EnQue(xLocal);    // 1. 标记数据就绪  2. 通知硬件可以等待
xLocal = inQueueX.DeQue(); // 1. 阻塞等待就绪  2. 获取可用 buffer
```

---

## 完整流水线模板

```cpp
__aicore__ inline void ProcessTile(uint32_t tileIdx)
{
    // ========== CopyIn 阶段 ==========
    // MTE2: GM → UB（异步）
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopyPad(xLocal, xGm[tileIdx * tileSize], copyParams, padParams);
    inQueueX.EnQue(xLocal);              // 同步点：标记就绪
    
    // ========== Compute 阶段 ==========
    // Vector: UB 计算（同步，需等待 MTE2）
    AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();  // 同步点：等待 MTE2
    AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    AscendC::Adds<float>(yLocal, xIn, 1.0f, tileSize);
    outQueueY.EnQue(yLocal);             // 同步点：标记就绪
    inQueueX.FreeTensor(xIn);
    
    // ========== CopyOut 阶段 ==========
    // MTE3: UB → GM（异步，需等待 Vector）
    AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();  // 同步点：等待 Vector
    AscendC::DataCopyPad(yGm[tileIdx * tileSize], yOut, copyParams);
    outQueueY.FreeTensor(yOut);
}
```

### 流水线时序图

```
时间 →
        
Tile 0:  [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                      ↑ DeQue等待    ↑ DeQue等待
Tile 1:          [MTE2]──EnQue──[Vector]──EnQue──[MTE3]
                  ↑ 并行！    ↑ DeQue等待    ↑ DeQue等待

关键：DeQue 阻塞等待上一个阶段的异步操作完成
```

---

## 调试技巧

### 检查缺少 EnQue/DeQue

```cpp
// ❌ 错误：AllocTensor 后直接用
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
Compute(x);  // 错！可能读到未完成搬运的数据

// ✅ 正确：DeQue 后再计算
LocalTensor<T> x = inQueue.AllocTensor<T>();
DataCopy(x, gm, size);
inQueue.EnQue(x);
LocalTensor<T> xIn = inQueue.DeQue<T>();  // 等待搬运完成
Compute(xIn);
```

### 临时加 PipeBarrier 调试

```cpp
DataCopy(x, gm, size);
PipeBarrier<PIPE_ALL>();  // 临时加，如果结果正确说明是同步问题
Compute(x);
```

**如果 PipeBarrier 能解决问题，说明是同步问题** → 修复方案：改为 EnQue/DeQue 机制

### 常见误区

| 误区 | 正确理解 |
|-----|---------|
| AllocTensor 后数据就可用 | AllocTensor 只分配内存，不等待搬运 |
| DataCopy 是同步的 | DataCopy 是异步 DMA，立即返回 |
| 不用 EnQue/DeQue 也能正常工作 | 必须用 EnQue/DeQue、SetFlag/WaitFlag 或 PipeBarrier 之一同步 |
| PipeBarrier 性能好 | **仅 `PipeBarrier<PIPE_ALL>` 是全流水线停顿**（性能差，仅临时调试用，定位后应移除）；`PipeBarrier<PIPE_V>` 等单 pipe 栅栏是廉价且必要的顺序依赖手段 |
