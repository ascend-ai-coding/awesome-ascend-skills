# AscendC 同步控制接口总结

## 概述
同步控制用于 AI Core 内部异步并行执行单元之间的协调，分为**核内同步**和**核间同步**。

## ⚠️ 关键概念：DMA 异步

MTE2（GM→UB）和 MTE3（UB→GM）是**异步**的：`DataCopyPad` 返回时数据搬运**尚未完成**。

**必须**在 DataCopyPad 之后通过 **EnQue/DeQue** 同步，才能安全访问数据:

```
AllocTensor → DataCopyPad → EnQue(VECIN)    ← 标记搬运完成
                            DeQue(VECIN)     ← 等待搬运完成，才能计算
                            ... 计算 ...
                            EnQue(VECOUT)    ← 标记计算完成
                            DeQue(VECOUT)    ← 等待计算完成，才能搬出
                            DataCopyPad → FreeTensor
```

**推荐 EnQue/DeQue 而非 PipeBarrier**：EnQue/DeQue 是精确的生产-消费同步，PipeBarrier 是粗粒度全流水线阻塞。

---

## 一、核内同步

### 1.1 流水类型

| 流水类型 | 含义 |
|----------|------|
| PIPE_S | 标量流水线（GetValue/SetValue） |
| PIPE_V | 矢量计算流水线 |
| PIPE_M | 矩阵计算流水线 |
| PIPE_MTE1 | L1→L0A/L0B 数据搬运 |
| PIPE_MTE2 | GM→L1/UB 数据搬运 |
| PIPE_MTE3 | UB→GM 数据搬运 |
| PIPE_FIX | L0C→GM/L1 数据搬运 |

### 1.2 多流水同步 (SetFlag/WaitFlag)

**功能**: 不同流水线间的同步，用于数据依赖场景。

**ISASI接口**（不保证跨版本兼容）:
```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID);

template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID);
```

**TQueSync接口**（保证跨版本兼容）:
```cpp
AscendC::TQueSync<PIPE_S, PIPE_MTE3> sync;
sync.SetFlag(0);
sync.WaitFlag(0);
```

**HardEvent类型**: `MTE2_V`, `V_MTE2`, `MTE3_V`, `V_MTE3`, `M_V`, `V_M`, `S_MTE3`等

**使用要点**:
- SetFlag/WaitFlag必须成对出现
- eventID需通过`AllocEventID()`或`FetchEventID()`获取
- 范围: Atlas训练0-3, 其他0-7

**示例**:
```cpp
dstLocal.SetValue(0, 0);
int32_t eventID = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3);
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventID);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventID);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

### 1.3 单流水同步 (PipeBarrier)

**功能**: 同一流水线内部同步，保证前序指令完成后执行后续指令。

**原型**:
```cpp
template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
```

**示例**:
```cpp
AscendC::Add(dst0Local, src0Local, src1Local, 512);
AscendC::PipeBarrier<PIPE_V>();  // 保证Add完成
AscendC::Mul(dst1Local, dst0Local, src2Local, 512);
```

> **注意**: PIPE_S禁止调用PipeBarrier，会引发硬件错误。

### 1.4 数据同步屏障 (DataSyncBarrier)

**功能**: 阻塞后续指令直到所有内存访问完成。

**原型**:
```cpp
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
```

**参数**: `ALL`(所有内存), `DDR`(GM), `UB`, `SEQ`(预留)

**示例**:
```cpp
AscendC::Mmad(...);
AscendC::DataSyncBarrier<MemDsbT::ALL>();
AscendC::Fixpipe(...);
```

---

## 二、核间同步

### 2.1 SyncAll (全核同步)

**功能**: 所有核同步，等待所有核执行完成。

**软同步**:
```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace,
                               const LocalTensor<int32_t>& ubWorkspace,
                               const int32_t usedCores = 0);
```

**硬同步**:
```cpp
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll();
```

**空间要求**: gmWorkspace ≥ 核数×32Bytes, ubWorkspace ≥ 核数×32Bytes

### 2.2 IBSet/IBWait (核间同步)

**功能**: 核间设置/等待同步标志。

**原型**:
```cpp
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(const GlobalTensor<int32_t>& gmWorkspace,
                             const LocalTensor<int32_t>& ubWorkspace,
                             int32_t blockIdx, int32_t eventID);

template <bool isAIVOnly = true>
__aicore__ inline void IBWait(...);  // 参数同上
```

**空间要求**: gmWorkspace ≥ 核数×32×eventID_max + blockIdx_max×32 + 32

### 2.3 确定性计算接口

**InitDetermineComputeWorkspace** - 初始化共享内存:
```cpp
__aicore__ inline void InitDetermineComputeWorkspace(
    GlobalTensor<int32_t>& gmWorkspace,
    LocalTensor<int32_t>& ubWorkspace);
```

**WaitPreBlock** - 等待前一个核:
```cpp
__aicore__ inline void WaitPreBlock(
    GlobalTensor<int32_t>& gmWorkspace,
    LocalTensor<int32_t>& ubWorkspace);
```

**NotifyNextBlock** - 通知下一个核:
```cpp
__aicore__ inline void NotifyNextBlock(
    GlobalTensor<int32_t>& gmWorkspace,
    LocalTensor<int32_t>& ubWorkspace);
```

### 2.4 CrossCoreSetFlag/CrossCoreWaitFlag (分离模式)

**功能**: 面向分离模式的核间同步。

**原型**:
```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

**modeId**:
- 0: AI Core核间同步
- 1: AIV核之间同步
- 2: AIC与AIV之间同步

**示例**:
```cpp
// 模式0: 同步所有AIV核
AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
AscendC::CrossCoreWaitFlag(0x8);
```

---

## 三、EventID管理

### AllocEventID
申请并占用EventID，需配合ReleaseEventID释放:
```cpp
AscendC::TEventID eventID = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
```

### FetchEventID
仅获取可用EventID，不占用:
```cpp
AscendC::TEventID eventID = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
```

---

## 产品支持汇总

| 接口 | Atlas A3 | Atlas A2 | Atlas 200I/500 A2 | Atlas 推理 AI Core | Atlas 训练 |
|------|----------|----------|-------------------|-------------------|------------|
| SetFlag/WaitFlag (ISASI) | √ | √ | × | √ | √ |
| TQueSync | √ | √ | √ | √ | √ |
| PipeBarrier | √ | √ | √ | √ | √ |
| DataSyncBarrier | × | √ | √ | × | × |
| SyncAll | √ | √ | × | √ | √ |
| CrossCoreSetFlag/WaitFlag | √ | √ | × | × | × |
