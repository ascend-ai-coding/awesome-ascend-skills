# 多核数据交换与同步

## 多核之间交换数据

如果需要多核之间交换数据，需要使用 **workspace**：kernel 侧直接使用 workspace 指针，生产者写、`SyncAll()` 后消费者读。

```cpp
// Kernel 侧
__gm__ float* ws = (__gm__ float*)workspace;
// ... 使用 workspace 交换数据 ...
AscendC::SyncAll();  // 全核同步
```

> **tiling 侧**（workspace 大小申请、`SetScheduleMode(1)`）属切分设计，见 [`ascendc-simt-tiling-design`](../../ascendc-simt-tiling-design/SKILL.md)；**凡 kernel 有 `SyncAll()`，tiling 必须 `SetScheduleMode(1)`**，否则多流死锁。

## 多个 VF_CALL 之间的同步

1. 同一个 kernel 内多个 `Simt::VF_CALL` 之间**仅能保证 block 内 VF 串行**
2. **不能保证全核同步**
3. 如果 `Simt::VF_CALL` 之间需要全核同步：
   - 在同步的位置加上 `SyncAll()`
   - tiling 侧须 `SetScheduleMode(1)`（见 `ascendc-simt-tiling-design`）

```cpp
// 两阶段 kernel 的标准编排（清零 → SyncAll → 累加）
asc_vf_call<InitOutputZeroSimt>(dim3(THREAD_NUM), outputSize, outGradGm);
SyncAll();
asc_vf_call<ComputeDeformableOffsetGrad>(dim3(THREAD_NUM), ...);
```

## 同步原语全景

| 原语 | 粒度 | 用途 | 前提 |
|------|------|------|------|
| `asc_syncthreads()` | 核内（block 内） | VF 内线程间同步 | 需配合 CUDA 风格 threadIdx |
| `AscendC::SyncAll()` | 跨核 | 多核数据交换/阶段分隔 | tiling 必须 `SetScheduleMode(1)` |
| `SetFlag/WaitFlag<HardEvent>` | 流水线 | VF 与 MTE2/MTE3/标量间的依赖 | 取 event id 配对 |
| `DataSyncBarrier<MemDsbT::UB>()` | 标量→VF | UB 参数可见性 | 传 `__ubuf__` 前 |

## SIMT 与 Vector/DMA 流水线的 event 依赖链

VF 与 MTE2/MTE3/标量之间的数据依赖必须用显式 event，不能靠函数返回顺序：

```cpp
// MTE2_V（加载完成才能算）→ V_MTE3（产出完成才能写回）→ MTE3_MTE2（写回完成才能覆盖输入缓冲）
event_t eventIdVToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
DataCopyPad(outValueGm_[gmOffset], finalValues_, valueCopyParam);
event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
// 标量写 UB → VF 读：S_V 或 DataSyncBarrier<UB>
```

## 头文件

```cpp
#include "simt_api/device_sync_functions.h"
```