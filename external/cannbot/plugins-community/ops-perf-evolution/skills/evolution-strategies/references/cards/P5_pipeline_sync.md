---
id: P5
bottlenecks: [scalar_compute, scalar_loading]
op_families: [attention, flash_attention]
complexity: L0
conflicts_with: []
synergizes_with: [P1, P20, P29, P30]
has_preconditions: true
has_playbook: true
---

# P5: 流水线同步优化 (Pipeline Sync)

## 核心思想
用细粒度的 PipeBarrier/SetFlag/WaitFlag 替代粗粒度的 SyncAll，只在真正有数据依赖的地方同步，减少不必要的等待。

## 代码骨架
```cpp
// === 改造前（粗粒度同步）===
ComputeMm1(info);
SyncAll();  // 所有核等待，包括不相关的核

ComputeVec1(info);
SyncAll();  // 再次全局等待

// === 改造后（细粒度同步）===
// 方式 A: PipeBarrier 按 pipe 类型同步
ComputeMm1(info);
PipeBarrier<PIPE_V>();  // 仅 Vector pipe 同步

// 方式 B: 事件同步（异步流水线）
DataCopy(inputLocal, inputGm, len);
SetFlag<HardEvent::MTE2_V>(eventID);   // 标记 DMA 完成
WaitFlag<HardEvent::MTE2_V>(eventID);  // Vector 等待 DMA
Compute(inputLocal);
SetFlag<HardEvent::V_MTE3>(eventID2);  // 标记计算完成
WaitFlag<HardEvent::V_MTE3>(eventID2); // MTE3 等待计算
CopyOut(outputLocal, outputGm, len);

// 方式 C: 跨核点对点同步（CV 融合场景）
// Cube 完成 MM1 后通知 Vector
CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(syncC1V1);
// Vector 等待后开始 Vec1
CrossCoreWaitFlag(syncC1V1);
```

## 关键修改点
1. **识别同步点**: 找出所有 `SyncAll()` 和全局 `PipeBarrier<PIPE_ALL>()`
2. **分析数据依赖**: 谁等谁？Cube→Vector？MTE2→Vector？
3. **选择同步方式**:
   - 同核内: `PipeBarrier<PIPE_V>` / `SetFlag/WaitFlag<HardEvent::MTE2_V>`
   - 跨核 Cube→Vector: `CrossCoreSetFlag/WaitFlag`
4. **移除不必要的 SyncAll**: 如果没有数据依赖，直接删除

## 常见陷阱
- ❌ **只加 PipeBarrier 不改同步粒度** → 仍用 PIPE_ALL，无优化
- ❌ **错误的 HardEvent 类型** → MTE2_V 写成了 MTE3_V，同步失败
- ❌ **遗漏同步点** → 数据未就绪就开始计算，精度错误
- ❌ **事件 ID 冲突** → 多个循环复用同一 eventID，死锁

## 代码搜索关键词
```bash
grep -n "SyncAll\|PipeBarrier" *.cpp *.h              # 找所有同步点
grep -n "SetFlag\|WaitFlag\|HardEvent" *.cpp *.h      # 找事件同步
grep -n "CrossCore.*Flag" *.cpp *.h                   # 找跨核同步
grep -n "FetchEventID" *.cpp *.h                      # 找事件分配
```
