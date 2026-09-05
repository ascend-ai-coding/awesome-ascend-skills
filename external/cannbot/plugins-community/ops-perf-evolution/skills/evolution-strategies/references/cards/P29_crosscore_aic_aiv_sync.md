---
id: P29
bottlenecks: [no_overlap, partial_overlap]
op_families: [cv_fusion, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P14, P30, P5, P87]
has_preconditions: true
has_playbook: true
---

# P29: 跨核同步（AIC-AIV CrossCore Sync）

## 核心思想
在 AIC（Cube 核）和 AIV（Vector 核）之间传递数据时，通过 CrossCoreSetFlag/CrossCoreWaitFlag 实现跨核同步。一个 AIC 对应两个 AIV，需要分别同步。

## 代码骨架

// === 改造后（专家模式）===
```cpp
if constexpr (bufferType == BufferType::L1) {
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
    } else {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
    }
}
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 实现 AIC/AIV 异构核间精确数据交接，支持 Cube-Vector 融合流水线

## 常见陷阱

⚠️ AIC 对应两个 AIV 需要两次 SetFlag/WaitFlag；跨核事件 ID 空间有限
⚠️ 跨核同步延迟高于核内同步，应尽量减少同步次数（批量数据交接优于逐 tile 交接）

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
