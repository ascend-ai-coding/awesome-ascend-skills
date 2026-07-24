---
id: R8
bottlenecks: []
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Uses fictional Mutex::Lock API. Overlaps with P28/P29/P30 sync strategies. Zero references."
---

# R8: Mutex 细粒度同步

## 核心思想
使用 Mutex 替代粗粒度 SetFlag/WaitFlag，减少流水线停顿。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Baseline: 粗粒度同步（等待整个流水完成）
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
// 所有 MTE2 操作都完成后才继续

// Evolved: Mutex 细粒度同步
Mutex::Lock(PIPE_MTE2);
// 仅锁定 MTE2 流水
DataCopy(ubTensor, gmTensor, count);
Mutex::Unlock(PIPE_MTE2);
// MTE2 解锁后其他流水可继续
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll

## 常见陷阱

⚠️ 收益**: 减少不必要的流水线等待
⚠️ 风险**: 过细粒度的同步增加编程复杂度
⚠️ 复杂度**: 高

## 代码搜索关键词

```bash
grep -n "GetBlockNum\|coreNum\|blockIdx\|SplitCore\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
