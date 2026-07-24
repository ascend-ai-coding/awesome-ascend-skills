---
id: P63
bottlenecks: [no_overlap]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P46, P47]
has_preconditions: true
has_playbook: true
---

# P63: Iterate 异步接口避免 AIC/AIV 同步依赖 (Iterate Async Interface)

## 核心思想
在 MIX（AIC+AIV 混合编程）场景中，使用 Matmul Iterate<false> 或 IterateAll<false> 异步接口替代同步 Iterate<true>，仅第一次调用需发送 AIV→AIC 消息，后续调用无需消息发送，减少 Cube 与 Vector 核间通信开销。需配合 SetWorkspace 使用。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：使用同步 Iterate<true>，每次调用都有核间通信开销
```

// === 改造后（专家模式）===
```cpp
// 同步方式（反例）：每次 Iterate 都发送消息
matmulObj.Iterate<true>(cGlobal);  // 每次触发 AIV→AIC 消息

// 异步方式（正例）：仅第一次发送消息
matmulObj.SetWorkspace(workspace);
matmulObj.Iterate<false>(cGlobal);  // 仅首次发送消息，后续无需消息交互
```

## 关键修改点

1. 预期收益: 减少核间消息发送次数，降低 AIC/AIV 同步等待延迟

## 常见陷阱

⚠️ 需要额外分配 Workspace 空间（singleCoreM * singleCoreN * sizeof(float)）
⚠️ 仅适用于 MIX 场景

## 代码搜索关键词

```bash
grep -n "SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
