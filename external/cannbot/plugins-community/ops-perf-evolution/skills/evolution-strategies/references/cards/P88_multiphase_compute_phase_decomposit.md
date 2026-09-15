---
id: P88
bottlenecks: [ub_memory_pressure]
op_families: [cv_fusion, moe, special]
complexity: L2
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P88: Multi-Phase Compute Phase Decomposition (多阶段计算拆分与资源重分配)

## 核心思想
当算子内部存在前置预处理、主计算、后处理或某个会被主循环重复引用的子计算时，可将其拆成多个彼此串行的独立阶段，并在阶段边界通过 `SyncAll()`、独立 `TPipe` 实例或 `pipe->Reset()` 重新分配 UB/L1/Workspace 等片上资源。该策略的核心不是单纯“多阶段”命名，而是**通过阶段化执行把互相竞争片上资源的子任务拆开**：要么让不同阶段拥有各自最合适的 buffer/TPipe 布局，要么把原本放在 Main 循环中的重复子计算前移为独立阶段并写入 workspace，供后续阶段直接消费。它适用于主阶段与前后处理阶段资源需求差异大、或某个子计算在主循环中被反复引用且值得前置抽取的场景。它不包含单纯的阶段内 buffer alias/zone reuse，也不包含跨 loop/batch/iteration 的常驻状态保留。

## 代码骨架

// === 改造前（基线）===
```cpp
void Process() {
    InitAllBuffers(pipe);
    MainLoop();
    PostProcess();
}
```

// === 改造后（专家模式）===
```cpp
void Process() {
    InitBuffersPhase1(pipe);
    MainLoop();

    SyncAll();
    pipe->Reset();

    vector2Service.InitBuffers(pipe);
    vector2Service.Process();
}
```

## 关键修改点

1. 预期收益: 每个阶段可独立最大化片上资源利用率，避免整条 kernel 生命周期内维持一套折中的 buffer 布局。

## 常见陷阱

⚠️ 阶段边界的 `SyncAll()`、独立 `TPipe` 初始化/销毁或 `pipe->Reset()` 会引入额外固定开销
⚠️ 多阶段之间通常无法像单一深流水那样完全重叠，阶段切换过多时可能抵消收益
⚠️ 若阶段拆分粒度过细，workspace 读写、控制流和初始化成本会变成新的瓶颈

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
