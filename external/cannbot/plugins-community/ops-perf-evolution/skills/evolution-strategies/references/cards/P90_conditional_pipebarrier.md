---
id: P90
bottlenecks: [scalar_loading, partial_overlap]
op_families: [matmul, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P87]
has_preconditions: true
has_playbook: true
---

# P90: 条件性 PipeBarrier 基于矩阵规模 (Conditional PipeBarrier by Matrix Size)

## 核心思想
在手动 Matmul 流水线中，当 L0C 处于累加模式（`isL0CAccum=true`）时，根据矩阵乘的 M/N 规模动态决定是否插入 `PipeBarrier<PIPE_M>()`。大矩阵的 Mmad 执行时间足够长，L0 搬运可以与 Mmad 完全重叠，不需要额外同步；小矩阵则需要 PipeBarrier 确保 Mmad 完成后再进行下一次 L0 搬运。

## 代码骨架

```cpp
// === 改造前（无条件 PipeBarrier）===
Mmad(l0c, l0a, l0b, mmadParams);
PipeBarrier<PIPE_M>();  // 无条件等待，大矩阵场景浪费性能

// === 改造后（条件性 PipeBarrier）===
void ManualMmad(const MmParam& mmParam, const MmadParams& mmadParams) {
    LoadData(l0a, l1Src, loadParams);
    Mmad(l0c, l0a, l0b, mmadParams);

    // 小矩阵：等待 Mmad 完成后再搬运 L0
    // 大矩阵：跳过，L0 搬运与 Mmad 自动重叠
    if (mmParam.isL0CAccum &&
        ((mmadParams.m / 16) * (mmadParams.n / 16) < 10)) {
        PipeBarrier<PIPE_M>();
    }
}
```

## 关键修改点

1. 识别手动 Matmul 流水线中 L0C 累加模式的 Mmad 调用点
2. 计算当前 Mmad 的 fractal 数量：`(M/16) * (N/16)`
3. 阈值设为 10 个 fractal：小于 10 插入 PipeBarrier，大于等于 10 跳过
4. 阈值是经验值（来自 SLI/LI 训练算子），不同硬件可能需要调整

## 常见陷阱

⚠️ 阈值 10 是经验值，A5 架构可能需要不同的阈值
⚠️ 仅在 L0C 累加模式（`isL0CAccum=true`）下生效，非累加模式需要无条件同步
⚠️ 跳过 PipeBarrier 的前提是大矩阵 Mmad 执行时间足够覆盖 L0 搬运延迟
