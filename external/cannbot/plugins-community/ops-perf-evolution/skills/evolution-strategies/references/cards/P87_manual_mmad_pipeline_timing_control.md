---
id: P87
bottlenecks: [mte2_stall, partial_overlap]
op_families: [matmul]
complexity: L2
conflicts_with: []
synergizes_with: [P29]
has_preconditions: true
has_playbook: true
---

# P87: Manual Mmad Pipeline Timing Control (手动 Mmad 流水线时序控制)

## 核心思想
当 AscendC 的 Matmul 高阶 API 无法满足特定时序需求时，可以完全绕过 Matmul 库，手动控制 L0A/L0B/L0C 的双缓冲、HardEvent 同步、`unitFlag` 驱动的 Mmad/Fixpipe 融合，以及基于矩阵规模的条件性 `PipeBarrier<PIPE_M>()`。这类优化的共同点不是改变数学逻辑或分核拓扑，而是**直接操控手动 Mmad 执行路径内部的 Cube 时序**：用 L0 双缓冲隐藏搬运延迟，用 `unitFlag` 把最后一次 Mmad 与 Fixpipe 融合，用条件性 `PipeBarrier` 避免小矩阵累加模式下的数据冒险。它适用于同一 kernel 内存在多个不同形状 Matmul、或需要精确安排 K 轴多次迭代时序的场景。它不包含 Matmul 库内部 `MatmulConfig` 调参，也不包含 Split-K/AtomicAdd 或 d=64 特化的算法级 Cube 融合。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：使用 Matmul 库 API
Matmul<...> mm;
mm.SetTensorA(queryGm);
mm.SetTensorB(keyGm);
mm.IterateAll(queryGm, keyGm, outGm);
// Matmul 库内部管理 L0 buffer 和同步
```

// === 改造后（专家模式）===
```cpp
// 手动 Mmad 流水线
TBuf<TPosition::A1> l0aBuf;  // 32K * 2 (ping/pong)
TBuf<TPosition::B1> l0bBuf;  // 32K * 2
TBuf<TPosition::C1> l0cBuf;  // 64K * 2

for (int k = 0; k < kLoops; k++) {
    int pp = k % 2;
    WaitFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);

    LoadData(l0a[pp], l1Key[kvIdx], load3DParams);
    LoadData(l0b[pp], l1Query[qpIdx], load2DParams);

    SetFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);
    WaitFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);

    MmadParams mmadParams;
    mmadParams.unitFlag = (k == kLoops - 1) ? 0b11 : 0b10;
    Mmad(l0c[pp], l0a[pp], l0b[pp], mmadParams);

    SetFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);
}
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 精确控制 L0 搬运与 Cube 执行时序；允许多个 Matmul 共享 L0 buffer，避免 Matmul 库调度限制。

## 常见陷阱

⚠️ 代码复杂度和维护成本极高，需要手动管理 L0 buffer 生命周期、HardEvent、`unitFlag` 和同步点
⚠️ 错误的 `unitFlag` 或 `PipeBarrier` 条件会直接导致数据错误、死锁或数据竞争
⚠️ 不同 headDim、K 切分层级和硬件环境下，最佳的 kL1/kL0 参数与同步阈值可能不同，通常需要 profiling 验证

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|DataCopy" op_kernel/*.cpp op_host/*_tiling.cpp
```
