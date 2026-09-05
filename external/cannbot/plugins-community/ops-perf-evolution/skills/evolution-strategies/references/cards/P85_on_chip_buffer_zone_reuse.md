---
id: P85
bottlenecks: [ub_memory_pressure]
op_families: [special]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P85: On-Chip Buffer Zone Reuse (片上 Buffer 分区与分时复用)

## 核心思想
当算子包含多个串行计算阶段，且各阶段的临时 buffer 生命周期不重叠时，可将它们映射到同一块 UB TBuf 或 GM workspace 的不同逻辑区域，或直接共享同一物理区间，从 `sum(stage buffers)` 压缩到 `max(concurrent buffers)`。该策略的核心不是常驻数据跨循环复用，而是**同一次 `Process` 或单个 kernel 内多个阶段的分时复用**：前一阶段结束后，后一阶段覆盖其临时空间。它适用于 RmsNorm/RoPE、SoftMax+后处理、IFA 多阶段中间结果编排等串行阶段明确的场景。它不依赖 `pipe->Reset()` 做阶段级资源重建，也不处理跨迭代状态常驻；主要收益是释放片上或 workspace 容量给更大的 tile、双缓冲或额外流水深度。

## 代码骨架

// === 改造前（基线）===
```cpp
pipe.InitBuffer(rmsBuf0, 1, zoneSize);
pipe.InitBuffer(rmsBuf1, 1, zoneSize);
pipe.InitBuffer(rmsBuf2, 1, zoneSize);
pipe.InitBuffer(ropeCosBuf, 1, zoneSize);
pipe.InitBuffer(ropeSinBuf, 1, zoneSize);
// RmsNorm 和 RoPE 各自独立分配临时空间，无法复用同一块 wsLocal
```

// === 改造后（专家模式）===
```cpp
int64_t xLocalFp32Offset = 0;                     // zone0: RmsNorm fp32
int64_t xSquareLocalOffset = rows * headSize;    // zone1: RmsNorm square / RoPE cos
int64_t xSumLocalOffset = rows * headSize * 2;   // zone2: RmsNorm sum / RoPE sin
LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];
```

## 关键修改点

1. 预期收益: 让一块 UB workspace 覆盖多个串行阶段，显著降低临时 buffer 总量。

## 常见陷阱

⚠️ 各阶段必须具备严格的串行边界，错误判断生命周期会导致后续阶段覆盖仍在使用的数据
⚠️ 共享区通常依赖手工 offset、MAX 大小估算或 alias 规划，维护成本高于独立分配
⚠️ 当后续优化把原本串行阶段改造成重叠流水时，原有 zone reuse 方案可能失效，需要重新规划

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
