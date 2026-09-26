---
id: P89
bottlenecks: [ub_memory_pressure]
op_families: [normalization, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: [P85]
has_preconditions: true
has_playbook: true
---

# P89: UB 高阶 API 临时 Buffer 共享复用 (High-Level API Temp Buffer Sharing)

## 核心思想
当算子使用需要临时 Buffer 的高阶 API（如 SoftMax）时，该临时空间会挤占其他计算空间。通过让高阶 API 的临时 Buffer 与其他计算阶段的临时 Buffer 共享同一块 `TBuf<VECCALC>` 空间（取 `std::max` 大小分配），在不同计算阶段分时复用，减少 UB 总占用和搬运次数。

## 代码骨架

```cpp
// === 改造前（独立分配）===
pipe.InitBuffer(softmaxBuf, 1, softmaxBufSize);  // SoftMax 临时空间
pipe.InitBuffer(sumBuf, 1, sumBufSize);            // Add 临时空间
// UB 占用 = softmaxBufSize + sumBufSize，搬运次数 = 16

// === 改造后（共享复用）===
uint32_t sharedSize = std::max(softmaxBufSize, sumBufSize);
pipe.InitBuffer(sharedBuf, 1, sharedSize);
// Phase 1: SoftMax 使用 sharedBuf
SoftMax(sharedBuf, input, params);
// Phase 2: Add 复用 sharedBuf（SoftMax 已完成，生命周期不重叠）
Add(sharedBuf, a, b);
// UB 占用 = max(softmaxBufSize, sumBufSize)，搬运次数从 16 → 8
```

## 关键修改点

1. 识别算子中所有使用临时 Buffer 的高阶 API 和普通计算阶段
2. 分析各阶段临时 Buffer 的生命周期，确认不重叠
3. 取所有阶段临时 Buffer 大小的 `std::max` 作为共享 Buffer 大小
4. 确保各阶段之间通过 PipeBarrier 或流程保证生命周期隔离

## 常见陷阱

⚠️ 共享 Buffer 的两个使用阶段必须生命周期不重叠，否则会数据竞争
⚠️ 与 P85（on-chip buffer zone reuse）互补：P85 针对 GM workspace 和 UB 的跨阶段复用，本策略专注 UB 内 TBuf 临时 Buffer 共享
⚠️ 高阶 API 的临时 Buffer 大小需要通过 API 文档或 profiling 确认
