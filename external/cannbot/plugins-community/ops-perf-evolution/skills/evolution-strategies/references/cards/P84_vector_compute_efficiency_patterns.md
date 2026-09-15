---
id: P84
bottlenecks: [compute_bound, scalar_compute]
op_families: [elementwise, normalization, optimizer]
complexity: L2
conflicts_with: [P67, P68, P69]
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P84: Vector Compute Efficiency Patterns (UB 内 Vector 计算效率优化模式)

## 核心思想
在纯 Vector 或 Vector 主导的算子中，性能瓶颈常来自三类 UB 内部效率损失：Scalar 侧为每个 Vector API 手动计算 repeat/mask，归约场景反复使用高延迟 reduce 指令，以及连续 Vector 计算链把中间结果写回 GM 后再读回 UB。该策略将这些问题统一为“Vector 计算阶段内部效率优化”：优先用 Counter 模式减少控制指令，用低延迟指令组合替代单一高延迟归约路径，并把连续 Vector 链的中间结果保留在 UB/VECCALC 中直通消费。它不改变算子数学逻辑和外部接口，主要通过 Vector API 调用方式、指令组合和 UB 内数据生命周期调整减少 Scalar 开销、Vector 指令延迟和 GM 往返搬运。三者均作用于单核 UB 内的 Vector 执行路径，不涉及分核策略、跨核同步或跨循环状态管理。

## 代码骨架

// === 改造前（基线）===
```cpp
for (uint32_t loop = 0; loop < loopCount; ++loop) {
    uint64_t mask = CalcMask(loop, totalLength);
    uint32_t repeat = CalcRepeat(loop, totalLength);
    Add(dst[offset], src0[offset], src1[offset], mask, repeat, params);
}
```

// === 改造后（专家模式）===
```cpp
// Normal 模式（反例）：手动计算 repeatTimes 和 tail mask
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;
Add(dst, src1, src2, FULL_MASK, repeatTimes, {...});
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst[offset], src1[offset], src2[offset], tailMask, 1, {...});
}

// Counter 模式（正例）：按总元素个数一次性覆盖
SetMaskCount();
SetVectorMask<MaskMode::COUNTER>(ELE_SIZE);
Add(dst, src1, src2, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
ResetMask();
```

## 关键修改点

1. 预期收益: 减少 repeat/tail/mask 的 Scalar 计算和分支，降低 Vector loop 的控制开销。

## 常见陷阱

⚠️ Counter 模式依赖硬件和 API 支持，使用后必须正确恢复 mask 模式，避免影响后续 Vector API
⚠️ 低延迟归约组合会增加临时 buffer、`PipeBarrier` 和 shape-specific 分支，复杂度高于直接调用单个 reduce API
⚠️ UB 融合连续 Vector 链会延长中间 tensor 生命周期，链条过长或 dtype 较宽时可能挤占双缓冲空间

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore\|DataCopy" op_kernel/*.cpp op_host/*_tiling.cpp
```
