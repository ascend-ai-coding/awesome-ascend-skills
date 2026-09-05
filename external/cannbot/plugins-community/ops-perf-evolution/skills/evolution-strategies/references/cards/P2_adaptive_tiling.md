---
id: P2
bottlenecks: [mte2_stall, partial_overlap, scalar_loading, tiling_imbalance, undersize_transfer]
op_families: [elementwise, normalization, optimizer, pooling_gather, quantization]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P2: Adaptive Tiling (自适应分块策略)

## 核心思想
专家实现的Tiling不是固定值，而是根据平台UB大小和数据类型动态计算。CalMaxFormerNum函数计算UB能容纳的最大行数，确保数据尽量驻留在UB中，减少GM访问。计算公式：availableUb = ubSize - RESERVED_UB_SIZE - idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB；maxFormerNum = (availableUb / (gradAlignNum * sizeof(float) * useGradNum)) * gradAlignNum。其中RESERVED_UB_SIZE (20KB)预留栈空间，USE_IDX_NUM_IN_UB (3)是索引队列数量，useGradNum根据数据类型变化（FP32=3, FP16=4）。

## 代码骨架

// === 改造前（基线）===
```cpp
// lingxi-code单一策略，仅按输出元素分块
uint32_t elems_per_core = (total_output_elems + BLOCK_DIM - 1) / BLOCK_DIM;
```

// === 改造后（专家模式）===
```cpp
// 动态策略选择
if (doubleC > tileLen) {
    mode = MODE_SPLIT_C;
    params.cTileLength = alignC > tileLen ? tileLen : alignC;
} else if (inputTileNum < params.maxWindowWLength) {
    mode = MODE_SPLIT_W;
} else {
    mode = MODE_MULTI_W;
}
```

## 关键修改点

1. 预期收益: 针对不同输入形状选择最优策略，最大化UB利用率和计算效率，预期性能提升20-50%

## 常见陷阱

⚠️ Tiling逻辑复杂，需要维护三种Kernel实现
⚠️ tiling逻辑复杂，需要维护多个kernel路径
⚠️ 增加tiling计算复杂度和代码量

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
