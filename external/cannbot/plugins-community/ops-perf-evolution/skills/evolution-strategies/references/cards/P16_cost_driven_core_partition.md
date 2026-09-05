---
id: P16
bottlenecks: [tiling_imbalance]
op_families: [cv_fusion, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P16: Cost-Driven Core Partition (基于代价模型的多粒度分核优化)

## 核心思想
与简单的按元素/行数均分不同，本策略通过构建计算代价模型（考虑对齐开销、轴权重等硬件特性），对每个计算块评估实际代价，然后采用多粒度层次化分配（Batch→Row→Block）将负载按代价均衡地分配到各核。同时支持Mask/稀疏感知跳过无效块、最优核数枚举搜索、容忍度机制避免碎片化、以及跨核归约(FD)的独立负载均衡。适用于计算量在不同块之间差异大、存在稀疏/Mask模式的复杂算子（如Flash Attention）。

## 代码骨架

// === 改造前（基线）===
```cpp
// baseline: 按元素数量均分，不考虑对齐和计算代价差异
uint32_t elementsPerCore = totalElements / coreNum;
```

// === 改造后（专家模式）===
```cpp
int64_t CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U;
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
}na

BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
    uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ?
        0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}
```

## 关键修改点

1. 预期收益: 精确反映硬件实际计算开销，尾块代价自动降低，避免按数量均分导致的隐性不均衡

## 常见陷阱

⚠️ Host 端 Tiling 计算复杂度显著增加，需要枚举核数 + 多级分配
⚠️ 需要维护代价模型参数（对齐系数、轴权重），模型不准会导致分核效果退化
⚠️ 代码量和调试难度大幅上升，适合高频调用的核心算子，不适合简单算子

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
