---
id: P51
bottlenecks: [tiling_imbalance]
op_families: [cv_fusion, matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P4, P47, P50, P73]
has_preconditions: true
has_playbook: true
---

# P51: 动态 AIC/AIV 核配比 (Dynamic AIC/AIV Core Ratio)

## 核心思想
根据 K 维大小动态选择 AIC:AIV 比例（1:1 vs 1:2 vs CUBE_ONLY），在 Cube 密集和 Vector 密集场景间自动切换，释放闲置核资源。

## 代码骨架

// === 改造前（基线）===
```cpp
// 始终使用 KERNEL_TYPE_MIX_AIC_1_2
// 大 K 场景: Vector 核闲置，系统资源浪费
```

// === 改造后（专家模式）===
```cpp
// Host 侧 tiling 决策
bool IsAivAicRatioTwoRequired() {
    return k < DOUBLE_VECTOR_THRESHOLD_K_UPPER;  // K < 2048
}

if (IsAivAicRatioTwoRequired()) {
    kernelType = KERNEL_TYPE_MIX_AIC_1_2;  // 小 K: Cube 快，需要更多 Vector
} else {
    kernelType = KERNEL_TYPE_MIX_AIC_1_1;  // 大 K: Vector 成瓶颈，释放多余 Vector
}

// Kernel 侧通过 TilingKey 分发
if (TILING_KEY_IS(0)) { /* 1:2 模式 kernel */ }
if (TILING_KEY_IS(1)) { /* 1:1 模式 kernel */ }
```

## 关键修改点

1. 预期收益: 大 K 场景系统吞吐提升 10-20%（释放的 Vector 核可服务其他算子）

## 常见陷阱

⚠️ 需要 Host 侧根据 K 值选择 kernel type，增加 tiling 逻辑
⚠️ 阈值参数（K=2048）需要根据具体算子调优
⚠️ 动态切换可能影响系统级多算子调度

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
