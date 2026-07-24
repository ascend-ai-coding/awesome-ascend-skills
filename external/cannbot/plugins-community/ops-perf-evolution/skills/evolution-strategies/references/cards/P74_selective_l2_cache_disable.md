---
id: P74
bottlenecks: [l2_cache_thrash]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P74: L2 Cache 选择性禁用 (Selective L2 Cache Disable)

## 核心思想
在 KV cache 数据量远大于 L2 容量的推理场景中，通过 `SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)` 选择性禁用 L2 Cache，避免 L2 cache thrashing 导致的性能抖动。Host 端 tiling 根据 KV layout 和数据总量动态决定是否禁用。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：默认启用 L2 Cache，长序列场景可能 thrashing
// 无 L2 Cache 控制逻辑
```

// === 改造后（专家模式）===
```cpp
// Host 端判断逻辑
bool GetL2CacheOffFlag() {
    // 条件1: BNSD/BnNBsD layout 连续访存，不需要 L2 预取
    if (kvLayout == BNSD || kvLayout == BnNBsD) {
        return true;  // 禁用 L2
    }
    // 条件2: KV 数据总量超过 L2 容量的 1.2 倍
    uint64_t kvDataSize = batchSize * kvHeadNum * seqLen * headDim * sizeof(half);
    if (kvDataSize > L2_CAPACITY * 1.2) {
        return true;  // 禁用 L2，避免 thrashing
    }
    return false;
}
```

## 关键修改点

1. 预期收益: 避免 L2 cache thrashing 导致的性能抖动；减少 L2 cache line 替换开销

## 常见陷阱

⚠️ 禁用后所有 DMA 走 DDR→L1 路径，短序列场景可能损失 L2 命中收益
⚠️ 需要 Host 端准确估算 KV 数据总量与 L2 容量的比值
⚠️ 与 P47 (L2 Cache 优化) 互补：P47 侧重利用 L2，本策略侧重规避 L2 thrashing

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
