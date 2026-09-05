---
id: P61
bottlenecks: [l2_cache_thrash]
op_families: [flash_attention]
complexity: L1
conflicts_with: [P52]
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P61: L2 Cache 优化 (L2 Cache Optimization)

## 核心思想
通过关闭双页表功能（开启 KV 的 L2 Cache），减少 GM 访问延迟。在 PageAttention 场景下，KV 数据通过 L2 Cache 缓存，显著提升数据复用效率。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：双页表开启，KV 不走 L2 Cache
// 每次 KV 访问都需要 GM 读取
```

// === 改造后（专家模式）===
```cpp
// 关闭双页表功能，开启 KV 的 L2 Cache
// 修改位置：ops_adv 代码

// 原始配置：l2CacheOffFlag = true
// KV 数据直接访问 GM，无 L2 缓存

// 优化配置：l2CacheOffFlag = false
// KV 数据进入 L2 Cache，后续访问命中缓存

// 性能收益
// 250us → 210us（约 16% 提升）
```

## 关键修改点

1. 预期收益: 开启 L2 Cache 后，KV 数据复用率提升，GM 访问延迟降低

## 常见陷阱

⚠️ 需要硬件支持 L2 Cache 功能
⚠️ 可能影响其他算子的 L2 Cache 使用
⚠️ 需要权衡 L2 Cache 与双页表功能

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
