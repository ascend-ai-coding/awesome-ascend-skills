---
id: P52
bottlenecks: [l2_cache_thrash]
op_families: [cv_fusion, matmul]
complexity: L1
conflicts_with: [P61]
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P52: L2 Cache Hint 优化 (L2 Cache Hint Optimization)

## 核心思想
通过 SetL2CacheHint 在多核竞争场景下主动禁用 L2 缓存，避免缓存抖动（cache thrashing）。在 blockDimM==1 或输出数据量超过 L2 容量时，禁用 L2 可提升整体带宽利用率。

## 代码骨架

// === 改造前（基线）===
```cpp
// 所有 buffer 使用默认 L2 缓存
// blockDimM==1 时: 多核竞争 → cache thrashing → 有效带宽下降
```

// === 改造后（专家模式）===
```cpp
if (blockDimM == 1) {
    // 多核加载同一权重列 → L2 cache line 频繁被驱逐
    // 禁用 L2，直接走 DDR 并行访问
    weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}
```

## 关键修改点

1. 预期收益: blockDimM==1 场景带宽提升 15-25%（避免 cache thrashing）

## 常见陷阱

⚠️ 禁用 L2 后依赖 DDR 带宽，小数据量场景可能反而变慢
⚠️ 需要根据 blockDimM 和数据量动态决策
⚠️ 仅两行 API 调用，实现成本极低

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue" op_kernel/*.cpp op_host/*_tiling.cpp
```
