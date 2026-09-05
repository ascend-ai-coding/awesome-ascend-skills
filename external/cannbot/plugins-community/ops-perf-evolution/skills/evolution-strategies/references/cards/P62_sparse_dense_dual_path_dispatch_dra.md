---
id: P62
bottlenecks: [tiling_imbalance]
op_families: [flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: [P76, P80]
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Explicitly marked Draft in title. Insufficient evidence width for general application."
---

# P62: Draft: Sparse/Dense Dual-Path Dispatch by Reachable Coverage（按可达覆盖度切换 sparse/dense 双路径，草稿）

## 核心思想
部分 sparse attention 训练算子并不会无条件执行 gather-merge sparse path。更优实现会先比较“当前 query 可达的有效稀疏跨度”与“已选 sparse/topK 预算是否已经足够覆盖”，再在运行时切换两条不同的数据流：
- 若 sparse/topK 覆盖已经足够，则直接走 dense GM path，避免额外 gather/merge
- 若可达跨度超出已选 sparse 预算，则保留 sparse gather/merge path，把不连续 KV 先整理到 workspace 再供后续阶段消费

这不是普通的 tiling 调节，而是基于可达覆盖度的运行时 dataflow dispatch：两条分支在输入来源、搬运模式、workspace 依赖以及后续流水结构上都显著不同。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// **Mechanism sketch:**
runInfo[mmPingPongIdx].isSmallS2 =
    (curMaxS2 <= actualSelectedBlockCount * selectedBlockSize);

if (runInfo.isSmallS2) {
    DensePath();
} else {
    SparseGatherPath();
}
```

## 关键修改点

1. 预期收益: 当 selected sparse coverage 已足够覆盖可达范围时，跳过不必要 gather/merge，减少 workspace 压力与额外搬运

## 常见陷阱

⚠️ 需要额外维护两套有效的数据流实现，而不仅是同一路径上的参数微调
⚠️ 运行时判定变量若定义不稳，会导致 dispatch 语义含糊，难以复用到其他算子
⚠️ 很容易被误写成“已经通用化的主策略”；当前证据宽度还不足，必须保留 draft 边界

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM" op_kernel/*.cpp op_host/*_tiling.cpp
```
