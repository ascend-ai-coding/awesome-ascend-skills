---
id: R3
bottlenecks: [compute_bound, no_overlap]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Uses fictional APIs. Dual-issue is compiler/hardware behavior, not a manually applicable strategy."
---

# R3: 指令双发射

## 核心思想
通过循环展开和指令重排，暴露无数据依赖的指令对，最大化 A5 Vector Core 的双发射能力。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Baseline
for (uint16_t i = 0; i < repTimes; ++i) {
    mask = UpdateMask<T>(count);
    LoadAlign(r0, src + i * repSize);
    Adds(r1, r0, scalar, mask);
    StoreAlign(dst + i * repSize, r1, mask);
}

// Evolved: 2x 展开，交错无依赖指令
uint16_t mainLoop = repTimes / 2;
uint16_t rem = repTimes % 2;
for (uint16_t i = 0; i < mainLoop; ++i) {
    uint16_t i0 = i * 2, i1 = i * 2 + 1;
    m0 = UpdateMask<T>(count);
    m1 = UpdateMask<T>(count);
    LoadAlign(r0, src + i0 * repSize);
    LoadAlign(r2, src + i1 * repSize);     // 双发射
    Adds(r1, r0, scalar, m0);
    Adds(r3, r2, scalar, m1);              // 双发射
    StoreAlign(dst + i0 * repSize, r1, m0);
// ... (truncated)
```

## 关键修改点

1. 预期收益: **: Load/Compute/Store 均可双发射
**Trade-off**: RegTensor 从 2 增加到 4，MaskReg 从 1 增加到 ...

## 常见陷阱

⚠️ 收益**: 理论 2x 吞吐提升（实际 1.3-1.8x）
⚠️ 风险**: RegTensor 使用量翻倍（2x 展开 → ~2x RegTensor）
⚠️ 复杂度**: 需处理尾部余数

## 代码搜索关键词

```bash
grep -n "GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
