---
id: P56
bottlenecks: [mte2_stall, mte3_stall]
op_families: [pooling_gather]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P56: DataCopy UB2GM 合并输出 (Batch DataCopy Output)

## 核心思想
将多次小批量 DataCopy UB2GM 操作合并为一次大批量写回，减少 MTE3 指令发射次数，提升写回效率。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：循环内多次小 DataCopy
for (int i = 0; i < loopCount; i++) {
    Compute();
    DataCopy(gm[offset], ub, size);  // 每次 16 个元素
}
```

// === 改造后（专家模式）===
```cpp
// 原始方案：循环内每次都 DataCopy
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1Inner(const ExtraInfoMla &info) {
    for (uint32_t i = 0; i < loopCount; i++) {
        DealBmm1ResBaseBlock(info, ...);  // 包含 DataCopy
    }
}

// 优化方案：循环外统一 DataCopy
template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1Inner(const ExtraInfoMla &info) {
    LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();

    // 循环内累积，不写回
    for (uint32_t i = 0; i < loopCount; i++) {
        DealBmm1ResBaseBlock(info, nInt32Out, ...);  // 不包含 DataCopy
    }

    // 循环外统一写回
    outputQue2.EnQue(nInt32Out);
    outputQue2.DeQue<int32_t>();
    uint32_t dealRowCount = (loopCount - 1) * gSplitSize + tailSplitSize;
    DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);  // 一次性写回
    outputQue2.FreeTensor(nInt32Out);
}
```

## 关键修改点

1. 预期收益: 减少 MTE3 指令发射次数，从 loopCount 次降为 1 次，显著提升写回效率

## 常见陷阱

⚠️ 需要额外的 UB 空间累积结果
⚠️ 增加代码复杂度，需要管理累积计数
⚠️ 首次迭代可能有额外判断逻辑

## 代码搜索关键词

```bash
grep -n "SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
