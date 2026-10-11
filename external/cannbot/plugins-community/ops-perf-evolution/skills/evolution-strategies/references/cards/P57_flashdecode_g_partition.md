---
id: P57
bottlenecks: [no_overlap]
op_families: [flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P57: FlashDecode G 分核归约 (FlashDecode G-Partition Reduction)

## 核心思想
在 FlashDecode 场景下，当 B 和 N2 都很小而 G 很大时，传统归约方案只使用很少的核数。通过对 G 轴进行分核，实现多核并行归约，充分利用硬件资源。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：仅按 BN2 分核，G 大时核利用率低
uint32_t usedCores = batchSize * kvHeadNum;  // 可能远小于总核数
```

// === 改造后（专家模式）===
```cpp
// G 分核逻辑
if (batchSize * kvHeadNum <= 24) {
    uint32_t coreNumPerGD = 48 / (batchSize * kvHeadNum);
    uint32_t gSizeMin = 16;  // G 最小切分大小，避免过小导致劣化
    uint32_t maxCoreNumPerGD = (gSize + gSizeMin - 1) / gSizeMin;
    coreNumPerGD = (coreNumPerGD > maxCoreNumPerGD) ? maxCoreNumPerGD : coreNumPerGD;
    gSizeSub = (gSize + coreNumPerGD - 1) / coreNumPerGD;

    // 修正核数
    uint32_t coreNumPerGDFix = (gSize + gSizeSub - 1) / gSizeSub;
    coreNumPerGD = (coreNumPerGD > coreNumPerGDFix) ? coreNumPerGDFix : coreNumPerGD;

    if (tmpBlockIdx >= batchSize * kvHeadNum * coreNumPerGD) {
        return;  // 超出分配核数，直接返回
    }

    gSizeTail = gSize - (coreNumPerGD - 1) * gSizeSub;
    gOuter = coreNumPerGD;
    gSizeStart = 0;
    FlashDecodeComputeSplitG();
    return;
}
```

## 关键修改点

1. 预期收益: 充分利用多核资源，提升并行度；G 切分最小 16 为经验最优值

## 常见陷阱

⚠️ 增加归约复杂度，需要多核协作
⚠️ G 切分粒度过小（≤4）可能导致性能劣化
⚠️ 需要额外的 workspace 存储中间结果

## 代码搜索关键词

```bash
grep -n "GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
