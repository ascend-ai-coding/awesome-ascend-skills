---
id: P58
bottlenecks: [tiling_imbalance]
op_families: [flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P58: TND 负载均衡分核 (TND Load-Balanced Partition)

## 核心思想
在 TND（T=B*S 合轴）场景下，通过线段抽象和贪心算法实现跨核负载均衡分配。将变长序列抽象为线段，按线段总长度均分到各核，避免传统 BS 切分方式导致的负载不均衡问题。支持 FlashDecode 归约机制处理跨核切分。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：按 BS 切分，不考虑实际序列长度
for (uint32_t b = 0; b < batchSize; b++) {
    assignToCore(b, coreIdx);  // 短序列和长序列可能分到同一核
}
```

// === 改造后（专家模式）===
```cpp
// 线段抽象与均分
// S1方向切分：QS * G <= 128
// S2方向切分：KVS = 512 为基本单位

// 计算线段总长度
uint32_t totalS2Length = 0;
for (uint32_t b = 0; b < batchSize; b++) {
    totalS2Length += (actualKVS[b] + 511) / 512;  // 向上取整
}

// 平均每个核需要计算的线段数目
avgS2Length = (totalS2Length + coreNum_ - 1) / coreNum_;  // 向上取整

// 分核时，记录跨核切分信息用于 FlashDecode 归约
for (each segment) {
    if (segment crosses core boundary) {
        recordSplitInfo(b, s1Outer, kvSplitNum);
    }
}
```

## 关键修改点

1. 预期收益: 实现真正的负载均衡，避免长序列导致的核间负载差异；支持变长序列的高效并行

## 常见陷阱

⚠️ 需要额外的 workspace 存储规约信息（lseMax/lseSum/accumOut）
⚠️ 线段被跨核切分时需要 FlashDecode 归约
⚠️ tiling 逻辑复杂度增加

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
