---
id: P45
bottlenecks: [undersize_transfer]
op_families: [attention, flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P45: 稀疏 Attention 离散块处理

## 核心思想
将 topK 选出的离散 block indices 预取到片上 cache 数组中，封装为 L1ChunkIterator 迭代器，按 L1 容量自动切分连续段，使上层调度逻辑与底层离散寻址解耦。完整的离散块处理流程包括：索引缓存与迭代（Variant A）、Gather-Merge 到连续 workspace（Variant B）、块内偏移追踪（Variant C）、PagedAttention 离散块搬运适配（Variant D）。

## 代码骨架

// === 改造后（专家模式）===
```cpp
int32_t blockIndexCache_[TOPK_CACHE_SIZE] = {0};
__aicore__ inline bool Next() {
    while (blockIndexCursor_ < range_->GetSparseBlockCount()) {
        auto [blockStart, blockEnd] = range_->GetBlockRange(blockIndexCursor_, ...);
        if (currentL1Size_ + (chunkEnd - blockCursor_) >= range_->GetL1Size()) {
            l1ChunkJustCompleted_ = true;
        }
        currentSegment_ = {blockCursor_, chunkEnd, l1ChunkJustCompleted_};
        blockCursor_ = chunkEnd;
        return true;
    }
}
```

## 关键修改点

1. 预期收益: 调度逻辑与离散寻址解耦，上层代码简洁且可复用

## 常见陷阱

⚠️ TOPK_CACHE_SIZE=128 限制可缓存数量；迭代器逻辑增加标量计算开销
⚠️ 每个 chunk 边界需要重新计算 GM 偏移，chunk 过小时标量开销占比增大
⚠️ Gather-Merge 需要额外 workspace GM 空间，V0 阶段增加核间同步开销

## 代码搜索关键词

```bash
grep -n "GetBlockNum\|coreNum\|blockIdx\|SplitCore\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
