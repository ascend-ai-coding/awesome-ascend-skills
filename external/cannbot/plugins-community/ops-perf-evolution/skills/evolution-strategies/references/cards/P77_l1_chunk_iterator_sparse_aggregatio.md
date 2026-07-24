---
id: P77
bottlenecks: [no_overlap]
op_families: [flash_attention]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P77: L1 Chunk Iterator 稀疏块自适应聚合 (L1 Chunk Iterator Sparse Block Adaptive Aggregation)

## 核心思想
在 Sparse Attention 推理场景中，多个不连续的 sparse block 可能无法单独填满一个 L1 buffer。L1ChunkIterator 将多个 sparse block 动态聚合为一个 L1 chunk，使每个 chunk 恰好填满一个 L1 KP buffer（64KB），最大化 L1 利用率。支持 FastAdvance 快速跳过 causal mask 裁剪后的无效区域。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：每个 sparse block 独立搬运到 L1
for (int i = 0; i < sparseBlockCount; i++) {
    LoadData(l1Tensor, kvGm[sparseIdx[i] * headDim], params);
    // 每个 block 一次 L1 搬运，L1 利用率低
}
```

// === 改造后（专家模式）===
```cpp
// L1ChunkIterator: 动态聚合多个 sparse block 到一个 L1 chunk
class L1ChunkIterator {
    int64_t Advance() {
        chunkStartIdx_ = chunkEndIdx_;
        int64_t accumulated = 0;
        while (chunkEndIdx_ < sparseBlockCount_) {
            int64_t blockSize = GetBlockSize(chunkEndIdx_);
            if (accumulated + blockSize > l1BufferCapacity_) break;
            accumulated += blockSize;
            chunkEndIdx_++;
        }
        return accumulated;  // 当前 chunk 的总大小
    }
};
```

## 关键修改点

1. 预期收益: L1 利用率最大化；自动适应不同 sparse 配置

## 常见陷阱

⚠️ 迭代器状态管理增加标量计算开销
⚠️ 不连续 sparse block 的 DMA 搬运可能产生碎片
⚠️ sparseBlockSize 较大时（>= L1 buffer 容量）退化为单块迭代

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue" op_kernel/*.cpp op_host/*_tiling.cpp
```
