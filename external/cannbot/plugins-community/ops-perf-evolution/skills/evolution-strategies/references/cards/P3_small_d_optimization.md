---
id: P3
bottlenecks: [tiling_imbalance, undersize_transfer]
op_families: [normalization]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P3: Small-D Multi-Row Merging (小D多行合并优化)

## 核心思想
当embedding_dim ≤ 512时，专家实现采用完全不同的算法：索引排序+连续处理。传统方法按原始顺序处理，相同索引分散在不同位置，导致频繁的GM随机访问。小维度优化先将索引排序（使用Sort指令），使相同索引连续，然后合并处理。具体流程：CreateVecIndex生成位置索引[0,1,2,...,n]；Sort对索引值排序，同时重排位置索引；Extract提取排序后的位置；按排序后的顺序处理，相同索引的梯度连续累加。这种优化将随机访问转化为顺序访问，大幅提升内存局部性。embedding_dim=512的阈值选择基于UB容量：512 * sizeof(float) = 2KB，可以在UB中缓存多行。

## 代码骨架

// === 改造前（基线）===
```cpp
// 通用处理，无特殊优化
for (uint32_t rowIdx = 0; rowIdx < this->rowsPerCore; rowIdx++) {
    for (uint32_t tileId = 0; tileId < this->nTiles; tileId++) {
        CopyInPass1(rowIdx, tileId);
        float tileSqSum = ComputePass1();
        rowSqSum += tileSqSum;
    }
}
```

// === 改造后（专家模式）===
```cpp
// SingleN模式极致优化
static constexpr int32_t MAXBUF = 195584;  // (192 - 1) * 1024 byte
Ppipe->InitBuffer(unitBuf, MAXBUF);
LocalTensor<float> ubLocal = unitBuf.Get<float>();
LocalTensor<T> xLocal = ubLocal.template ReinterpretCast<T>();
LocalTensor<T> x1Local = xLocal[0];
LocalTensor<T> x2Local = xLocal[ubFactor];
LocalTensor<float> xFp32Local = ubLocal[ubFactor];
LocalTensor<float> sqxLocal = ubLocal[ubFactor * 2];
LocalTensor<float> tmpLocal = ubLocal[ubFactor * 3];
```

## 关键修改点

1. 预期收益: 对于小batch推理场景，单核性能达到极致，减少GM访问次数

## 常见陷阱

⚠️ 仅适用于特定场景（每核1行），通用性降低
⚠️ 增加了Tiling逻辑的复杂度
⚠️ 排序需要额外的UB空间和计算（O(n log n)），大维度时不划算

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
