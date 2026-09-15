---
id: P76
bottlenecks: [mte2_stall, undersize_transfer]
op_families: [flash_attention]
complexity: L2
conflicts_with: []
synergizes_with: [P18, P60]
has_preconditions: true
has_playbook: true
---

# P76: V_TEMPLATE KV 预合并 Workspace (V_TEMPLATE KV Pre-Merge Workspace)

## 核心思想
在 Sparse Attention 的 V_TEMPLATE 模式下，由 Vector 核在 Cube 计算之前将分散在 PageAttention block table 中的稀疏 KV 数据预合并到连续的 GM workspace 中。Cube 核直接从连续 workspace 读取，避免处理不连续 PA 地址的复杂性。通过 sparseBlockSize 阈值（<=4）自动选择 V_TEMPLATE 或 C_TEMPLATE 路径。

## 代码骨架

// === 改造前（基线）===
```cpp
// C_TEMPLATE: Cube 核直接从 GM 随机读取
for (int i = 0; i < sparseBlockCount; i++) {
    int realS2Idx = CalcTopKBlockInfo(topKIndices, i);
    LoadData(l1Tensor, kvGm[realS2Idx * headDim], loadParams);
    // 每个 sparse block 一次随机 DMA
}
```

// === 改造后（专家模式）===
```cpp
// V_TEMPLATE: Vector0 阶段 MergeKv
void MergeKv(const ExtraInfo& info) {
    for (int i = 0; i < sparseBlockCount; i++) {
        int realS2Idx = GetRealS2Idx(topKIndices, i);
        // 从 PA block table 中 gather KV 数据
        DataCopyPad(ubBuf, kvGm[realS2Idx * headDim], copyParams);
        // ND→NZ 格式转换
        TransDataTo5HD(nzBuf, ubBuf, transParams);
        // 写入连续 workspace
        DataCopy(kvMergeGm[mergeOffset], nzBuf, outParams);
        mergeOffset += blockSize * headDim;
    }
}
// Cube 阶段直接从 kvMergeGm 读取连续数据
LoadData(l1Tensor, kvMergeGm[offset], loadParams);
```

## 关键修改点

1. 预期收益: Cube 核只需顺序读取连续数据；ND→NZ 转换在 Vector 阶段完成，与 Cube 计算重叠

## 常见陷阱

⚠️ 需要额外的 4*512*576*sizeof(half) ≈ 2.25MB workspace 空间
⚠️ Vector0 阶段增加了额外的搬运开销和 V0-C1 同步
⚠️ sparseBlockSize 较大时 C_TEMPLATE 更优（无需预合并）

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|DataCopy\|CopyIn\|CopyOut\|Fixpipe\|MergeKv" op_kernel/*.cpp op_host/*_tiling.cpp
```
