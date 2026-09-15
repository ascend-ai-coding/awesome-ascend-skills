---
id: P9
bottlenecks: [tiling_imbalance]
op_families: [optimizer]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P9: Deterministic Output via Workspace (确定性输出)

## 核心思想
在多核并行场景下，如果多个核同时更新同一权重行，由于执行顺序不确定，最终结果虽然是正确的（原子加），但浮点累加顺序不同可能导致微小差异（非结合性）。这在分布式训练中会导致不同节点的梯度不一致，影响收敛。确定性模式通过反向处理和边界检测解决这一问题：每个核从后向前处理自己的数据块；检测与前一个核的边界索引是否相同；如果相同，当前核跳过该索引，由前一个核处理；通过SyncAll保证全局顺序。

## 代码骨架

// === 改造后（专家模式）===
```cpp
if (this->oldDouble || (this->outQuant2Flag == 1 || this->outQuant1Flag == 1)) {
    workspaceGm.SetGlobalBuffer((__gm__ float*)(workspace) + 2 * this->blockIdx_ * this->numLastDim);
}
CopyOutSmoothNorm(yLocalFp32, 0, rowGmOffset, elementCount);  // 写入smooth1结果
CopyOutSmoothNorm(zLocalFp32, this->numLastDim, rowGmOffset, elementCount);  // 写入smooth2结果
CopyInSmoothNorm(xLocalFp32, 0, rowGmOffset, elementCount, this->localMax1);  // 读取
```

## 关键修改点

1. 预期收益: 双输出量化性能接近单输出的2倍，而非简单重复的2倍开销

## 常见陷阱

⚠️ 增加GM访存，需要额外的Workspace内存
⚠️ 约5-10%的性能损失（边界检测和反向遍历的开销）
⚠️ 原子操作有一定性能开销，但这是正确性必需的

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|SyncAll" op_kernel/*.cpp op_host/*_tiling.cpp
```
