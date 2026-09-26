---
id: P72
bottlenecks: [tiling_imbalance]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P71]
has_preconditions: true
has_playbook: true
---

# P72: Matmul 多核切 K (Matmul Multi-Core Split-K)

## 核心思想
当矩阵 M 和 N 较小无法有效切分多核时，可沿 K 轴切分实现多核并行。各核计算部分 K 的矩阵乘结果，最终通过 AtomicAdd 在 GM 上累加合并。需在首次写入 GM 前对 C 矩阵清零。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Tiling 侧：使能多核切 K
cubeTiling.SetOrgShape(M, N, K);
cubeTiling.SetShape(M, N, K);
cubeTiling.EnableMultiCoreSplitK(true);
cubeTiling.GetTiling(tilingData);

// Kernel 侧：GM 清零 + AtomicAdd
cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c), tiling.M * tiling.N);
Fill(cGlobal, tiling.M * tiling.N, (cType)0);  // GM 清零

uint8_t enAtomic = 1;
matmulObj.IterateAll(cGlobal, enAtomic);  // AtomicAdd 累加
```

## 关键修改点

1. 预期收益: aic_time 从 19.60us 降至 13.70us（单核→双核并行，提升 30%）

## 常见陷阱

⚠️ 需要 GM 清零 + AtomicAdd，有额外开销
⚠️ 不支持 Bias 参与矩阵乘
⚠️ 仅支持输出到 GM

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
