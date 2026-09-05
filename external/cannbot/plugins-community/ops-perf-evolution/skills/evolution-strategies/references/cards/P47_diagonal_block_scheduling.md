---
id: P47
bottlenecks: [compute_bound, tiling_imbalance]
op_families: [cv_fusion, matmul]
complexity: L2
conflicts_with: []
synergizes_with: [P46, P51, P63]
has_preconditions: true
has_playbook: true
---

# P47: 对角线块调度 (Diagonal Block Scheduling)

## 核心思想
用 MNBlockIdxCompute 对角线分配替代线性行优先分块，优化多核场景下的 L2 cache 复用和负载均衡。当 blockDimM > 5 时启用对角线映射，小 shape 自动回退到线性调度。

## 代码骨架

// === 改造前（基线）===
```cpp
mIdx = (curBlock - count) / blockDimN;
nIdx = (curBlock - count) % blockDimN;
// 同列块被不同核加载 → L2 cache 抖动
```

// === 改造后（专家模式）===
```cpp
void MNBlockIdxCompute(uint32_t curBlock, ...) {
    if (blockDimM > thresholdDimM) {  // thresholdDimM = 5
        uint32_t thresholdBlockNum = 8;
        uint32_t curThresholdM = min(blockDimM, thresholdBlockNum);
        uint32_t curThresholdN = min(blockDimN, thresholdBlockNum);
        uint32_t thresholdM_dimN = curThresholdM * blockDimN;
        uint32_t curThresholdM_thresholdN = curThresholdM * curThresholdN;
        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN))
               % curThresholdN + relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * curThresholdN;
    } else {
        // 线性回退
        mIdx = (curBlock - count) / blockDimN;
        nIdx = (curBlock - count) % blockDimN;
    }
}
```

## 关键修改点

1. 预期收益: L2 cache 命中率提升 10-20%，负载均衡改善 10-30%

## 常见陷阱

⚠️ 对角线映射增加少量地址计算开销（LCM 计算）
⚠️ 小 shape（blockDimM ≤ 5）无收益，需 threshold 控制
⚠️ 与 P51（动态核配比）互补：P47 优化块分配，P51 优化核分配

## 代码搜索关键词

```bash
grep -n "MNBlockIdxCompute" op_kernel/*.cpp op_host/*_tiling.cpp
```
