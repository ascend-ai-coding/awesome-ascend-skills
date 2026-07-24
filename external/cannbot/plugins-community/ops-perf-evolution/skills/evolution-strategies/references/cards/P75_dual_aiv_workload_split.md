---
id: P75
bottlenecks: [tiling_imbalance]
op_families: [cv_fusion, moe]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P75: 双 AIV M/S1 轴工作量分裂 (Dual AIV M/S1-Axis Workload Split)

## 核心思想
利用 Ascend 910B 的 1:2 AIC:AIV 核比例，将 Vector 阶段的 M 轴或 S1 轴工作量在两个 AIV 核之间分裂。每个 AIC 核的 Cube 计算结果被 2 个 AIV 核并行处理，实现 Cube:Vector = 1:2 的计算比例匹配，充分利用双 AIV 核的 Vector 算力。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：单 AIV 处理全部 M 轴
vecStartM = 0;
vecDealM = mSize;  // 单核处理，Vector 成为瓶颈
```

// === 改造后（专家模式）===
```cpp
// M 轴分裂公式
info.mSizeV = (info.mSize <= 16) ? info.mSize :
    (((info.mSize + 15) / 16 + 1) / 2 * 16);

// AIV 核区分
uint32_t aivIdx = GetBlockIdx() % 2;
if (aivIdx == 0) {
    vecStartM = 0;
    vecDealM = mSizeV;
} else {
    vecStartM = mSizeV;
    vecDealM = mSize - mSizeV;
}
```

## 关键修改点

1. 预期收益: Vector 阶段吞吐量翻倍；两个 AIV 核并行处理不同 M 行，无数据依赖

## 常见陷阱

⚠️ M/S1 <= 16 时无法分裂，退化为单 AIV
⚠️ 分裂边界需要 16 对齐，可能有少量负载不均
⚠️ 两个 AIV 核需要独立的 workspace 区域，增加内存开销

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
