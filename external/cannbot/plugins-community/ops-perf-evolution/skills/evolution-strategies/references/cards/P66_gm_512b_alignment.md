---
id: P66
bottlenecks: [mte2_stall, undersize_transfer]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P66: GM 地址 512B 对齐优化带宽 (GM 512B Alignment Bandwidth)

## 核心思想
AI 处理器内部设计约束下，GM 地址 512B 对齐可最有效发挥带宽效率。实测数据显示 32B 对齐场景带宽最差时仅为 512B 对齐的 70%。Kernel 入参（包括 Workspace/Tiling）地址已保证 512B 对齐，开发者需关注偏移量是否保持 512B 对齐。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// 确保偏移量保持 512B 对齐
uint32_t offset = AlignUp(rawOffset, 512);
DataCopy(ubTensor, gmTensor[offset], dataSize);

// 实测带宽对比（GM→UB）：
// 512B 对齐: 100% 带宽效率
// 256B 对齐: ~90% 带宽效率
// 32B 对齐:  ~70% 带宽效率（最差情况）
```

## 关键修改点

1. 预期收益: 带宽效率提升最高 30%（32B 对齐→512B 对齐）

## 常见陷阱

⚠️ 可能需要调整数据布局以保持 512B 对齐
⚠️ 与 P7（32B 对齐）互补：P7 关注最低对齐要求，本策略关注最优带宽对齐

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
