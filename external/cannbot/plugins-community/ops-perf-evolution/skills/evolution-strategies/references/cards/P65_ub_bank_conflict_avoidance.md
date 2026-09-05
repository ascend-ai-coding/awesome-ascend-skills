---
id: P65
bottlenecks: [bus_contention]
op_families: [datapath]
complexity: L2
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P65: UB Bank 冲突规避 (UB Bank Conflict Avoidance)

## 核心思想
UB（192KB）划分为 48 个 bank（16 个 bank group，每组 3 个 bank），每 bank 4KB。同一 bank group 的并发读写会导致 bank 冲突，使单 Repeat 从 1 拍退化到 8 拍。通过优化计算逻辑或地址分配规避冲突。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// 反例：跳读连续写，blk_stride=16 导致 8 个 DataBlock 在同一 bank group
Adds(dst, src, scalar, MASK_PLACEHOLDER, 1, {1, 16, 1, 16});

// 正例：连续读跳写，读操作连续不冲突
Adds(dst, src, scalar, MASK_PLACEHOLDER, 1, {16, 1, 16, 1});
```

## 关键修改点

1. 预期收益: 消除 bank 冲突，单 Repeat 从 8 拍降至 1-2 拍

## 常见陷阱

⚠️ 地址优化方案需多申请 UB 空间（如 256 字节 padding）
⚠️ 计算逻辑优化可能增加代码复杂度
⚠️ 需要理解 UB bank 结构

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue" op_kernel/*.cpp op_host/*_tiling.cpp
```
