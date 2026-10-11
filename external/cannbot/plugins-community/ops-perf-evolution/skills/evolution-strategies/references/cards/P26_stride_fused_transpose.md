---
id: P26
bottlenecks: [partial_overlap]
op_families: [matmul, flash_attention, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P26: Stride 搬运模式（转置与列提取）

## 核心思想
通过 srcStride=0（连读）+ dstStride=跳步（跳写）实现零拷贝转置，将 UB 上连续数据写回 GM 上不连续布局。反向地，通过 blockCount/blockLen/srcStride 参数还可实现非连续列的提取搬运，一次 DataCopyPad 搬运多行，每行跳过无关列（见 Variant B）。

## 代码骨架

// === 改造后（专家模式）===
```cpp
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = gCountOneS1;
dataCopyParams.blockLen = headDim * sizeof(OUT_T);
dataCopyParams.srcStride = 0;                                     // 连读
dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T); // 跳写
DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
```

## 关键修改点

1. 预期收益: 零额外拷贝完成布局转换，节省 UB 空间和搬运带宽

## 常见陷阱

⚠️ DataCopyParams stride 为 uint16_t 最大 65535；超限需切换到 DataCopyExtParams
⚠️ 仅适用于单维度转置，多维度复杂重排需要组合多次 stride 搬运或使用 Gather
⚠️ 列提取时 srcStride 和 blockLen 必须满足 32B 对齐，每次提取一个子字段需多次搬运调用

## 代码搜索关键词

```bash
grep -n "DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
