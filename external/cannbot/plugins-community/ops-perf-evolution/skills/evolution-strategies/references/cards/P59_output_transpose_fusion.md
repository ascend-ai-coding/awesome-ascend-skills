---
id: P59
bottlenecks: [mte2_stall, mte3_stall]
op_families: [pooling_gather]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P59: 输出 Transpose 融合 (Output Transpose Fusion)

## 核心思想
在 IFA 等算子输出阶段，通过 DataCopy stride 参数实现输出格式转换（Transpose），避免额外的转置算子调用。支持 BNSD→NBSD、BSND→NBSD、BSH→NBSD、TND→NTD 等多种布局转换。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：输出后调用 Transpose 算子
DataCopy(outputGm, attenOutUb, size);  // BNSD 格式
Transpose(outputGm, tempGm, ...);      // 额外转置操作
```

// === 改造后（专家模式）===
```cpp
// BNSD → NBSD 转置
// attenOutUb: gCount * S1 * D（连续）
// 输出: N * B * S * D（N 轴在外层）

DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCount;  // 处理多少个 G
dataCopyParams.blockLen = s1Size * headDim * sizeof(OUT_T) / 32U;  // 一个 S1*D
dataCopyParams.srcStride = 0;  // 连读
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;  // 跳写

uint64_t attenOutOffset = n2Idx * gSize * batchSize * qSeqSize * headDim +  // N2轴
                          gStartIdx * batchSize * qSeqSize * headDim +       // G轴
                          bIdx * qSeqSize * headDim;                         // B轴
DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParams);
```

## 关键修改点

1. 预期收益: 消除 Transpose 算子调用，减少 kernel 启动开销和数据搬运

## 常见陷阱

⚠️ DataCopy stride 模式效率低于连续搬运
⚠️ 需要精确计算各轴偏移量
⚠️ 头块和尾块需要单独处理

## 代码搜索关键词

```bash
grep -n "DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
