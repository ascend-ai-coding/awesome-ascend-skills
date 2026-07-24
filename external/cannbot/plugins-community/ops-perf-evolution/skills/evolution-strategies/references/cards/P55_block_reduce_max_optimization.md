---
id: P55
bottlenecks: [compute_bound, scalar_compute]
op_families: [attention, normalization]
complexity: L1
conflicts_with: []
synergizes_with: [P67]
has_preconditions: true
has_playbook: true
---

# P55: BlockReduceMax 替代 DataCopy slice (Block-level Reduction Optimization)

## 核心思想
当需要从 UB 中按固定间隔提取元素（如每 8 个元素取 1 个）并写回 GM 时，使用 BlockReduceMax 指令替代 DataCopy slice 操作，在 UB 内完成归约后再写回，提升搬运效率。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：DataCopy slice 效率低
DataCopyPad(nUpdateGm[...], nInt32Out, dataCopyParams);  // stride 模式
```

// === 改造后（专家模式）===
```cpp
// 原始方案：DataCopy stride slice，效率低
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = dealRowCount;  // 16
dataCopyParams.blockLen = 1 * sizeof(int32_t);  // 4
dataCopyParams.srcStride = (8 - 1) / (BYTE_BLOCK / sizeof(int32_t));  // stride=0 实际 32 字节
dataCopyParams.dstStride = 0;
DataCopyPad(nUpdateGm[...], nInt32Out, dataCopyParams);

// 优化方案：BlockReduceMax + 连续 DataCopy
LocalTensor<int32_t> nInt32Out = outputQue2.template AllocTensor<int32_t>();
int32_t repeatTime = (dealRowCount * BLOCK_ELEMENT_NUM) / FP28_MAX_MASK_ELEMENT_NUM;  // =2
int32_t srcRepStride = FP28_MAX_MASK_ELEMENT_NUM / BLOCK_ELEMENT_NUM;  // =8

// BlockReduceMax：每个 datablock 内取最大值，实现 slice 效果
BlockReduceMax(nUpdateTmp, nUpdateTmp, repeatTime, FP28_MAX_MASK_ELEMENT_NUM, 1, 1, srcRepStride);
pipe_barrier(PIPE_V);

// 后续处理
Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
pipe_barrier(PIPE_V);
ShiftLeft(nInt32Out, nInt32Out, (int32_t)23, dealRowCount);
pipe_barrier(PIPE_V);

// 连续 DataCopy，效率更高
DataCopy(nUpdateGm[...], nInt32Out, dealRowCount);
```

## 关键修改点

1. 预期收益: BlockReduceMax 在 UB 内完成归约，后续 DataCopy 为连续搬运，效率显著提升

## 常见陷阱

⚠️ 需要额外的 UB 空间存放归约结果
⚠️ 仅支持 half/float 数据类型
⚠️ 需要精确计算 repeat、mask、stride 参数

## 代码搜索关键词

```bash
grep -n "DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
