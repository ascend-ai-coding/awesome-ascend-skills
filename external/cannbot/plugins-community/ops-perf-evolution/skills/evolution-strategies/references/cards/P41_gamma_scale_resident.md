---
id: P41
bottlenecks: [mte2_stall]
op_families: [normalization]
complexity: L1
conflicts_with: []
synergizes_with: [P34]
has_preconditions: true
has_playbook: true
---

# P41: gamma/scale/offset 常驻 UB

## 核心思想
gamma（RmsNorm 权重）、quantization scale/offset 等小张量在整个 kernel 生命周期内只搬入一次，常驻 UB 供所有 ubLoop 迭代复用。利用 ReinterpretCast 在同一块 buffer 中先存 half 再原地 Cast 为 fp32。

## 代码骨架

// === 改造后（专家模式）===
```cpp
LocalTensor<float> gammaLocalFp32 = inQueueGamma.AllocTensor<float>();
LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.ReinterpretCast<KV_DTYPE>()[RMS_NORM_LENGTH];
DataCopyPad(gammaLocal, gammaGm, copyParams, padParams);
Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, RMS_NORM_LENGTH);
// gammaLocalFp32 在所有 ubLoop 中复用
```

## 关键修改点

1. 预期收益: 小张量只搬运一次，消除重复 DMA 开销

## 常见陷阱

⚠️ 常驻 buffer 占用约 2KB UB 空间
⚠️ ReinterpretCast 原地 Cast 技巧要求 float buffer 后半段空间足够存放 half 数据

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
