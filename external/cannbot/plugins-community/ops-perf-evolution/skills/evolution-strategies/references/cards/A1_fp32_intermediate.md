---
id: A1
bottlenecks: [compute_bound]
op_families: [elementwise, moe, normalization, optimizer]
complexity: L0
conflicts_with: []
synergizes_with: [A2, A3, A5, A6]
has_preconditions: true
has_playbook: true
---

# A1: FP32 Intermediate Computation (FP32中间计算)

## 核心思想
专家实现在DAG中显式地进行了类型转换：输入数据类型U (BF16/FP16) 被转换为T (FP32) 后进行所有计算，最后转换回U输出。这种设计确保了累加精度（AdaGrad中accum的累加操作在FP32上进行，避免BF16/FP16的精度损失）、除法精度（sqrt(accum)和除法操作在FP32上进行，减少数值误差）、梯度精度（梯度平方和乘法在更高精度下进行）。lingxi-code实现全程使用float32，虽然精度足够但失去了BF16/FP16的内存带宽优势。

## 代码骨架

// === 改造前（基线）===
```cpp
// lingxi-code仅FP32，无需转换
AscendC::Add(accumLocal, accumLocal, inputLocal, C);
```

// === 改造后（专家模式）===
```cpp
// 升精度累加
if constexpr (std::is_same_v<T, float>) {
    Add(sumBufLocal, sumBufLocal, inputLocal, len);
} else {
    LocalTensor<float> castBufLocal = castBuf.Get<float>();
    Cast(castBufLocal, inputLocal, RoundMode::CAST_NONE, len);
    Add(sumBufLocal, sumBufLocal, castBufLocal, len);
}

// 降精度输出
Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
```

## 关键修改点

1. 预期收益: 有效避免FP16/BF16多次累加产生的精度损失，数值稳定性提升

## 常见陷阱

⚠️ 需要额外的castBuf和sumBuf(FP32)存储空间
⚠️ 需要在Tiling阶段计算每个输出点的kernel大小
⚠️ 增加了类型转换操作的开销，但通常远小于内存带宽节省的收益

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
