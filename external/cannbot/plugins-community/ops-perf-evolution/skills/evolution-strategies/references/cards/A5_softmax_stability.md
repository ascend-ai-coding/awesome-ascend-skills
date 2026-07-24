---
id: A5
bottlenecks: []
op_families: [attention, broadcast_mask, flash_attention]
complexity: L0
conflicts_with: []
synergizes_with: [A1, A3]
has_preconditions: true
has_playbook: true
---

# A5: Numerical Safety & Special Value Handling (数值安全与特殊值处理)

## 核心思想
专家实现在计算输出值时，通过Compare+Select组合实现了对NaN（Not a Number）的特殊处理。具体逻辑是：Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount) - 如果x是NaN，则x != x，比较结果为false；然后通过Select(curTemp, maskTemp, curTemp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, ...)，当mask为false（即x为NaN）时，输出0.0f。这种处理确保了量化结果在遇到NaN输入时不会传播NaN，而是输出一个确定值（0.0f）。这对于神经网络的量化训练尤为重要，因为NaN的传播可能导致训练不稳定。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基础形状检查，无索引范围检查
```

// === 改造后（专家模式）===
```cpp
OP_CHECK_IF((xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM),
    OP_LOGE(context_->GetNodeName(), "Input dim num should equal = %lu", NCDHW_DIM_NUM),
    return false);

OP_CHECK_IF(maxPoolGradParams.diDim * maxPoolGradParams.hiDim * maxPoolGradParams.wiDim > MAX_INT32,
    OP_LOGE(context_->GetNodeName(), "Shape too big"),
    return ge::GRAPH_FAILED);

maxPoolGradParams.dGcd = Gcd(maxPoolGradParams.doDim, maxPoolGradParams.diDim);
```

## 关键修改点

1. 预期收益: 保证极端输入下的数值稳定性和正确性

## 常见陷阱

⚠️ 增加tiling阶段的开销
⚠️ 需要更多的UB内存用于FP32缓冲区
⚠️ 增加UB内存占用(4字节 vs 2字节)

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
