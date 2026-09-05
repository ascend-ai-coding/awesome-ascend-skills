---
id: P8
bottlenecks: [bus_contention, mte3_stall, partial_overlap, ub_memory_pressure]
op_families: [datapath]
complexity: L0
conflicts_with: []
synergizes_with: [P1, P20, P34, P35]
has_preconditions: true
has_playbook: true
---

# P8: UB Memory Partitioning (UB内存分区管理)

## 核心思想
针对BF16（BFloat16）数据类型，专家实现采用了高精度中间计算策略。由于BF16只有8位指数和7位尾数，直接计算可能导致精度损失。专家实现通过InnerComputer模板的特化版本，在BF16场景下将数据先Cast到float32进行计算，然后再Cast回bfloat16_t。这种策略的核心优势在于：1) 计算过程使用FP32保证数值稳定性；2) 存储和传输使用BF16节省内存带宽。代价是增加了两次Cast操作的开销和额外的UB内存占用。

## 代码骨架

// === 改造前（基线）===
```cpp
// lingxi-code运行时计算索引
uint32_t d_start = d_out * D_in / D_out;
uint32_t d_end = (d_out + 1) * D_in / D_out;
if ((d_out + 1) * D_in % D_out != 0) d_end += 1;
```

// === 改造后（专家模式）===
```cpp
// 预计算索引到buffer
for (int64_t i = offset, j = 0; i < offset + len; ++i, ++j) {
    OutputOffsetToInputIndex(i, outputShape, inputShape, index);
    startDIndexLocal.SetValue(j, index.dstart);
    // ...
}

// 从buffer获取复用
index.dstart = startDIndexLocal.GetValue(start);
```

## 关键修改点

1. 预期收益: 减少Kernel中复杂的除法/取模运算，预期性能提升5-15%

## 常见陷阱

⚠️ 占用更多UB空间存储索引buffer
⚠️ 需要维护偏移常量；代码可读性降低
⚠️ 增加两次Cast操作开销和额外UB内存占用

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
