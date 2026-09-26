---
id: D4
bottlenecks: [ub_memory_pressure]
op_families: [quantization]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# D4: FP8/INT4 Quantization Conversion (FP8/INT4量化输出类型)

## 核心思想
lingxi-code 使用固定的 CAST_ROUND 模式进行类型转换，这可能在某些场景下引入系统性偏差。专家实现支持多种 rounding mode：MODE_RINT (默认，四舍五入到最近偶数/Banker's rounding)、MODE_ROUND (标准四舍五入)、MODE_FLOOR (向下取整)、MODE_TRUNC (向零取整)、MODE_CEIL (向上取整)。不同场景需要不同的 rounding mode：训练量化通常使用 rint 以减少累积误差、推理量化可能需要 floor 或 trunc 以满足特定硬件要求、FP8 输出强制使用 MODE_RINT 以符合标准。

## 代码骨架

// === 改造前（基线）===
```cpp
this->Input("x1").DataType({ge::DT_FLOAT16});
this->Output("output").DataType({ge::DT_INT8});
```

// === 改造后（专家模式）===
```cpp
static const std::vector<ge::DataType> xDataType91095 = {
    ge::DT_FLOAT16, ge::DT_BF16, ...
};
static const std::vector<ge::DataType> yDataType91095 = {
    ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8
};
if constexpr (is_same<T, half>::value) {
    Cast(xOut, xLocalFp32, RoundMode::CAST_NONE, elementCount);
} else { // BF16
    Cast(xOut, xLocalFp32, RoundMode::CAST_RINT, elementCount);
}
```

## 关键修改点

1. 预期收益: 支持多种数据类型组合，适配不同硬件和场景需求，避免算子碎片化

## 常见陷阱

⚠️ 增加编译时间和代码复杂度，需要更多测试覆盖
⚠️ 代码复杂度增加，需要维护多个模板类实现
⚠️ 需要额外的条件判断和地址计算

## 代码搜索关键词

```bash
grep -n "tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum\|coreNum\|blockIdx\|SplitCore" op_kernel/*.cpp op_host/*_tiling.cpp
```
