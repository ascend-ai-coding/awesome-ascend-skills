---
id: D3
bottlenecks: []
op_families: [matmul, moe, quantization, special]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# D3: TilingKey-Driven Type Dispatch (TilingKey驱动类型分发)

## 核心思想
专家实现通过模板类FakeQuantAffineCachemaskFp32和FakeQuantAffineCachemaskFp16实现了对FP16和FP32两种数据类型的完整支持。这种设计允许根据输入数据类型自动选择最优的计算路径，充分利用昇腾芯片对不同精度计算的硬件支持。在算子定义文件中，通过配置多组DataType实现了运行时数据类型的灵活选择。Tiling阶段通过SetTilingKeyMode函数根据数据类型设置不同的Tiling Key（FP16_MODE=2, FP32_MODE=1），从而在kernel入口处分发到对应的模板实例。这种策略的优势在于：1) 避免了运行时的类型判断开销；2) 允许针对每种数据类型进行专门的优化；3) 通过编译期多态实现零开销抽象。值得注意的是，FP16和FP32的实现细节有所不同，例如FP16实现中使用了更多的Cast操作进行精度转换，而FP32实现则尽量保持原生精度计算。

## 代码骨架

// === 改造前（基线）===
```cpp
// lingxi-code仅支持FP32
this->Input("x")
    .DataType({ge::DT_FLOAT});
```

// === 改造后（专家模式）===
```cpp
// 三种数据类型支持
this->Input("x")
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16});

// 差异化RoundMode处理
if constexpr (std::is_same_v<T, float>) {
    DataCopy(outputLocal, sumBufLocal, len);
} else if constexpr (std::is_same_v<T, half>) {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, len);
} else {
    Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
}
```

## 关键修改点

1. 预期收益: 支持FP16/BF16/FP32三种精度，通过低精度计算获得2x性能提升，同时保持数值稳定性

## 常见陷阱

⚠️ 代码复杂度增加，需要模板编程和编译期分支处理
⚠️ Tiling逻辑复杂度增加
⚠️ 增加代码量和维护成本

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
