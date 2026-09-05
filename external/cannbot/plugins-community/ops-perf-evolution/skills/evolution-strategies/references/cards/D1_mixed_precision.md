---
id: D1
bottlenecks: [compute_bound]
op_families: [datapath]
complexity: L0
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# D1: Mixed Precision Architecture (混合精度架构)

## 核心思想
lingxi-code 实现仅支持 float32 输入类型，这在实际 AI 推理场景中是不足的。现代大语言模型和推理框架通常使用 FP16 或 BF16 进行计算以减少内存带宽和存储需求。专家实现通过模板参数 typename T 支持 half (FP16) 和 bfloat16_t (BF16) 两种输入类型。关键技术细节包括：类型特化的计算精度（当输入为 half 时，计算类型 calcType 提升为 float，避免 FP16 精度损失；当输入为 bfloat16_t 时，直接以 BF16 计算）、条件类型推导使用 std::conditional 实现编译期类型选择、MicroAPI 自动处理中 FP16 输入会自动通过 Cast 转换为 FP32 进行计算而 BF16 保持原格式。

## 代码骨架

// === 改造前（基线）===
```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

// === 改造后（专家模式）===
```cpp
this->Input("x")
    .ParamType(DYNAMIC)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
this->Input("scalar")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

## 关键修改点

1. 预期收益: 支持更广泛的业务场景，减少数据类型转换开销，提高端到端性能; 覆盖更广的精度需求，支持2-4倍内存带宽优化，适应不同硬件平台和精度要求; 一个算子支持多种数据类...

## 常见陷阱

⚠️ 代码复杂度增加，需要维护多类型模板实例
⚠️ 模板代码复杂度增加，编译时间可能增加
⚠️ 需要在运行时检查属性，略微增加开销

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
