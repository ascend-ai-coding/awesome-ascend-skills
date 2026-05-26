---
schema_version: 1
op_name: {operator_name}
supported_dtypes: [bfloat16]
p0_shapes: [[1024, 128], [1024, 256], [1024, 512]]
tolerance: {tolerance}
dynamic_axes: {axes_list}
dynamic_axes_ranges: {axes_ranges}
shape_constraints: {shape_constraints}
default_params: {'eps': 1e-4, 'min_v': -128.0, 'max_v': 127.0}
perf_target: {performance_target}
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: {operator_name}
- **算子分类**: {category}  <!-- element-wise / reduction / matmul / attention / custom -->

### 1.1 功能描述

{description}

### 1.2 算法参数

{algorithm_params}  <!-- 如 epsilon, momentum, beta 等超参数，无则填写 "无" -->

### 1.3 数学公式

$$
{formula}
$$

### 2. 关键特性
<!-- 复杂算子必须填写，简单算子可省略 -->

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| {feature_name} | {need_or_not} | {confidence} | {impl_note} | {priority} |

### 3. 算法描述
<!-- 当公式无法完整表述计算流程时填写，简单算子省略此节 -->

```
Algorithm: {algorithm_name}
────────────────────────────────────
{带编号的伪代码步骤，展示循环结构、分块策略、状态更新等流程}
```

### 4. 数据流图

{ASCII数据流图}

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| {name} | {shape} | {dtype} | {dynamic_axes} | {confidence} | {description} |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| {name} | {shape} | {dtype} | {dynamic_axes} | {confidence} | {description} |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 |  | 0.001 | 0.001 | 默认 |

### 7. 精度要求
- **atol**: {atol}
- **rtol**: {rtol}

### 8. 动态轴说明
- **动态轴**: {axes_list}
- **轴含义**: {axes_meanings}
- **取值范围**: {axes_ranges}

### 9. 边界条件处理
- **零值**: {zero_handling}
- **极值**: {inf_handling}
- **NaN/Inf**: {nan_handling}

### 10. 性能要求
- **性能目标**: {performance_target}

### 11. 参考信息
- **参考实现**: {reference_impl}
- **论文**: {paper}
- **类似算子**: {similar_ops}

### 12. 应用场景
- **目标模型**: {model}
- **使用位置**: {layer}

**典型配置**（建议至少提供一个，用于下游 golden 验证和设计方案生成）:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| {config_name} | {type} | {priority} | {params} | {input_shapes} | {output_shapes} | {config_desc} |

---
*生成时间: {timestamp}*
*确认状态: 已确认*
*置信度说明: ✓ 高（自身知识库/框架知识） / ⚠ 中（外部材料提取，需确认）*
