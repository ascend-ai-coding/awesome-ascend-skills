## 算子需求规范

### 1. 基础信息
- **算子名称**: {name}
- **算子分类**: {category}  <!-- element-wise / reduction / matmul / attention / custom -->
- **数学公式**: ${formula}$
- **功能描述**: {description}

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

### 5. 数据规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| {name} | {shape} | {dtype} | {dynamic_axes} | {description} |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 说明 |
|------|-------|-------|--------|------|
| {name} | {shape} | {dtype} | {dynamic_axes} | {description} |

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
