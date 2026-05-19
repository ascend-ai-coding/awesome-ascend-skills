# {operator_name} 算子设计文档

> **算子名称**: {operator_name}
> **算子分类**: {category}
> **生成时间**: {timestamp}
> **基于**: SPEC.md

---

## 1. 概述

### 1.1 功能描述

{description}

### 1.2 数学公式

${formula}$

### 1.3 算法描述

<!-- 简单算子省略此节；复杂算子（涉及分块、循环、在线更新等）填写 -->

```
Algorithm: {algorithm_name}
────────────────────────────────────
{算法伪代码步骤}
```

### 1.4 数据流图

```
{ASCII数据流图，从 SPEC.md §3 复制}
```

---

## 2. API 映射设计

### 2.1 数学公式分解

将公式拆解为基本操作步骤：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | {sub_formula_1} | {step_desc_1} |
| 2 | {sub_formula_2} | {step_desc_2} |

### 2.2 PyPTO API 映射表

| 步骤 | 数学表达 | PyPTO API | 参数 | 文档路径 |
|------|----------|-----------|------|----------|
| 1 | {sub_formula_1} | {pypto_api_1} | {params_1} | {doc_path_1} |
| 2 | {sub_formula_2} | {pypto_api_2} | {params_2} | {doc_path_2} |

### 2.3 计算步骤序列

```python
# 伪代码展示计算流程
{step_1_code}
{step_2_code}
{output_code}
```

### 2.4 设计依据

- 来源：{spec / api_report / docs / example}
- 说明：{为何选择这些 API}

---

## 3. 数据规格设计

### 3.1 OperatorInput dataclass

```python
@dataclass
class {OperatorName}Input:
    {input_field_1}: Tensor  # {input_desc_1}, shape: {input_shape_1}, dtype: {input_dtype_1}
    {input_field_2}: Tensor  # {input_desc_2}, shape: {input_shape_2}, dtype: {input_dtype_2}
```

### 3.2 OperatorOutput dataclass

```python
@dataclass
class {OperatorName}Output:
    {output_field}: Tensor  # {output_desc}, shape: {output_shape}, dtype: {output_dtype}
```

### 3.3 中间 Tensor 定义

| 名称 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| {intermediate_name} | {intermediate_shape} | {intermediate_dtype} | {intermediate_desc} |

### 3.4 数据格式选择

| Tensor | 格式 | 说明 |
|--------|------|------|
| {tensor_name} | ND / NZ | {format_reason} |

### 3.5 动态轴定义

<!-- 如有动态轴 -->

| 轴名称 | 含义 | 取值范围 |
|--------|------|----------|
| {axis_name} | {axis_meaning} | {axis_range} |

### 3.6 JIT 装饰器配置

```python
@pypto.frontend.jit(
    runtime_options={"{runtime_option_key}": {runtime_option_value}}
)
def {operator_name}(inputs: {OperatorName}Input) -> {OperatorName}Output:
    ...
```

---

## 4. Tiling 策略

### 4.1 算子类型判断

- **类型**: {Cube / Vector / 混合}
- **判断依据**: {type_reason}

### 4.2 TileShape 初值设置

```python
{tiling_api_call}
```

### 4.3 设置依据

{tiling_rationale}

### 4.4 注意事项

- {tiling_note_1}
- {tiling_note_2}

### 4.5 判断依据与适用条件

- 判断依据：{为什么采用该 tiling}
- 适用条件：{适用于哪些 shape / dtype / 动态轴场景}
- 不适用场景：{哪些情况需要人工调整}

---

## 5. Loop 结构设计

<!-- 根据 quick_ref.md §2.1 判据表的结论选择对应模板 -->

### 场景 A：不需要 Loop

> 适用于所有轴编译期已知、单次 Tile 可处理的算子（如逐元素运算）。

- **结论**：不需要 pypto.loop
- **原因**：{no_loop_reason}（如"所有轴编译期已知，单次 Tile 可覆盖全部数据"）
- **适用条件**：{no_loop_applicable_conditions}
- **限制**：{no_loop_limitations}
- **处理方式**：编译器自动处理数据切分，无需手动循环

### 场景 B：需要 Loop

#### 5.1 Loop 判断结论

- **结论**: 需要 Loop
- **原因**: {loop_reason}
- **Loop 类型**: {pypto.loop / pypto.loop_unroll / Python for}
- **适用条件**: {loop_applicable_conditions}
- **限制**: {loop_limitations}

#### 5.2 静态轴 vs 动态轴处理

| 轴 | 类型 | 处理方式 |
|----|------|----------|
| {axis} | 静态 / 动态 | Python for / pypto.loop |

#### 5.3 Loop 合并策略

{loop_merge_strategy}

#### 5.4 数据依赖处理

{data_dependency_handling}

#### 5.5 尾块处理策略

{tail_block_strategy}

#### 5.6 loop_unroll 配置

<!-- 动态轴范围跨度大（如 1~64k）时使用，可在编译期生成多版本代码 -->

```python
{loop_unroll_config}
```

---

## 6. 验证方案

### 6.1 Golden 函数设计

```python
def {operator_name}_golden({golden_params}) -> {golden_return_type}:
    """{operator_name} 参考实现"""
    {golden_impl}
```

### 6.2 测试用例设计

#### 基于 SPEC.md 所有典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| {config_name} | {type} | {priority} | {params} | {input_shapes} | {output_shapes} | {config_desc} |

#### 边界情况测试（可选）

| 场景 | 参数 | 说明 |
|------|------|------|
| {boundary_scenario} | {boundary_params} | {boundary_desc} |

### 6.3 精度验证标准

| Dtype | atol | rtol |
|-------|------|------|
| {dtype} | {atol} | {rtol} |

---

## 7. 性能指标与开箱配置

### 7.1 性能目标

基于 SPEC.md 典型配置（性能类）的预期性能：

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 预期 kernel 耗时 |
|----------|------|--------|------|------------|------------|------------------|
| {perf_config_name} | 性能 | {priority} | {params} | {input_shapes} | {output_shapes} | {expected_time} |

### 7.2 开箱性能配置

```python
{tiling_config}
```

### 7.3 pass_options 配置

{pass_options_config}

### 7.4 runtime_options 配置

```python
{runtime_options_config}
```

---

## 8. 风险点与注意事项

### 8.1 已知约束

- {constraint_1}
- {constraint_2}

### 8.2 常见错误规避

| 风险 / 错误 | 触发场景 | 影响 / 原因 | 规避方法 |
|-------------|----------|-------------|----------|
| {error_1} | {trigger_1} | {reason_1} | {solution_1} |

### 8.3 特殊场景处理

{special_scenario_handling}

### 8.4 实现建议

<!-- 记录影响后续 PyPTO 算子实现的关键提示 -->

| 建议项 | 说明 |
|--------|------|
| {impl_hint_1} | {hint_desc_1} |

---

## 9. 交付件清单

### 9.1 目录结构

```
custom/{operator_name}/
├── SPEC.md                          # 需求规范（已有）
├── DESIGN.md                        # 设计文档（本文件）
├── {operator_name}_golden.py        # Golden 参考实现
├── {operator_name}_impl.py          # 算子实现代码
├── test_{operator_name}.py          # 测试代码
└── output/                          # 运行输出（自动生成）
```

### 9.2 文件清单

| 文件 | 类型 | 说明 | 生成方式 |
|------|------|------|----------|
| SPEC.md | 需求 | 算子需求规范 | pypto-intent-understand |
| DESIGN.md | 设计 | 算子设计文档 | pypto-op-design（本 skill） |
| {operator_name}_golden.py | 代码 | Golden 参考实现 | pypto-golden-generate |
| {operator_name}_impl.py | 代码 | 算子核心实现 | 后续实现 |
| test_{operator_name}.py | 代码 | 测试用例 | 后续实现 |

### 9.3 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 算子名称 | 小写字母 + 下划线 | `fast_gelu` |
| 目录名 | 与算子名称一致 | `custom/fast_gelu/` |
| Golden 文件 | `{op}_golden.py` | `fast_gelu_golden.py` |
| 实现文件 | `{op}_impl.py` | `fast_gelu_impl.py` |
| 测试文件 | `test_{op}.py` | `test_fast_gelu.py` |

### 9.4 生成顺序

```
SPEC.md → DESIGN.md → {op}_golden.py → {op}_impl.py → test_{op}.py
```
