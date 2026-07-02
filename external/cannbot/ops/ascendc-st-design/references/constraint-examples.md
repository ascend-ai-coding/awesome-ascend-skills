# 约束定义详细示例

> 本文档从 SKILL.md 迁出，包含所有约束类型的完整 YAML 示例。在编写 `05_约束定义.yaml` 时按需查阅。

## 目录

- [1. 因子节点定义（按参数类型）](#1-因子节点定义按参数类型)
- [2. 约束类型速查](#2-约束类型速查)
- [3. 约束定义示例](#3-约束定义示例)
  - [3.1 计算约束 (calculate)](#31-计算约束-calculate)
  - [3.2 指定维度广播 (broadcast_dim)](#32-指定维度广播-broadcast_dim)
  - [3.3 张量广播 (broadcast_shape)](#33-张量广播-broadcast_shape)
  - [3.4 条件约束 (conditional)](#34-条件约束-conditional)
  - [3.5 匹配约束 (match)](#35-匹配约束-match)
  - [3.6 存在性约束 (existential)](#36-存在性约束-existential)
  - [3.7 可转换约束 (convertible)](#37-可转换约束-convertible)
  - [3.8 链式依赖约束 (inferable_filter)](#38-链式依赖约束-inferable_filter)
  - [3.9 可推导约束 (inferable)](#39-可推导约束-inferable)
- [4. 两步约束模式](#4-两步约束模式)
- [5. 完整约束文件示例](#5-完整约束文件示例)

---

## 1. 因子节点定义（按参数类型）

因子节点标识规范：`{parameter}.{attribute}`

**通用属性**：

| 属性 | 说明 | 示例 |
|------|------|------|
| `exist` | 存在性 | `bias.exist` |
| `dtype` | 数据类型 | `self.dtype` |
| `format` | 数据格式 | `input.format` |
| `dimensions` | 维度数 | `self.dimensions` |
| `value` | 值（枚举值/标量值/数组值） | `axis.value` |

**按参数类型分组的属性**：

| 参数类型 | 包含的属性 | 说明 |
|----------|-----------|------|
| `aclTensor` | exist, dtype, shape, format, dimensions | shape 由 dimensions 隐式推导 |
| `aclTensorList` | exist, dtype, length, shape_list, format, dimensions | length 由 length_ranges 推导；shape_list 由 length+dimensions 推导 |
| `aclIntArray` / `aclFloatArray` / `aclBoolArray` / `aclScalarList` | exist, dtype, length, value | length 由 length_ranges 推导；value 由 length+value_range 推导 |
| 18种标量类型（int4_t~string） | exist, dtype, value | value 由 value_range 推导（非枚举时） |

**因子节点 YAML 定义示例**（按参数类型分组）：

```yaml
factors:
  # ── Tensor 类 ──
  self.dtype: {type: dtype, param: self, io_type: input}
  self.shape: {type: shape, param: self, io_type: input}
  self.format: {type: format, param: self, io_type: input}
  self.exist: {type: exist, param: self, io_type: input}
  self.dimensions: {type: dimensions, param: self, io_type: input}

  # ── TensorList 类 ──
  tensors.dtype: {type: dtype, param: tensors, io_type: input}
  tensors.format: {type: format, param: tensors, io_type: input}
  tensors.exist: {type: exist, param: tensors, io_type: input}
  tensors.dimensions: {type: dimensions, param: tensors, io_type: input}
  tensors.length_ranges: {type: length_ranges, param: tensors, io_type: input}
  tensors.length: {type: length, param: tensors, io_type: input}
  tensors.shape_list: {type: shape_list, param: tensors, io_type: input}
  tensors.value_range: {type: value_range, param: tensors, io_type: input}

  # ── Array 类（aclIntArray/aclFloatArray/aclBoolArray）──
  sizes.dtype: {type: dtype, param: sizes, io_type: input}
  sizes.exist: {type: exist, param: sizes, io_type: input}
  sizes.length_ranges: {type: length_ranges, param: sizes, io_type: input}
  sizes.length: {type: length, param: sizes, io_type: input}
  sizes.value_range: {type: value_range, param: sizes, io_type: input}
  sizes.value: {type: value, param: sizes, io_type: input}

  # ── ScalarList 类 ──
  scalars.dtype: {type: dtype, param: scalars, io_type: input}
  scalars.exist: {type: exist, param: scalars, io_type: input}
  scalars.length_ranges: {type: length_ranges, param: scalars, io_type: input}
  scalars.length: {type: length, param: scalars, io_type: input}
  scalars.value_range: {type: value_range, param: scalars, io_type: input}
  scalars.value: {type: value, param: scalars, io_type: input}

  # ── Scalar 类 ──
  alpha.dtype: {type: dtype, param: alpha, io_type: input}
  alpha.exist: {type: exist, param: alpha, io_type: input}
  alpha.value_range: {type: value_range, param: alpha, io_type: input}
  alpha.value: {type: value, param: alpha, io_type: input}

  # ── 18种标量类型（枚举/非枚举）──
  cubeMathType.dtype: {type: dtype, param: cubeMathType, io_type: input}
  cubeMathType.exist: {type: exist, param: cubeMathType, io_type: input}
  cubeMathType.value: {type: value, param: cubeMathType, io_type: input}

  axis.dtype: {type: dtype, param: axis, io_type: input}
  axis.exist: {type: exist, param: axis, io_type: input}
  axis.value_range: {type: value_range, param: axis, io_type: input}
  axis.value: {type: value, param: axis, io_type: input}

  # ── 输出参数 ──
  out.dtype: {type: dtype, param: out, io_type: output}
  out.shape: {type: shape, param: out, io_type: output}
  out.format: {type: format, param: out, io_type: output}
  out.exist: {type: exist, param: out, io_type: output}
  out.dimensions: {type: dimensions, param: out, io_type: output}

  # ── 中间因子（派生值，用于两步约束）──
  _broadcast_target.shape: {type: list, param: _broadcast_target, io_type: intermediate}
```

---

## 2. 约束类型速查

| 类型 | 语义 | 典型场景 | 关键字段 |
|------|------|----------|----------|
| `calculate` | 计算约束 | 类型等值、形状计算 | sources, target, expression |
| `broadcast_dim` | 指定维度广播 | 单维度广播兼容性 | sources, source_index, target, target_index |
| `broadcast_shape` | 张量广播 | 完整形状广播（单向/双向） | sources, target, mode |
| `conditional` | 条件约束 | if-then-else 分支 | condition, then, else |
| `match` | 匹配约束 | 维度/属性双向匹配 | sources, source_index, target, target_index |
| `existential` | 存在性约束 | 可选参数联动 | trigger, trigger_value, effect |
| `convertible` | 可转换约束 | 类型兼容性检查 | sources, target, target_domain |
| `inferable_filter` | 链式依赖 | 多 Tensor 类型互推导 | sources, target, target_domain |
| `inferable` | 可推导约束 | 类型推导检查 | sources, target, mode |

**设计原则**：
- `calculate` 是通用计算约束，等值场景使用 `expression: "sources[0]"`
- `match` 是**双向约束**，`calculate` 是**单向计算**，语义不同不应合并
- `broadcast_dim` 用于单维度广播，`broadcast_shape` 用于完整形状广播
- 固定取值范围（如 `cubeMathType.enum_values: [0, 1, 2, 3]`）已在 `04_测试因子.yaml` 中定义，无需添加约束

---

## 3. 约束定义示例

### 3.1 计算约束 (calculate)

```yaml
# 等值计算：输出类型等于self类型
- id: "TYPE-002"
  type: calculate
  sources: ["self.dtype"]
  target: "out.dtype"
  expression: "sources[0]"
  description: "输出类型等于self类型"

# 形状计算：out.shape 由多个输入形状计算
- id: "SHAPE-001"
  type: calculate
  sources: ["batch1.shape", "batch2.shape"]
  target: "out.shape"
  expression: "[sources[0][1], sources[1][2]]"
  description: "out.shape = [M, N]"

# 自引用：固定某一维度
- id: "SHAPE-006"
  type: calculate
  sources: ["grid.shape"]
  target: "grid.shape"
  expression: "sources[0][:3] + [2]"
  description: "grid.shape最后一维度固定为2"

# 多shape广播结果计算
- id: "SHAPE-007"
  type: calculate
  sources: ["self.shape", "other.shape", "beta.shape"]
  target: "out.shape"
  expression: "get_broadcast_result(sources[0], sources[1], sources[2])"
  description: "out.shape是self、other、beta广播后的shape"
```

### 3.2 指定维度广播 (broadcast_dim)

```yaml
- id: "BCAST-001"
  type: broadcast_dim
  sources: ["batch1.shape"]
  source_index: 0
  target: "batch2.shape"
  target_index: 1
  description: "batch1[0]与batch2[1]需满足广播关系"
```

### 3.3 张量广播 (broadcast_shape)

```yaml
# 单向广播
- id: "BCAST-002"
  type: broadcast_shape
  sources: ["out.shape"]
  target: "bias.shape"
  mode: unidirectional
  description: "bias.shape可广播到out.shape"

# 双向广播
- id: "BCAST-003"
  type: broadcast_shape
  sources: ["x.shape"]
  target: "y.shape"
  mode: bidirectional
  description: "x.shape和y.shape双向广播"
```

**重要限制**：`broadcast_shape` 不支持 `source_shape_expr` 字段。如需表达"广播到派生形状"，应使用两步约束模式（见[第4节](#4-两步约束模式)）。

### 3.4 条件约束 (conditional)

```yaml
- id: "C007"
  type: conditional
  condition:
    factor: "beta.value"
    expression: "== 0"
  then:
    type: ignore
    target: "self"
  else:
    type: require
    target: "self"
  description: "beta=0时self被忽略"
```

### 3.5 匹配约束 (match)

```yaml
- id: "MATCH-001"
  type: match
  sources: ["batch1.shape"]
  source_index: 2
  target: "batch2.shape"
  target_index: 1
  description: "K维度匹配: batch1[2] == batch2[1]"
```

### 3.6 存在性约束 (existential)

> 仅用于**可选参数**之间的依赖关系，必选参数无需此约束。

```yaml
# ✅ 正确：bias 是可选参数
- id: "C-EXIST-001"
  type: existential
  trigger: "bias.exist"
  trigger_value: false
  effect:
    target: "bias.dtype"
    action: set_na
  description: "bias不存在时，bias.dtype设为NA"

# ❌ 错误：self 和 batch1 都是必选参数，不应添加存在性约束
```

### 3.7 可转换约束 (convertible)

```yaml
- id: "C-TYPE-004"
  type: convertible
  sources: ["self.dtype"]
  target: "beta.dtype"
  target_domain: ["float32", "float16", "int32", "int64", "bool"]
  description: "beta类型需要可转换成self类型"
```

### 3.8 链式推导约束 (inferable_filter)

用于多 Tensor 类型互推导的链式求解场景。选择一个锚点独立采样，其他因子逐层推导：

```yaml
# Step 1: 选择锚点（Level 0 独立采样）
# batch1.dtype 作为锚点

# Step 2: Level 1 推导
- id: "C-TYPE-001"
  type: inferable_filter
  sources: ["batch1.dtype"]
  target: "batch2.dtype"
  target_domain: ["float32", "float16", "bfloat16"]
  description: "batch2.dtype 需与 batch1.dtype 兼容"

# Step 3: Level 2 推导
- id: "C-TYPE-002"
  type: inferable_filter
  sources: ["batch1.dtype", "batch2.dtype"]
  target: "self.dtype"
  target_domain: ["float32", "float16", "bfloat16"]
  description: "self.dtype 需与 (batch1, batch2) 的推导结果兼容"
```

### 3.9 可推导约束 (inferable)
用于Tensor-Tensor、Tensor-Scalar类型互推导的链式求解场景。

```yaml
# Tensor-Tensor 推导（默认模式）
- id: "C012"
  type: inferable
  mode: tensor_tensor
  sources: ["batch1.dtype"]
  target: "batch2.dtype"          # 推导结果赋值给目标
  description: "batch1和batch2类型需要可相互推导"

# Tensor-Scalar 推导
- id: "C013"
  type: inferable
  mode: tensor_scalar            # Tensor-Scalar 模式
  sources: ["input.dtype"]  # 第一个为Tensor类型，第二个为Scalar类型
  target: "scalar.dtype"
  description: "Tensor和Scalar类型推导"
```

---

## 4. 两步约束模式

当需要表达"某个 Tensor 的 shape 需广播到一个派生形状"时，`broadcast_shape` 不支持 `source_shape_expr`，需要拆分为两步约束：

```yaml
# ❌ 错误写法
# - id: "BCAST-004"
#   type: broadcast_shape
#   sources: ["batch1.shape", "batch2.shape"]
#   target: "self.shape"
#   mode: unidirectional
#   source_shape_expr: "[sources[0][1], sources[1][2]]"  # ❌ 不支持

# ✅ 正确写法：两步约束

# 步骤1：计算派生形状（使用 intermediate 类型因子）
- id: "BCAST-004-CALC"
  type: calculate
  sources: ["batch1.shape", "batch2.shape"]
  target: "_broadcast_target.shape"
  expression: "[sources[0][1], sources[1][2]]"
  description: "计算广播目标形状 [M, N]"

# 步骤2：检查广播兼容性
- id: "BCAST-004-BCAST"
  type: broadcast_shape
  sources: ["_broadcast_target.shape"]
  target: "self.shape"
  mode: unidirectional
  description: "self.shape可广播到[M, N]"
```

**注意**：需要在 factors 中定义中间因子：
```yaml
_broadcast_target.shape: {type: list, param: _broadcast_target, io_type: intermediate}
```

---

## 5. 完整约束文件示例

```yaml
metadata:
  operator: "aclnnAddbmm"
  version: "1.0"
  description: "Addbmm算子完整约束定义"

factors:
  self.dtype: {type: dtype, param: self, io_type: input}
  self.shape: {type: shape, param: self, io_type: input}
  self.format: {type: format, param: self, io_type: input}
  self.exist: {type: exist, param: self, io_type: input}
  self.dimensions: {type: dimensions, param: self, io_type: input}
  out.dtype: {type: dtype, param: out, io_type: output}
  out.shape: {type: shape, param: out, io_type: output}
  out.format: {type: format, param: out, io_type: output}
  out.exist: {type: exist, param: out, io_type: output}
  out.dimensions: {type: dimensions, param: out, io_type: output}
  beta.dtype: {type: dtype, param: beta, io_type: input}
  beta.exist: {type: exist, param: beta, io_type: input}
  cubeMathType.dtype: {type: dtype, param: cubeMathType, io_type: input}
  cubeMathType.exist: {type: exist, param: cubeMathType, io_type: input}
  cubeMathType.value: {type: value, param: cubeMathType, io_type: input}

constraints:
  - id: "SHAPE-001"
    type: match
    sources: ["batch1.shape"]
    source_index: 2
    target: "batch2.shape"
    target_index: 1
    description: "K维度匹配: batch1[2] == batch2[1]"

  - id: "SHAPE-002"
    type: calculate
    sources: ["batch1.shape", "batch2.shape"]
    target: "out.shape"
    expression: "[sources[0][1], sources[1][2]]"
    description: "out.shape = [M, N] = [batch1[1], batch2[2]]"
```
