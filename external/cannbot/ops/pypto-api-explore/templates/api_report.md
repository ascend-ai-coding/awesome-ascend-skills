---
schema_version: 1
op_name: {operator_name}
supported_dtypes: {supported_dtypes}
dynamic_axes: {axes_list}
shape_constraints: {shape_constraints}
tiling_required: {tiling_required}
feasibility: {feasibility}
---

# API 探索报告

> **生成时间**: {timestamp}

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

{输入内容摘要}

### 1.2 算子分类

- **类型**: {Vector / Cube / 混合}
- **判断依据**: {type_reason}

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | {op_type} | {math_expr} | {desc} |

---

<!-- REQUIRED -->
## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | {expr} | `{api}` | {direct/substitute/unsupported} | {✓/⚠/✗} |

### 3.2 Substitute 配方

<!-- 仅 substitute 时填写 -->

```
{operation}: {recipe}
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | {supported} | {input_dtype} | {✓/✗} |
| contiguous | 必须 | — | {✓/需确保} |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| {api} | dtype | {list} | {✓/✗} |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| {type} | `pypto.set_{vec/cube}_tile_shapes()` |

---

## 6. 参考实现

### 6.1 匹配示例

<!-- 无匹配时填写：无匹配参考实现，需从零设计 -->
<!-- 置信度：正式算子实现为「高」，golden 用法为「高」，experimental 实现为「中」 -->
<!-- 注意：tests/ 中的 golden/用法仅作 API 用法参考，不是 production 实现标准；简化写法（如 pypto.Tensor([])）可能违反门禁要求，冲突时以门禁要求为准，不得据此判定门禁误报。 -->

| 参考路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `{ref_path}` | {算子实现/golden}（`pypto-docs-search` 命中） | {高/中/低} | {高/中} | {reuse_points} |

### 6.2 可复用模式

- **API 调用模式**：{api_usage_pattern}
- **Tiling 策略**：{tiling_pattern}
- **Loop 结构**：{loop_pattern}
- **边界处理**：{boundary_pattern}

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| {diff} | {example_approach} | {current_need} | {suggestion} |

---

<!-- REQUIRED -->
## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| {issue} | {reason} | {suggestion} |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| {warning} | {desc} |

---

<!-- REQUIRED -->
## 8. 证据索引

证据路径（已知具体文档给 raw URL；参考实现给 `pypto-docs-search` 命中的 算子参考实现路径）：

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/operation/index.md` |
| {api} 文档 | `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/operation/pypto-{api}.md` |
| 入口约束 | `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/others/pypto-from_torch.md` |
| 参考实现 | `pypto-docs-search` 命中的 算子参考实现路径（如有） |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: {可行 / 需调整 / 不可行}
- **主要问题**: {main_issue}
