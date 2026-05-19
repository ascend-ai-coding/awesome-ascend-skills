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
<!-- 置信度：models/（排除 experimental）和 examples/ 为「高」，models/experimental/ 为「中」 -->

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `{example_path}` | {models/examples} | {高/中/低} | {高/中} | {reuse_points} |

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

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| {api} 文档 | `docs/api/operation/pypto-{api}.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| 参考实现 | `{example_path}`（如有） |

---

<!-- REQUIRED -->
## 9. 结论

- **可行性**: {可行 / 需调整 / 不可行}
- **主要问题**: {main_issue}
