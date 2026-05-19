---
name: external-cannbot-ops-pypto-api-explore
description: '探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。当需要查找 PyPTO 是否支持某个操作、验证
  API 约束、分析算子可行性时使用。Triggers: API 探索、查找 API、PyPTO 有没有 xxx、支持什么 dtype、约束是什么、tiling
  怎么配、API 映射、可行性分析、这个算子能做吗。'
original-name: pypto-api-explore
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# pypto-api-explore

使用 `Explore` subagent 探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。

## 输入

接受任意形式的输入，提取算子计算逻辑：
- 自然语言描述（如"计算 softmax"）
- 数学公式（如 softmax(x) = exp(x)/sum(exp(x))）
- 代码片段（PyTorch 或伪代码）
- 已有的 spec 文档内容

如果信息不足，向用户提问补充。

## 输出

- **输出件**：API_REPORT.md
- **格式**：markdown，使用 [templates/API_REPORT.md](templates/API_REPORT.md) 模板
- **输出路径**：当前目录或用户指定位置

---

## 核心工作流

**注意**：必须使用 `Explore` subagent 进行 API 和约束探索，确保搜索的全面性和准确性。

### Stage 1: 输入解析

接受任意形式输入，提取：
- 算子名称（如有）
- 数学公式 / 计算逻辑
- 输入输出规格（shape、dtype）
- 其他约束条件

### Stage 2: 公式分解

将计算逻辑分解为原子操作序列：

操作类型：
- elementwise: add, sub, mul, div, exp, log, sin, cos...
- reduction: sum, max, min, mean, var...
- matmul: matmul, bmm, linear...
- shape: reshape, transpose, concat...
- index: gather, scatter, index_select...
- activation: relu, sigmoid, softmax...

### Stage 3: API 探索

搜索 PyPTO 文档获取 API 信息：

1. 查 `docs/api/operation/index.md` 确认 API 存在性
2. 读取具体 API 文档 `docs/api/operation/pypto-*.md` 获取参数和约束
3. 未找到 → 标记 unsupported，尝试 substitute 方案

### Stage 4: 参考实现搜索

在 `models/` 和 `examples/` 目录中搜索与当前算子相关的官方示例（具体目录见「搜索目录」），提取可复用的实现模式。

> **⚠️ 不要找到一个就停止**
> - 遍历所有候选目录，收集**所有匹配的参考实现**
> - 对多个候选进行对比评估，选择**最佳匹配**（相似度最高、置信度最高、可复用点最多）
> - 若存在多个高质量参考，在报告中列出 Top 3，并说明推荐首选及理由

排除 `models/experimental/`，该目录为实验性实现，未充分验证，禁止参考。

**提取内容**：
- API 实际调用方式和参数用法
- Tiling 配置（tile shape 设置、分块策略）
- Loop 结构（循环方式、边界处理）
- 数据类型处理、cast 用法

**输出**：
- 找到匹配 → 记录路径、相似度、置信度、可复用点，写入报告「参考实现」章节
- 未找到匹配 → 在报告中标注「无匹配参考实现」

### Stage 5: 约束探索

三层验证：

**Layer 1 - 入口约束（from_torch）**：
- dtype、contiguous、format
- 证据：`docs/api/others/pypto-from_torch.md`

**Layer 2 - API 约束（从具体 API 文档提取）**：
- dtype 支持、shape 范围、广播规则、特殊值限制

**Layer 3 - Tiling 约束（从 config API 文档提取）**：
- Vector: `set_vec_tile_shapes()`
- Cube: `set_cube_tile_shapes()`

### Stage 6: 生成报告

基于 [templates/API_REPORT.md](templates/API_REPORT.md) 模板生成 API_REPORT.md。

---

## 搜索目录

| 目录 | 搜索内容 | 优先级 |
|------|----------|--------|
| `docs/api/operation/index.md` | API 列表，确认存在性 | **入口** |
| `docs/api/operation/pypto-*.md` | 具体 API 文档 | **主要** |
| `docs/api/others/pypto-from_torch.md` | 入口约束 | **必查** |
| `docs/api/config/pypto-set_vec_tile_shapes.md` | Vector Tiling | 条件 |
| `docs/api/config/pypto-set_cube_tile_shapes.md` | Cube Tiling | 条件 |
| `docs/api/datatype/` | DataType、TileOpFormat 枚举 | 参考 |
| `models/`（排除 experimental） | 生产级模型算子实现 | 首选 |
| `examples/02_intermediate/operators/` | 完整算子参考实现 | 次选 |
| `examples/03_advanced/patterns/` | 高级组合模式 | 参考 |

---

## 内嵌知识

### 操作 → API 映射速查

| 操作类别 | 常见操作 | PyPTO API |
|----------|----------|-----------|
| 逐元素 | add, sub, mul, div, neg | `pypto.{op}` |
| 数学 | exp, log, sin, cos, sqrt, rsqrt | `pypto.{op}` |
| 比较 | eq, ne, lt, le, gt, ge | `pypto.{op}` |
| 位运算 | bitwise_and, bitwise_or, bitwise_xor | `pypto.bitwise_{op}` |
| 归约 | sum, amax, amin, prod, var | `pypto.{op}` |
| 矩阵 | matmul | `pypto.matmul` |
| 形状 | reshape, transpose, concat, view | `pypto.{op}` |
| 索引 | gather, scatter, index_select | `pypto.{op}` |
| 激活 | relu, sigmoid, softmax | `pypto.{op}` |
| 构造 | zeros, ones, full, arange | `pypto.{op}` |
| 类型转换 | cast | `pypto.cast` |

**常见 Substitute（无直接 API）**：
- `mean` → `sum/count`
- `gelu` → 组合 mul/add/tanh/pow

### 算子类型判断

```
公式分析
    │
    ├── 含 matmul/@ → Cube 类型 → set_cube_tile_shapes
    │
    ├── 仅逐元素/归约 → Vector 类型 → set_vec_tile_shapes
    │
    └── matmul + 逐元素 → 混合类型 → 两者都需要
```

### 硬约束速查

| 约束类型 | 规则 | 来源 |
|----------|------|------|
| dtype 入口 | FP16/BF16/FP32/INT8-64/BOOL | from_torch 文档 |
| shape 入口 | 非空 Tensor | from_torch 文档 |
| contiguous | 必须连续 | from_torch 文档 |
| TileShape | 每维 > 0，最多 4 维 | set_vec_tile_shapes 文档 |
| shape size | ≤ INT32_MAX | 各 API 文档 |

---

## Checklist

验证 API_REPORT.md 的门禁条件：
1. 文件存在
2. 以下 5 个章节存在且内容不为空：
   - `## 1. 概述`
   - `## 3. API 映射`
   - `## 6. 参考实现`（可标注「无匹配」但不可缺失）
   - `## 8. 证据索引`
   - `## 9. 结论`

---

## 错误处理

| 场景 | 处理 |
|------|------|
| 输入无法解析 | 引导用户提供公式或代码 |
| API 不存在 | 标记 unsupported，在风险中说明 |
| 约束不满足 | 标记 ✗，在风险中给出替代方案 |
| 无匹配参考实现 | 在「参考实现」章节标注「无匹配」，不阻断流程 |
