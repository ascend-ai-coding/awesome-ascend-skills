---
name: external-cannbot-ops-pypto-api-explore
description: 探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。当需要查找 PyPTO 是否支持某个操作、验证
  API 约束、分析算子可行性时使用。触发词：API 探索、查找 API、PyPTO 有没有 xxx、支持什么 dtype、约束是什么、tiling 怎么配、API
  映射、可行性分析、这个算子能做吗。
original-name: pypto-api-explore
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
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
- **格式**：markdown，使用 [templates/api_report.md](templates/api_report.md) 模板
- **输出路径**：当前目录或用户指定位置
- **front matter**：至少填写 `schema_version`、`op_name`，并建议填充 `supported_dtypes`、`axes_list`、`shape_constraints`、`tiling_required`、`feasibility`（其中 `axes_list` 应为 YAML 列表，例如 `['N']` 或 `['N','M']`）

---

## 核心工作流

**注意**：必须使用 `Explore` subagent 进行 API 和约束探索，确保搜索的全面性和准确性。

### 步骤 1: 输入解析

接受任意形式输入，提取：
- 算子名称（如有）
- 数学公式 / 计算逻辑
- 输入输出规格（shape、dtype）
- 其他约束条件

### 步骤 2: 公式分解

将计算逻辑分解为原子操作序列：

操作类型：
- elementwise: add, sub, mul, div, exp, log, sin, cos...
- reduction: sum, max, min, mean, var...
- matmul: matmul, bmm, linear...
- shape: reshape, transpose, concat...
- index: gather, scatter, index_select...
- activation: relu, sigmoid, softmax...

### 步骤 3: 并行探索

将 API 探索、参考实现搜索和约束探索合并为**三个并行的 Explore subagent**，分别负责不同的一级目录。必须在**同一条消息中同时发起所有 Agent 调用**，确保并行执行。

#### Explore subagent 1: `pypto/docs/zh/` — API 文档与约束

**搜索范围**：`pypto/docs/zh/` 目录

**任务**：
1. 查 `pypto/docs/zh/api/operation/index.md` 确认 API 存在性
2. 读取具体 API 文档 `pypto/docs/zh/api/operation/pypto-*.md` 获取参数和约束
3. 未找到 → 标记 unsupported，尝试 substitute 方案
4. 提取入口约束（`pypto/docs/zh/api/others/pypto-from_torch.md`）：dtype、contiguous、format
5. 提取 API 约束：dtype 支持、shape 范围、广播规则、特殊值限制
6. 提取 Tiling 约束（`pypto/docs/zh/api/config/pypto-set_vec_tile_shapes.md`、`pypto/docs/zh/api/config/pypto-set_cube_tile_shapes.md`）
7. 查阅 `pypto/docs/zh/api/datatype/` 获取 DataType、TileOpFormat 枚举信息

**返回内容**：API 映射表、三层约束检查结果、Tiling 需求、证据索引路径列表

#### Explore subagent 2: `models/` — 生产级参考实现

**搜索范围**：`models/` 目录，**排除 `models/experimental/`**（实验性实现，未充分验证，禁止参考）

**任务**：
1. 搜索与当前算子相关的生产级模型算子实现
2. 提取可复用的实现模式：API 调用方式、Tiling 配置、Loop 结构、数据类型处理
3. 不要找到一个就停止，遍历所有候选，收集**所有匹配的参考实现**

**返回内容**：每个匹配实现的路径、相似度、置信度、可复用点；若无匹配标注「无匹配」

#### Explore subagent 3: `examples/` — 官方示例实现

**搜索范围**：`examples/` 目录

**任务**：
1. 在 `pypto/examples/02_intermediate/operators/` 搜索完整算子参考实现
2. 在 `pypto/examples/03_advanced/patterns/` 搜索高级组合模式
3. 提取可复用的实现模式：API 调用方式、Tiling 配置、Loop 结构、边界处理
4. 不要找到一个就停止，遍历所有候选，收集**所有匹配的参考实现**

**返回内容**：每个匹配实现的路径、相似度、置信度、可复用点；若无匹配标注「无匹配」

---

#### 并行探索结果汇总

等待三个 Explore subagent 全部返回后：
1. 合并 API 映射与约束检查结果（来自 subagent 1）
2. 合并参考实现搜索结果（来自 subagent 2 和 3），对比评估选择**最佳匹配**
3. 若存在多个高质量参考，在报告中列出 Top 3，并说明推荐首选及理由

### 步骤 4: 生成报告

基于 [templates/api_report.md](templates/api_report.md) 模板生成 API_REPORT.md。

---

## 搜索目录

按一级目录分配到对应的 Explore subagent：

| Subagent | 目录 | 搜索内容 | 优先级 |
|----------|------|----------|--------|
| **1: pypto/docs/zh/** | `pypto/docs/zh/api/operation/index.md` | API 列表，确认存在性 | **入口** |
| | `pypto/docs/zh/api/operation/pypto-*.md` | 具体 API 文档 | **主要** |
| | `pypto/docs/zh/api/others/pypto-from_torch.md` | 入口约束 | **必查** |
| | `pypto/docs/zh/api/config/pypto-set_vec_tile_shapes.md` | Vector Tiling | 条件 |
| | `pypto/docs/zh/api/config/pypto-set_cube_tile_shapes.md` | Cube Tiling | 条件 |
| | `pypto/docs/zh/api/datatype/` | DataType、TileOpFormat 枚举 | 参考 |
| **2: models/** | `models/`（排除 experimental） | 生产级模型算子实现 | 首选 |
| **3: examples/** | `pypto/examples/02_intermediate/operators/` | 完整算子参考实现 | 次选 |
| | `pypto/examples/03_advanced/patterns/` | 高级组合模式 | 参考 |

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
- `gelu` → 组合 `mul + sigmoid + mul`（x * sigmoid(1.702 * x)），与 activation.py 一致
- `sigmoid` → 仅支持 FP32；需在其他 dtype 场景下先 `cast` 到 FP32 再调用

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
| dtype 入口 | FP16/BF16/FP32/FP64/INT8/INT16/INT32/INT64/UINT8/UINT16/UINT32/UINT64/BOOL | from_torch 文档 |
| shape 入口 | 非空 Tensor | from_torch 文档 |
| contiguous | 必须连续 | from_torch 文档 |
| TileShape | 每维 > 0，最多 4 维 | set_vec_tile_shapes 文档 |
| Cube TileShape | 32 字节对齐；buffer 空间需满足 K 轴 × 2 ≤ L1 容量 | set_cube_tile_shapes 文档 |
| shape size | ≤ INT32_MAX | 各 API 文档 |
| sigmoid dtype | 仅支持 DT_FP32 | pypto.sigmoid API 文档 |
| **动态 shape 兼容性** | **matmul / 归约类 等计算 API 在编译期需要 concrete shape，不接受含 DYNAMIC 维度的 tensor（报错：`has invalid shape value: -1`）。若算子有动态轴且用到这类 API，必须在风险评估中标注，并说明需采用"loop 切 tile"策略（见 execution-constraints.md §5）

---

## Checklist

验证 API_REPORT.md 的门禁条件：
1. 文件存在
2. 以下 6 个章节存在且内容不为空：
   - `## 1. 概述`
   - `## 3. API 映射`
   - `## 6. 参考实现`（可标注「无匹配」但不可缺失）
   - `## 7. 风险评估`
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
