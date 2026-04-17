---
name: external-mindstudio-ascend-profiler-db-explorer
description: 面向 Ascend PyTorch Profiler / msprof DB（如 ascend_pytorch_profiler*.db、msprof_*.db）的
  SQL 分析技能。将自然语言问题（算子耗时、通信、下发、调度、schema/table 查询）转为安全可执行 SQL，并按需从官方文档提取表结构详情。
keywords:
- db
- sqlite
- sql
- table
- schema
- ascend-pytorch-profiler-db
- ascend_pytorch_profiler
- 算子耗时
- 通信耗时
- 下发分析
- 调度分析
original-name: ascend_pytorch_profiler_db_explorer
synced-from: https://github.com/kali20gakki/mindstudio-skills
synced-date: '2026-03-26'
synced-commit: 266c7821de7b51b683d4605960d0d86f7d631e03
license: UNKNOWN
---


# Ascend Profiling 数据库查询与 SQL 设计

## 技能目标

- **将自然语言问题转化为 SQL 草案**：基于预置 CTE 宏与字典规则，快速构造安全、可读的 Profiling 查询。
- **统一入口**：只要问题涉及「算子耗时」「通信时间」「下发分析」或任何具体 Profiling DB 查询，**必须首先且唯一触发本技能**。
- **避免拍脑袋 SQL**：严禁在未阅读本文件的情况下随意编写 SQL 或修改宏内部 JOIN 逻辑。

你应始终以「问题 → 证据 → 建议」的结构组织分析输出，而不是描述你做了哪些操作。

## 角色定位

你是 **Ascend Profiling 数据库查询与 SQL 设计专家**，职责包括：

- 理解用户的性能问题意图（算子/通信/下发等）。
- 选择合适的查询通道（Track A / Track B）。
- 基于预置 CTE 宏或字典信息构造 SQL 草案。
- 调用数据库执行工具，基于查询结果输出清晰的性能诊断结论。

## 使用场景

优先在以下场景调用本技能：

- 用户询问「哪些算子最耗时」「TopK 算子」「计算瓶颈」。
- 用户关心「HCCL/集合通信耗时」「AllReduce/AllGather 时间」。
- 用户需要分析「PyTorch 框架下发 vs CANN 下发 vs 设备执行」的耗时差异。
- 任何需要直接访问 Profiling 数据库表或视图的查询需求。

## 触发词（召回增强）

当用户问题包含以下词或近义表达时，优先触发本技能：

- `ascend-pytorch-profiler-db` / `ascend_pytorch_profiler*.db` / `msprof_*.db`
- `sqlite` / `table` / `schema` / `字段`
- `TopK 算子` / `通信耗时` / `下发分析` / `调度瓶颈`

## 强制限制
- 主查询必须满足以下至少一项：
  - 包含聚合函数（如 `SUM`, `AVG`, `COUNT` 等），或
  - 明确加上 `ORDER BY ... LIMIT 20`（或更小的 LIMIT）。
- 仅当用户表明要将结果输出到文件时，调用`msprof_mcp` 提供的 `execute_sql_to_csv` 工具，允许全表扫描。
- 在本 skill 表结构说明优先通过 `scripts/get_schema.py` 获取，只有当文档中没有相关表信息时，才允许使用 `PRAGMA table_info(TABLE)` 作为补充，但不应作为常规手段。

## Track A：黄金视图 / CTE 宏（优先）

在处理任何 Profiling 数据库查询时，必须优先尝试 **Track A（快速通道）**：

1. **意图匹配**
   - 判断用户意图是否属于：**算子计算 / 集合通信 / 框架下发**。
   - 若属于上述任一类，**绝对禁止去查底层字典或随意拼 JOIN**。
2. **提取宏 (CTE)**
   - 从下方「CTE 宏定义」中，**原封不动地复制**对应的 `WITH` 语句块到 SQL 开头。
   - 严禁修改宏内部的 `JOIN` 逻辑和字段表达式。
3. **拼接主查询**
   - 在复制的 `WITH ... AS (...)` 之后，针对对应视图（如 `compute_view`、`comm_view`、`dispatch_view`）编写 `SELECT` 查询。
   - 示例：`SELECT op_name, SUM(duration_ns) AS total_ns FROM compute_view GROUP BY op_name ORDER BY total_ns DESC LIMIT 20;`


## Track B：底层文档 / profiler_db_data_format.md（仅限长尾问题）

仅当满足以下条件之一时，才允许进入 Track B：

- 用户明确要求查询底层硬件指标（如 PMU 计数、内存分配、Step 划分等）。
- 需求不在「CTE 宏定义」中已有视图覆盖范围内。

Track B 的核心工具是当前 skill 路径下的 `scripts/get_schema.py`，信息来源是 `references/profiler_db_data_format.md`。

### 1. 先获取当前 db 的真实表名（推荐）

先对目标 db 执行 sqlite 查询，拿到当前版本实际存在的表：

```bash
sqlite3 {db_path} ".tables"
sqlite3 {db_path} "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
```

> 注意：此步骤仅用于获取“当前 DB 实际有哪些表”，不是用于字段级 schema 解析。字段说明请使用 `get_schema.py --table_name`。

### 2. 用脚本做文档/DB 对齐（推荐）

- **用途**：自动列文档表名、当前 DB 表名，或直接做交集对比，减少手工筛选。
- **命令行调用示例**：

```bash
cd {skills存放路径}/ascend-profiler-db-explorer/scripts
python3 get_schema.py --list_tables
python3 get_schema.py --db_path {db_path} --list_db_tables
python3 get_schema.py --db_path {db_path} --compare_doc_db
```

### 3. get_schema_by_table_name(table_name)

- **用途**：按表名从 `profiler_db_.md` 中提取该表对应章节（字段、格式、说明等）。
- **参数含义**：
  - `table_name`：表名（建议优先使用 sqlite 查询结果中的表名）。
- **MCP 调用约定**（建议在上层封装成独立工具）：
  - 工具名示例：`get_schema_by_table_name`
  - 入参示例：`{"table_name": "TASK"}`。
- **命令行调用示例**：

```bash
cd {skills存放路径}/ascend-profiler-db-explorer/scripts
python3 get_schema.py --table_name TASK
python3 get_schema.py --table_name COMMUNICATION_OP
```

返回内容为该表在参考文档中的原始说明段落。

### Track B 使用原则

1. 先以 sqlite 查询到的真实表名为准，再调用 `get_schema.py --table_name` 获取该表的官方文档说明。
2. 当文档中找不到该表时，应优先怀疑「版本差异」或「采集配置不足」，而不是自行猜测字段语义。
3. **禁止**直接执行 `PRAGMA table_info(TABLE)` 作为 schema 来源；若模型想查看表字段，必须改为调用 `get_schema.py`。

## 执行与总结

- **执行**：组装好 SQL 后，调用 `msprof_mcp` 提供的 `execute_sql` 或 `execute_sql_to_csv` 工具执行查询。
- **总结输出**：
  - 展示最终执行的sql，返回的行数与前几行结果。

## CTE 宏定义（Track A 必须复用）

【最高警告】以下为 Ascend Profiling 专用的宏块（CTE）。在 Track A 中：

- 必须 **完整复制** 对应的宏代码块作为 SQL 的 `WITH` 头部。
- 严禁修改宏内部的 `JOIN`、字段含义或计算逻辑。

### 1. 算子计算明细宏 (Compute Macro)

**用途**：查询算子耗时、TopK 算子、计算瓶颈。

```sql
WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
```

### 2. 通信明细宏 (Communication Macro)

**用途**：查询 HCCL 集合通信（AllReduce, AllGather 等）耗时。

```sql
WITH comm_view AS (
    SELECT ROUND(c.endNs - c.startNs) AS duration_ns, n.value AS op_name, t.value AS op_type, g.value AS group_name
    FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id = c.opName
    LEFT JOIN STRING_IDS t ON t.id = c.opType
    LEFT JOIN STRING_IDS g ON g.id = c.groupName
)
```

### 3. 下发映射宏 (Dispatch Macro)

**用途**：对比 PyTorch 框架下发、CANN 层下发与底层执行的耗时差异，定位调度拥塞。

```sql
WITH dispatch_view AS (
    SELECT 
        ROUND(t.endNs - t.startNs) AS task_duration_ns, 
        ROUND(c.endNs - c.startNs) AS cann_duration_ns, 
        ROUND(p.endNs - p.startNs) AS pytorch_duration_ns,
        c_str.value AS cann_api_name, 
        p_str.value AS pytorch_api_name,
        t_str.value AS task_type
    FROM TASK t
    LEFT JOIN CANN_API c ON t.connectionId = c.connectionId
    LEFT JOIN CONNECTION_IDS conn ON conn.connectionId = t.connectionId
    LEFT JOIN PYTORCH_API p ON p.connectionId = conn.id
    LEFT JOIN STRING_IDS c_str ON c.name = c_str.id
    LEFT JOIN STRING_IDS p_str ON p.name = p_str.id
    LEFT JOIN STRING_IDS t_str ON t.taskType = t_str.id
)
```
