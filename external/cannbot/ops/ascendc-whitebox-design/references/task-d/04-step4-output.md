# Step 4：输出

> **前置条件**：Step 1-3 全部完成

本步骤负责运行 builder 转换 Step 3 增量写入的产出、校验一致性、更新路径列表，并生成可追溯性报告。

## 输入

| 数据项 | 来源 | 说明 |
|--------|------|------|
| groups 推导数据 | S2P2_param_def_groups.json | Step 3 由 pick_dims.py 生成 |
| reachability + dead_reason | Step 1 内存 | 每条路径的可达性判定及不可达原因 |
| group 字段 | Step 2 内存 | 每条 reachable 路径的 group 分配 |
| groups 列表 | Step 2 内存 | 所有 group id |
| dtype 路由声明 | S2P1_operator_model.json inputs | tensor 名与 dtype_param 名的映射 |
| S2P1_path_list.json | 文件 | 原始路径数据（含 conditions、source_constraints） |

## 输出

| 产出 | 目标文件 | 对应子步骤 |
|------|---------|-----------|
| 参数定义文件 | S2P2_param_def.json（由 builder 生成） | 4.1 |
| 路径列表更新 | S2P1_path_list.json（原位覆写） | 4.2 |
| 可追溯性报告 | S2P2_traceability.md | 4.3（校验性产出） |

执行顺序：4.1 → 4.2 → 4.3，严格顺序。4.2 校验失败修正 reachability_data.json 后重跑；4.3 constraint_note 校验不一致回到 Step 3。

## 4.1 运行 builder 生成 S2P2_param_def.json

使用 Bash 执行 builder 脚本，将 groups 数据展开为完整的参数定义文件：

```bash
python3 {skill_base}/scripts/build_param_def.py \
  --groups S2P2_param_def_groups.json \
  --output S2P2_param_def.json
```

builder 脚本职责：
- 读取 S2P2_param_def_groups.json
- 保留每个 group 的 `group_dims` 嵌套结构（不展平）
- 按固定字段顺序（id → mode → group_dims → per_dtype → constraint_note）输出
- 使用紧凑 JSON 格式（简单数组单行、compound dict 单行）。此格式仅适用于 S2P2_param_def.json，与 S2P2_param_def_groups.json 的格式约束独立

**完成标志**：S2P2_param_def.json 已生成，python3 可正常 `json.load()` 读取。

## 4.2 更新 S2P1_path_list.json

### 执行步骤

1. 将 Step 1（可达性标注）和 Step 2（路径分组）的内存数据序列化为 S2P2_reachability_data.json：

```json
{
  "groups": ["G1", "G2", ...],
  "paths": [
    {"id": "P1", "reachability": "reachable", "group": "G1"},
    {"id": "P2", "reachability": "dead", "dead_reason": "..."}
  ]
}
```

- `groups`：所有 group id 的有序列表
- `paths`：每条路径的 id + reachability + group（reachable 时）+ dead_reason（dead/api_dead/api_warn 时）

2. 运行更新脚本：

```bash
python3 {skill_base}/scripts/update_path_list.py \
  --path-list S2P1_path_list.json \
  --param-def S2P2_param_def.json \
  --reach-data S2P2_reachability_data.json \
  --op-name {op_name} \
  --platform "{platform}"
```

3. 检查 exit code：
   - 0 → 6 项校验通过，继续 4.3
   - 1 → 根据输出修正 S2P2_reachability_data.json 后重跑（最多 3 轮）

**完成标志**：S2P1_path_list.json 已更新，6 项校验全部通过。

## 4.3 生成 S2P2_traceability.md

本节为校验性产出——数据在 Step 3 已准备好，此处格式化并核验一致性。

### 执行规则

- **触发条件表**：tiling 源码中该 group 对应模式的分支判断条件
- **等价推导表**：Step 3 中所有内部变量条件的回溯过程。每行一条内部变量条件；若一个条件同时影响取值列表和约束文字，拆为两行分别记录「写入位置」
- 若 group 无内部变量回溯，等价推导表注明此情况
- **constraint_note 校验**：生成此表后，逐行核验 Step 3 写入的 constraint_note 中所有数值和条件都能在本表「写入位置 = constraint_note」的行中找到对应，确保无中间变量残留

### 复合维度映射

| path | 复合维度名 | 耦合类型 | 子维度 | 耦合关系 | 基准维 |
|------|-----------|---------|--------|---------|--------|
| `{path_id}` | `{compound_name}` | 确定性 | `{dim_a}`, `{dim_b}` | `{关系描述}` | `{基准维名}` |
| `{path_id}` | `{compound_name}` | 约束 | `{dim_a}`, `{dim_b}` | `{约束表达式}` | `{基准维名}` |

- 同一 group 内不同 path 可能有不同的耦合关系，按 path 分行记录
- 若该 group 无耦合维度，此表注明"本 group 无耦合维度，所有路由维度独立取值"

### 格式参考

```markdown
# 参数推导可追溯性报告：{op_name} ({platform})

## Group: {group_id}

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `条件表达式` | tiling.cpp:行号 | 简要说明 |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `{internal_var} op {V}` | L行号: `计算表达式` | `{param} op {V′}` | per_dtype.{dtype}.{param}（取值边界） |
| `{internal_var} op {V}` | L行号: `计算表达式` | `{param} op {V′}` | constraint_note（约束文字） |
```
