# Step 5：生成 S2P2_gen_cases.py

> **前置条件**：Step 4 已完成，S2P2_param_def.json 已就绪
>
> **职责边界**：本文档为规范文档，仅定义规则、约束和校验标准。完整脚本模板在 `{skill_base}/scripts/gen_cases_template.py` 中，直接复制使用。

## 输入

脚本运行时从 `S2P2_param_def.json` 读取所有数据。LLM **不硬编码**任何维度取值，仅生成提取逻辑。

| param_def 字段 | 运行时提取方式 |
|---|---|
| `dtype_tensors[0].param` | `DTYPE_PARAM` 变量 |
| `groups[].per_dtype[{dtype}][*].path` / `key` | 直接读取 entry 的 `path`/`key` 字段 |
| `groups[].per_dtype[{dtype}][*].{其他字段}` | `extract_entry_dims(entry)` 自动识别并构建 dim_dicts |
| `groups[].group_dims.{dim}` 数组（group 级维度） | `extract_group_dims(group)` 自动识别并构建 dim_dicts |

## 输出

| 产出 | 文件 | 说明 |
|------|------|------|
| 生成脚本 | S2P2_gen_cases.py | JSON 驱动的可执行 Python 脚本 |
| 测试用例 | S2P2_cases.json | 脚本执行后自动产出的 JSON 文件 |

## 硬性规则

1. **JSON 驱动**。脚本运行时从 `S2P2_param_def.json` 读取所有数据。LLM 仅生成提取逻辑（`extract_entry_dims`、`extract_group_dims` 函数调用）和通用循环，不硬编码任何维度取值。

2. **约束隐含在取值中**。所有路由约束（dtype 依赖、对齐要求、范围互斥）在 **Task D Step 3 选值** 时已消化完毕。脚本中不含 `if` 约束过滤逻辑。

3. **运行时提取，运行时采样**。`extract_entry_dims()` 从 per_dtype entry 自动识别字段类型并构建 dim_dicts；`extract_group_dims()` 从 group 对象自动提取 group 级维度。提取结果传入 `compress_per_dtype()` 和 `compress_group_pool()` 进行采样。

4. **去重**。全字段去重：对完整 case dict 排序后构造元组作为去重键，防止完全重复的 case。去重仅作为安全网捕获极端边界情况。

5. **零外部依赖**。仅使用 Python 标准库（`json`、`os`、`random`、`argparse`、`collections`、`zlib`）。不 import 第三方包。

6. **自执行**。`python3 S2P2_gen_cases.py` 即可产出 `S2P2_cases.json`，无需命令行参数。

## 压缩函数接口

以下函数由 `gen_cases_template.py` 提供完整实现，此处仅声明接口契约。生成脚本时不得修改函数签名或违反约束条件。

| 函数 | 输入 | 输出 | 约束 |
|------|------|------|------|
| `compress_per_dtype(dim_dicts, cap)` | `dict[str, list[dict]]`：维度名 → 取值列表；`int`：每个 per_dtype 路径生成的组合数 | `list[dict]`：生成的 dict 列表（≤ cap 条） | 循环生成 cap 个组合，每维度独立随机选值，固定 seed 保证可复现；可选值空间不足 cap 时以实际唯一组合数为准 |
| `compress_group_pool(dim_dicts)` | `dict[str, list[dict]]`：维度名 → 取值列表 | `list[dict]`：合并后的 POOL | 单维度直接返回原 list；多维度池大小 = 各维度最小长度 |
| `shuffled_pool(base, seed)` | `list[dict]`（base POOL）+ `int`（seed） | `(list[dict], int)`：(打乱后的池, 位置指针 0) | 每个 group 独立 seed；池耗尽时用相同 seed 重新打乱 |

## 提取函数规则

### extract_entry_dims(entry)

从 per_dtype entry 中提取维度字段，构建 `dim_dicts` 供 `compress_per_dtype` 使用。

**字段识别规则**：
- `path`、`key` → 跳过（不作为维度）
- 值为 dict list（如 compound `{compound_dim}`）→ 直接作为 dim_dicts 的一个 key
- 值为 flat array（如 `["NHWC"]`、`[1]`）→ 转为 `[{field: v} for v in values]`

### extract_group_dims(group)

从 `group["group_dims"]` 提取 group 级维度。

**提取规则**：遍历 `group["group_dims"]` 的所有字段，值为 list 的 → 转为 `[{field: v} for v in values]`

## 生成逻辑

所有 group 统一使用通用循环模式，无需为每个 group 单独编写代码。

### 生成步骤

1. 遍历 `param_def["groups"]`
2. 对每个 group：`extract_group_dims(group)` → `compress_group_pool()` → `shuffled_pool()`
3. 遍历 group 的 `per_dtype` 每个 dtype 的每个 entry：`extract_entry_dims(entry)` → `compress_per_dtype()` → 对每个组合从 pool 抽取 1 项 → 组装 case
4. 池耗尽时用相同 seed 重新打乱

### case dict 结构

每条 case 必须包含以下字段：

| 字段 | 来源 | 说明 |
|------|------|------|
| `_group` | group.id | 当前 group 标识 |
| `{dtype_param}` | dtype 名称 | 主 dtype 的取值（如 `"float16"`） |
| `path` | entry.path | 字符串 |
| `key` | entry.key | 整数 |
| per_dtype 维度 | `**p` 解包 | 当前 per_dtype 组合的所有维度 |
| group 级维度 | `**gp` 解包 | 当前 pool 项的所有维度 |

完整代码模板见 `{skill_base}/scripts/gen_cases_template.py`。

## 池化抽样规则

- `shuffled_pool(base, seed)` 函数：返回 `(pool, 0)`，其中 pool 是 base 打乱后的副本
- 每个 group 使用独立 `seed`（`(group_idx + 1) * 100`），不同 group 错开切片实现互补覆盖
- `_a` 由脚本运行时从 JSON 累加计算（遍历所有 group × 所有 dtype 的 per_dtype entry 总数）
- `_default_cap = max(min(10, 100 // _a), 1)`
- cap 通过 `--cap` 命令行参数传入；无参数运行时使用 default_cap

  | a 范围 | cap | 效果 |
  |--------|-----|------|
  | ≤ 10 | 10 | 小算子，每 entry 可组合 10 种代表配置 |
  | 11 | 9 | cap 随 a 增长逐步收敛 |
  | 12 | 8 | |
  | 13–14 | 7 | |
  | 15–16 | 6 | |
  | 17–20 | 5 | 中等 |
  | 21–25 | 4 | |
  | 26–33 | 3 | 多 dispatch |
  | 34–50 | 2 | 高度分支 |
  | ≥ 51 | 1 | 密集算子，每 entry 仅 1 种代表配置，依赖 group POOL 提供多样性 |

- 池耗尽时调用 `shuffled_pool` 用 **相同 seed** 重新打乱（结果与上一轮完全一致），从头消费

## 输出格式

`S2P2_cases.json`：

```json
[
  {"_group": "{group_a}", "{dtype_param}": "{dtype_val}", "{dim1}": {v1}, "{dim2}": {v2}},
  {"_group": "{group_b}", "{dtype_param}": "{dtype_val}", "{dim1}": {v1}}
]
```

- 每条 case 必含 `_group`（对应 `S2P2_param_def.json` 的 group `id`）
- 每条 case 必含 dtype key（key 名为 `S2P2_param_def.json` 的 `dtype_tensors[0].param`）
- 每条 case 必含 `path`（字符串）和 `key`（整数）
- 每条 case 必含所有路由维度（通过 `**` 解包写入；复合维度展开为各子维度的独立字段）

## 脚本模板

使用 Bash 复制模板到产出目录：

```bash
cp {skill_base}/scripts/gen_cases_template.py S2P2_gen_cases.py
```

模板为通用 JSON 驱动脚本，无需任何修改。

## 生成后自检

`S2P2_gen_cases.py` 写完后，主 Agent 执行以下自检：

**阶段 1：执行**。`python3 S2P2_gen_cases.py` 返回码为 0，并检查输出中的 `a=N, default_cap=M, cap=M` 信息（无参数运行时 cap==default_cap；若需覆盖，用 `python3 S2P2_gen_cases.py --cap X` 运行，此时 cap==X；default_cap 由公式 `max(min(10, 100//a), 1)` 计算，可人工复核）。

**阶段 2：完整性**。S2P2_cases.json 中：
- 每个 group 的每类 dtype 至少有一条 case
- 所有 case 无完全重复（全字段去重，first-win）
- JSON 语法有效
- 所有 case 均包含 `path`（string）和 `key`（int）字段，无缺失
- 所有 case 的顶层 key 均为标量字段（string/int/float/bool），不存在 dict 或 array 类型的值（复合维度必须已展开为独立字段）

**阶段 3：一致性**：
- 脚本正确从 `S2P2_param_def.json` 读取数据（验证 `extract_entry_dims` 和 `extract_group_dims` 函数逻辑）
- `constraint_note` 描述的约束在取值中全部体现（人工核对——constraint_note 为自由文本，不要求自动化校验）
