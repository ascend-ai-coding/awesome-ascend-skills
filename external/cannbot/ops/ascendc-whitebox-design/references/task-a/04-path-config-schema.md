# Task A Step 4：path_config 完整 Schema（判定逻辑唯一权威）

本文件对应 `01-code-analyzer.md` 的步骤 4a/4b。LLM 分析源码后写出 `S2P1_path_config.json`，脚本 `build_path_list.py` 消费此文件产出最终文件。

**本文件是 path_config 判定逻辑的唯一权威**（conditions 紧凑格式、dtype RHS 规则、变量三分类、glossary、source_constraints、完整性配置、降级）。`00-overview.md` 只给概览与指针。

- **步骤 4a**（读 §路径主体）：写 paths + glossary + source_constraints。
- **步骤 4b**（读 §完整性配置）：写 completeness_checklist + orphan_explanations + tiling_no_kernel_keys + degradations。

---

## 设计原则

- LLM 负责源码分析结果：路径枚举、conditions 提取、变量三分类、约束表。
- 脚本作为黑盒调用，Task A 只关心运行成功并产出 `S2P1_path_list.json` 和 `S2P1_tiling_glossary.md`。
- LLM 每条路径只写 5 个路径主体字段，富化字段由脚本产出。

---

# 一、路径主体（步骤 4a）

## 顶层字段（主体部分）

| 字段 | 必填 | 说明 |
|------|------|------|
| `operator` | 是 | 算子名称 |
| `tiling_file` | 是 | tiling 源码文件名，如 `{op_name}_tiling.cpp` |
| `kernel_file` | 是 | kernel 源码文件名，如 `{op_name}.cpp` |
| `paths` | 是 | 正常路径数组，LLM 从源码分析得出 |
| `glossary` | 是 | 变量含义表数组，含 `category` 字段 |
| `source_constraints` | 是 | 源码约束表数组 |

完整性配置字段 `completeness_checklist`、`orphan_explanations`、`tiling_no_kernel_keys`、`degradations` 见下方「二、完整性配置」。

---

## paths 数组

每条路径 LLM 填写 5 个核心字段：

```json
{
  "id": "T1K1",
  "tiling_key": 100,
  "conditions": "varA==0\nvarB>8\nvarC<=varD",
  "kernel_class": "KernelClassName<template_param, N>",
  "tiling_line": 100
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 路径 ID，格式 `T{t}K{k}` |
| `tiling_key` | int | 该路径的 tiling key 值 |
| `conditions` | string | 分支条件，换行分隔的紧凑格式 |
| `kernel_class` | string | kernel 侧 dispatch 的模板类名，含模板参数 |
| `tiling_line` | int | tiling 源码中该分支的行号 |

脚本自动补全字段：`name`、`source`、`key_instructions`、`input_variables`、`caller_options`、`internal_variables`。

---

## Conditions 紧凑格式

`conditions` 使用换行分隔的紧凑字符串，每行一个条件。脚本自动解析为标准 JSON 数组。

| 紧凑写法 | 解析结果 | 说明 |
|----------|---------|------|
| `varA==0` | `{"var":"varA","op":"==","value":0}` | 右侧数字 → value |
| `varA!="ENUM_VALUE"` | `{"var":"varA","op":"!=","value":"ENUM_VALUE"}` | 引号 → 字符串常量 |
| `dtype_key=="float32"` | `{"var":"dtype_key","op":"==","value":"float32"}` | dtype 编码变量使用语义字符串 |
| `varA==true` | `{"var":"varA","op":"==","value":true}` | 布尔常量 |
| `varA>8` / `varA>=8` / `varA<8` / `varA<=8` | 对应 op | 数值比较 |
| `varA==varB` | `{"var":"varA","op":"==","ref":"varB"}` | 右侧标识符 → ref |
| `varA in [1,2,3]` | `{"var":"varA","op":"in","value":[1,2,3]}` | 枚举 |
| `varA range 1..10` | `{"var":"varA","op":"range","min":1,"max":10}` | 范围 |
| `varA%32==0` | `{"var":"varA","op":"mod_eq","divisor":32,"remainder":0}` | `%` 右侧数字 → mod_eq |
| `varA%varB==0` | `{"expr":"varA % varB","op":"==","value":0}` | `%` 右侧标识符 → expr |
| `boundary:varA==0` | `{"boundary_check":"varA==0"}` | 边界检查 |
| `{"expr":"(a+b)%c","op":"==","value":0}` | 原样保留 | 混合模式兜底 |

解析顺序：JSON 对象 → `boundary:` → `range` → `in` → `%` → 比较运算符。

变量命名：conditions 中 `var` / `ref` / `expr` 的变量名必须使用 tiling 源码中的原始变量名，禁止自行发明语义化名称。

### dtype 编码变量 RHS 规则

当 tiling 源码变量表示输入 tensor dtype 的内部编码时，`conditions` 左侧仍使用 tiling 源码中的原始变量名，右侧必须使用后续流程可直接消费的 dtype 语义字符串，禁止使用内部编码整数。

示例：

| 源码语义 | conditions 写法 |
|----------|----------------|
| float16 / FP16 | `dtype_key=="float16"` |
| float32 / FP32 | `dtype_key=="float32"` |
| bfloat16 / BF16 | `dtype_key=="bfloat16"` |

反例：`dtype_key==1`、`dtype_key==2`、`dtype_key==3`。这些内部编码会丢失 dtype 语义，后续流程不能稳定判断其对应的真实 dtype。

普通数值变量和内部计算变量不做语义化替换，仍保留源码变量关系或数值边界，例如 `numCol>ubFactor`、`numColAlign<=2000`、`blockFactor==1`、`rowFactor!=0`。

dtype 编码到语义字符串的源码依据必须记录在 `source_constraints` 中，保留原始赋值、case 分支或相关源码位置。

例外：dispatch 变量通过框架 API 管理且不存在源码局部变量名时，允许使用描述性名称（如 `tilingKey`），但必须在 `glossary.desc` 中说明其框架 API 来源。

---

## 条件来源约束

每条路径的 `conditions` 只包含该路径自身分支的条件，来源限定为两类：

1. 当前 IsMatch 函数的正向条件：该策略匹配函数内部使函数 return true 的所有条件。
2. 当前 DoTiling / kernel dispatch 块内部的条件判断：同一 tiling key 内不同 kernel dispatch 分支构成独立路径，conditions 需包含 kernel 侧区分条件。

禁止将同一 if-else 链中前置兄弟分支的内部条件或否定形式加入当前路径的 conditions。前置 IsMatch 函数 return false 只需其内部任一条件不满足，将所有条件取反后 AND 到当前路径会引入不存在的约束。

---

## 路径 ID 命名规则

路径 `id` 采用 `T{t}K{k}` 格式，编码 tiling/kernel 两级分支结构。

| 路径类型 | ID 格式 | 说明 |
|----------|---------|------|
| 常规路径 | `T{t}K{k}` | t = tiling 分支序号，k = kernel dispatch 分支序号 |
| 降级路径 | `T{t}K{k}d{d}` | 继承父路径，d = 降级序号 |
| 孤儿 Dispatch | `D{N}` | 无对应 tiling 分支，由脚本自动分配 |

命名约束：同一 tiling 分支共享相同 `T{t}`；同一 tiling 分支内的不同 kernel dispatch 分支用 `K{k}` 区分；T/K 序号按源码出现顺序分配。

Task A 不指定 group 归属。Group 划分由 Task D 在 Phase 2 完成。

---

## 变量三分类判定规则

对 conditions 中出现的每个变量，按以下规则分类：

- `input_variable`：用户可控制的输入属性，包括 tensor shape/dtype/rank 和标量属性直接或间接派生的量。
- `caller_option`：调用者通过 API 调用方式控制的执行路径选项，非 tensor shape/dtype/属性。
- `internal_variable`：路径内部派生量，计算链不经过任何用户可控属性，仅依赖平台常量、编译时常量或其他内部变量。

判定流程：直接由用户设置 → `input_variable`；反映调用者抽象选择 → `caller_option`；反映框架编码信号 → `internal_variable`；内部计算量必须追溯计算链后再分类。

当内部变量边界检查导致不同代码路径时，conditions 中记录 `boundary:内部变量==边界值`，并在源码约束或后续说明中保留完整计算链。边界值数学反推由 Task D 完成，Task A 只记录原始条件。

只有 `input_variable` 和 `caller_option` 会映射为 `S2P2_param_def.json` 的维度。

---

## glossary 数组

```json
{
  "tiling_var": "{tiling源码变量名}",
  "semantic_name": "{param}_{attr}",
  "category": "input_variable",
  "type": "int",
  "desc": "该变量的一句话含义描述",
  "shape_contribution": null
}
```

规则：

1. `tiling_var` 使用 tiling 源码中的原始变量名，与 conditions 中变量名严格一致。
2. `semantic_name` 使用 `{参数名}_{属性}`，仅用于文档可读性。
3. `category` 为 `input_variable` / `caller_option` / `internal_variable` 三选一。
4. `type` 为变量数据类型，如 int / float / bool / enum。
5. `desc` 用一句话描述变量含义。
6. `shape_contribution` 为必选字段，用于描述当前 tiling 变量对 input/output shape 的贡献，只服务于变量的 shape 语义理解。
7. conditions 中出现的每个变量都必须在 glossary 中有对应记录。

`shape_contribution` 规则：

- `shape_contribution` 为必选字段，用于记录当前 tiling 变量与 input/output shape 的关系。
- 不影响 shape 的变量写 `null`。
- 影响 shape 的变量写 object，必须包含 `shape_relation`。
- `representative` 可选，用于记录代表性 shape 构造。
- 如果变量表示多个轴的乘积、shape size 或 aligned size，应在 `shape_relation` 中直接写明。
- 如果填写 `representative`，需说明它是否只是代表构造。

`shape_contribution` object 示例：

```json
{
  "tiling_var": "outerSize",
  "semantic_name": "input_leading_axes_product",
  "category": "input_variable",
  "type": "uint32_t",
  "desc": "Product of the leading axes before the normalized or reduced axes.",
  "shape_contribution": {
    "shape_relation": "outerSize = product(input leading axes before normalized/reduced axes)",
    "representative": "input.shape=[outerSize, innerSize] is only a rank-2 representative when innerSize describes the trailing normalized/reduced axes; it is not the only valid shape construction."
  }
}
```

---

## source_constraints 数组

```json
{
  "id": "C1",
  "source_expr": "{源码中的原始表达式}",
  "source_location": "{tiling_file}:{line}",
  "variables": ["{变量名}"],
  "semantics": "{该约束的含义}"
}
```

`source_expr` 必须逐字抄录源码表达式，不能改写或简化。

---

# 二、完整性配置（步骤 4b）

Task A 不需要理解 `build_path_list.py` 的内部实现，只需在步骤 4 写出完整配置，在步骤 5 调用固定命令并确认输出文件生成。

## completeness_checklist

`completeness_checklist` 是必填字段，每次 Task A 都必须写入。

```json
{
  "api_variants": {"status": "covered", "evidence": ["示例：OpName + InplaceOpName 共享入口"]},
  "format_variants": {"status": "na", "evidence": []},
  "mode_variants": {"status": "covered", "evidence": ["示例：N norm × M mode × K dtype"]},
  "quant_variants": {"status": "na", "evidence": []},
  "optional_input_combos": {"status": "covered", "evidence": ["示例：output 存在性决定 mode"]},
  "tiling_analysis": {"status": "covered", "evidence": ["示例：Tiling4OpName 完整分析"]}
}
```

检查项说明：

| 检查项 | 说明 |
|--------|------|
| `api_variants` | 算子是否有多种调用方式，如 Tensor vs Scalar、inplace vs outplace |
| `format_variants` | 是否支持多种数据格式，如 NCHW/NHWC/ND/5D |
| `mode_variants` | 是否有 static/dynamic、training/inference 等模式切换 |
| `quant_variants` | 是否覆盖所有量化类型 |
| `optional_input_combos` | 每个可选输入 present/absent 是否都在某条路径中出现 |
| `tiling_analysis` | tiling 逻辑是否完整分析；委托通用框架时 status=delegated |

状态值：`covered` / `missing` / `na` / `delegated`。`delegated` 仅用于 `tiling_analysis`。

`dispatch_coverage` 字段由脚本自动填充，LLM 无需填写。

---

## orphan_explanations

孤儿 dispatch 指 kernel 侧存在 dispatch key，但 tiling 侧无法产生该 key。LLM 不直接创建 dead 路径，只在 `orphan_explanations` 中解释。

集合运算（`orphan_keys = active_keys - declared_keys`）由脚本执行，LLM 只对脚本可能报出的 orphan key 提供解释文本（识别过程见 `03-step3-kernel.md` §Dispatch 覆盖规则）。

### 字段 Schema

```json
{
  "100": {
    "dead_detail": "{mode}模式要求{var}=={value} ({tiling_file}:{line})，{dtype}无法进入此模式",
    "key_instructions": ["KernelClassName<dtype_template, N>"]
  }
}
```

字段规则：

- key 为孤儿 tiling_key 值，使用字符串形式。
- `dead_detail` 必须溯源到 tiling 源码具体行号。
- `key_instructions` 从 kernel P0 源码提取 kernel 类名，作为单元素数组。

约束：

- 只使用 Scout-K、source_scope 和已读取源码中的已有信息，不新增范围外读取。
- 禁止将孤儿 dispatch 标记为 reachable 或 disputed。
- 禁止直接创建 `D{N}` 路径。

---

## tiling_no_kernel_keys

```json
[<key1>, <key2>, <key3>]
```

tiling 公式能计算出的 key 值，但 kernel 侧没有对应 dispatch 时填写。LLM 不将这些组合写入 `paths`，而是在此声明，供步骤 5 固定脚本处理。

---

## degradations 数组

当 tiling 代码中存在“先进入某 mode 分支，再因内部变量边界检查回退到另一个 mode 或使用不同 kernel”的模式时，降级后的路径必须作为独立条目声明。

判定标准满足任一即需独立声明：

1. 内部变量边界检查导致 key 分量被重新赋值为其他值。
2. 内部变量边界检查导致实际执行的 kernel 指令与该分支主路径不同。

禁止合并降级路径到降级目标 mode 的路径中；禁止省略降级路径，即使它与已有路径使用相同 kernel。

### 字段 Schema

```json
{
  "parent_id": "T4K1",
  "id": "T4K1d1",
  "tiling_key": 100,
  "trigger": "varA==0",
  "kernel_class": "KernelClassName<dtype_template, N>",
  "tiling_line": 200
}
```

| 字段 | 说明 |
|------|------|
| `parent_id` | 降级前的父路径 ID |
| `id` | 降级路径 ID，格式 `{parent_id}d{d}` |
| `tiling_key` | 降级后的 tiling key 值 |
| `trigger` | 触发降级的条件，紧凑格式字符串 |
| `kernel_class` | 降级后实际执行的 kernel 类名 |
| `tiling_line` | 降级代码的行号 |

LLM 只填写上述字段；其他富化信息由步骤 5 固定脚本产物体现。

---

## 脚本调用（步骤 5）

```bash
python3 {skill_base}/scripts/build_path_list.py \
  --config {output_dir}/S2P1_path_config.json \
  --scout-k {output_dir}/S2P0_scout_k.json \
  --output-dir {output_dir}
```

产出：
- `S2P1_path_list.json`（含 paths / source_constraints / completeness_checklist）
- `S2P1_tiling_glossary.md`（变量含义表）
