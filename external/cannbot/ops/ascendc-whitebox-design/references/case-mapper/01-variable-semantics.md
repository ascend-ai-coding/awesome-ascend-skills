# Step 5.1：变量语义与 Mapper 指导书生成

> **职责**：将 Step 2 最终事实产物翻译为 Step 5.2 可直接实现的 mapper 代码指导书，并生成算子级 `S5_data_range_policy.json`。本步骤不重新分析源码。

`S5_variable_semantics.md` 是 Step 5.2 的 mapper 代码指导书，不是仅供阅读的语义摘要。它必须把 Step 5.2 需要实现的 dtype 规则、format 规则、`attributes` 构造规则、`const_inputs` 构造规则、input shape assembly plan、rank 覆盖目标、output 派生规则、network mapping 和 mapper 注意事项写到可直接编码的程度。

Step 5.2 只负责翻译执行这份指导书，生成 `S5_case_mapper.py` 和 low 档 mapped cases；Step 5.2 不负责重新设计变量语义、rank 覆盖策略或 shape 组装方案。若 Step 5.2 仍需自行猜测变量语义、rank 选择、axis group、shape 拆分、execution value 构造或 output 派生规则，则 Step 5.1 未完成。

## Step 5.1 输入输出

输入：

- `S2P1_operator_model.json`
- `S2P2_cases.json` 前 30 行
- `S2P1_low_configs.json` 前 30 行
- `S2P1_tiling_glossary.md`
- `infershape.cpp` output shape 相关片段，仅在 operator model output shape 信息不足时按需读取

输出：

- `S5_variable_semantics.md`：Step 5.2 的 mapper 代码指导书，必须写到可直接实现 `S5_case_mapper.py` 的程度。
- `S5_data_range_policy.json`：data_range 扩展策略，供 Step 5.4 静态展开阶段使用。

不输出：

- `S5_case_mapper.py`
- `S5_mapped_cases_path.json`、`S5_mapped_cases_network.json`
- `S5_mapped_cases_low_shape.json`、`S5_cases_low.json`、`S5_cases_high.json`
- 任何 `S1*`、`S2*`、`S3*`、`S4*` 文件

## 输入读取

Step 5.1 只读取 mapper 消费所需的最终事实产物。禁止读取 Step 2 推理链、生成规格或源码补字段语义。

### 必读

| 文件 | 用途 |
|------|------|
| `S2P1_operator_model.json` | input/output/attribute/const input 结构、dtype/rank/shape 约束、param_type、output shape rule。 |
| `S2P2_cases.json` 前 30 行 | 确认 path case 字段形态。批量处理由 Step 5.2 脚本运行时 `json.load` 完成。 |
| `S2P1_low_configs.json` 前 30 行 | 确认 network config 字段形态。批量处理由 Step 5.2 脚本运行时 `json.load` 完成。 |
| `S2P1_tiling_glossary.md` | 解释 case/config 字段最终语义，并提供变量对 input/output shape 的贡献关系（`shape_contribution`）。 |

读取 `S2P2_cases.json` 和 `S2P1_low_configs.json` 样本时，必须使用 Read 工具的 `offset=1, limit=30`，禁止全文读取。

### 按需

仅当 operator model 的 output shape 信息不足时，读取 `infershape.cpp` 的 output shape 片段。读取范围只限 output shape，不得用于补 tiling、kernel、接口或 case 字段语义。

如果 `S2P1_tiling_glossary.md` 无法解释 mapper 必需字段，Step 5.1 必须报告 Step 2 语义产物不完整，不得读取推理链、生成规格或路径枚举补齐。

## 功能 A：生成 Step 5.2 代码指导书

本功能输出 `S5_variable_semantics.md`。该文件是 Step 5.2 生成 `S5_case_mapper.py` 的 mapper 代码指导书，必须把 mapper 需要实现的语义和构造规则写到可直接编码的程度。

`S5_variable_semantics.md` 的正文说明必须使用中文。代码标识符、JSON 字段名、函数名、dtype 名称、源码变量名、case 字段名和伪代码变量名保持原文，不翻译。若上游事实产物中的自然语言为英文，Step 5.1 应将其语义转写为中文说明，但不得改写其中的代码表达式或字段名。

Step 5.1 必须把 Step 5.2 要实现的逻辑写清楚，包括 dtype 规则、format 规则、`attributes` / `const_inputs` 构造规则、input shape assembly plan、rank 覆盖目标、output 派生规则、network mapping 和 mapper notes。

### 必填主题

`S5_variable_semantics.md` 必须按以下章节组织。每个章节都应提供 Step 5.2 生成 mapper 代码所需的规则；不适用内容写“不适用”，不得省略章节。

| 章节 | 指导书内容 |
|------|------------|
| `dtype` | dtype 控制字段、合法值、名称归一化、input/output dtype 同步或固定规则。 |
| `format` | input/output format 构造规则；默认 `ND`，若 path/config 明确存在 format 约束，则按当前 case 的 path/config 信息写入对应 format。 |
| `input shapes` | 主输入判定、shape assembly plan、依赖 input shape 规则；必须包含 Step 5.2 可直接实现的自然语言规则或伪代码。 |
| `shape coverage plan` | 主输入合法维度结构覆盖目标，并明确主输入 rank 覆盖目标。依赖 input 和 output 不作为独立覆盖目标。 |
| `outputs` | output dtype、shape、optional placeholder 规则；shape 只从 input 或明确字段派生。 |
| `execution values` | `attributes` 和 `const_inputs` 构造规则；说明 operator attributes、`.ValueDepend()` const input values、默认值如何注入，以及哪些字段只进入 dtype/format/shape/meta。 |
| `network mapping` | network config 到算子参数空间的映射；network-only 信息放入 `meta`。 |
| `mapper notes` | 保留的 meta 字段、边界条件、禁止猜测项。 |

### Shape Assembly Plan 要求

`input shapes` 是 `S5_variable_semantics.md` 中最关键的代码指导章节。Step 5.1 必须在这里写出 shape assembly plan；Step 5.2 只能实现该 plan，不得重新设计 shape 组装策略。

shape assembly plan 至少说明：

- 主输入是谁。
- 主输入 rank 如何选择或覆盖。
- 主输入各轴如何取值。
- 参与 shape construction 的字段及其语义。
- 每个 product / shape_size / aligned_size 字段对应的 axis group。
- 依赖 input 如何同步、broadcast 或派生。
- optional input 不输入时如何保留 descriptor 并写 `shape = null`。
- output shape 派生所需的 input 关系。
- Step 5.2 可直接实现的自然语言规则或伪代码。

如果存在 `shape_contribution.shape_relation`，必须使用其中的变量关系约束 shape 组装。product 变量必须拆到对应 axis group；shape_size 或 aligned_size 变量必须说明它们如何约束 shape 或派生字段。

如果不存在 shape 变量关系，也必须根据 `S2P1_operator_model.json` 的 rank/shape 约束给出默认 shape 采样方案。默认方案应说明 rank 覆盖目标、单轴上限、元素量上限和 dependent input 派生规则。

`representative` 只能作为一个构造样例，不能替代 rank 覆盖方案或 shape assembly plan。

### Execution Values 要求

`execution values` 章节必须明确每个非 tensor 执行值写入哪个字段：

- `.Attr("...")` 注册的 operator attribute 写入 `attributes`。
- `.ValueDepend()` 标记的 const input value 写入 `const_inputs`。
- Tensor/TensorList descriptor 写入 `inputs` 或 `outputs`，不得写入 `const_inputs`。
- shape、dtype、format、data_range、tiling/router 信息和 mapper 审计字段不得写入 `attributes` 或 `const_inputs`。
- `attributes` 和 `const_inputs` 不得同名重复。

Step 5.1 必须说明默认值如何注入 `attributes` 或 `const_inputs`，以及哪些 case/config 字段仅用于 dtype、format、shape 或 `meta`。

### Output 要求

`outputs` 章节必须说明每个 output 的 dtype、format、param_type 和 shape 派生规则。outputs 必须由 mapper 完整生成，后续脚本不得推导 outputs。

optional output 不输出时必须保留原 output 名并写 `shape = null`。output 任何层级不得包含 `data_range`。

### S5_variable_semantics.md 输出契约

必须写入以下章节。缺少不适用内容时写“不适用”，不要省略章节。

```markdown
# {op_name} Step 5 Variable Semantics

> 本文件是 Step 5.2 的 mapper 代码指导书。Step 5.2 必须将本文规则翻译为 `S5_case_mapper.py`，不得重新设计变量语义、rank 覆盖策略、execution value 构造或 shape 组装方案。
> 正文说明使用中文；代码标识符、JSON 字段名、函数名、dtype 名称、源码变量名、case 字段名和伪代码变量名保持原文。

## dtype
说明 dtype 控制字段、合法值、名称归一化、input/output dtype 同步或固定关系。

## format
说明 input/output format 构造规则。默认情况下所有 input/output 使用 `ND`；若 Step 2 path/tiling 分析、path case 字段、network config 或 operator model 明确存在 format 约束，Step 5 mapper 必须按当前 case 的 path/config 信息写入对应 format。empty case 和 high data_range case 必须继承原始 low case 的 format，不得修改。

## input shapes
说明主输入判定、主输入 shape construction、依赖 input shape 关系，并给出 shape assembly plan。无论是否存在 shape_contribution，shape assembly plan 都必须说明 Step 5.2 如何生成具体 input shape；若存在 product/shape_size/aligned_size 变量，必须说明字段到 axis group 的映射、rank 覆盖或过滤规则、shape 拆分方案和 dependent input 规则。必须给出自然语言规则或伪代码。

## shape coverage plan
说明主输入合法维度结构覆盖目标和主输入 rank 覆盖目标；若未覆盖 operator model 的完整 rank 范围，说明过滤后的 rank 集合或范围及原因。依赖 input 和 output 不作为独立覆盖目标。

## outputs
说明 output dtype、format、param_type、shape 和 optional placeholder 规则。optional output 不输出时保留 descriptor，并使用 `shape = null`。

## execution values
说明 Mapper-v1 的 `attributes` 和 `const_inputs` 如何构造。`attributes` 只写 operator attributes，`const_inputs` 只写 `.ValueDepend()` 标记的 const input values；默认值如何注入也必须在本节说明。

本节还必须明确哪些 case/config 字段仅用于 dtype、format、shape 或 `meta`，不得进入 `attributes` 或 `const_inputs`。白盒路径、tiling key、group、shape 构造变量、case 来源、网络来源等 tiling/router 信息和 mapper 审计字段必须放入 `meta`。

## network mapping
说明 network config 如何映射到 `attributes`、`const_inputs`、`inputs`、`outputs` 和 `meta`。

## mapper notes
说明边界条件、保留 meta 字段和禁止猜测项。
```

## 功能 B：data_range policy 建模

`S5_data_range_policy.json` 描述每个 input tensor 是否参与 data_range expansion，以及支持哪些标准非 `normal` data_range。

本功能只消费功能 A 已明确的 input tensor 语义，不重新推理字段语义，不读取额外输入。

### 标准 data_range

Step 5.1 只能从以下标准 data_range 中选择，禁止自造类型。

| data_range | 用途 |
|------------|------|
| `normal` | base/low case 默认数据范围，不允许出现在 policy 的 `supported` 中 |
| `zero` | 0 值边界 |
| `extreme` | dtype 范围内极大/极小值 |
| `negative` | 负值数据 |
| `tiny_pos` | 很小的正数 |
| `all_ones` | 全 1 |
| `near_zero` | 接近 0 的非零值 |
| `with_inf` | 包含 inf，仅适用于支持 inf 的浮点输入 |
| `with_nan` | 包含 nan，仅适用于支持 nan 的浮点输入 |

`S5_data_range_policy.json` 中的 `supported` 只能包含非 `normal` 类型：`zero`、`extreme`、`negative`、`tiny_pos`、`all_ones`、`near_zero`、`with_inf`、`with_nan`。

### S5_data_range_policy.json 输出形式

输出文件 `S5_data_range_policy.json`，供 Step 5.4 静态展开读取。

文件必须使用以下格式：

```json
{
  "version": 1,
  "mode": "per_input_cyclic",
  "inputs": {
    "input_name": {
      "participates": true,
      "supported": ["zero", "extreme", "negative", "tiny_pos", "all_ones", "near_zero", "with_inf", "with_nan"]
    }
  }
}
```

字段规则：

- `version` 固定为 `1`。
- `mode` 固定为 `per_input_cyclic`。
- `inputs` 必须覆盖算子的所有 input descriptor，key 使用 input 名。
- 每个 input 只包含 `participates` 和 `supported`。
- `participates=true` 时，`supported` 写该 input 支持的标准非 `normal` data_range。
- `participates=false` 时，`supported` 必须为 `[]`。

`S5_data_range_policy.json` 不得包含 case 级内容：

- mapped case 字段，如 `id`、`source`、`attributes`、`const_inputs`。
- tensor 实例内容，如 input/output shape、dtype、format、param_type、data_range。
- low/high/empty/shape/range variants。
- range 展开状态，如 `range_by_input`、`range_index` 或展开结果。

Step 5.2 不生成或修改 `S5_data_range_policy.json`。Step 5.4 直接读取该文件，并由静态区执行 per-input cyclic expansion。

## 输出 descriptor 规则

优先使用 `S2P1_operator_model.json.outputs[*]` 中的 output dtype、shape、param_type 和 TensorList 信息：

- `same_as_input`：直接复用对应 input shape。
- `fixed`：使用固定 shape。
- `derived`：按 operator model 表达式或结构化规则派生。

仅当 operator model 信息不足时，读取 `infershape.cpp` 的 output shape 片段。不得借此补 tiling、kernel、接口或 case 字段语义。

output descriptor 派生规则必须写入 `S5_variable_semantics.md` 的 `outputs` 章节，达到 Step 5.2 可直接实现 `derive_outputs(inputs, attributes, const_inputs, meta)` 的程度。

## 完成条件

- `S5_variable_semantics.md` 已写入，且内容足以指导 Step 5.2 生成 `S5_case_mapper.py`。
- `S5_variable_semantics.md` 包含完整 shape assembly plan；无论是否存在 shape 变量，Step 5.2 都能据此生成具体 input shape。
- 若存在 product / shape_size / aligned_size 变量，已说明字段语义、axis group、rank 覆盖或过滤、shape 拆分方案、dependent input 规则和 output 派生关系。
- 若不存在 shape 变量，已说明默认 shape 采样方案、rank 覆盖目标、单轴上限、元素量上限和 dependent input 派生规则。
- `S5_variable_semantics.md` 已说明 input/output format 构造规则；默认 `ND`，存在 path/config format 约束时已说明如何按当前 case 写入对应 format。
- `S5_variable_semantics.md` 已说明 `attributes` 与 `const_inputs` 如何构造，以及哪些字段不进入二者。
- `S5_variable_semantics.md` 已说明 outputs 完整 descriptor 的生成规则，且能直接指导 `derive_outputs(...)` 实现。
- `S5_data_range_policy.json` 已写入，且不包含 case 级内容或展开结果。
- 若 Step 5.2 仍需自行猜测变量语义、rank 选择、axis group、shape 拆分或 output 派生规则，则 Step 5.1 未完成。

## 禁止事项

输入边界：

- 禁止读取 `S2P2_traceability.md`、`S2P2_param_def.json`、`S2P2_dim_spec.json`、`S2P1_path_list.json` 补语义。
- 禁止读取 tiling、kernel、`_def.cpp` 或注册源码补 shape、接口、mode、tiling key 或 case 字段语义。
- 禁止修改 `S2P1_*`、`S2P2_*` 输入产物。

语义边界：

- 禁止凭字段名猜测语义。
- 禁止只写 representative 示例而不写 shape assembly plan。
- 禁止把 Step 5.2 的 shape 组装设计留空，让 Step 5.2 自行发挥。
- 禁止把 shape、dtype、format、data_range、tiling/router 信息或 mapper 审计字段写入 `attributes` 或 `const_inputs`。
