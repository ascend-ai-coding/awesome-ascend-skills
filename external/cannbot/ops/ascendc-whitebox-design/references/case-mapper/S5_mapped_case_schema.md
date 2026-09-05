# S5 MappedCase Schema

> 本文定义 Step 5 mapped cases 的固定数据结构。Mapper-v1 的唯一字段契约以 `S5_case_json_schema.md` 为准；本文只补充 Step 5 各阶段产物的 source、文件级约束和生命周期要求。

## 1. Schema Authority

所有 Step 5 输出 JSON case 必须遵守 `S5_case_json_schema.md`。

固定顶层字段为：

```json
{
  "id": "low_case_00",
  "source": "shape",
  "attributes": {},
  "const_inputs": {},
  "inputs": {},
  "outputs": {},
  "meta": {}
}
```

执行参数分为：

- `attributes`：`.Attr("...")` 注册的 operator attributes。
- `const_inputs`：`.ValueDepend()` 标记的 const input values。

Tensor、TensorList、`shape = null`、`data_range`、`meta` 和 TTK CSV 映射规则均以 `S5_case_json_schema.md` 为准。

## 2. Source And Files

`source` 表示当前 case 在 Step 5 流程中的来源或变体类型。

| source | 使用位置 | 含义 |
|--------|----------|------|
| `path` | `S5_mapped_cases_path.json` | 由 `S2P2_cases.json` 映射得到的 path base case。 |
| `network` | `S5_mapped_cases_network.json` | 由 `S2P1_low_configs.json` 映射得到的 network base case。 |
| `shape` | `S5_mapped_cases_low_shape.json`、`S5_cases_low.json`、`S5_cases_high.json` | shape low case。 |
| `empty` | `S5_cases_low.json`、`S5_cases_high.json` | empty low case。 |
| `range` | `S5_cases_high.json` | data_range high case。 |

文件级 source 约束：

| 文件 | 允许 source |
|------|-------------|
| `S5_mapped_cases_path.json` | `path` |
| `S5_mapped_cases_network.json` | `network` |
| `S5_mapped_cases_low_shape.json` | `shape` |
| `S5_cases_low.json` | `shape`、`empty` |
| `S5_cases_high.json` | `shape`、`empty`、`range` |

## 3. Base Case

`build_low_base_case(record, source, index)` 输出 base case。path 和 network 的 base case 使用同一固定 schema，区别只在字段取值来源和 mapper 指导书指定的映射规则。

base case 要求：

- `source` 必须为 `path` 或 `network`。
- `attributes` 只包含 `S5_variable_semantics.md` 声明的 operator attributes。
- `const_inputs` 只包含 `S5_variable_semantics.md` 声明的 `.ValueDepend()` const input values。
- `inputs` 和 `outputs` 必须按 `S5_case_json_schema.md` 构造完整 descriptor。
- `outputs` 必须由 mapper 完整生成，不得留给后续脚本根据 input 或 operator model 推导。
- optional input/output 不出现时不得删除 descriptor，必须保留原位置并写 `shape = null`。
- 所有 input descriptor 的 `data_range` 必须为 `normal`。
- `meta` 只写入 mapper 和审计信息，不得写入执行字段。

## 4. Shape Low Case

`make_path_shape_case(case)` 和 `make_network_shape_case(case)` 输出 shape low case。

shape low case 要求：

- `source` 必须为 `shape`。
- 顶层 schema 与 base case 保持一致。
- `inputs` 必须包含最终用于 low 档执行的完整 input shape/dtype/format/data_range 信息。
- `outputs` 必须按最终 `inputs`、`attributes`、`const_inputs` 和 `meta` 重新派生并完整写入。
- `meta` 必须保留可审计的 base case 关联和 shape variant 信息。
- 所有 input Tensor 的 `data_range` 必须为 `normal`。
- 所有 input TensorList 外层和 child Tensor 的 `data_range` 必须为 `normal`。

## 5. Empty Low Case

empty low case 由 Step 5.3 固定脚本基于 `S5_mapped_cases_low_shape.json` 中的 shape low case 机械追加，输出到最终 low 用例文件 `S5_cases_low.json`。

empty low case 要求：

- `source` 必须为 `empty`。
- 顶层 schema 与 shape low case 保持一致。
- empty tensor 使用含 0 维度的 `shape` 表达。
- optional placeholder 仍使用 `shape = null` 表达。
- TensorList 整体 empty 使用 `tensor_count = 0` 和 `tensors = []` 表达。
- `meta` 必须保留 empty variant 的审计信息。
- 所有 input descriptor 的 `data_range` 必须为 `normal`。

## 6. Range High Case

range high case 由 Step 5.4 固定脚本基于 `S5_cases_low.json` 和 `S5_data_range_policy.json` 机械生成，输出到最终 high 用例文件 `S5_cases_high.json`。

range high case 要求：

- `source` 必须为 `range`。
- 顶层 schema 与 low case 保持一致。
- `attributes`、`const_inputs`、`inputs`、`outputs` 和 `meta` 的结构不得改变。
- 至少一个参与 data_range expansion 的 input 使用非 `normal` data_range。
- output 任何层级不得写入 `data_range`。
- `meta` 必须保留 range variant 的审计信息。

## 7. Data Range

Step 5.2 只生成 low 档 mapped cases，所有 input 的 `data_range` 必须为 `normal`。

TensorList input 需要同时满足：

- 外层 `data_range = "normal"`。
- 每个 child Tensor 的 `data_range = "normal"`。

非 `normal` data_range 扩展由 Step 5.4 处理，不属于 Step 5.2 schema 生成范围。

## 8. Final Validation Requirements

最终 low/high validator 必须以 `S5_case_json_schema.md` 为唯一结构契约，检查 `S5_cases_low.json` 和 `S5_cases_high.json` 的结构、字段完整性和基础类型。文件级 source 约束由本文第 2 章定义，并由 Step 5.3/5.4 的固定生成脚本保证。

禁止 validator 或后续脚本根据以下信息补全输出：

- 算子名。
- 字段名。
- shape 相等关系。
- 样例分布。
- operator model。

outputs 必须由 mapper 的 `derive_outputs(...)` 或等价动态区逻辑完整生成。后续脚本只能校验、复制、机械变换 empty/data_range variant，不得推导缺失 output descriptor。
