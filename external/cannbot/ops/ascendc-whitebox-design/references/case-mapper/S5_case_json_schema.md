# Mapper-v1 Case JSON Schema

> 本文固定 Mapper-v1 输出 JSON 的唯一结构契约。后续 mapper 脚本、schema validator 和 TTK converter 必须只依赖本文字段，不得根据算子名、字段名、shape 相等关系、样例分布或 operator model 猜测补全输入输出语义。

## 1. Top-Level Case

每个 case 必须是 object，顶层字段固定为：

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

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | case 唯一 ID。 |
| `source` | string | 是 | case 来源或变体类型。 |
| `attributes` | object | 是 | operator attributes。 |
| `const_inputs` | object | 是 | `.ValueDepend()` 标记的 const input values。 |
| `inputs` | object | 是 | 输入 descriptor，key 为算子 input 名。 |
| `outputs` | object | 是 | 输出 descriptor，key 为算子 output 名。 |
| `meta` | object | 是 | 审计信息，不参与算子执行。 |

顶层字段必须精确等于上述 7 个字段。

## 2. Source

`source` 固定取值：

| source | 使用位置 | 含义 |
|--------|----------|------|
| `path` | `S5_mapped_cases_path.json` | 由 Step 2 path case 映射得到的 base case。 |
| `network` | `S5_mapped_cases_network.json` | 由 network config 映射得到的 base case。 |
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

## 3. Execution Values

执行值分为 `attributes` 和 `const_inputs`。

```json
{
  "attributes": {
    "keep_dims": false
  },
  "const_inputs": {
    "axis": 0
  }
}
```

| 字段 | 内容 | 进入 TTK CSV `attributes` 列 |
|------|------|------------------------------|
| `attributes` | `.Attr("...")` 注册的 operator attributes。 | 是，按 Attr 白名单过滤。 |
| `const_inputs` | `.ValueDepend()` 标记的 const input values。 | 是，按 ValueDepend input 白名单过滤。 |

约束：

- `attributes` 和 `const_inputs` 都必须是 object。
- `attributes` 和 `const_inputs` 不得重复写入同名字段。
- shape、dtype、format、data_range、tiling/router 信息和 mapper 审计字段不得写入 `attributes` 或 `const_inputs`。
- Tensor/TensorList descriptor 必须写入 `inputs` 或 `outputs`，不得写入 `const_inputs`。
- 元素数固定的 Tensor 输入按普通 Tensor 输入处理：descriptor 写入 `inputs`（`shape=[n]`），取值由 `data_range` 表达；其值不得写入 `const_inputs`，不得通过 `attributes` 覆盖。

## 4. Tensor Descriptor

Tensor descriptor 使用 `kind = "tensor"`。Tensor descriptor 表示单个接口 Tensor 参数，区别于 TensorList child Tensor。

### 4.1 Common Fields

Input 和 output Tensor 公共字段：

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `kind` | string | 是 | 固定为 `tensor`。 |
| `dtype` | string | 是 | 非空 dtype 名称；`shape = null` 时仍必须保留。 |
| `format` | string | 是 | 非空 format，默认 `ND`；`shape = null` 时仍必须保留。 |
| `shape` | list[int] 或 null | 是 | `null` 见 §6，仅允许非 `REQUIRED` 参数；空 tensor 用含 0 维度的 shape（如 `[0]`）表达（TL3），不得为空列表 `[]`。 |
| `param_type` | string | 是 | 接口参数类型，例如 `REQUIRED` 或 `OPTIONAL`。 |

Input Tensor 额外包含：

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `data_range` | string | 是 | 输入数据范围标签；`shape = null` 时仍必须保留，建议为 `normal`。 |

Output Tensor 不包含 `data_range`。

### 4.2 Input Tensor

Input Tensor 固定包含 6 个字段：`kind/dtype/format/shape/param_type/data_range`。

Input Tensor 示例：

```json
{
  "kind": "tensor",
  "dtype": "float16",
  "format": "ND",
  "shape": null,
  "param_type": "OPTIONAL",
  "data_range": "normal"
}
```

`shape = null` 表示该 input 位置不存在（仅限 `param_type != REQUIRED`；`REQUIRED` 输入禁止 `null`，见 §6）。descriptor 不得删除。input `shape = null` 时，`data_range` 仍必须保留。

### 4.3 Output Tensor

Output Tensor 固定包含 5 个字段：`kind/dtype/format/shape/param_type`。

Output Tensor 示例：

```json
{
  "kind": "tensor",
  "dtype": "float16",
  "format": "ND",
  "shape": null,
  "param_type": "OPTIONAL"
}
```

`shape = null` 表示该 output 位置不存在（仅限 `param_type != REQUIRED`）。`REQUIRED` 输出为空（某 branch 不产生该输出数据）时必须写空 tensor `shape = [0]`（TL3，TTK `(0,)`），不得写 `null`（见 §6）。descriptor 不得删除。Output Tensor 不包含 `data_range`。

## 5. TensorList Descriptor

TensorList descriptor 使用 `kind = "tensor_list"`。TensorList 外层表示接口参数，`tensors[]` 表示 child Tensor 列表。

TensorList 基础要求：

- TL1：`tensor_count` 满足 `1 <= tensor_count <= 50`，且等于 `len(tensors)`。
- TL2：`tensors` 至少包含 1 个 child；禁止整体为空的 TensorList（不得 `tensor_count = 0` / `tensors = []`）。
- TL3：空 tensor 用含 0 维度的一维 shape 表达（`[0]`，TTK 写作 `(0,)`）；不得使用空 shape 列表 `[]`。

`shape = null` 的语义见 §6，与空 tensor 严格区分。

### 5.1 Outer Descriptor

Input 和 output TensorList 外层公共字段：

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `kind` | string | 是 | 固定为 `tensor_list`。 |
| `dtype` | string | 是 | 非空外层 dtype。 |
| `format` | string | 是 | 非空外层 format，默认 `ND`。 |
| `param_type` | string | 是 | 固定为 `DYNAMIC`。 |
| `tensor_count` | int | 是 | 必须等于 `len(tensors)`（TL1）。 |
| `tensors` | list[object] | 是 | child Tensor 列表（TL2）。 |

Input TensorList 外层额外包含：

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `data_range` | string | 是 | 输入数据范围标签。 |

Output TensorList 外层不得包含 `data_range`。TensorList 外层不得使用顶层 `shape`。

### 5.2 Input TensorList

Input TensorList 外层固定包含 7 个字段：`kind/dtype/format/param_type/tensor_count/data_range/tensors`。

Input TensorList 示例：

```json
{
  "kind": "tensor_list",
  "dtype": "float16",
  "format": "ND",
  "param_type": "DYNAMIC",
  "tensor_count": 2,
  "data_range": "normal",
  "tensors": [
    {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": [2, 3],
      "data_range": "normal"
    },
    {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": null,
      "data_range": "normal"
    }
  ]
}
```

Input TensorList child Tensor 固定包含 5 个字段：

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `kind` | string | 是 | 固定为 `tensor`。 |
| `dtype` | string | 是 | 非空 dtype；`shape = null` 时仍必须保留。 |
| `format` | string | 是 | 非空 format；`shape = null` 时仍必须保留。 |
| `shape` | list[int] 或 null | 是 | `null` 表示该 child 位置不输入（见 §6）。 |
| `data_range` | string | 是 | 必须与 TensorList 外层 `data_range` 一致。 |

Input TensorList child Tensor 不包含 `param_type`。child `data_range` 必须与 TensorList 外层 `data_range` 一致。`shape = null` 的 child Tensor 仍计入 `tensor_count`。

### 5.3 Output TensorList

Output TensorList 外层固定包含 6 个字段：`kind/dtype/format/param_type/tensor_count/tensors`。

Output TensorList 示例：

```json
{
  "kind": "tensor_list",
  "dtype": "float16",
  "format": "ND",
  "param_type": "DYNAMIC",
  "tensor_count": 2,
  "tensors": [
    {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": [2, 3]
    },
    {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": null
    }
  ]
}
```

Output TensorList child Tensor 固定包含 4 个字段：

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `kind` | string | 是 | 固定为 `tensor`。 |
| `dtype` | string | 是 | 非空 dtype；`shape = null` 时仍必须保留。 |
| `format` | string | 是 | 非空 format；`shape = null` 时仍必须保留。 |
| `shape` | list[int] 或 null | 是 | `null` 表示该 child 位置不输出（见 §6）。 |

Output TensorList child Tensor 不包含 `param_type` 或 `data_range`。`shape = null` 的 child Tensor 仍计入 `tensor_count`。

## 6. Shape Null Placeholder（接口位置不存在）

`shape = null` 只表示**该接口参数位置本身不存在**（例如接口声明为可选、且该 case 未提供该参数），**不等于**「kernel 某 branch 不产生该输出数据」。两者必须严格区分：

- 位置不存在 → `shape = null`（TTK `None`，不传递）。
- 位置存在但该 branch 为空输出 → 空 tensor `shape = [0]`（TL3，TTK `(0,)`，传 shapeSize==0 的空张量）。

TTK kernel 模式以 `shape = None` 判定该位置不参与实际传递；Mapper-v1 JSON 使用 JSON `null` 表达该占位。

规则：

- `shape = null` 仅允许出现在 `param_type != REQUIRED`（如 `OPTIONAL`）的 descriptor 上。**`param_type == REQUIRED` 的 descriptor 禁止 `shape = null`**。
- `REQUIRED` 输出在某 branch 为空时，必须使用空 tensor `shape = [0]`（TL3），保留该位置并传递空张量；禁止写 `null`，禁止删除 descriptor。
- 不输入或不输出（位置不存在）时，不删除 descriptor；保留原位置并写 `shape = null`。
- `shape = null` 时，`dtype` 和 `format` 仍必须保留对应占位值，不能写 `null`。
- input `shape = null` 时，`data_range` 仍必须保留，建议为 `normal`。
- TensorList child `shape = null` 时，该 child 仍计入 `tensor_count`。

TTK 依据：

- `shapelike_stc_nested()` 允许 top-level 和 TensorList child shape 为 `None`。
- `TestcaseOp._construct_op_xput_tensor_dict()` 遇到 `shape is None` 时生成 `None` tensor。
- TTK profiling 根据 kernel json 的 optional placeholder 配置决定运行时是否过滤 `None`。
- 对 `REQUIRED` 参数传 `None` 会触发算子参数校验错误（如 `add_rms_norm` 的 `rstd_out_` 缺失，`errCode E80001`）。

## 7. Data Range

`data_range` 只属于 input descriptor。

合法值固定为：

| data_range | 含义 |
|------------|------|
| `normal` | 常规随机数据范围。 |
| `zero` | 全 0 数据。 |
| `extreme` | 接近 dtype 最大值的数据。 |
| `negative` | 负数数据范围。 |
| `tiny_pos` | 极小正数数据范围。 |
| `all_ones` | 全 1 数据。 |
| `near_zero` | 接近 0 的数据。 |
| `with_inf` | 包含 inf 的数据。 |
| `with_nan` | 包含 nan 的数据。 |

low case 要求：

- 所有 input Tensor 的 `data_range` 必须为 `normal`。
- 所有 input TensorList 外层和 child Tensor 的 `data_range` 必须为 `normal`。

high case 要求：

- `range` case 至少一个参与 data_range expansion 的 input 使用非 `normal`。
- input TensorList 外层 `data_range` 必须与所有 child Tensor 的 `data_range` 保持一致。

## 8. Meta

`meta` 是 mapper 和 variant 审计信息，不参与算子执行。

```json
{
  "base_id": "case00000",
  "variant_kind": "shape",
  "source_kind": "path",
  "path": "path_name",
  "tiling_key": "tiling_key_name",
  "network_name": "network_name"
}
```

约束：

- `meta` 必须是 object。
- `meta.network_name` 可被 TTK converter 用于 CSV `network_name` 列。
- `meta` 可被 TTK converter 用于 CSV `remark` 列。
- 不得写入 `meta.supported_data_ranges`。
- 不得把应进入 `attributes`、`const_inputs`、`inputs` 或 `outputs` 的执行字段写入 `meta`。

## 9. TTK Consumption Contract

Mapper-v1 JSON 使用 `shape = null` 表达接口位置不存在（仅限非 `REQUIRED` 参数，见 §6）；`REQUIRED` 参数的空输出使用 `[0]`。写入 TTK Kernel CSV / Python 结构时，JSON `null` 必须转换为 `None`，空 tensor `[0]` 转换为 `(0,)`。converter 必须保留 mapper 输出的 input/output 参数顺序和 TensorList child 顺序，不得按名称排序，不得过滤 `shape = null` 的 descriptor 或 child Tensor。

所有 Tensor/TensorList 输入（含元素数固定的输入）的取值由 `input_data_ranges` 表达；`attributes` 列不用于覆盖任何输入的值。

Mapper-v1 → TTK Kernel CSV 映射：

| CSV 字段 | Mapper-v1 JSON 来源 | 映射规则 |
|----------|---------------------|----------|
| `input_shapes` | Tensor: `inputs.<name>.shape`；TensorList: `inputs.<name>.tensors[].shape` | JSON `null` 转为 TTK `None`，并保留位置。 |
| `input_dtypes` | 同结构位置读取 `dtype` | `shape = null` 位置仍保留 dtype。 |
| `input_formats` | 同结构位置读取 `format` | `shape = null` 位置仍保留 format。 |
| `output_shapes` | Tensor: `outputs.<name>.shape`；TensorList: `outputs.<name>.tensors[].shape` | JSON `null` 转为 TTK `None`，并保留位置。 |
| `output_dtypes` | 同结构位置读取 `dtype` | `shape = null` 位置仍保留 dtype。 |
| `output_formats` | 同结构位置读取 `format` | `shape = null` 位置仍保留 format。 |
| `attributes` | `attributes` + `const_inputs` | 合并后写入 TTK `attributes` 列；只允许 Attr 和 ValueDepend const input，不得包含 shape/dtype/format/data_range/meta 字段。 |
| `input_data_ranges` | Input Tensor: `inputs.<name>.data_range`；Input TensorList: `inputs.<name>.tensors[].data_range` | 只从 input descriptor 读取；output 不参与。 |
| `precision_tolerances` | output dtype | 按 output 结构生成；`shape = null` 位置仍保留 dtype 对齐。 |
| `network_name` | `meta.network_name` | 缺失时为空。 |
| `remark` | `id`、`source` 和可审计 `meta` 字段 | 不得影响测试语义。 |

TensorList CSV 嵌套要求：

| 字段 | JSON 来源 | CSV 形态 |
|------|-----------|----------|
| `input_shapes` / `output_shapes` | 每个 child Tensor 的 `shape`。 | 展开嵌套，允许 `None`。 |
| `input_dtypes` / `output_dtypes` | 每个 child Tensor 的 `dtype`，或外层一致 dtype。 | 可压缩或展开，但必须与 shape 分布对齐。 |
| `input_formats` / `output_formats` | 每个 child Tensor 的 `format`，或外层一致 format。 | 可压缩或展开，但必须与 shape 分布对齐。 |
| `input_data_ranges` | 每个 input child Tensor 的 `data_range`。 | 展开嵌套。 |
| `precision_tolerances` | 每个 output child Tensor 的 `dtype`。 | 展开嵌套。 |

## 10. Validation Checklist

validator 必须检查：

- 顶层只包含 `id/source/attributes/const_inputs/inputs/outputs/meta`。
- Input Tensor 必须包含 `kind/dtype/format/shape/param_type/data_range`。
- Output Tensor 必须包含 `kind/dtype/format/shape/param_type`。
- Input TensorList 外层必须包含 `kind/dtype/format/param_type/tensor_count/data_range/tensors`。
- Output TensorList 外层必须包含 `kind/dtype/format/param_type/tensor_count/tensors`。
- Input TensorList child Tensor 必须包含 `kind/dtype/format/shape/data_range`。
- Output TensorList child Tensor 必须包含 `kind/dtype/format/shape`。
- 所有 descriptor 的 `dtype` 和 `format` 必须是非空字符串。
- `shape` 只能是 `list[int]` 或 `null`：`null` 遵循 §6（仅限非 `REQUIRED` 参数，`REQUIRED` 禁止 `null`；`REQUIRED` 空输出用 `[0]`），空 tensor 遵循 TL3。
- TensorList 必须包含 `tensor_count` 和 `tensors`，满足 TL1 与 TL2。
- TensorList child Tensor 不得包含 `param_type`。
- outputs 任何层级不得包含 `data_range`。
- output TensorList 不得使用顶层 `shape`。
- input TensorList 不得使用顶层 `shape` 替代 `tensors[]`。
