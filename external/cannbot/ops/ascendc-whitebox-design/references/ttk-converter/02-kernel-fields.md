# Kernel 模式 CSV 字段规格

适用于 `python3 -m ttk kernel` 命令，使用 `UniversalTestcaseStructure`。

静态 shape 通过 `input_shapes` / `output_shapes` 直接指定，动态 shape 由框架自动推导（将正数维度替换为 `-1`）。

## Kernel 专有字段（17 个）

| 序号 | 列名 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| 10 | `op_name` | STRING | 是 | *(无)* | 算子名称（用户输入的规范名 = camel_to_snake(OP_ADD(op_type))，可能与目录名不一致）。如 `add`、`mat_mul_v3`、`lamb_next_mv`。 |
| 11 | `input_shapes` | SHAPE_NESTED | 是 | `()` | 输入张量 shape。支持 TensorList 嵌套。用 `None` 表示可选输入。 |
| 12 | `input_dtypes` | DTYPE_NESTED | 是 | `()` | 输入数据类型。支持 TensorList 嵌套。 |
| 13 | `output_shapes` | SHAPE_INFER_NESTED | 是 | `None` | 输出张量 shape。支持 TensorList 嵌套和自动推断关键字。 |
| 14 | `output_dtypes` | DTYPE_NESTED | 是 | *(无)* | 输出数据类型。支持 TensorList 嵌套。 |
| 15 | `input_formats` | DTYPE_NESTED | 否 | `('ND',)` | 输入张量格式。 |
| 16 | `input_ori_shapes` | SHAPE_NESTED | 否 | → `input_shapes` | 原始输入 shape（格式转换前）。 |
| 17 | `input_ori_formats` | DTYPE_NESTED | 否 | `('ND',)` | 原始输入格式。 |
| 18 | `output_formats` | DTYPE_NESTED | 否 | `('ND',)` | 输出张量格式。 |
| 19 | `output_ori_shapes` | SHAPE_INFER_NESTED | 否 | `None` | 原始输出 shape。 |
| 20 | `output_ori_formats` | DTYPE_NESTED | 否 | `('ND',)` | 原始输出格式。 |
| 21 | `attributes` | DICT | 否 | `{}` | 算子属性（编译期和运行期合并）。 |
| 22 | `output_inplace_indexes` | INT_TUPLE | 否 | `()` | inplace 输入的索引。Kernel CSV 生成阶段固定写入 `()`；必要的同名 inplace 由 TTK op_info 自动推导。详见 `03-kernel-mode.md`。 |
| 23 | `output_shape_unknown_indexes` | INT_TUPLE | 否 | `()` | 编译期 shape 未知的输出索引。 |
| 24 | `dump_file_prefix` | STRING | 否 | `None` | 数据 dump 文件的自定义文件名前缀。 |
| 25 | `manual_input_binaries` | EVAL | 否 | `()` | 手动输入二进制文件路径。 |
| 26 | `manual_golden_binaries` | EVAL | 否 | `()` | 手动 Golden 输出二进制文件路径。 |

## CSV 列顺序（严格固定）

```
testcase_name, network_name, op_name, input_shapes, input_dtypes, input_formats, output_shapes, output_dtypes, output_formats, input_ori_shapes, input_ori_formats, output_ori_shapes, output_ori_formats, attributes, input_data_ranges, precision_tolerances, absolute_precision, output_inplace_indexes, output_shape_unknown_indexes, is_enabled, remark, soc_series, priority, dump_file_prefix, manual_input_binaries, manual_golden_binaries
```

## TensorList 嵌套结构

当输入/输出为 TensorList（S5_mapping_spec.md 中 param_type=DYNAMIC）时，以下字段使用嵌套格式：外层 tuple = 各输入/输出，内层 tuple = 该 TensorList 的各子 tensor。

各字段因值类型不同，DYNAMIC 格式有差异：

| 字段 | REQUIRED 格式 | DYNAMIC 格式 | 说明 |
|------|-------------|-------------|------|
| `input_shapes` | `((1,2), (3,4))` | `(((1,2), (3,4), (5,6)),)` | 展开：每个子 tensor 对应一个 shape |
| `input_dtypes` | `("float16", "float32")` | `(("float16",),)` | 压缩：1 dtype 广播到所有子 tensor |
| `output_shapes` | `((1,2),)` | `(((1,2), (3,4), (5,6)),)` | 展开：每个子 tensor 对应一个 shape |
| `output_dtypes` | `("float16",)` | `(("float16",),)` | 压缩：1 dtype 广播到所有子 tensor |
| `input_data_ranges` | `((-10,10), (-1,1))` | `(((-10,10),(-10,10),(-10,10)),)` | **展开**：每个子 tensor 对应一个 range |
| `precision_tolerances` | `((0.001,0.001),)` | `(((0.001,0.001),(0.001,0.001),(0.001,0.001)),)` | **展开**：每个子 tensor 对应一个 tolerance |

**格式选择原因**（经 TTK 源码验证）：

- **`input_shapes` / `output_shapes`**：必须展开（每个子 tensor 有独立 shape，无法广播）
- **`input_dtypes` / `output_dtypes`**：压缩即可。TTK `_normalize_field_by_dist` 负责归一化 dtype 这类 string/scalar 字段，`_flatten_by_distribution` 的 `len(val)==1` 分支会将 `('float16',)` 自动广播到所有子 tensor
- **`input_data_ranges` / `precision_tolerances`**：**必须展开**。range pair `(min, max)` 本身是 2 元素 tuple，TTK `_normalize_range_field_by_dist` 的 `len(field)==1` 广播分支会将 `field[0]` 整体广播，压缩嵌套 `(((-10,10),),)` 会导致每个子 tensor 收到 `((-10,10),)` 而非 `(-10,10)`，解析错误。扁平格式 `((-10,10),)` 在纯 TensorList（1 个输入）时可用，但在混合场景（TensorList + 普通 tensor）下 `_flatten_by_distribution` 会拆散 range pair，解析错误。展开嵌套是唯一通用格式

`input_formats` / `output_formats` / `ori` 系列字段保持扁平格式不变（TTK 通过 `get()` 回退机制处理，无需嵌套）。

混合示例（2 个输入：第 1 个 TensorList 含 3 子 tensor，第 2 个普通 tensor）：

```
input_shapes      = (((1,2), (3,4), (5,6)), (7,8))
input_dtypes      = (("float16",), "float32")
input_data_ranges = (((-10,10),(-10,10),(-10,10)), (-1,1))
```
