# CSV 公共字段与约定

## 公共字段定义（9 个，所有模式通用）

| 序号 | 列名 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| 1 | `testcase_name` | STRING | 是 | 自动生成 | 用例唯一名称。缺失时自动生成为 `auto_testcase_name_N`。 |
| 2 | `network_name` | STRING | 否 | `None` | 网络/模型名称标签（如 `model_name_train`）。 |
| 3 | `input_data_ranges` | FLOAT_RANGE_NESTED | 否 | `((None, None),)` | 每个输入张量的随机数据范围。每个元素为 `(min, max)`。TensorList 输入使用展开嵌套格式（每个子 tensor 对应一个 range），如 `(((-10, 10), (-10, 10), (-10, 10)),)`，详见 02-kernel-fields.md「TensorList 嵌套结构」。 |
| 4 | `precision_tolerances` | FLOAT_RANGE_NESTED | 否 | `None` | 每个输出的精度容差对 `(rtol, atol)`。如 `"((0.001, 0.001),)"`。TensorList 输出使用展开嵌套格式，如 `(((0.001, 0.001), (0.001, 0.001), (0.001, 0.001)),)`。 |
| 5 | `absolute_precision` | FLOAT_OR_NESTED | 否 | `1e-8` | 默认绝对精度容差。可以是单个浮点数或嵌套容器实现逐输出控制。 |
| 6 | `is_enabled` | BOOL | 否 | `True` | 设为 `False` 跳过此用例。 |
| 7 | `remark` | STRING | 否 | `None` | 自由备注信息。 |
| 8 | `soc_series` | STRING_TUPLE | 否 | `None` | SoC 过滤。前缀 `-` 表示排除。如 `('Ascend910A', '-Ascend310P')` |
| 9 | `priority` | INT | 否 | `0` | 优先级，用于选择性执行。 |

## CSV 格式规则（填写规范）

1. **禁止无故设 None**：未明确要求为空的字段，使用列规格默认值
2. **字符串带单引号**：tuple 内 dtype/format 等字符串用 `'float16'`、`'ND'`；顶层字段（testcase_name、op_name）不加引号
3. **双引号包裹特殊字段**：含括号、逗号的字段必须用双引号包裹（CSV 标准）
4. **dict key 双引号**：`{"epsilon": 1e-05}`
5. **禁止 repr()**：字符串用 `str()`，单引号在构造 tuple 时嵌入
6. **单元素 tuple 尾逗号**：1 维 shape 如 `(12289,)` 必须带尾逗号
7. **precision_tolerances 尾逗号兼容**：校验脚本需同时接受 `((a, b))` 和 `((a, b),)` 两种格式
8. **input_shapes / output_shapes**：只填静态值，动态 shape 由框架自动推导
9. **ori 字段**：默认留空，TTK 框架自动回退

## 通用禁止（所有模式，约束）

1. 禁止不从 S5 JSON 取值而自行推导 shape/dtype
2. 禁止修改固定列名和列顺序
3. 禁止跳过验证步骤
