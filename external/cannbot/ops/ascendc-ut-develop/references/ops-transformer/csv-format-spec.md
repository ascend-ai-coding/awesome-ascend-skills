# CSV 格式规范

本文档定义 Host UT CSV 测试用例文件的格式规范。

---

## 文件格式要求

⚠️ **必须严格遵守以下规则：**

1. **禁止注释行** — 不能有 `#` 开头的行
2. **禁止空行** — 文件中不能有任何空行
3. **第一行为列名行** — 定义所有 CSV 列名
4. **后续行为数据行** — 每行一个测试用例
5. **无尾随逗号** — 每行最后一个字段后不加逗号
6. **英文逗号分隔** — 不要使用中文逗号

### 正确格式示例

```csv
case_name,expectResult,query_shape,query_dtype,query_format,num_heads
test_0,SUCCESS,4 13 8192,FLOAT16,ND,16
test_1,SUCCESS,32 1 1024,FLOAT16,ND,8
```

### 错误格式示例（❌）

```csv
# 注释行 - 错误！
case_name,expectResult,...

test_0,SUCCESS,...  # 错误！有空行
```

---

## 列名定义规则

### 通用字段

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `case_name` | string | 测试用例名称 | `test_0` |
| `expectResult` | string | 期望执行结果 | `SUCCESS` 或 `FAILED` |
| `case_annotation` | string | 用例注释（可选） | `基础场景测试` |

### Tensor 字段

#### Tiling/InferShape 测试（完整 Tensor 信息）

每个 Tensor 占 3 列：

| 列名格式 | 类型 | 说明 | 示例 |
|---------|------|------|------|
| `{tensor_name}_shape` | string | Tensor 形状，空格分隔 | `4 13 8192` |
| `{tensor_name}_dtype` | string | 数据类型 | `FLOAT16` |
| `{tensor_name}_format` | string | 内存格式 | `ND` |

**示例：**
```csv
query_shape,query_dtype,query_format,key_shape,key_dtype,key_format
4 13 8192,FLOAT16,ND,32 1 1024,FLOAT16,ND
```

#### InferDataType 测试（仅 dtype）

每个 Tensor 占 1 列：

| 列名格式 | 类型 | 说明 | 示例 |
|---------|------|------|------|
| `{tensor_name}_dtype` | string | 数据类型 | `FLOAT16` |

**示例：**
```csv
query_dtype,key_dtype,value_dtype,attention_out_dtype
FLOAT16,FLOAT16,FLOAT16,INT8
```

### 属性字段

直接使用算子定义中的属性名：

| 属性类型 | CSV 列名 | 示例值 | 读取方法 |
|---------|---------|--------|---------|
| Int | `num_heads` | `16` | `std::stoll` |
| Float | `scale` | `0.04166` | `std::stof` |
| String | `input_layout` | `BSND` | `ReadMap` |
| Bool | `softmax_lse_flag` | `0` 或 `1` | `std::stoi` |

### 期望结果字段

#### Tiling 测试专用

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| `expectTilingKey` | uint64 | 期望的 tiling key | `104000000010500001` |
| `expectTilingDataHash` | string | 期望的 tiling data hash | `3bc1d4b865d7985d` |

⚠️ 仅在 `expectResult=SUCCESS` 时填写。

#### InferShape 测试

期望输出 shape 已在输出 Tensor 的 `_shape` 列中定义。

#### InferDataType 测试

期望输出 dtype 已在输出 Tensor 的 `_dtype` 列中定义。

---

## Tensor 列填写规则

根据 Tensor 状态和测试类型，CSV 列的填写规则如下：

| 测试类型 | Tensor 状态 | shape | dtype | format | 说明 |
|---------|-----------|-------|-------|--------|------|
| **Tiling** | REQUIRED Tensor | 必填 | 必填 | 必填 | 正常填写 |
| **Tiling** | Optional Tensor 不使用 | 空 | 空 | 空 | 三列都留空 |
| **Tiling** | 失败用例的输出 Tensor | 空 | 空 | 空 | 三列都留空 |
| **InferShape** | REQUIRED Tensor | 必填 | 必填 | 必填 | 正常填写 |
| **InferShape** | Optional Tensor 不使用 | **`0`** | 必填 | 必填 | ⚠️ **shape 必须填 `0`**，不能空着 |
| **InferShape** | 失败用例的输出 Tensor | **`0`** | 必填 | 必填 | ⚠️ **shape 必须填 `0`**，不能空着 |
| **InferDataType** | REQUIRED Tensor | - | 必填 | - | 仅 dtype 列 |
| **InferDataType** | Optional Tensor 不使用 | - | 空 | - | dtype 列留空 |

⚠️ **重要提示**：InferShape 测试中，Optional Tensor 的 **shape 列必须填 `0`**，不能留空！

### 示例对比

**Tiling 测试 CSV：**
```csv
case_name,expectResult,query_shape,query_dtype,query_format,pse_shift_shape,pse_shift_dtype,pse_shift_format
success_case,SUCCESS,4 13 8192,FLOAT16,ND,,,
failed_case,FAILED,1 1,FLOAT16,ND,,,
```

**InferShape 测试 CSV：**
```csv
case_name,expectResult,query_shape,query_dtype,query_format,pse_shift_shape,pse_shift_dtype,pse_shift_format
success_case,SUCCESS,4 13 8192,FLOAT16,ND,0,FLOAT16,ND
failed_case,FAILED,1 1,FLOAT16,ND,0,FLOAT16,ND
```

**InferDataType 测试 CSV：**
```csv
case_name,expectResult,query_dtype,pse_shift_dtype,attention_out_dtype
success_case,SUCCESS,FLOAT16,FLOAT16,INT8
optional_case,SUCCESS,FLOAT16,,FLOAT16
```

---

## 失败测试用例处理

`expectResult=FAILED` 的用例：

- **期望结果字段不填写**
  - Tiling：不填写 `expectTilingKey` 和 `expectTilingDataHash`
  - InferShape：输出 Tensor 的 `_shape` 填 `0`（或空）
  - InferDataType：输出 Tensor 的 `_dtype` 填空（或任意值）

- **属性值必须完整填写** — 失败原因通常与参数相关

- **建议标注失败原因** — 在 `case_annotation` 中说明

示例：
```csv
case_name,expectResult,query_shape,...,num_heads,case_annotation
fail_test,FAILED,1 1,...,0,num_heads=0 导致失败
```

---

## 数据类型映射表

| ge::DataType | CSV 写法 | 说明 |
|--------------|----------|------|
| `ge::DT_FLOAT` | `FLOAT` | 单精度浮点 |
| `ge::DT_FLOAT16` | `FLOAT16` | 半精度浮点 |
| `ge::DT_BF16` | `BF16` | BFloat16 |
| `ge::DT_INT8` | `INT8` | 8位整数 |
| `ge::DT_INT4` | `INT4` | 4位整数 |
| `ge::DT_INT16` | `INT16` | 16位整数 |
| `ge::DT_INT32` | `INT32` | 32位整数 |
| `ge::DT_INT64` | `INT64` | 64位整数 |
| `ge::DT_UINT8` | `UINT8` | 无符号8位整数 |
| `ge::DT_UINT16` | `UINT16` | 无符号16位整数 |
| `ge::DT_UINT32` | `UINT32` | 无符号32位整数 |
| `ge::DT_UINT64` | `UINT64` | 无符号64位整数 |
| `ge::DT_BOOL` | `BOOL` | 布尔类型 |
| `ge::DT_DOUBLE` | `DOUBLE` | 双精度浮点 |
| `ge::DT_STRING` | `STRING` | 字符串类型 |

---

## 格式映射表

| ge::Format | CSV 写法 | 说明 |
|-----------|----------|------|
| `ge::FORMAT_ND` | `ND` | N维格式 |
| `ge::FORMAT_NCHW` | `NCHW` | NCHW格式 |
| `ge::FORMAT_NHWC` | `NHWC` | NHWC格式 |
| `ge::FORMAT_HWCN` | `HWCN` | HWCN格式 |
| `ge::FORMAT_NC1HWC0` | `NC1HWC0` | 5维格式 |

---

## Shape 写法规则

- 使用**空格**分隔各维度值
- 不要使用逗号或其他分隔符
- 示例：`4 13 8192` 表示 shape 为 [4, 13, 8192]

---

## 列顺序建议

虽然列顺序无强制要求，但建议按以下顺序组织：

1. 通用字段（`case_name`, `expectResult`）
2. 输入 Tensor（按算子定义顺序）
3. 输出 Tensor（按算子定义顺序）
4. 属性字段（按算子定义顺序）
5. 期望结果字段
6. 注释字段（`case_annotation`）

---

## 常见错误

| 错误类型 | 表现 | 解决方法 |
|---------|------|---------|
| 注释行 | CSV 中有 `#` 开头的行 | 删除注释行 |
| 空行 | CSV 中有空行 | 删除空行 |
| 尾随逗号 | 行末有多余逗号 | 删除尾随逗号 |
| 中文逗号 | 使用了中文逗号 | 替换为英文逗号 |
| shape 格式错误 | 使用逗号分隔维度 | 使用空格分隔 |
| dtype 格式错误 | 使用 `DT_FLOAT16` | 使用 `FLOAT16` |

---

## 文件路径规范

| 测试类型 | CSV 文件路径 | CPP 文件路径 |
|---------|-------------|-------------|
| Tiling | `{算子目录}/tests/ut/op_host/arch32/test_{算子名}_tiling.csv` | `test_{算子名}_tiling.cpp` |
| InferShape | `{算子目录}/tests/ut/op_host/test_{算子名}_shape_infershape.csv` | `test_{算子名}_shape_infershape.cpp` |
| InferDataType | `{算子目录}/tests/ut/op_host/test_{算子名}_dtype_infershape.csv` | `test_{算子名}_dtype_infershape.cpp` |

**文件命名规范**：
- Tiling: `test_{算子名}_tiling`
- InferShape: `test_{算子名}_shape_infershape`（注意：包含 `_shape_` 关键字）
- InferDataType: `test_{算子名}_dtype_infershape`（注意：包含 `_dtype_` 关键字）

**示例**：
```
tests/ut/op_host/
├── fused_infer_attention_score_param.h
├── test_fused_infer_attention_score_shape_infershape.cpp
├── test_fused_infer_attention_score_shape_infershape.csv
├── test_fused_infer_attention_score_dtype_infershape.cpp
└── test_fused_infer_attention_score_dtype_infershape.csv
```