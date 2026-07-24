# 从旧代码提取测试用例参数

本文档提供从硬编码测试代码提取参数到 CSV 格式的通用方法。

---

## 旧代码结构识别

旧格式的测试代码通常包含以下结构：

```cpp
TEST_F(OpTest, test_case_0) {
    gert::InfershapeContextPara para(
        "OpName",
        {
            // 输入 Tensor 列表
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // 空 Tensor
            // ... 更多输入
        },
        {
            // 输出 Tensor 列表
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            // ... 更多输出
        },
        {
            // 属性列表
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166f)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            // ... 更多属性
        }
    );
    
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 13, 16, 512},
        {0},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

---

## 提取参数通用方法

### 方法一：Tensor 参数提取

#### 旧代码格式

```cpp
{{{shape1, shape2, ...}, {shape1, shape2, ...}}, dtype, format}
```

#### 提取规则

| 旧代码组成部分 | 提取方法 | CSV 格式 |
|--------------|---------|---------|
| **shape** | 从第一个花括号提取 | 空格连接：`4 13 16 512` |
| **dtype** | 从 `ge::DT_XXX` 提取 | 去掉前缀：`DT_FLOAT16` → `FLOAT16` |
| **format** | 从 `ge::FORMAT_XXX` 提取 | 去掉前缀：`FORMAT_ND` → `ND` |

#### 不同状态的 Tensor 处理

| Tensor 状态 | 旧代码特征 | CSV 处理规则 |
|-----------|-----------|-------------|
| **正常 Tensor** | `{{{4, 13, ...}, {...}}, dtype, format}` | 正常填写 shape/dtype/format |
| **空 Tensor（Tiling）** | `{{{}, {}}, dtype, format}` | 三列都留空 |
| **空 Tensor（InferShape）** | `{{{}, {}}, dtype, format}` | shape 填 `0`，dtype/format 正常填写 |
| **空 Tensor（InferDataType）** | 不存在 shape 信息 | dtype 列留空 |

#### 示例对比

| 旧代码 | CSV（Tiling） | CSV（InferShape） |
|--------|--------------|------------------|
| `{{{4, 13, 16, 512}, {...}}, ge::DT_FLOAT16, ge::FORMAT_ND}` | `4 13 16 512,FLOAT16,ND` | `4 13 16 512,FLOAT16,ND` |
| `{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}` | `,,` | `0,FLOAT16,ND` |
| `{{{32, 4096, 128}, {...}}, ge::DT_INT8, ge::FORMAT_ND}` | `32 4096 128,INT8,ND` | `32 4096 128,INT8,ND` |

---

### 方法二：属性值提取

#### 旧代码格式

```cpp
{"attr_name", Ops::Transformer::AnyValue::CreateFrom<type>(value)}
```

#### 提取规则

| 属性类型 | 旧代码特征 | CSV 写法 |
|---------|-----------|---------|
| **Int** | `CreateFrom<int64_t>(16)` | `16` |
| **Float** | `CreateFrom<float>(0.04166f)` | `0.04166`（去掉 f 后缀） |
| **String** | `CreateFrom<std::string>("BSND")` | `BSND`（去掉引号） |
| **Bool** | `CreateFrom<bool>(false)` | `0` 或 `1` |

#### 类型转换表

| C++ 类型 | CSV 写法 | 读取方法 |
|---------|---------|---------|
| `int64_t` | 数字 | `std::stoll` |
| `float` | 数字（可带小数） | `std::stof` |
| `std::string` | 字符串 | `ReadMap` |
| `bool` | `0` 或 `1` | `std::stoi` |

---

### 方法三：期望结果提取

#### 期望执行结果

| 旧代码 | CSV 写法 |
|-------|---------|
| `ge::GRAPH_SUCCESS` | `SUCCESS` |
| `ge::GRAPH_FAILED` | `FAILED` |

#### 期望输出 Shape（InferShape）

旧代码中通过 `expectOutputShape` 定义：

```cpp
std::vector<std::vector<int64_t>> expectOutputShape = {
    {4, 13, 16, 512},
    {0},  // 空 Tensor
};
```

转换为 CSV：
- 第一个输出：`4 13 16 512`（写入 `output0_shape` 列）
- 第二个输出：`0`（写入 `output1_shape` 列）

#### 期望 Tiling Key/DataHash（Tiling）

旧代码可能不显式定义，需要在重构后：
1. 先运行一次测试获取实际值
2. 或从已有测试结果中提取

---

## 完整转换示例

### 旧代码（InferShape）

```cpp
TEST_F(FusedInferAttentionScoreProto, infershape_0)
{
    gert::InfershapeContextPara para(
        "FusedInferAttentionScore",
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // query
            {{{4, 10347, 1, 512}, {4, 10347, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // key
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // pse_shift (空)
            {{{4, 13, 10347}, {4, 13, 10347}}, ge::DT_INT8, ge::FORMAT_ND}, // atten_mask
        },
        {
            {{{4, 13, 16, 512}, {4, 13, 16, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // attention_out
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_lse (空)
        },
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.04166666666666666f)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        });
    
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 13, 16, 512},
        {0},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
```

### 转换为 CSV

```csv
case_name,expectResult,query_shape,query_dtype,query_format,key_shape,key_dtype,key_format,pse_shift_shape,pse_shift_dtype,pse_shift_format,atten_mask_shape,atten_mask_dtype,atten_mask_format,attention_out_shape,attention_out_dtype,attention_out_format,softmax_lse_shape,softmax_lse_dtype,softmax_lse_format,num_heads,scale,input_layout,softmax_lse_flag,case_annotation
infershape_0,SUCCESS,4 13 16 512,FLOAT16,ND,4 10347 1 512,FLOAT16,ND,0,FLOAT16,ND,4 13 10347,INT8,ND,4 13 16 512,FLOAT16,ND,0,FLOAT,ND,16,0.04166666666666666,BSND,0,基础场景
```

---

## 大量 Tensor 的处理建议

当输入/输出 Tensor 数量 > 10 时，建议使用自动化方法：

### 建议 1：使用脚本生成 CSV 列名

```bash
python scripts/generate_csv_template.py {算子名}_def.cpp
```

脚本会自动：
- 解析输入/输出 Tensor 列表
- 解析属性列表
- 生成 CSV 列名行
- 生成参数结构体框架代码

### 建议 2：手动提取时按功能分组

将 Tensor 按功能分组排列，便于理解和填写：

```
基础输入组：query, key, value
量化组：dequant_scale1, quant_scale1
PagedAttention组：block_table, kv_padding_size
Optional组：pse_shift, antiquant_scale
```

### 建议 3：在注释中标注关键信息

```csv
case_annotation
基础场景：query + key + value
量化场景：启用 antiquant_scale + antiquant_offset
PagedAttention：启用 block_table
```

---

## 验证转换结果

转换完成后，需要验证：

1. **编译验证**
   ```bash
   bash build.sh -u --ops={算子名}
   ```

2. **用例数量验证**
   - 新旧用例数量应该一致
   - 每个旧用例都应该对应一行 CSV

3. **参数正确性验证**
   - 运行结果与旧代码一致
   - 期望结果匹配

---

## 常见问题处理

### Q1: 旧代码中有多个 TEST_F，如何映射到 CSV？

**A:** 每个 `TEST_F` 对应一行 CSV，`case_name` 使用测试名称。

示例：
```cpp
TEST_F(OpTest, test_case_0) { ... }
TEST_F(OpTest, test_case_1) { ... }
```

转换为：
```csv
case_name,...
test_case_0,...
test_case_1,...
```

### Q2: 旧代码中有复杂的逻辑判断，如何处理？

**A:** 将逻辑判断的结果作为参数写入 CSV。

示例：
```cpp
if (condition) {
    shape = {4, 13, 512};
} else {
    shape = {1, 1};
}
```

转换时，根据 condition 的值，将对应 shape 写入 CSV。

### Q3: 旧代码中有循环生成的多个用例，如何处理？

**A:** 每个循环迭代对应一行 CSV。

示例：
```cpp
for (int i = 0; i < 3; i++) {
    TEST_F(OpTest, test_i) { ... }
}
```

转换为：
```csv
case_name,...
test_0,...
test_1,...
test_2,...
```

### Q4: 如何处理精度损失？

**A:** 浮点属性建议使用旧代码中的精确值，避免转换过程中的精度损失。

示例：
```cpp
CreateFrom<float>(0.04166666666666666f)  // 精确值
```

CSV：
```
0.04166666666666666  // 保留完整精度
```

---

## 迁移检查清单

完成迁移后，检查以下事项：

- [ ] 所有旧用例都已转换为 CSV 行
- [ ] case_name 与旧测试名称一致
- [ ] Tensor 参数（shape/dtype/format）正确
- [ ] 属性值正确（类型匹配）
- [ ] 期望结果正确（SUCCESS/FAILED）
- [ ] Optional Tensor 处理正确（按测试类型）
- [ ] 空 Tensor 处理正确
- [ ] 编译成功
- [ ] 所有用例通过
- [ ] 运行结果与旧代码一致