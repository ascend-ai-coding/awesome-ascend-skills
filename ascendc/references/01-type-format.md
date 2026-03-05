# 类型与格式约定（op_host）

op_host 中 Input/Output 的 DataType、Format、UnknownShapeFormat 必须与 **CANN 算子工程规范**及头文件 `graph/types.h` 一致，且三者的元素个数一一对应。**类型与格式的权威定义以 CANN 安装目录下 `graph/types.h` 及官方算子开发文档为准。**

## 枚举来源

- **DataType**：`graph/types.h` 中 `enum DataType`（约 L80–L123），路径示例：`/usr/local/Ascend/cann-8.5.0-beta.1/aarch64-linux/include/graph/types.h`
- **Format**：同一文件中 `enum Format`（约 L189–L247）

常用示例：

| 用途     | DataType           | Format             |
|----------|--------------------|--------------------|
| 浮点     | DT_FLOAT, DT_FLOAT16, DT_BF16 | FORMAT_ND          |
| 整型     | DT_INT8, DT_INT32, DT_BOOL    | FORMAT_ND          |
| 布局     | —                  | FORMAT_NCHW, FORMAT_NHWC, FORMAT_FRACTAL_NZ |

## JSON → C++ 映射

| JSON type/format | ge 枚举        |
|------------------|----------------|
| "fp16"           | ge::DT_FLOAT16 |
| "bf16"           | ge::DT_BF16    |
| "float"          | ge::DT_FLOAT   |
| "int32"          | ge::DT_INT32   |
| "ND"             | ge::FORMAT_ND  |

## 个数一致规则

每个 `Input("xxx")` / `Output("yyy")` 的以下三个列表**元素个数必须相同**：

- `.DataType({ ... })`
- `.Format({ ... })`
- `.UnknownShapeFormat({ ... })`

示例（3 种类型）：

```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

示例（单一类型）：

```cpp
this->Input("idx")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT32})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

新算子应只使用 `graph/types.h` 中已定义的枚举，并与算子 JSON 的 type/format 保持一致。**官方参考**：CANN 算子工程开发文档中关于 Input/Output 描述与数据类型、格式的说明；`graph/types.h` 位于 CANN 安装路径 `include/graph/` 下。
