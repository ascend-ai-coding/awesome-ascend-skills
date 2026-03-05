# op_host 定义示例（Input/Output/Attr）

以下为 **FFN、GMM、MoE 三类算子的典型 op_host 接口示例**（Input/Output/Attr 片段），供复制后按需修改。**规范与示例以 CANN 算子工程开发文档为准**；DataType/Format 需与 [01-type-format.md](01-type-format.md) 及 CANN 头文件 `graph/types.h` 一致，且 DataType/Format/UnknownShapeFormat 个数一致。

## FFN

```cpp
Input("x").ParamType(REQUIRED).DataType({DT_FLOAT16, DT_BF16, DT_INT8}).Format({FORMAT_ND});
Input("weight1").ParamType(REQUIRED).DataType({DT_FLOAT16, DT_BF16, DT_INT8}).Format({FORMAT_ND});
Input("weight2").ParamType(REQUIRED).DataType({DT_FLOAT16, DT_BF16, DT_INT8}).Format({FORMAT_ND});
Input("bias1").ParamType(OPTIONAL).DataType({DT_FLOAT16, DT_BF16, DT_INT32}).Format({FORMAT_ND});
Output("y").DataType({DT_FLOAT16, DT_BF16, DT_INT8}).Format({FORMAT_ND});
Attr("activation").AttrType(OPTIONAL).Int({0});   // 0:GELU, 1:RELU, 2:FASTGELU, 3:SILU, 4:SIGMOID, 5:TANH
Attr("inner_precise").AttrType(OPTIONAL).Int({0}); // 0:BF16, 1:FLOAT32
```

## GMM

```cpp
Input("x").ParamType(REQUIRED).DataType({DT_FLOAT16, DT_BF16, DT_INT8}).Format({FORMAT_ND});
Input("weight").ParamType(REQUIRED).DataType({DT_FLOAT16, DT_BF16, DT_INT8}).Format({FORMAT_ND});
Input("bias").ParamType(OPTIONAL).DataType({DT_FLOAT16, DT_BF16, DT_INT32}).Format({FORMAT_ND});
Output("y").DataType({DT_FLOAT16, DT_BF16, DT_INT32, DT_INT8}).Format({FORMAT_ND});
Attr("split_item").AttrType(OPTIONAL).ListInt({});
Attr("dtype").AttrType(OPTIONAL).Int({0});
Attr("transpose_weight").AttrType(OPTIONAL).Int({0});
```

## MoE

```cpp
Input("x").ParamType(REQUIRED).DataType({DT_FLOAT16, DT_BF16}).Format({FORMAT_ND});
Input("rowIdx").ParamType(REQUIRED).DataType({DT_INT32}).Format({FORMAT_ND});
Input("expertIdx").ParamType(REQUIRED).DataType({DT_INT32}).Format({FORMAT_ND});
Output("expandedXOut").DataType({DT_FLOAT16, DT_BF16}).Format({FORMAT_ND});
Output("expandedRowIdx").DataType({DT_INT32}).Format({FORMAT_ND});
Output("expandedExpertIdx").DataType({DT_INT32}).Format({FORMAT_ND});
Attr("activeNum").AttrType(OPTIONAL).Int({0});
```

新算子：以 CANN 算子工程规范为准，从同类型算子完整复制类与构造函数后只改类名、输入输出名与个数、DataType/Format、属性与默认值；无特殊原因不随意改 AICore 与 ExtendCfgInfo；需 aclnn 时沿用 `"aclnnSupport.value", "support_aclnn"`。**官方参考**：CANN 文档中「基于自定义算子工程的算子开发」及 op_host 定义说明。
