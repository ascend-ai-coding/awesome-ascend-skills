# 代码审查技能文件 - 算子接口规范

本文档例举昇腾算子开发中特有的接口规范条款, 为Ascend C 代码检视过程提供编码规范指导


# 二、昇腾算子接口规范

算子接口规范涉及TilingID语义、Runtime接口调用、动态Shape处理、内核执行异常处理等关键规范


### 2.26 TilingID语义不可变

**【描述】**
新增tiling_id字段时，已有tiling_id语义、字段顺序绝对不能修改。TilingID语义发生变化会导致缓存的调优数据失效，引发一些未知问题。

**【风险】**
导致缓存调优数据失效，引发未知运行异常

**【核心要求】**
新字段追加在原有字段末尾，不修改历史字段顺序

**【错误代码示例】**
```cpp
// TilingKey 为UINT64，十进制表示，每一位0-9，输出结果为生成倒序; 新增xxx1,xxx2字段时，导致原有tiling_id语义发生变化
tilingKey_ = GET_TILINGKEY(xxx1, xxx2, tilingEnableLow, tilingEnableHigh)；
```

**【正确代码示例】**
```cpp
// 新字段追加在原有字段末尾
tilingKey_ = GET_TILINGKEY(tilingEnableLow, tilingEnableHigh，xxx1, xxx2)；
```


### 2.27 Runtime2.0属性获取规范

**【描述】**
Runtime2.0场景，必须从context->GetAttrs()获取属性，禁止用CompileInfo传递。CompileInfo能够带入编译态的信息，执行态无法动态获取到算子的属性信息并传递给Tiling。

**【风险】**
CompileInfo仅支持编译态，执行态无法动态获取，导致Tiling参数错误

**【错误代码示例】**
```cpp
auto compileInfo = GetCompileInfoPtr<GroupNormGradCompileInfo>(context);
OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
int64_t groups = compileInfo->groups;
```

**【正确代码示例】**
```cpp
auto attrs = context->GetAttrs();
OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
const int64_t* num_groups = attrs->GetAttrPointer<int64_t>(INDEX_NUM_GROUPS);
tiling_params->num_groups = *num_groups;
```


### 2.28 输入Shape/Dtype获取规范

**【描述】**
禁止用GetInputTensor获取Shape和Dtype。GetInputTensor获取的对象，只有在IMPL_OP实现算子时，将对应输入设置为数据依赖后，才可以调用此接口获取tensor，否则行为是未定义的。

- Dtype：用context->GetInputDesc()->GetDataType()
- Shape：用context->GetInputShape()

**【风险】**
非IMPL_OP算子调用GetInputTensor行为未定义，触发内核崩溃

**【错误代码示例】**
```cpp
auto inputTensor = context->GetInputTensor(0);
OPS_CHECK_NULL_WITH_CONTEXT(context, inputTensor);
auto shape = inputTensor->GetShape(); 
auto dtype = inputTensor->GetDataType();
```

**【正确代码示例】**
```cpp
auto inputDesc = context->GetInputDesc(0);
OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
auto shape = context->GetInputShape(0); 
auto dtype = inputDesc->GetDataType();
```


### 2.29 可选输入获取规范

**【描述】**
InferShape阶段获取可选输入，必须用GetOptionalInputTensor。对于可选输入一定要用GetOptionalInputTensor接口，不然会产生未定义的错误。

**【风险】**
用普通输入接口获取，触发未定义内存错误

**【错误代码示例】**
```cpp
REG_OP(StridedSliceV3)
    .INPUT(x, TensorType::BasicType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType()) // 可选输入
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType()) // 可选输入
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(StridedSliceV3)

static ge::graphStatus StridedSliceV3InferShape(gert::InferShapeContext* context) {
  const gert::Tensor* strides_tensor = context->GetInputTensor(INDEX_STRIDES); // 按照必要输入的方式获取strides
  xxxx
}
```

**【正确代码示例】**
```cpp
REG_OP(StridedSliceV3)
    .INPUT(x, TensorType::BasicType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(axes, TensorType::IndexNumberType()) // 可选输入
    .OPTIONAL_INPUT(strides, TensorType::IndexNumberType()) // 可选输入
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(StridedSliceV3)

static ge::graphStatus StridedSliceV3InferShape(gert::InferShapeContext* context) {
  const gert::Tensor* strides_tensor = context->GetOptionalInputTensor(INDEX_STRIDES); // 应该使用对应的可选接口去获取tensor
  xxxx
}
```

### 2.31 ACLNN L0接口返回值校验

**【描述】**
KERNEL LAUNCHER必须判断返回值，失败直接返回nullptr。在Kernel Launch阶段，Common层异常情况下还是会将正常的out返回，导致aclnn继续执行二阶段，产生aic error或者core dump错误。

**【风险】**
Kernel异常仍继续执行二阶段，触发AIC Error/Core Dump

**【错误代码示例】**
```cpp
static const aclTensor* AbsAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor) {
     L0_DFX(AbsAiCore, self, out);
     ADD_TO_LAUNCHER_LIST_AICORE(Abs, OP_INPUT(self), OP_OUTPUT(out));
     return out;
   }
```

**【正确代码示例】**
```cpp
static const aclTensor* AbsAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor) {
     L0_DFX(AbsAiCore, self, out);
     // 需要校验一阶段接口的返回值，如果是fail情况需要返回nullptr，终止计算
     auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Abs, OP_INPUT(self), OP_OUTPUT(out));
     OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AbsAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
       return nullptr);
     return out;
   }
```