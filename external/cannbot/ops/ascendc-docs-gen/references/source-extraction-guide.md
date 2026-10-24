# 源码信息提取指南

本文档详细说明如何从Ascend CANN算子源码中提取文档所需信息，以 [`example-aclnn-md.md`](example-aclnn-md.md) 和 [`example-readme.md`](example-readme.md) 为参考范例。

**MANDATORY - READ ENTIRE FILE**：阶段1源码分析时，必须完整阅读本指南。

---

## 1. 从 `*_def.cpp` 提取

### 1.1 输入参数信息

```cpp
this->Input("grad_output")
    .ParamType(REQUIRED)                              // → 输入/输出列取值
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})  // → 数据类型
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})  // → 数据格式
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

**提取映射**：

| 源码 | aclnn MD | README |
|------|----------|--------|
| `ParamType(REQUIRED)` | 输入/输出列="输入"，使用说明列按需填写 | 输入/输出/属性列="输入" |
| `ParamType(OPTIONAL)` | 输入/输出列="可选输入"，使用说明列加`<ul><li>可选参数，传入空指针表示不使用该参数。</li></ul>` | 输入/输出/属性列="可选输入" |
| `ge::DT_FLOAT16` | FLOAT16 | FLOAT16 |
| `ge::DT_BF16` | BFLOAT16 | BFLOAT16 |
| `ge::DT_FLOAT` | FLOAT32 | FLOAT32 |
| `ge::FORMAT_ND` | ND | ND |
| `ge::FORMAT_NCHW` | NCHW | NCHW |
| `.AutoContiguous()` | 非连续Tensor列标√ | 不涉及 |
| 无`.AutoContiguous()` | 非连续Tensor列标× | 不涉及 |

**注意**：def.cpp中Input名称使用snake_case（如`grad_output`），对应README参数名；aclnn接口参数名使用camelCase（如`gradOutput`），对应aclnn MD参数名。

### 1.2 输出参数信息

```cpp
this->Output("grad_input")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

**注意**：输出mean/rstd的DataType固定为`ge::DT_FLOAT`，与输入数据类型无关。

### 1.3 属性信息

```cpp
this->Attr("output_size").AttrType(REQUIRED).ListInt();    // → 属性名、类型ListInt、无默认值
this->Attr("input_size").AttrType(REQUIRED).ListInt();     // → 属性名、类型ListInt、无默认值
this->Attr("align_corners").AttrType(OPTIONAL).Bool(false); // → 属性名、类型bool、默认值false
this->Attr("scales_h").AttrType(OPTIONAL).Float();          // → 属性名、类型float、无默认值（空）
this->Attr("scales_w").AttrType(OPTIONAL).Float();          // → 属性名、类型float、无默认值（空）
```

**Attr类型映射**：

| 源码Attr类型 | aclnn接口类型 | aclnn MD参数名写法 | README数据类型 | README输入/输出/属性 |
|-------------|-------------|-------------------|---------------|---------------------|
| `.Int(x)` | `int64_t` | `attrName（int64_t）` | INT64 | 属性/可选属性 |
| `.Float(x)` | `double` | `attrName（double）` | FLOAT32 | 属性/可选属性 |
| `.Bool(x)` | `bool` | `attrName（bool）` | BOOL | 属性/可选属性 |
| `.ListInt()` | `aclIntArray*` | `attrName（aclIntArray*）` | INT64 | 属性/可选属性 |

**关键差异**：
- `.Float(x)` 在aclnn接口中映射为`double`类型，但README中写FLOAT32
- `.ListInt()` 在aclnn接口中映射为`aclIntArray*`，需在使用说明列注明size约束
- `AttrType(REQUIRED)` → README中标"属性"
- `AttrType(OPTIONAL)` → README中标"可选属性"，描述列用`<ul><li>`包裹默认值

### 1.4 平台配置

```cpp
this->AICore().AddConfig("ascend910b");    // → Atlas A2系列产品，仍需核验接口校验与发布支持
this->AICore().AddConfig("ascend910_93");  // → Atlas A3系列产品，仍需核验接口校验与发布支持

OpAICoreConfig regbaseConfig;
regbaseConfig.Input("grad_output")
    .ParamType(REQUIRED)
    .DataType(xDtype)
    .Format(xFormat)
    .UnknownShapeFormat(xFormat);
regbaseConfig.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("opFile.value", "upsample_bicubic2d_aa_grad_apt");
this->AICore().AddConfig("ascend950", regbaseConfig);  // → Ascend 950PR&950DT系列产品，仍需核验接口与发布支持
```

**注意**：950平台可能有独立的`OpAICoreConfig`配置，含DynamicCompileStaticFlag等扩展信息，且对应独立的tiling文件（`*_tiling_arch35.cpp`）。

**无AddConfig的平台标×**。

---

## 2. 从 `*_base.h` 提取

### 2.1 枚举定义

```cpp
enum class NormType : int64_t {
    NONE = 0,
    LAYER_NORM,              // = 1
    LAYER_NORM_AFFINE,       // = 2
    RMS_NORM,                // = 3
    RMS_NORM_AFFINE,         // = 4
};
```

**转换为文档描述**：

| 枚举值 | 数值 | aclnn MD使用说明列 | README描述列 |
|--------|------|-------------------|-------------|
| NONE | 0 | 枚举值支持0，其中0表示不做正则化 | 0表示不做正则化 |
| LAYER_NORM | 1 | 1表示LayerNorm | 1表示LayerNorm |
| LAYER_NORM_AFFINE | 2 | 2表示LayerNormAffine | 2表示LayerNormAffine |
| RMS_NORM | 3 | 3表示RmsNorm | 3表示RmsNorm |
| RMS_NORM_AFFINE | 4 | 4表示RmsNormAffine | 4表示RmsNormAffine |

**注意**：仅文档化aclnn接口暴露的枚举值，内部枚举（如LAYER_NORM_ACROSS_HEADS、L2_NORM）不写入文档。

### 2.2 维度常量

```cpp
constexpr size_t INPUT_DIM_NUM = 4;   // B, S, H, D
```

**用于推断参数shape维度**。

### 2.3 参数索引枚举

```cpp
enum class InputIndexForward : size_t {
    QUERY_INDEX = 0,
    KEY_INDEX,
    ...
};
```

**用于确认参数顺序和数量**。

**注意**：部分算子（如upsample_bicubic2d_aa_grad）不存在`*_base.h`文件，此时从def.cpp和aclnn接口头文件提取参数信息即可。

---

## 3. 从 `*_infershape.cpp` 提取

### 3.1 输出shape推导

```cpp
*queryOutputShape = {batch, head, querySeq + encoderQuerySeq, dim}; // B, H, S, D
```

**提取规则**：
- 输入shape为(B, S, N, D)即BSND
- 输出shape为(B, N, S, D)即BNSD（注意维度顺序变化！）
- 拼接场景：S = querySeq + encoderQuerySeq

### 3.2 条件输出

```cpp
if (*isTraining) {
    if (*normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
        *normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
        *normQueryMeanShape = {batch, querySeq, head, 1};
        *normQueryRstdShape = {batch, querySeq, head, 1};
    }
}
```

**提取规则**：
- mean/rstd仅在isTraining=true时有效
- mean/rstd仅在LayerNorm/LayerNormAffine时有效（RmsNorm不需要输出mean）
- mean/rstd的shape为[BSN1]或[BS]

**aclnn MD使用说明列写法**：
```html
<ul><li>normType=1或2且isTraining=true时有效。</li></ul>
```

**README描述列写法**：
```
LayerNorm中的query均值输出，用于反向
```

### 3.3 空infershape文件

部分算子的infershape.cpp仅包含版权声明，无实际推导逻辑（如upsample_bicubic2d_aa_grad）。此时shape推导由框架自动处理，文档中的shape信息从aclnn接口头文件注释和def.cpp推断。

---

## 4. 从 `aclnn_*.cpp/h` 提取

### 4.1 函数签名

```cpp
aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(
    const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize,
    bool alignCorners, double scalesH, double scalesW, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

**提取规则**：
- 参数顺序必须与函数声明完全一致
- 参数类型映射为参数名列，保留原型中的 `const`：`gradOutput（const aclTensor*）`、`outputSize（const aclIntArray*）`、`alignCorners（bool）`、`scalesH（double）`
- 最后两个参数固定为 `workspaceSize（uint64_t*）` 和 `executor（aclOpExecutor**）`

### 4.2 内部接口转发

```cpp
aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(...) {
    // 参数检查
    auto checkRet = CheckParams(gradOutput, outputSize, inputSize, scalesH, scalesW, out);
    // Contiguous处理
    auto selfContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    // 调用L0算子
    const aclTensor *upsampleOut = l0op::UpsampleBicubic2dAAGrad(
        selfContiguous, outputSize, inputSize, out, alignCorners, realScalesH, realScalesW, uniqueExecutor.get());
    // ViewCopy
    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
}
```

**提取规则**：
- 外部接口是内部接口的简单转发，文档只写外部接口
- `l0op::Contiguous` 调用说明支持非连续Tensor
- CheckParams逻辑是错误码描述的主要来源

### 4.3 参数注释

```cpp
* @param [in] gradOutput: Device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * 支持非连续的Tensor,数据格式支持NCHW，shape仅支持四维Tensor。数据类型与出参out的数据类型一致。
 * @param [in] outputSize: Host侧的aclIntArray，数据类型支持INT64，size大小为2。
 * 表示输入gradOutput在H和W维度上的空间大小。
```

**提取规则**：
- 注释中的描述是文档参数描述的主要来源
- "支持非连续的Tensor" → 非连续Tensor列标√
- "数据格式支持NCHW" → 数据格式列填NCHW、ND
- "shape仅支持四维Tensor" → 维度列填4
- "数据类型与出参out的数据类型一致" → 使用说明列加`<ul><li>数据类型和数据格式与`out`保持一致。</li></ul>`
- "size大小为2" → 使用说明列写"size为2，且各元素均大于零。"

### 4.4 CheckParam逻辑提取错误码

```cpp
static bool CheckNotNull(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclTensor *out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(...)
{
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == EXPECT_SIZE, ..., return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUTSIZE, ..., return false);
    return true;
}

static bool CheckScalesValid(const double width, const double high)
{
    if ((width < 0) || (high < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scales_w and scales_h cannot be negative ...");
        return false;
    }
    return true;
}
```

**转换为错误码表格**：

| 返回码 | 错误码 | 描述 |
|--------|--------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的gradOutput、inputSize或out是空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | gradOutput或out的数据类型不在支持的范围之内。 |
| | | gradOutput与out的数据类型不一致。 |
| | | gradOutput的shape不是4维。 |
| | | inputSize的size不为4。 |
| | | outputSize的size不为2。 |
| | | scalesH或scalesW的值为负数。 |

---

## 5. 从 `*_tiling.cpp` 提取约束

### 5.1 参数校验

```cpp
if (headDim_ < 1 || headDim_ > 1024) {
    OP_LOGE(ctx, "headDim must be in [1, 1024].");
    return ge::GRAPH_FAILED;
}
```

**转换为约束说明**：
- "headDim长度在[1~1024]之间。"

### 5.2 依赖关系约束

```cpp
if (ropeSeq_ > std::min(querySeq_ + encoderQuerySeq_, keySeq_ + encoderKeySeq_)) {
    OP_LOGE(ctx, "seqRope must <= min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey).");
}
```

**转换为约束说明**：
- "seqRope长度大小在[1~Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)]之间。"

### 5.3 平台差异化约束

tiling.cpp中常通过`socVer_`分支实现平台特定校验，必须提取到文档中。

```cpp
if (socVer_ == SOC_VER_950_CODE) {
    if (innerPrecise_ != BsaInnerCalcPrec::LOW_HIGH_MIXED) {
        OP_LOGE(bsaContext->GetNodeName(), "On chip 950, only innerPrec = 4 is supported, "
            "but got %u.", innerPrecise_);
        return ge::GRAPH_FAILED;
    }
} else {
    if (innerPrecise_ != BsaInnerCalcPrec::ALL_HIGH && innerPrecise_ != BsaInnerCalcPrec::ALL_LOW) {
        OP_LOGE(bsaContext->GetNodeName(), "On chip 910 & 910_93, only innerPrec = 0 or 1 is supported, "
            "but got %u.", innerPrecise_);
        return ge::GRAPH_FAILED;
    }
}
```

**转换为参数表下方的产品差异化描述**（仅对上述已核验分支适用）：
```markdown
<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：innerPrecise仅支持配置为4。
<!-- end id1 -->
<!-- npu="910b,A3" id2 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：innerPrecise仅支持配置为0或1。
<!-- end id2 -->
```

**socVer_映射表**：

| socVer_值 | 平台名称 |
|-----------|---------|
| SOC_VER_950_CODE (4) | Ascend 950PR&950DT系列产品 |
| 其他 | Atlas A2/A3系列 |

### 5.4 RegBase平台判断

```cpp
bool regBase = Ops::Cv::OpTiling::IsRegbaseSocVersion(tilingContext);
if (regBase) {
    OP_LOGI(tilingContext->GetNodeName(), "enter Tiling4UpsampleBicubic2dAAGradRegbase");
    return Tiling4UpsampleBicubic2dAAGradRegbase(tilingContext);
}
```

**提取规则**：
- `IsRegbaseSocVersion` 为true时走950平台的tiling逻辑（`*_tiling_arch35.cpp`）
- 非RegBase平台走910b/910_93的tiling逻辑

---

## 6. 从 `*_tiling_arch35.cpp` 提取确定性计算信息

### 6.1 确定性计算分支

```cpp
baseTiling_.isDeterministic = context_->GetDeterministic();

// CalTilingData中的分支
if (isDataCopy) {
    baseTiling_.schId = CONST_0; // 纯copy模板
    baseTiling_.isDeterministic = 1;  // DataCopy场景强制确定性
} else if (baseTiling_.isDeterministic) {
    baseTiling_.schId = CONST_1; // nchw 分线程 deterministic
} else {
    baseTiling_.schId = CONST_1; // nchw 分线程 non-deterministic
}
```

**转换为约束说明中的确定性计算描述**（须额外核验默认配置和各平台 Kernel，不能仅凭分支存在推断默认行为）：
```markdown
- 确定性计算：

  <!-- npu="A3,910b" id1 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：aclnn<OpName>默认确定性实现。
  <!-- end id1 -->
  <!-- npu="950" id2 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnn<OpName>默认非确定性实现，支持通过aclrtSetSysParamOpt开启确定性。
  <!-- end id2 -->
```

**判断规则**：
- `isDeterministic` 分支说明实现区分确定性模式；默认值和是否支持切换仍须核验配置及 Kernel。已确认支持切换时，文档使用 `aclrtSetSysParamOpt`。
- 逐个平台核验默认确定性行为，不因平台是910b/910_93就推断为确定性；缺少证据时标注“待确认”。
- 上述示例仅在 `isDataCopy` 分支显式设置了确定性标志；其他算子仍须检查实际分支。

---

## 7. 从 `examples/test_aclnn_*.cpp` 提取调用示例

### 7.1 提取要点

1. ACL初始化代码（Init函数）
2. 输入Tensor构造函数名和格式
   - `CreateAclTensor` → `ACL_FORMAT_ND`
   - `CreateAclNchTensor` → `ACL_FORMAT_NCHW`
3. aclIntArray构造方式（`aclCreateIntArray`）
4. 两段式接口调用
5. workspace申请和释放
6. 结果同步和拷贝
7. 资源释放

### 7.2 示例不存在时

根据函数签名构造标准调用示例，必须包含：
- CHECK_RET/LOG_PRINT宏定义
- GetShapeSize辅助函数
- Init辅助函数
- CreateAclTensor或CreateAclNchTensor模板函数（根据def.cpp的Format选择）
- aclCreateIntArray用法（若接口含aclIntArray参数）
- 7步流程完整呈现
- 开头固定话术："示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。"

### 7.3 CreateAclTensor变体选择

| def.cpp Format | 使用函数 | aclFormat |
|---------------|---------|-----------|
| `FORMAT_ND` | `CreateAclTensor` | `ACL_FORMAT_ND` |
| `FORMAT_NCHW` | `CreateAclNchTensor` | `ACL_FORMAT_NCHW` |
| `FORMAT_ND` + `FORMAT_NCHW` | `CreateAclNchTensor` | `ACL_FORMAT_NCHW` |

---

## 8. 从 `config/` 目录提取平台信息

### 8.1 目录结构

```
config/
├── ascend910b/          → Atlas A2系列支持
│   ├── *_binary.json
│   └── *_simplified_key.ini
├── ascend910_93/        → Atlas A3系列支持
│   ├── *_binary.json
│   └── *_simplified_key.ini
└── ascend950/           → Ascend 950系列支持
    ├── *_binary.json
    └── *_simplified_key.ini
```

**规则**：有对应子目录 = 该平台支持。与def.cpp的AddConfig交叉验证。

---

## 9. 参数名映射：def.cpp → aclnn接口 → 文档

def.cpp中的Input/Output名称（snake_case）与aclnn接口参数名（camelCase）的映射关系：

| def.cpp名称 | aclnn接口参数名 | aclnn MD参数名列 | README参数名列 |
|-------------|----------------|-----------------|---------------|
| `grad_output` | `gradOutput` | `gradOutput（const aclTensor*）` | `grad_output` |
| `grad_input` | `out` | `out（aclTensor*）` | `grad_input` |
| `output_size` | `outputSize` | `outputSize（const aclIntArray*）` | `output_size` |
| `input_size` | `inputSize` | `inputSize（const aclIntArray*）` | `input_size` |
| `align_corners` | `alignCorners` | `alignCorners（bool）` | `align_corners` |
| `scales_h` | `scalesH` | `scalesH（double）` | `scales_h` |
| `scales_w` | `scalesW` | `scalesW（double）` | `scales_w` |

**映射规则**：
- def.cpp名称用snake_case
- aclnn接口参数名用camelCase
- aclnn MD参数名列用camelCase + 中文括号类型
- README参数名列用def.cpp的snake_case（不含类型）

**特殊情况**：某些参数在def.cpp中声明为Input（如output_size、input_size），但在aclnn接口中为`aclIntArray*`类型。在README中按def.cpp声明归类：
- def.cpp `this->Attr("output_size")` → README标"属性"
- def.cpp `this->Input("blockShape")` → README标"可选输入"

---

## 10. 反向算子信息提取

### 10.1 前向算子关联

当算子名称以"Grad"结尾时，需关联前向算子：

```
算子名: UpsampleBicubic2dAAGrad
前向算子名: UpsampleBicubic2dAA
前向算子路径: ../upsample_bicubic2d_aa/
```

**提取规则**：
1. 从算子名称去掉"Grad"后缀得到前向算子名
2. 在同级目录查找前向算子目录
3. 生成交叉引用链接：
   - aclnn MD: `[aclnnUpsampleBicubic2dAA](../../upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)`
   - README: `[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)`

### 10.2 功能说明写法

- aclnn MD: `接口功能：[aclnnUpsampleBicubic2dAA](../../upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。`
- README: `算子功能：[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。`
