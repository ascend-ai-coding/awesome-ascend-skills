# 源码信息提取指南

本文档详细说明如何从Ascend CANN算子源码中提取文档所需信息，以 [`example-aclnn-md.md`](example-aclnn-md.md) 和 [`example-readme.md`](example-readme.md) 为参考范例。

**MANDATORY - READ ENTIRE FILE**：阶段1源码分析时，必须完整阅读本指南。

---

## 使用边界

输入状态、资料读取顺序和冲突采用策略统一见[父入口的资料来源与冲突处理说明](../SKILL.md#资料来源与冲突处理说明)。解释冲突处理方法时，先说明“未提供规格、需求或设计文档时直接依据源码生成”和“有资料时先完整读取并提取设计意图，再完整扫描源码核验”两条路径，再解释当前冲突；已声明提供但不可读的资料不能当作未提供。本指南负责“源码线索如何核验为文档事实”，不另设咨询或交付流程。

全文代码、参数值、shape 和产品结论都是提取方法的示例。生成或解释规则时，区分格式与事实：模板决定怎样写，目标接口的证据决定能写什么；示例本身不证明其他算子的支持度、默认值或约束。

资料冲突咨询的填写示例（仅依据用户陈述，未读取原文件）：

| 事实项 | 资料结论及位置 | 源码结论及位置/版本 | 一致性 | 待确认项及采用理由 |
|---|---|---|---|---|
| 输入dtype范围 | 用户称REQUIREMENTS.md要求FP32，章节待确认 | 用户称def.cpp仅注册FP16，路径、行号及版本待确认 | 需求与已述注册范围不一致；调用链尚未核验 | 先记录已述实现FP16，不宣称FP32已支持；完整核验接口、Tiling及Kernel后确认实现范围，并请开发者判断实现缺口或资料未同步 |

这不是对实际文件的核验结论。FP16/FP32写入参数的数据类型说明，不写成产品支持行；源码不可读时，所有未核验字段均标注“待确认”，不只标冲突字段。

## 证据文件覆盖

按下表核对目标接口的证据类别。文件路径是常见布局下的定位线索，不是排除其他目录、共享文件或依赖的过滤条件；实际范围由接口声明和可达调用链确定。

| 证据类别 | 常见文件或目录 | 核验用途 |
|----------|----------------|----------|
| 接口声明 | `op_host/op_api/aclnn_*.h` | 递归扫描全部 `ACLNN_API` 声明，确认接口数量、实际函数签名和参数顺序 |
| 接口实现 | `op_host/op_api/aclnn_*.cpp` | 参数校验、空指针语义、调用链及实际错误码分支 |
| 算子注册 | `op_host/*_def.cpp` | Input/Output/Attr、ParamType、dtype、format 和 AddConfig；注册不单独证明运行时限制 |
| 公共定义 | `op_host/*_base.h` | 枚举、维度常量、参数索引及其实际使用含义 |
| Shape 推导 | `op_host/*_infershape.cpp` | 输出 shape、条件输出和推导条件 |
| Tiling 头文件 | `op_host/*_tiling.h` | 数据结构、字段类型、常量以及 Host/Kernel 间参数传递；不能用读取 cpp 代替头文件核验 |
| Tiling 实现 | `op_host/*_tiling.cpp`、可用的 `op_host/*_tiling_arch35.cpp` | shape/attr 校验、平台分支、workspace 和确定性调度路径 |
| Kernel 实现与分派 | `op_kernel/*.cpp`、`op_kernel/*_tiling_key.h` | 实际执行分支、类型转换、访存及数值约束；核对 tiling key 与实现的对应关系，不能由 Host 注册代替 |
| 平台配置 | `op_host/config/*/` | 与 AddConfig、接口及实现交叉核验逐平台支持情况，目录存在或缺失不独立证明支持度 |
| 调用与测试 | `examples/test_aclnn_*.cpp`、`tests/st/` | 目标接口的实际调用、参数构造和已有测试范围；不将示例取值泛化为约束 |

在可用范围内完整读取目标证据文件，并沿调用链读取必要定义；缺失文件或未解析依赖记录其影响和替代证据，不能默认为对应约束不存在。Tiling 头文件、Tiling 实现和 Kernel 是不同证据类别，合并说明时仍须保留各自的文件范围与核验用途。

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
| `ParamType(OPTIONAL)` | 输入/输出列="可选输入"，使用说明列加`<ul><li>按实际外部校验说明是否允许nullptr及其含义，不仅依据OpDef的OPTIONAL推断。</li></ul>` | 输入/输出/属性列="可选输入" |
| `ge::DT_FLOAT16` | FLOAT16 | FLOAT16 |
| `ge::DT_BF16` | BFLOAT16 | BFLOAT16 |
| `ge::DT_FLOAT` | FLOAT32 | FLOAT32 |
| `ge::FORMAT_ND` | ND | ND |
| `ge::FORMAT_NCHW` | NCHW | NCHW |
| `.AutoContiguous()` 或接口中的连续化处理 | 作为支持非连续 Tensor 的核验线索，沿目标接口调用链确认后填表 | 不涉及 |
| 未发现连续化配置 | 不能直接标×；继续检查接口校验和实现，缺证时待确认 | 不涉及 |

**注意**：README取def.cpp原始名称，aclnn参数名从具体API接口的入参声明获取；通过转发建立映射，不能按命名风格转换。

### 1.2 输出参数信息

```cpp
this->Output("grad_input")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

输出类型按目标输出的注册、接口校验及实现分别核验；不能根据 mean/rstd 等参数名推断固定类型，也不能默认与输入类型一致。

### 1.3 属性信息

```cpp
this->Attr("output_size").AttrType(REQUIRED).ListInt();    // → 属性名、类型ListInt、无默认值
this->Attr("input_size").AttrType(REQUIRED).ListInt();     // → 属性名、类型ListInt、无默认值
this->Attr("align_corners").AttrType(OPTIONAL).Bool(false); // → 属性名、类型bool、默认值false
this->Attr("scales_h").AttrType(OPTIONAL).Float();          // → 属性名、类型float、无默认值（空）
this->Attr("scales_w").AttrType(OPTIONAL).Float();          // → 属性名、类型float、无默认值（空）
```

**Attr类型映射示例**（aclnn 类型须读取实际函数声明，不能仅由注册属性类型推导）：

| 源码Attr类型 | aclnn接口类型 | aclnn MD参数名写法 | README数据类型 | README输入/输出/属性 |
|-------------|-------------|-------------------|---------------|---------------------|
| `.Int(x)` | `int64_t` | `attrName（int64_t）` | INT64 | 属性/可选属性 |
| `.Float(x)` | `double` | `attrName（double）` | FLOAT32 | 属性/可选属性 |
| `.Bool(x)` | `bool` | `attrName（bool）` | BOOL | 属性/可选属性 |
| `.ListInt()` | `aclIntArray*` | `attrName（aclIntArray*）` | INT64 | 属性/可选属性 |

**核验要点**：
- 注册中的属性类型用于 README；aclnn 参数的类型、指针和 const 以实际原型为准，上表只演示两层类型可能不同
- 列表参数的 size、元素范围和空值语义分别检查，不能由列表类型补造约束
- `AttrType(REQUIRED)` → README中标"属性"
- `AttrType(OPTIONAL)` → README中标"可选属性"，描述列用`<ul><li>`包裹默认值

### 1.4 平台配置

```cpp
this->AICore().AddConfig("ascend910b");    // → Atlas A2系列产品，结合接口校验判断支持情况
this->AICore().AddConfig("ascend910_93");  // → Atlas A3系列产品，结合接口校验判断支持情况

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
this->AICore().AddConfig("ascend950", regbaseConfig);  // → Ascend 950PR&950DT系列产品，结合接口校验判断支持情况
```

**注意**：950平台可能有独立的`OpAICoreConfig`配置，含DynamicCompileStaticFlag等扩展信息，且对应独立的tiling文件（`*_tiling_arch35.cpp`）。

`AddConfig` 是平台注册证据，不是完整发布支持结论。逐平台交叉核验 config、接口检查及实际实现；未找到注册不等于已证明不支持，证据不足时记录“待确认”。

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

仅文档化目标 aclnn 接口实际暴露并接受的枚举值。是否属于内部枚举须核验接口与参数校验，不能依据名称名单判断。

### 2.2 维度常量

```cpp
constexpr size_t INPUT_DIM_NUM = 4;   // B, S, H, D
```

检查常量在目标参数校验或 shape 推导中的实际用途，再记录维度；常量存在本身不能证明所有参数具有该维度。

### 2.3 参数索引枚举

```cpp
enum class InputIndexForward : size_t {
    QUERY_INDEX = 0,
    KEY_INDEX,
    ...
};
```

用于定位内部参数绑定；对外参数顺序和数量仍以实际 aclnn 声明及转发关系核对。

**注意**：部分算子（如upsample_bicubic2d_aa_grad）不存在`*_base.h`文件，此时从def.cpp和aclnn接口头文件提取参数信息即可。

---

## 3. 从 `*_infershape.cpp` 提取

### 3.1 输出shape推导

```cpp
*queryOutputShape = {batch, head, querySeq + encoderQuerySeq, dim}; // B, H, S, D
```

**上述代码的示例转换**（实际维度含义与顺序须由目标实现确认）：
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

**上述分支可确认的信息**：
- isTraining=true 且 normType 为示例中两个值时，该分支为 mean/rstd 设置 `[batch, querySeq, head, 1]`
- 其他模式是否输出、是否允许空指针及输出 shape，仍需读取完整推导与接口使用路径；不能从未展示的分支补出结论

**aclnn MD使用说明列写法**：
```html
<ul><li>normType=1或2且isTraining=true时有效。</li></ul>
```

**README描述列写法**：
```
LayerNorm中的query均值输出，用于反向
```

### 3.3 空infershape文件

infershape.cpp 为空或不存在时，沿实际调用链查找输出构造及框架推导证据，结合接口注释和注册核验。不能仅因文件为空就认定由框架自动处理；无法确认的 shape 保留待确认。

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
- 参数名列采用“参数名（C/C++类型）”：`gradOutput（const aclTensor*）`、`outputSize（const aclIntArray*）`、`alignCorners（bool）`、`scalesH（double）`；名称、const和指针层级与原型一致
- 最后两个参数写为 `workspaceSize（uint64_t*）` 和 `executor（aclOpExecutor**）`

记录参数到OpDef的转发映射及来源位置，按[参数名映射](#9-参数名映射defcpp--aclnn接口--文档)逐项核对名称和顺序。

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
- 文档描述目标外部接口；沿转发链核对类型转换、参数重排、默认值和输出复用，不假定内外语义完全相同
- `l0op::Contiguous` 是非连续输入处理的线索，须确认目标参数可达该调用且未被前置校验拒绝
- 错误码须扫描完整接口实现及辅助校验，不能只读取CheckParams；提取边界见[接口返回分支提取错误码](#44-接口返回分支提取错误码)。

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
- "数据格式支持NCHW" → 记录 NCHW；只有目标实现另有证据时才增加 ND 或其他格式
- "shape仅支持四维Tensor" → 维度列填4
- "数据类型与出参out的数据类型一致" → 只记录数据类型与 out 一致；数据格式是否一致另行核验，不扩展原注释的含义
- "size大小为2" → 记录 size 为 2；各元素范围须另查校验分支，不能自行补充正数限制

### 4.4 接口返回分支提取错误码

扫描对应阶段的完整接口实现及辅助函数，展开影响返回值的校验宏，追踪实际返回的异常码及原因；辅助函数的bool结果、日志中的错误码须结合调用方确认最终返回值。不能只扫描CheckParam/Validate。

- 返回值表覆盖接口实现中显式返回的异常码，包括通用错误码。按实际返回码类型和数值归组，同一码只占一行，合并全部已核验原因；不能仅在表外说明，也不能按`ACLNN_ERR_INNER_`等前缀排除。
- 例如，workspaceSize或executor为空、连续化结果为空及ViewCopy失败均显式返回ACLNN_ERR_INNER_NULLPTR（561103）时，合并入561103一行；创建执行器失败显式返回ACLNN_ERR_INNER_CREATE_EXECUTOR（561101）时，另列一行。这些是提取示例，不是所有算子必有的错误码。
- 数值从实际返回码定义或目标仓库返回码文档核对；名称和数值不得从示例套用。仅由框架或运行时透传、接口内未明确列出的通用异常通过返回码文档链接说明，不枚举整个通用码表，不从Tiling日志推断固定aclnn错误码。

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

| 返回码/状态码 | 码值 | 描述 |
|--------|--------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的gradOutput、inputSize或out是空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | gradOutput或out的数据类型不支持或不一致；gradOutput不是4维；inputSize或outputSize长度不符；scalesH或scalesW为负数。 |

---

## 5. 从 `*_tiling.cpp` 提取约束

按“接口声明及转发 → CheckParam/Validate与辅助函数 → 全部适用Tiling/IsCapable → Kernel读取和Cast”的路径完整扫描。日志只用于定位，最后读取完整分支。按参数组核对单参数与组合约束，记录触发条件、允许值/关系、影响及平台；可调用范围取入口与实现的交集。

先记录目标源码提交、分支及相关未提交修改，再与已提供规格和既有README/aclnn文档逐条对照。资料中的能力与目标源码不一致时，记录各自版本和采用依据；未经授权不更新源码或将远端新版本能力混入本地旧版文档。范围不明时标注待确认，不能把版本差异误判为同版本提取遗漏，也不能直接沿用旧文档结论。

当实际接口包含maskType、winLeft、winRight、attenMaskOptional时，不能只找到其中一个参数便停止。若分支确认仅允许maskType=1、winLeft=winRight=-1、attenMaskOptional=nullptr，应集中写成这组明确要求，不写“不启用时为-1”暗示还有其他窗口。

精度约束从实际类型转换取证：double转float时说明scaleValue以FLOAT32参与计算及低位舍入；FLOAT32转FLOAT16时说明发生的条件及数值舍入影响；扩精度不直接描述损失。不要写“计算中存在低精度中间量”“不能视为全程FLOAT32”“NaN/Inf及溢出未提供保证”。特殊值或溢出限制只有实际检查、实现语义或确认规格支持时才写。workspace分配公式不是自动的内存限制；有容量边界时再用不等式说明。

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
- "seqRope 不超过 Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)。" 下界须由其他校验分支另行确认。

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

**平台分支核验**：读取目标代码中的枚举定义和所有可达分支，再映射公开产品名。不能把示例常量的数值或 else 分支当成跨算子通用的产品映射。

### 5.4 RegBase平台判断

```cpp
bool regBase = Ops::Cv::OpTiling::IsRegbaseSocVersion(tilingContext);
if (regBase) {
    OP_LOGI(tilingContext->GetNodeName(), "enter Tiling4UpsampleBicubic2dAAGradRegbase");
    return Tiling4UpsampleBicubic2dAAGradRegbase(tilingContext);
}
```

**提取规则**：读取目标版本 `IsRegbaseSocVersion` 的定义及调用分支，确认实际覆盖的平台和 tiling 路径。函数名或 `arch35` 文件后缀只用于定位，不独立证明产品支持或 else 分支覆盖范围。

---

### 5.5 约束分组与文档覆盖

提取完整不等于可以任意压缩成品。每条已核验约束记录适用接口、模式/布局、平台、触发条件、允许值/关系、调用影响、源码证据及最终文档位置；该覆盖记录用于核验，不要求把记录表原样放进成品。

- **按适用范围分组**：仅真正通用的限制写为公共约束；模式、布局或平台独有的限制就近标明条件或分组说明。按实际入口分派扫描全部校验路径，不因函数名不是Validate/CheckParam而漏掉分支；不能把某模式的固定D、头数范围或Metadata要求扩大为整个接口的限制。
- **README保留关键调用边界**：存在实现特有限制时，保留支持模式、所需前置算子、关键输入依赖、主要属性组合、尺寸/头数/分块边界及重要布局要求，使读者能判断场景是否适用。按模式及平台分别说明不同要求，不固定条数，也不将一份完整清单压成几条泛化摘要。
- **区分摘要与细节**：dtype、单参数值域等细节可在参数表就近说明；跨参数条件集中写在约束章节。允许在约束中简要重述影响是否可调用的关键条件，不因“避免重复”删掉模式前置条件、关键组合或边界；不要求逐字重复所有参数项或内部计算细节。
- **链接补充而不代替关键条件**：完整shape关系、复杂平台差异、数值行为及接口层确定性说明可放在aclnn文档并引用。引用须指向适用接口和实际位置，正文仍说明关键适用条件，不能仅写“所有约束见接口文档”。源码中的失败、忽略、自动调整等行为分别核验，不能互相替换。
- **覆盖验收**：将既有约束和新提取事实逐项归入“保留于约束章节、移至参数表、由适用详细说明覆盖、因版本/实现冲突不采用”。移动项须给出位置，冲突项须保留证据与原因；关键调用边界不能仅以外链视为已覆盖。无约束时才按模板省略或写“无”。

例如，已确认模式A无需Metadata、D在某范围内，模式B需先生成Metadata、D固定时，应分别说明两条路径；不能合并为“D固定且必须提供Metadata”。这些只是组织方式示例，实际模式、数值和前置步骤均由目标版本源码确认。

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

**转换为约束说明中的确定性计算描述**：分别核验各平台的默认模式、是否支持切换和实际实现。下例只有在这些事实全部证实时才可采用，不能据产品名选择默认值：
```markdown
- 确定性计算：

  <!-- npu="A3,910b" id1 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：{各平台已核验的默认行为与切换能力；不同时分开描述}。
  <!-- end id1 -->
  <!-- npu="950" id2 -->
  - <term>Ascend 950PR&950DT系列产品</term>：{该平台已核验的默认行为与切换能力}。
  <!-- end id2 -->
```

**判断规则**：
- `isDeterministic` 是模式相关分支的线索；默认值、实际模式差异和是否支持切换仍须核验配置及 Kernel。已确认支持手动开启时配置 API 使用 `aclrtSetSysParamOpt`，但不能因此断言默认非确定性。
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

文档内嵌示例从第一条必要include等代码开始，不复制或新加Copyright/License文件头；原始源码文件的许可声明保持原样。

提取或生成示例前读取目标仓库的代码规范和格式配置，检查缩进、命名、宏续行、参数顺序、返回值及资源释放，避免原样继承源样例的格式或遗漏。可用时使用仓库已有格式化器；没有工具时说明实际检查范围。未编译或运行不得声称通过，冒烟数据不代表完整精度验证。

### 7.2 示例不存在时

没有现成样例时，只有在原型、参数约束和调用语义已核验后才能构造调用示例；关键事实缺失时按父入口保留草稿，不用常见值填空。完整示例必须包含：
- CHECK_RET/LOG_PRINT宏定义
- GetShapeSize辅助函数
- Init辅助函数
- CreateAclTensor或CreateAclNchTensor模板函数（根据def.cpp的Format选择）
- aclCreateIntArray用法（若接口含aclIntArray参数）
- 7步流程完整呈现
- 开头固定话术（以下相对链接属于生成文档，不相对于本指南）：`示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。`

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
├── ascend910b/          → Atlas A2系列的配置线索
│   ├── *_binary.json
│   └── *_simplified_key.ini
├── ascend910_93/        → Atlas A3系列的配置线索
│   ├── *_binary.json
│   └── *_simplified_key.ini
└── ascend950/           → Ascend 950系列的配置线索
    ├── *_binary.json
    └── *_simplified_key.ini
```

目录名仅提供平台配置线索。读取配置内容，与 AddConfig、接口限制、实现和目标发布版本交叉核验；有目录不直接等于支持，缺目录也不直接等于不支持。分别记录支持、不支持和证据不足，不能照抄模板中的 √/× 或支持度。

---

## 9. 参数名映射：def.cpp → aclnn接口 → 文档

下表为OpDef原名与外部声明原名的映射示例；实际映射须核对声明和转发，不按命名风格推导：

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
- def.cpp名称使用Input/Output/Attr原始字符串
- aclnn接口参数名逐字取自具体API接口的入参声明
- aclnn MD参数名列采用“参数名（C/C++类型）”，名称及类型逐项核对具体API声明，保留const和指针层级
- README参数名列取def.cpp原始字符串（不含类型），不按风格改写

**特殊情况**：某些参数在def.cpp中声明为Input（如output_size、input_size），但在aclnn接口中为`aclIntArray*`类型。在README中按def.cpp声明归类：
- def.cpp `this->Attr("output_size")` → README标"属性"
- def.cpp `this->Input("blockShape")` 且 `.ParamType(OPTIONAL)` → README标"可选输入"；只有 Input 声明不能推出可选

---

## 10. 反向算子信息提取

### 10.1 前向算子关联

反向算子的前向关联须由接口语义、需求或实现确认。名称的 Grad 后缀只能作为查找线索，不能独立证明关联关系。以下是已确认关联后的路径示例：

```
算子名: UpsampleBicubic2dAAGrad
前向算子名: UpsampleBicubic2dAA
前向算子路径: ../upsample_bicubic2d_aa/
```

**提取规则**：
1. 用接口语义和目标实现确认前向算子及实际接口名
2. 在已提供的项目范围内核对目标目录和文档是否存在；缺少目标时记录缺口，不构造虚假链接
3. 按实际目录生成交叉引用链接，以下仅为该示例的写法：
   - aclnn MD: `[aclnnUpsampleBicubic2dAA](../../upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)`
   - README: `[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)`

### 10.2 功能说明写法

- aclnn MD: `接口功能：[aclnnUpsampleBicubic2dAA](../../upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。`
- README: `算子功能：[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。`

## 11. 功能特点提取

从设计意图和实际计算路径提取用途、输入输出、参与计算的数据范围、模式或权重等差异，说明其对结果的影响。不要用内部调度分块代替功能特点，不把某个打样算子的机制列为所有算子的要求。反向算子保留前向引用，并说明梯度如何传播及依赖的前向结果。


## 12. layout与关联Tensor

**layout独立章节和关系表均为可选项；布局事实本身仍须说明。** 读取接口校验、infershape及Kernel寻址，核验逻辑轴、shape、输入输出关系、辅助索引/计数Tensor、实际支持组合和平台差异。不能因省略章节而遗漏使用信息，也不能因有layout参数、多种布局、TND或稀疏索引就自动增加章节。ND等数据格式、逻辑layout及非连续Tensor是不同概念，不能混写。

### 12.1 选择说明位置

| 情况 | 呈现方式 |
|---|---|
| shape固定、轴含义清楚，或只是BSH/BSND/BNSD/TND等常规排布；枚举与各Tensor的shape可在参数表清楚表达 | 在layout参数的使用说明、对应Tensor的shape列及表前维度说明中就近写明，不另设layout章节。多种枚举或输入输出排布不同本身不要求成章。 |
| 布局与dtype、长度、平台等少量条件有关，但相关参数说明及已有场景约束足以表达 | 继续在参数表及相关场景约束说明，避免建立重复总表。 |
| 多个布局参数的合法组合、原始KV与Paged KV、模式导致的索引方向/共享关系/batch维变化等跨Tensor关系，逐参数说明会反复引用、分散或难以对照 | 可在约束中增加`### layout对应关系说明`集中解释；按实际复杂度决定，不将其中任一关键词作为强制触发条件。用户明确要求layout章节时也可增加。 |
| Metadata或配套接口沿用主算子的相同关系，主文档已有完整说明 | 就近链接该章节，补充本接口的支持值和差异，不复制整表；核对接口版本、关系适用性、目标文件及锚点。没有适用说明时不造链接。 |

attention目录中的写作样式可参考：FlashAttentionScore/VarLenScore在参数表列出shape及inputLayout；LightningIndexer分别说明Q/K布局和维度；FusedInferAttentionScoreV5将输入输出转换写入inputLayout说明并结合场景约束；GenericBlockSparseAttention在独立章节集中解释Q/KV组合及稀疏索引；其Metadata接口引用主文档。这些是组织方式参考，不是其他算子的支持规格。

### 12.2 增加章节时怎么写

1. 只集中解释需要对照的关系。先说明实际支持值/合法组合及控制参数，再根据内容选择简短段落、组合表、shape关系表或稀疏模式表，不要求全部表格，也不预设所有算子均由稀疏表开头。
2. 多个layout参数存在组合关系时，可用“实际布局参数名/实际布局参数名/描述”的组合表；需要比较shape时，列“布局参数及值/对应Tensor/shape/关系说明”。输入、输出、LSE等分别核验，不默认跟随同一layout或强制两个layout相等。
3. 稀疏索引关系需要单独解释时，可采用[模板稀疏布局示例](aclnn-api-doc-template.md#layout对应关系说明)的四列表，表头取实际API原名，如`layoutSparsePattern / sparseBlockIdx / sparseBlockCount / 描述`。表前同时声明实际支持值、其他已确认枚举的非支持状态，以及当前支持模式在各已确认layout下是否保留batch维，避免完整规格表被误读为支持清单；不能把batch维声明仅放在表后。表内说明Q选KV还是KV选Q、组内共享还是独立，表后解释维度与容量。实际布局组合复杂时可先说明组合关系，不强制所有接口使用同一组表格。
4. shape沿用已确认规格中清晰的维度名，并定义与源码符号的映射；只定义本次使用的符号。按需说明TND累计偏移、分块计算、尾块、有效计数与容量，不把“至少为S1”写成“等于S1”，不把索引容量误写为有效索引数。
5. 默认只列当前支持且含义已确认的值/组合。用户要求完整规格表或理解差异确需其他枚举时，才列已确认语义的其他行，并明确不支持或待确认。不得将模板0～7自动复制为目标能力。
6. 常规Tensor shape保留在参数表，章节仅补充跨参数关系及必需差异；平台内容保留产品标签，避免整表重复及无关内部地址/调度细节。省略独立章节时，同样将这些必要信息写在参数说明、已有场景约束或有效引用中。

需要通用shape关系表时可采用：

| 布局参数及值 | 对应Tensor | shape | 关系说明 |
|---|---|---|---|
| <已确认值> | <外部参数原名> | <轴顺序> | <符号含义、参数关联及支持组合> |

README只保留必要参数语义和接口文档链接；没有独立layout章节时链接整个接口文档或实际相关章节，不生成悬空锚点，不重复详细关系表。

### 12.3 平台差异的产品化表达

LSE等中间结果的已核验平台差异也属于成品文档的产品差异，不能仅写裸`Arch22/Arch35`。先核对目标版本的架构别名、枚举和配置，再按[产品映射配置](aclnn-full-api-docs-gen/config/product_mapping.yml)转为正式产品名和模板标签：

| 已确认的架构分支 | 对外产品名 | npu标签值 |
|---|---|---|
| Arch22对应2201 | Atlas A3系列产品、Atlas A2系列产品 | `A3,910b`；两产品行为不同时分开 |
| Arch35对应3510 | Ascend 950PR&950DT系列产品 | `950` |

在表后用产品差异段落保留各自shape，产品名带`<term>`，整段用配对`<!-- npu="..." idN -->`与`<!-- end idN -->`包裹；id在同一文档中唯一。名称映射不证明接口支持、shape或发布范围，不能把同架构产品的行为自动视为相同；映射或具体行为缺证据时标待确认，不猜产品范围。

## 13. 参数使用说明写法

使用说明面向接口调用者，表达调用要求，不把源码核验笔记直接作为成品。以参数所在行提供的主语为上下文，写清动作、条件和对象；“不支持空Tensor。”“数据类型与query保持一致。”等清楚的短句可以保留，不为完整句式机械重复参数名。

- **各列分工**：描述列解释参数是什么、用于什么；使用说明列写合法输入、依赖和调用要求；dtype、format、shape列填写各自规格。条件导致规格变化时就近写清对应关系，不把所有信息压成分号分隔的关键词，也不重复抄写各列。
- **不重复常规必选性**：“输入”“输出”等分类已经表达必选性时，不在使用说明中再写“必传”“必须配置”或无额外含义的“显式传入”。保留“不支持空Tensor”、维度正值、shape/dtype匹配等实际约束；nullptr与空Tensor分别核验，不能互相替代。没有额外使用要求时填“-”，不编造要求填满单元格。
- **条件必需性**：对标为可选输入但某条件下必需的参数，写成“当〈具体参数及取值〉时，必须传入〈实际内容或来源〉；当〈其他条件〉时，传入〈nullptr或相应内容〉”。单独的“TND必传”“定长布局传nullptr”不能代替完整条件；源码只检查元素数时写元素数，不把建议排布写成强制rank，也不以“推荐”模糊实际约束。
- **空指针含义**：可传nullptr时，说明表示不使用、由接口生成还是采用何种已确认行为。禁止以“允许nullptr”代替已核验的行为说明；缺少行为证据时记录待确认，不猜测默认值。名称或OpDef标为可选、实际入口不接受nullptr时，直接说明当前调用要求及来源，不写“名称含Optional但当前必传”等命名分析。
- **条件与引用**：写实际API参数名和支持值，不用“TND”“该模式”“有差异”等孤立词让读者推断条件。较长的平台或组合约束可链接到具体有效位置，但表内仍交代该参数的基本用途及触发条件，不只写“见约束说明”。不使用“前三轴一致”等可能有歧义的简写，必要时列出对应维度或关系。

例如，以下事实已核验时，可将`cuSeqLengthsQOptional`的使用说明写为：“layoutQ为\"TND\"时，必须传入记录各batch的Q序列累计偏移的Tensor；layoutQ为\"BNSD\"或\"BSND\"时传nullptr。”输入/输出列保留“可选输入”，不能因TND条件下必需而删掉分类；dtype列填INT64，shape列按已核验要求说明包含B+1个元素，B表示本例的batch数。累计偏移的首项、末项和单调性等要求有证据时就近补充。这里的参数、值及行为仅为写作示例，不套用于其他接口。


## 14. 章节引用

引用其他章节时必须提供可点击链接，不能只写“具体关系见约束说明”“详见上文”等纯文本。Markdown正文使用“具体关系见[约束说明](#55-约束分组与文档覆盖)。”；HTML表格单元格使用“具体关系见<a href="#55-约束分组与文档覆盖">约束说明</a>。”。跨文件引用使用实际相对路径及章节锚点；优先链接具体相关小节，并核验目标文件、标题及锚点存在。链接用于补充细节，不能替代参数基本用途、触发条件及关键调用边界。


引用可定位的内容、样例代码、源码文件或接口文档时，应提供可点击链接，方便开发者直接查找；调用说明表的“样例代码”列不得仅写“本地torch_extension/flash_attn_grad.py”等纯文本路径。使用有意义的文件名或接口名作为链接文字，仓内文件按最终文档位置计算相对路径，例如[flash_attn_grad.py](torch_extension/flash_attn_grad.py)；HTML表格使用<a href="torch_extension/flash_attn_grad.py">flash_attn_grad.py</a>。文档输出到源码仓外时，使用可访问的仓库文件链接或正确的相对路径，不沿用已失效的仓内相对路径。核验文件及目标章节存在；缺少目标时如实说明，不编造链接。


### 调用方式完整性（强制）

生成README前，递归检查算子examples、tests/st及相关torch_extension、Python封装、C++绑定和GE注册，建立“调用方式—实际入口—绑定/实现—样例—README落点”清单。扫描全部样例类型，包括test_aclnn_*.cpp、test_torch_*.py及其他实际存在的调用样例；不得只搜索aclnn文件或只按文件名判断支持。

逐项读取样例的实际调用，追踪导入、注册或绑定到目标算子，核验PyTorch/torch.ops、自定义扩展、aclnn、GE等已存在的调用方式。发现PyTorch样例后必须核验并在README调用说明中单列PyTorch入口，提供可点击样例链接和必要调用条件；不能因为本Skill重点生成aclnn文档而省略其他方式。多个入口共用样例时仍分别列出，不以链接去重合并入口。

交付前将发现清单与README逐项对账：每个已核验调用方式及入口均须有对应行。绑定、依赖或样例缺失时，在核验记录中写明具体缺口及影响，不将未核验方式直接宣称为支持，也不能静默遗漏。仅当用户明确限定调用方式时缩小该清单范围。不存在某方式时不编造入口或占位链接。

典型漏项检查：目录同时存在test_aclnn_xxx.cpp与test_torch_xxx.py时，必须核验两条调用链；确认属于目标算子后，README分别保留aclnn和PyTorch行。仓外输出按最终位置计算样例链接，禁止写入编辑器临时预览URL。
