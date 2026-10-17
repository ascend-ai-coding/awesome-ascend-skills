# Ascend C Host 侧日志上报规范

<适用>
语言: C++
侧别: Host
领域: true
触发: OP_LOGE, OP_LOGI, OP_LOGW, OP_LOGD
默认启用: true
适用场景: op_host/**/*.cpp, op_graph/**/*.cpp, op_api/**/*.cpp（覆盖 tiling、InferShape、InferDataType、aclnn 实现）
介绍: 两条检视规则：LOG-1 语义可映射时禁止独立使用（裸调用）OP_LOGE 打印错误日志，必须替换为 log.h 预定义宏 OP_LOGE_WITH_*/OP_LOGE_FOR_*，检视意见须给出应替换的宏名称与宏定义所在文件路径；LOG-2 日志中打印算子名称必须从 TilingContext 获取——优先直接传 context（框架自动携带算子标识），context->GetNodeName() 合规，禁止自定义算子名称的字符串（context 判空分支内豁免）
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 1
</检视负载>

> **适用范围**：AscendC host 侧代码——`op_host/`（tiling、InferShape、InferDataType）、`op_graph/`（infer 实现）、`op_api/`（aclnn 实现）中的错误日志打印
>
> **背景**：opbase 在 `include/op_common/log/log.h` 中为常见参数校验场景预定义了带专属错误码（EZ 系列）的日志宏，打印日志的同时上报结构化错误信息

## 快速索引

| 规范编号 | 规范名称 | 类别 | 侧别 | 严重级别 |
|---------|---------|------|------|---------|
| LOG-1 | 语义可映射时禁止独立使用 OP_LOGE，必须替换为 log.h 预定义宏 | 日志规范 | Host | 中 |
| LOG-2 | 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 | 日志规范 | Host | 中 |

---

## LOG-1: 语义可映射时禁止独立使用 OP_LOGE `[适用: Host]`

**严重级别**：中

### 问题描述

裸 `OP_LOGE`（无下划线后缀）内部上报**通用错误码 EZ9999**，不携带结构化字段；而 `log.h` 预定义宏（`OP_LOGE_WITH_*` / `OP_LOGE_FOR_*`）在打印日志的同时，通过 `REPORT_PREDEFINED_ERR_MSG` 上报**专属 EZ 错误码（EZ0001–EZ0038）**及结构化 msgKey/msgValue，供错误诊断系统采集与聚合。

当裸 `OP_LOGE` 的日志语义属于预定义宏覆盖的场景（参数 shape/维度/shapeSize/dtype/format/属性取值/tensor 数量/列表长度/stride/文件/配置项等校验失败）时，必须替换为对应宏，否则错误信息无法进入结构化诊断链路。

**覆盖的代码范围**：`op_host/**/*.cpp`（tiling、InferShape、InferDataType）、`op_graph/**/*.cpp`、`op_api/**/*.cpp`（aclnn 实现）。Kernel 侧（`op_kernel/`）不适用本条。

### 关键定义："独立使用"（裸调用）

检索正则：`\bOP_LOGE\s*\(`

| 用法 | 是否本条检视对象 |
|------|-----------------|
| `OP_LOGE(...)` —— 裸调用，无下划线后缀 | 是 |
| `OP_LOGE_WITH_*(...)` / `OP_LOGE_FOR_*(...)` —— log.h 预定义宏 | 否，合规用法 |
| `OP_LOGE_LIBOPAPI_REPORT(...)` / `OP_LOGE_WITHOUT_REPORT(...)` —— 底层/别名宏 | 否 |

### 检查点

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | op_host/ op_graph/ op_api/ 下是否存在裸 `OP_LOGE(` | 检索 `\bOP_LOGE\s*\(`，带后缀宏跳过 |
| 2 | 裸调用的日志语义是否命中宏语义映射表 | 按下方映射表逐组比对格式串与实参 |
| 3 | 宏所需参数在当前上下文是否可构造 | 检查 entityName/paramName/实际值/期望值等实参来源 |

### 反例

```cpp
// 反例1: InferShape 中裸 OP_LOGE 打印维度校验失败（op_host/xxx_infershape.cpp）
static ge::graphStatus InferShape4Xxx(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    if (xShape->GetDimNum() != 4) {
        OP_LOGE(context, "the shape of x must be 4D, but got %lu dims.", xShape->GetDimNum());   // ❌ 裸 OP_LOGE
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

```cpp
// 反例2: tiling 中裸 OP_LOGE 打印属性取值校验失败（op_host/xxx_tiling.cpp）
static ge::graphStatus XxxTiling(gert::TilingContext* tilingContext)
{
    const int64_t kernelSize = tilingContext->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (kernelSize <= 0) {
        OP_LOGE(tilingContext, "kernel_size must be positive, but got %ld.", kernelSize);   // ❌ 裸 OP_LOGE
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

```cpp
// 反例3: aclnn 实现中裸 OP_LOGE 打印入参数量校验失败（op_api/src/aclnn_xxx.cpp）
if (numInputs != 2) {
    OP_LOGE(opName, "the num of inputs must be 2, but got %ld.", numInputs);   // ❌ 裸 OP_LOGE
    return ACLNN_ERR_PARAM_INVALID;
}
```

### 正例

```cpp
// 正例1: 替换为 OP_LOGE_FOR_INVALID_SHAPEDIM（EZ0011）
// 日志语义 = 参数 shape 维度错误且含期望值；实际维度可由 GetDimNum() 构造
OP_LOGE_FOR_INVALID_SHAPEDIM("Xxx", "x", std::to_string(xShape->GetDimNum()) + "D", "4D");

// 正例2: 替换为 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON（EZ0026）
// 日志语义 = 参数取值错误，约束为描述性（positive），无具体期望值 → 选 _WITH_REASON 变体
OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Xxx", "kernel_size", std::to_string(kernelSize), "it must be positive");

// 正例3: 替换为 OP_LOGE_FOR_INVALID_TENSORNUM（EZ0022）
// incorrectNum 为 int64_t 直接传入，correctNum 为字符串
OP_LOGE_FOR_INVALID_TENSORNUM(opName, "inputs", numInputs, "2");

// 正例4【PASS】: 过程信息，映射表中无对应语义 → 不报
OP_LOGE(opName, "enter XxxTiling, tilingKey = %lu.", tilingKey);
```

### 宏语义映射表（38 个预定义宏）

> 来源：opbase 仓 `include/op_common/log/log.h`（master 分支，commit 245f557d）。
> `entityName` 为算子名（REG_OP 注册名）或 aclnn 接口名（如 `"aclnnXxx"`），infershape/tiling 中通常传字符串字面量。

**A. WITH_ 系列（EZ0001–EZ0007）——校验第 index 个输入/输出时优先使用**

| 宏名称 | EZ码 | 日志语义 | 参数签名 |
|--------|------|---------|---------|
| OP_LOGE_WITH_INVALID_INPUT_SHAPE | EZ0001 | 第 index 个输入 shape 错误，应为 correctShape | entityName, index, incorrectShape, correctShape |
| OP_LOGE_WITH_INVALID_ATTR | EZ0002 | 属性值错误，应为 correctVal | entityName, attrName, incorrectVal, correctVal |
| OP_LOGE_WITH_INVALID_ATTR_SIZE | EZ0003 | 属性 size 错误，应为 correctSize | entityName, attrName, incorrectSize, correctSize |
| OP_LOGE_WITH_INVALID_INPUT | EZ0004 | 参数必填但为空 | entityName, paramName |
| OP_LOGE_WITH_INVALID_INPUT_SHAPESIZE | EZ0005 | 第 index 个输入 shape size（元素总数）错误 | entityName, index, incorrectSize, correctSize |
| OP_LOGE_WITH_INVALID_INPUT_FORMAT | EZ0006 | 输入 format 错误，应为 expectedFormatList | entityName, paramName, dataFormat, expectedFormatList |
| OP_LOGE_WITH_INVALID_INPUT_DTYPE | EZ0007 | 输入 dtype 错误，应为 expectedDtypeList | entityName, paramName, dataDtype, expectedDtypeList |

**B. FOR_ 系列——shape / 维度 / shapeSize（EZ0008–EZ0016）**

| 宏名称 | EZ码 | 日志语义 | 参数签名 |
|--------|------|---------|---------|
| OP_LOGE_FOR_INVALID_SHAPE | EZ0008 | 单参数 shape 错误（含期望值） | entityName, paramName, incorrectShape, correctShape |
| OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON | EZ0009 | 单参数 shape 错误（含原因） | entityName, paramName, incorrectShape, reason |
| OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON | EZ0010 | 多参数 shape 错误（含原因） | entityName, paramNames, incorrectShapes, reason |
| OP_LOGE_FOR_INVALID_SHAPEDIM | EZ0011 | 单参数 shape 维度错误（含期望值） | entityName, paramName, incorrectDim, correctDim |
| OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON | EZ0012 | 单参数维度错误（含原因） | entityName, paramName, incorrectDim, reason |
| OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON | EZ0013 | 多参数维度错误（含原因） | entityName, paramNames, incorrectDims, reason |
| OP_LOGE_FOR_INVALID_SHAPESIZE | EZ0014 | 单参数 shape size 错误（含期望值） | entityName, paramName, incorrectSize, correctSize |
| OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON | EZ0015 | 单参数 shape size 错误（含原因） | entityName, paramName, incorrectSize, reason |
| OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON | EZ0016 | 多参数 shape size 错误（含原因） | entityName, paramNames, incorrectSizes, reason |

**C. FOR_ 系列——format / dtype（EZ0017–EZ0021, EZ0035）**

| 宏名称 | EZ码 | 日志语义 | 参数签名 |
|--------|------|---------|---------|
| OP_LOGE_FOR_INVALID_FORMAT | EZ0017 | 单参数 format 错误（含期望值） | entityName, paramName, incorrectFormat, correctFormat |
| OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON | EZ0018 | 多参数 format 错误（含原因） | entityName, paramNames, incorrectFormats, reason |
| OP_LOGE_FOR_INVALID_DTYPE | EZ0019 | 单参数 dtype 错误（含期望值） | entityName, paramName, incorrectDtype, correctDtype |
| OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | 单参数 dtype 错误（含原因） | entityName, paramName, incorrectDtype, reason |
| OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | 多参数 dtype 错误（含原因） | entityName, paramNames, incorrectDtypes, reason |
| OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON | EZ0035 | 单参数 format 错误（含原因） | entityName, paramName, incorrectFormat, reason |

**D. FOR_ 系列——取值 / 数量 / 长度 / stride / 通用非法（EZ0022–EZ0028, EZ0037, EZ0038）**

| 宏名称 | EZ码 | 日志语义 | 参数签名 |
|--------|------|---------|---------|
| OP_LOGE_FOR_INVALID_TENSORNUM | EZ0022 | 参数 tensor 数量错误（含期望值） | entityName, paramName, incorrectNum(int64_t), correctNum |
| OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON | EZ0023 | 多参数 tensor 数量错误（含原因） | entityName, paramNames, incorrectNums, reason |
| OP_LOGE_FOR_INVALID_VALUE | EZ0024 | 单参数取值错误（含期望值） | entityName, paramName, incorrectValue, correctValue |
| OP_LOGE_FOR_INVALID_LISTSIZE | EZ0025 | 参数列表长度错误（含期望值） | entityName, paramName, incorrectSize, correctSize |
| OP_LOGE_FOR_INVALID_VALUE_WITH_REASON | EZ0026 | 单参数取值错误（含原因） | entityName, paramName, incorrectValue, reason |
| OP_LOGE_FOR_INVALID_VALUES_WITH_REASON | EZ0027 | 多参数取值错误（含原因） | entityName, paramNames, incorrectValues, reason |
| OP_LOGE_FOR_INVALID_STRIDE | EZ0028 | 参数 stride 错误（含期望值） | entityName, paramName, incorrectStride, correctStride |
| OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON | EZ0037 | 参数非法（含原因，无期望值） | entityName, paramName, reason |
| OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON | EZ0038 | 参数列表长度错误（含原因） | entityName, paramName, incorrectSize, reason |

**E. FOR_ 系列——文件 / 配置项 / 图节点（EZ0029–EZ0034, EZ0036）**

| 宏名称 | EZ码 | 日志语义 | 参数签名 |
|--------|------|---------|---------|
| OP_LOGE_FOR_FILE_PATH | EZ0029 | 输入路径非法 | opName, filePath, reason |
| OP_LOGE_FOR_FILE_OPEN | EZ0030 | 文件打开失败 | opName, filePath, reason |
| OP_LOGE_FOR_FILE_PARSE | EZ0031 | 文件解析失败 | opName, fileName, reason |
| OP_LOGE_FOR_INVALID_CONFIG | EZ0032 | 配置项取值错误（含期望值） | opName, configFile, item, value, expect |
| OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON | EZ0033 | 配置项取值错误（含原因） | opName, configFile, item, value, reason |
| OP_LOGE_FOR_INVALID_CONFIGS_WITH_REASON | EZ0034 | 多配置项取值错误（含原因） | opName, configFile, items, values, reason |
| OP_LOGE_FOR_INVALID_GRAPH_NODE | EZ0036 | 输入图含非法节点 | nodeName, reason |

### 专属检视方法

**步骤1 — 定位裸调用**：在 `op_host/` `op_graph/` `op_api/` 范围内检索 `\bOP_LOGE\s*\(`；带后缀宏（`OP_LOGE_WITH_*` / `OP_LOGE_FOR_*` / `OP_LOGE_LIBOPAPI_REPORT` / `OP_LOGE_WITHOUT_REPORT`）跳过；`op_kernel/` 下的命中跳过（Kernel 侧不适用）。

**步骤2 — 语义判定**：逐条提取裸调用的格式串与实参，按映射表比对。常见信号：

| 裸日志中的语义信号 | 候选宏组 |
|-------------------|---------|
| "shape of xx must/should be [具体 shape]" | A 组 SHAPE / B 组 SHAPE/SHAPESIZE |
| "dim of xx / must be 2D/4D"（维度数错误） | B 组 SHAPEDIM 类 |
| "shape size / element nums" | B 组 SHAPESIZE 类 |
| "dtype of xx must be" | A 组 WITH_INVALID_INPUT_DTYPE / C 组 DTYPE 类 |
| "format of xx must be" | A 组 WITH_INVALID_INPUT_FORMAT / C 组 FORMAT 类 |
| "attr xxx must be" | A 组 ATTR 类 / D 组 VALUE 类 |
| "num of inputs/outputs/tensors" | D 组 TENSORNUM 类 |
| "list size / length of list" | D 组 LISTSIZE 类 |
| "stride of xx" | D 组 FOR_INVALID_STRIDE |
| 文件路径无效 / 打开失败 / 解析失败 | E 组 FILE_* |
| 配置项取值非法 | E 组 CONFIG 类 |

变体选择规则：
1. 校验明确第 index 个输入/输出（如 aclnn 入参、`context->GetInputShape(i)`）→ 优先 WITH_ 系列（EZ0001–EZ0007）；不区分输入输出的通用参数 → FOR_ 系列
2. 有明确期望值（"must be 4D" / "should be [1,128]"）→ 基础变体；只有描述性原因（positive / in range 等）→ `_WITH_REASON` 变体
3. 单个参数 → 单数形式；多个参数 → 复数形式（`*_S_WITH_REASON`）

**步骤3 — 参数可得性判定**：
- 语义命中，且宏所需参数（entityName/paramName/实际值/期望值）在当前上下文可构造 → **FAIL(HIGH)**
- 语义命中但个别参数确实不可得（如上下文无 opName 且算子名无法确认）→ **SUSPICIOUS(MED)**，意见中说明缺失参数
- 语义不匹配（过程/调试/性能打点等，映射表无对应场景）→ **PASS**，不报

**步骤4 — 意见输出（FAIL/SUSPICIOUS 强制三要素）**：
1. 应替换的**宏名称**（如 `OP_LOGE_FOR_INVALID_SHAPEDIM`）
2. 宏定义所在的**文件路径**：opbase 仓 `include/op_common/log/log.h`（https://gitcode.com/cann/opbase/blob/master/include/op_common/log/log.h ；若被检视仓依赖路径下可见本地 log.h，优先引用本地路径）
3. 按宏参数签名给出的**替换写法示例**

意见模板：

```
[LOG-1] 独立使用 OP_LOGE 打印参数校验日志，语义可由预定义宏替代：
- 应替换宏: OP_LOGE_FOR_INVALID_SHAPEDIM（EZ0011）
- 宏定义位置: opbase 仓 include/op_common/log/log.h
  （https://gitcode.com/cann/opbase/blob/master/include/op_common/log/log.h）
- 建议写法: OP_LOGE_FOR_INVALID_SHAPEDIM("Xxx", "x", std::to_string(xShape->GetDimNum()) + "D", "4D");
- 原因: 裸 OP_LOGE 上报通用码 EZ9999 且无结构化字段，预定义宏上报专属 EZ 码供错误诊断系统采集
```

### 注意事项

1. 替换的根本动机：裸 `OP_LOGE` 上报通用码 EZ9999、无结构化字段；预定义宏上报专属 EZ 码 + 结构化 msgKey/msgValue
2. 替换后错误消息由宏内部固定模板生成，原裸调用中的自定义格式串删除，只按参数签名传值
3. 映射表来源为 log.h master 分支（commit 245f557d，2026-09-05），宏清单可能随 opbase 演进；检视时若被检视仓内可见实际依赖的 log.h，以其为准核对宏存在性与签名
4. 语义无法映射的裸 `OP_LOGE`（过程/调试信息）按 PASS 处理，不强制改用 `OP_LOGE_WITHOUT_REPORT`
5. 上述宏"仅供算子或 aclnn 实现使用"（log.h 原文注释），本条覆盖的 op_host/op_graph/op_api 代码即属此类

---

## LOG-2: 日志中打印算子名称必须从 TilingContext 获取，禁止自定义算子名字符串 `[适用: Host]`

**严重级别**：中

### 问题描述

Host 侧代码打印日志时需要算子名称的，必须从宿主上下文（TilingContext 等）获取，**禁止在代码中自定义一个算子名称的字符串**。算子名获取的三级层级：

| 层级 | 用法 | 判定 |
|------|------|------|
| 优先 | 日志宏直接传 `context`（如 `OP_LOGE(context, ...)`）——框架自动携带算子标识，无需额外获取 | PASS |
| 合规 | `context->GetNodeName()` 显式获取；tiling 类成员在构造/Init 时从 context 获取后复用（如 `opName_`） | PASS，不算报错 |
| 违规 | 自定义算子名称的字符串——出现在第一参数、格式串前缀、或预定义宏 entityName | FAIL |

> **例外：context 判空分支豁免**。当获取到的 context 为空（代码处于 `context == nullptr` / `!context` 判空保护分支内）时，无 context 可传、也无法调用 `GetNodeName()`——该分支内的日志**允许使用算子名字符串**（建议与注册名一致），不判违规。豁免**仅限判空分支体内**；分支外的日志仍须从 context 获取；若存在 context 之外的算子名来源（如 aclnn 实现的入参 `opName`），仍优先使用该来源。

**危害**：

- 算子改名/代码复制到新算子时，硬编码的算子名与实际算子不符，误导问题定位
- 同一算子多文件日志口径不一（有的手写、有的从 context 获取）
- 调试阶段遗留的假算子名（如 `"xxx-test"`）混入生产日志

### 真实检视案例

案例1 — 第一参数硬编码算子名（ops-math PR#2195）：

> 评论：「第一个参数用了硬编码字符串 "Pow"，其他所有文件统一用的是 context_->GetNodeName()。建议改为 context_->GetNodeName() 保持一致，也避免算子名变更后日志不匹配。」

案例2 — OP_LOGI 调试遗留假名（ops-transformer PR#4089）：

> 评论：「这行 OP_LOGI 用了硬编码的 "alltoallmatmul-test" 作为算子名，看起来是调试阶段留下的日志。生产代码中不应该用 test 后缀的标识符做日志tag，要么删掉这行，要么改成使用实际的 opName 参数。」

案例3 — context 已携带算子标识又手动加前缀（ops-math PR#1767 / #1994）：

> 评论：「context->GetNodeName() 这个已经获取了算子名称，后面打印可以不需要再添加。」
> 作者采纳后回复：「日志输出中不再手动添加算子名前缀，依赖 context->GetNodeName() 或 context 自动携带的算子标识。」

### 反例

```cpp
// 反例1: 第一参数硬编码算子名（ops-math PR#2195 同类）
OP_LOGE("Pow", "tiling failed, powNum is %ld.", powNum);                     // ❌ 自定义算子名字符串

// 反例2: 消息里手写算子名前缀——context 已自动携带算子标识（ops-math PR#1767 同类）
OP_LOGE(context, "AddCustom tiling failed, totalNum is %ld.", totalNum);     // ❌ 手动前缀冗余

// 反例3: 调试遗留假名（ops-transformer PR#4089 同类）
OP_LOGI(context, "alltoallmatmul-test enter, tilingKey = %lu.", tilingKey);  // ❌ 调试遗留

// 反例4: 预定义宏 entityName 传字符串字面量
OP_LOGE_FOR_INVALID_SHAPEDIM("Pow", "x", "3D", "4D");                        // ❌ entityName 硬编码
```

### 正例

```cpp
// 正例1【优先】: 直接传 context，框架自动携带算子标识
OP_LOGE(context, "tiling failed, totalNum is %ld.", totalNum);

// 正例2【合规】: 显式 GetNodeName()，不算报错
OP_LOGE(context->GetNodeName(), "tiling failed, totalNum is %ld.", totalNum);

// 正例3【合规】: 预定义宏 entityName 从 context 获取（宏参数为 const char*，无法直接传 context）
OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", "3D", "4D");

// 正例4【合规】: tiling 类成员初始化时从 context 获取后复用
// Init 中: opName_(context->GetNodeName());
OP_LOGE(opName_, "fail to get platform info");

// 正例5【豁免】: context 判空分支内无 context 可传，允许使用算子名字符串
if (context == nullptr) {
    OP_LOGE("Xxx", "TilingContext is null, cannot get node name.");   // 豁免：判空分支内（建议与注册名一致）
    return ge::GRAPH_FAILED;
}
OP_LOGE(context, "tiling failed, totalNum is %ld.", totalNum);        // 分支外仍须从 context 获取
```

### 专属检视方法

**步骤1 — 定位日志调用**：检索 `\bOP_LOG[A-Z]*\s*\(`（含 `OP_LOGE_WITH_*` / `OP_LOGE_FOR_*` 预定义宏与 `OP_LOGI` / `OP_LOGW` / `OP_LOGD`）。

**步骤2 — 算子名佐证**：从概要/代码中提取本算子注册名——`IMPL_OP(XXX)`、`REG_OP(XXX)`、文件路径 `op_host/{op}_tiling.cpp`、aclnn 接口名。

**步骤3 — 逐调用判定**：

| 检查对象 | 判定 |
|---------|------|
| 第一参数为 `context` / `context->GetNodeName()` / 初始化自 context 的成员变量（Grep 其声明与初始化链） | PASS |
| 第一参数、格式串前缀或实参中存在算子名风格字符串字面量（与注册名/路径吻合的大驼峰、`*-test` 调试遗留） | FAIL |
| 无 context 的静态工具函数中的日志（纯工具函数无宿主上下文） | 豁免，不报 |
| 日志调用位于 context 判空保护分支内（向上追溯所在 `if (context == nullptr)` / `if (!context)` 分支体） | 豁免，不报（无 context 可传；建议字符串与注册名一致） |
| 疑似算子名但无法确认（字面量与注册名不吻合） | SUSPICIOUS |

**步骤4 — 意见输出**：建议改为**直接传 context（优先）**或 `context->GetNodeName()`；理由引用：算子名变更/复制代码到新算子后日志不匹配、多文件口径不一致。

### 注意事项

1. 优先级：直接传 context > `context->GetNodeName()` > 硬编码字符串（本条报错对象仅为硬编码）
2. LOG-1 管宏选择（用哪个宏）、LOG-2 管算子名来源（名字从哪来），两者独立判定——裸 OP_LOGE 且算子名硬编码可同时报两条
3. 预定义宏（`OP_LOGE_WITH_*`/`FOR_*`）的 entityName 参数为 `const char*`，从 `context->GetNodeName()` 获取
4. 融合算子/多算子共用 tiling 基类场景，可通过基类虚函数返回算子名（每个继承类重写自己的实现），禁止各文件各自硬编码
5. context 判空分支豁免的边界：仅限 `context == nullptr` / `!context` 保护分支体内，不得以"context 可能为空"为由在正常路径硬编码；存在 context 之外的算子名来源（如 aclnn 入参 `opName`）时仍优先使用该来源；判空分支内使用的字符串建议与算子注册名一致
