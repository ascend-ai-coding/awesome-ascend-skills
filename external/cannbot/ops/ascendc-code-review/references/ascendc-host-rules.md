# Ascend C Host 侧规则

```
<适用>
语言: C++
侧别: Host
默认启用: true

领域: true
触发: 必须触发
</适用>
```


## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| HOST-1 | InferDataType 注册与实现必须放置在 op_graph 目录 | 工程布局 | 高 | [适用: Host] |
| HOST-2 | REG_OP 注册的算子必须实现 InferDataType | 算子注册 | 高 | [适用: Host] |
| HOST-3 | 获取的 platform 参数必须校验取值合法性 | 参数校验 | 高 | [适用: Host] |
| HOST-4 | platform 参数须在 Tiling 方法中直接获取，禁止 TilingParse 经 compileInfo 传递 | 参数获取 | 高 | [适用: Host] |

---

## HOST-1: InferDataType 注册与实现必须放置在 op_graph 目录

**严重级别**：高

### 问题描述

算子的 `InferDataType` 实现函数与注册语句必须位于 `op_graph/` 目录下，规范形态为 `op_graph/xxx_graph_infer.cpp` 中定义实现函数并用 `IMPL_OP(OpName).InferDataType(...)` 注册。

禁止放置在 `op_host/` 目录下。典型违规形态：`op_host/xxx_infershape.cpp` 中通过 `IMPL_OP_INFERSHAPE(OpName).InferShape(...).InferDataType(...)` 链式调用注册 InferDataType。

InferDataType 属于图推断（infer）职责，与 InferShape 的注册位置解耦：InferShape 放在 `op_host/xxx_infershape.cpp` 是合法的，但 InferDataType 不允许跟随 InferShape 留在 op_host。

**存量代码同等判定**：op_host 目录下已存在的 `IMPL_OP_INFERSHAPE(...)...InferDataType(...)` 链式写法（历史遗留）同样判 FAIL，不做 SUSPICIOUS 降级。

### 错误示例

```
pooling/adaptive_avg_pool2d/op_host/adaptive_avg_pool2d_infershape.cpp
pooling/max_pool_grad_with_argmax/op_host/max_pool_grad_with_argmax_infershape.cpp
pooling/max_pool_with_argmax/op_host/max_pool_with_argmax_infershape.cpp
pooling/adaptive_max_pool3d/op_host/adaptiva_max_pool3d_infershape.cpp
pooling/max_pool_grad_with_argmax_v3/op_host/max_pool_grad_with_argmax_v3_infershape.cpp
```

```cpp
// 反例: InferDataType 实现与注册放在 op_host 的 infershape 文件中（adaptive_avg_pool2d_infershape.cpp）
static ge::graphStatus InferDtype4AdaptiveAvgPool2d(gert::InferDataTypeContext* context)
{
    const auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AdaptiveAvgPool2d)
    .InferShape(InferShape4AdaptiveAvgPool2d)
    .InferDataType(InferDtype4AdaptiveAvgPool2d);
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//  InferDataType 在 op_host 目录通过链式调用注册 ❌（存量代码同样判 FAIL）
```

### 正确示例

```
activation/celu_v2/op_graph/celu_v2_graph_infer.cpp
```

```cpp
// 正例: 独立 infer 文件，实现与注册均在 op_graph 目录（celu_v2_graph_infer.cpp）
static ge::graphStatus InferDataTypeCeluV2(gert::InferDataTypeContext* context)
{
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);
    return GRAPH_SUCCESS;
}
IMPL_OP(CeluV2).InferDataType(InferDataTypeCeluV2);
// 实现函数 + IMPL_OP 注册都在 op_graph/xxx_graph_infer.cpp ✅
```

### 注意事项

1. 本条与 HOST-2 的分工：目录内完全无 InferDataType 实现 → 报 HOST-2；有实现但位置在 op_host → 报本条 HOST-1。二者互斥，不重复报
2. `op_api/` 目录的 aclnn 代码不涉及 InferDataType 注册，命中本条触发词但无注册语句时 PASS
3. 检视意见中须注明迁移范围：仅 InferDataType，防止误删 InferShape

### 专属检视方法

**步骤1 — 定位**：在被检视算子目录下 grep `InferDataType`，收集实现函数定义（函数名含 InferDataType / InferDtype 且签名为 `gert::InferDataTypeContext*`）与注册语句（`IMPL_OP(...).InferDataType(...)` 或 `IMPL_OP_INFERSHAPE(...)...InferDataType(...)`）所在的文件路径。

**步骤2 — 目录判定**：
- 实现与注册均在 `op_graph/*.cpp` → PASS
- 实现或注册任一位于 `op_host/*.cpp` → FAIL(HIGH)。链式写法 `IMPL_OP_INFERSHAPE(OpName).InferShape(...).InferDataType(...)` 出现在 op_host 即 FAIL，**存量代码不做降级**

**步骤3 — 修复建议输出**（迁移仅限 InferDataType，InferShape 不动）：
1. 新增 `op_graph/xxx_graph_infer.cpp`：将 InferDataType 实现函数迁移至此，并用 `IMPL_OP(OpName).InferDataType(fn)` 注册
2. `op_host/xxx_infershape.cpp`：仅删除链式调用中的 `.InferDataType(fn)` 一节，**InferShape 的实现与 `IMPL_OP_INFERSHAPE(OpName).InferShape(fn)` 注册原地保留**（InferShape 放 op_host 合法，不受本条约束）

---

## HOST-2: REG_OP 注册的算子必须实现 InferDataType

**严重级别**：高

### 问题描述

当算子通过 `REG_OP(OP_NAME)` 在 `op_graph/xxx_proto.h` 中注册后，必须实现 `InferDataType` 函数。
- **实现位置**：`op_graph/xxx_graph_infer.cpp`（独立 infer 文件，使用 `IMPL_OP(OpName).InferDataType(...)` 注册）。**位置约束见本文件 HOST-1：InferDataType 禁止放置在 op_host 目录，op_host 链式写法属 HOST-1 违规**

缺少 `InferDataType` 会导致框架无法正确推导输出 dtype，引发运行时 dtype 不匹配错误。

### 错误示例

> 以下算子在 `op_graph/*_proto.h` 中有 `REG_OP`，但**整个算子目录下（含 `op_host/` 和 `op_graph/`）均无 `InferDataType` 实现**。

```
pooling/avg_pool_v2/op_host/avg_pool_v2_infershape.cpp      ← 仅 .InferShape()，无 .InferDataType()
pooling/avg_pool/op_host/avg_pool_infershape.cpp             ← 仅 .InferShape()，无 .InferDataType()
pooling/avg_pool3_d/op_host/avg_pool3_d_infershape.cpp       ← 仅 .InferShape()，无 .InferDataType()
pooling/avg_pool3_d_grad/op_host/avg_pool3_d_grad_infershape.cpp  ← 仅 .InferShape()，无 .InferDataType()
```

```cpp
// 反例1: 仅有 InferShape，整个算子目录无 InferDataType（avg_pool_v2_infershape.cpp）
// op_graph/avg_pool_v2_proto.h: REG_OP(AvgPoolV2) ✅ 已注册
// 检查 range: 整个 pooling/avg_pool_v2/ 目录 — 无任何 InferDataType ❌
IMPL_OP_INFERSHAPE(AvgPoolV2).InferShape(InferShapeForAvgPoolV2);
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                            只有 InferShape，缺少 .InferDataType(...) ❌

// 反例2: 仅有 InferShape，缺失 InferDataType（avg_pool_infershape.cpp）
// op_graph/avg_pool_proto.h: REG_OP(AvgPool) ✅ 已注册
// 检查 range: 整个 pooling/avg_pool/ 目录 — 无任何 InferDataType ❌
IMPL_OP_INFERSHAPE(AvgPool).InferShape(InferShapeForAvgPool);
//                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                         缺少 .InferDataType(...) ❌

// 反例3: 仅有 InferShape，缺失 InferDataType（avg_pool3_d_infershape.cpp）
// op_graph/avg_pool3_d_proto.h: REG_OP(AvgPool3D) ✅ 已注册
// 检查 range: 整个 pooling/avg_pool3_d/ 目录 — 无任何 InferDataType ❌
IMPL_OP_INFERSHAPE(AvgPool3D).InferShape(InferShapeForAvgPool3D);
//                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                           缺少 .InferDataType(...) ❌
```

### 正确示例

**唯一合规位置**（`op_graph/xxx_graph_infer.cpp`）：

```
activation/celu_v2/op_graph/celu_v2_graph_infer.cpp
```

```cpp
// 正例1: 独立 infer 文件，IMPL_OP 模式（celu_v2_graph_infer.cpp）
static ge::graphStatus InferDataTypeCeluV2(gert::InferDataTypeContext* context)
{
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);
    return GRAPH_SUCCESS;
}
IMPL_OP(CeluV2).InferDataType(InferDataTypeCeluV2);
//      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 有 InferDataType ✅
```

### 注意事项

1. 本条与 HOST-1 的分工：目录内完全无 InferDataType 实现 → 报本条 HOST-2；有实现但位置在 op_host → 报 HOST-1。二者互斥，不重复报
2. 检查范围为整个算子目录（含 `op_host/` 与 `op_graph/`），不只看 infershape.cpp——无任何 InferDataType 匹配即为违规
3. 历史上曾将 `op_host/xxx_infershape.cpp` 中的 `IMPL_OP_INFERSHAPE(...)...InferDataType(...)` 链式写法认可为兼容位置（如 pooling/adaptive_avg_pool2d 等 5 个文件），现已禁止，位置约束见本文件 HOST-1

### 专属检视方法

```bash
# 1. 查找所有 REG_OP 注册
grep -rn "REG_OP(" --include='*proto*.h' op_graph/ 2>/dev/null

# 2. 对每个已注册的算子，搜索整个算子目录下是否有 InferDataType
grep -rn "InferDataType" <op_dir>/ --include='*.cpp' --include='*.h' 2>/dev/null | grep -v test

# 3. 无任何匹配即为违规（需搜索算子目录下的所有文件，不只看 infershape.cpp）
```

**逐步检视**：
1. 在 `op_graph/*_proto.h` 中找到 `REG_OP(OpName)`
2. 在 `op_graph/xxx_graph_infer.cpp` 中检查是否定义了 `InferDataType` 函数并用 `IMPL_OP(OpName).InferDataType(...)` 注册
3. 在 `op_host/xxx_infershape.cpp` 中发现 `IMPL_OP_INFERSHAPE(OpName)` 链式调用 `.InferDataType(...)` 的，位置违规转判 HOST-1（见本文件），本条不重复报
4. 仅 `.InferShape(...)` 无 `.InferDataType(...)` 则判为违规

---

## HOST-3: 获取的 platform 参数必须校验取值合法性

**严重级别**：高

### 问题描述

Host 侧（tiling/InferShape 等）获取的 platform 参数——coreNum/aicNum/aivNum、ubSize、blockSize 等——来自运行时环境查询，存在非法取值场景（查询失败返回 0、非目标平台、框架异常）。未校验直接使用将导致除零（如 `totalLength / coreNum`）、切分异常、buffer 计算错误等运行时故障。

**标准拦截写法**（来自真实修复范例 ops-nn PR#3963）：

```cpp
OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be > 0, got %ld", coreNum), return ge::GRAPH_FAILED);
```

### 错误示例

```cpp
// 反例1: coreNum 未校验直接作除数（ops-math PR#93 同类）
auto coreNum = platformInfo.GetCoreNumAiv();
int64_t perCoreNum = totalLength / coreNum;        // ❌ coreNum 可能为 0，除零

// 反例2: 无符号类型使用错误判断条件（ops-math PR#347）
uint64_t ubSize = platformInfo.GetUBSize();
OP_CHECK_IF(ubSize <= 0, ...);                     // ❌ uint64_t 恒非负，<= 0 语义错误，应为 == 0

// 反例3: GetPlatformInfo 返回未判空（ops-math PR#319 同类）
auto platformInfo = context->GetPlatformInfo();    // ❌ 返回指针需判空
auto coreNum = platformInfo->GetCoreNumAiv();
```

### 正确示例

```cpp
// 正例: 判空 + 按类型语义校验（修复范例结构，ops-nn PR#3963）
auto platformInfo = context->GetPlatformInfo();
OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"),
            return ge::GRAPH_FAILED);

auto coreNum = platformInfo->GetCoreNumAiv();      // int64_t（有符号）
OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be > 0, got %ld", coreNum),
            return ge::GRAPH_FAILED);

uint64_t ubSize = platformInfo->GetUBSize();       // uint64_t（无符号）
OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize must be > 0"),
            return ge::GRAPH_FAILED);
```

### 注意事项

1. **类型匹配是本条要点**：有符号（int64_t）用 `<= 0`；无符号（uint64_t/uint32_t）用 `== 0`——对无符号写 `<= 0` 本身即判违规（语义错误）
2. 编译期常量（constexpr）不适用本条
3. 校验位置：Host/tiling 侧 `OP_CHECK_IF` 拦截并返回 `ge::GRAPH_FAILED`；不放 kernel 侧 ASSERT（ops-transformer PR#4377：「不要在kernel侧ASSERT，可以在tiling侧加corenum为0情况的拦截」）
4. 与 methodology 变量溯源的衔接：概要/溯源表中「硬件配置 → 硬件保证，构造阶段已校验」仅适用于框架基类流程已含校验的场景；代码独立获取（直接调 `GetCoreNumAiv()` 等）须自行校验，否则报本条
5. **关联防御提示（不计 FAIL，从严建议）**：platform 值参与无符号减法时防下溢（如 `ubSize - reservedBytes` 前校验 `ubSize > reservedBytes`，ops-math PR#2663/#2422、ops-nn PR#5364）；派生核数建议 `std::min(needCoreNum, coreNum)` 上限钳制（ops-math PR#2652/#2365）——发现缺失时在意见中附带提示，不单独判违规

### 专属检视方法

**步骤1 — 定位获取点**：`grep -nE "GetPlatformInfo|GetCoreNumAic|GetCoreNumAiv|GetUbBlockSize|GetUBSize" {file}`，收集每个 platform 参数获取位置及其后的首次使用点。

**步骤2 — 逐获取点核验**：
- 获取点与首次使用之间是否存在对应校验（判空 / `<= 0` / `== 0`，按类型匹配）
- 校验条件与变量类型匹配：有符号 `<= 0`；无符号 `== 0`——对无符号写 `<= 0` 本身即违规

逐对象判定依据：

| 检查对象 | 要求 | 判断依据 |
|---------|------|---------|
| `GetPlatformInfo()` 返回值 | 指针判空 | 返回指针解引用前必须非空（ops-math PR#319、ops-nn PR#431） |
| coreNum / aicNum / aivNum（有符号类型） | `<= 0` 拦截并返回失败 | 0 或负值导致除零、空切分（ops-math PR#93/#337/#399、ops-nn PR#3104） |
| ubSize / blockSize（无符号类型） | `== 0` 拦截并返回失败 | **类型匹配**：无符号类型恒非负，`<= 0` 语义错误（ops-math PR#347） |

**步骤3 — 豁免排除**：
- 编译期常量（`constexpr`，如 `BLOCK_SIZE = 32`）无需判 0（ops-nn PR#380）
- 依赖的框架基类流程中已含同项校验（如 tiling_base 基类 `GetPlatformInfo()` 已校验 coreNum==0，ops-nn PR#3347）——Grep 基类源码确认后可作负向证据，不报

**步骤4 — 范围过滤与意见输出**：PR 模式仅报 diff 新增/修改的获取点（存量未动代码按 methodology 范围外规则不进正文）；FAIL + confidence HIGH，fix_suggestion 给出按类型匹配的标准拦截写法。

---

## HOST-4: platform 参数须在 Tiling 方法中直接获取，禁止 TilingParse 经 compileInfo 传递

**严重级别**：高

### 问题描述

获取 platform 参数（coreNum/aicNum/aivNum、ubSize、blockSize 等）必须**在 Tiling 方法中通过 `context->GetPlatformInfo()` 直接获取**，禁止在 TilingParse 方法中获取后经 compileInfo 传递给 Tiling 方法。

- TilingParse 在**编译期**执行，Tiling 在**执行期**执行；platform 参数是运行时环境属性，编译期经 compileInfo 固化后，编译/执行环境不一致时（如 UT 未 mock、异构部署）取到过期值——ops-transformer PR#3712 中「原本通过 platform 获取 aivcore，改为 compileInfo 传递后 UT 行为漂移」即此差异的实证
- 传递链路引入赋值时序脆弱性：ops-math PR#3086 / #3151 中 `compileInfo->ubSize` 在赋值前被错误消息引用，显示 0 而非实际值——compileInfo 传递 platform 参数引发的实际缺陷
- 传递链路引入类型隐患：ops-math PR#2276 / #2951 中 CompileInfo 结构体 coreNum/ubSize 字段的类型不一致与格式化错误
- platform 参数可随时从 context 查询，经 compileInfo 序列化传递徒增体积与链路负担

### 错误示例

```cpp
// ❌ TilingParse（编译期）获取 platform 参数，经 compileInfo 传递给 Tiling
struct XxxCompileInfo {
    uint32_t coreNum;      // ❌ platform 参数进入 CompileInfo 结构
    uint64_t ubSize;       // ❌
};

ge::graphStatus TilingParse(gert::TilingParseContext* context)
{
    auto platformInfo = context->GetPlatformInfo();      // ❌ 编译期获取
    compileInfo->coreNum = platformInfo.GetCoreNumAiv();
    compileInfo->ubSize = platformInfo.GetUBSize();
    ...
}

ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto coreNum = compileInfo->coreNum;                 // ❌ 执行期读取固化值
    ...
}
```

### 正确示例

```cpp
// ✅ Tiling 方法中直接获取，并按 HOST-3 校验
ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"),
                return ge::GRAPH_FAILED);

    auto coreNum = platformInfo->GetCoreNumAiv();        // ✅ 执行期直接获取
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be > 0, got %ld", coreNum),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = platformInfo->GetUBSize();
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize must be > 0"),
                return ge::GRAPH_FAILED);
    ...
}
```

### 注意事项

1. 与 TOPK-4（ascendc-topk.md）的边界：TOPK-4 管**算子属性**禁止 CompileInfo 传递（`GetAttrPointer` 等）；本条管 **platform 参数**的获取位置。对象不同，独立判定，同现可双报
2. 与 HOST-3 联动：Tiling 中直接获取后，须按 HOST-3 校验取值合法性（判空 + 类型匹配零值拦截）
3. compileInfo 本身是合法机制（传递编译期确定的配置信息）——本条只禁 platform 参数（运行时可从 context 查询的环境参数）走此通道
4. 本条为团队规范指令（规则 owner 明确提供），真实案例（时序缺陷/类型隐患/UT 语义漂移）作为该模式实际危害的佐证

### 专属检视方法

**步骤1 — 定位**：`grep -n "TilingParse" {file}` 定位 TilingParse 方法体；`grep -nE "coreNum|aicNum|aivNum|ubSize|blockSize" {file}` 收集 platform 参数的使用点。

**步骤2 — 传递路径判定（任一端命中即 FAIL）**：
- **写端**：TilingParse 方法体内调用 `GetPlatformInfo`/`GetCoreNumAic`/`GetCoreNumAiv` 等获取 platform 参数并写入 compileInfo → FAIL
- **读端**：Tiling 方法内从 compileInfo 读取 platform 参数字段（`compileInfo->coreNum` 等）使用 → FAIL
- **辅助线索（不计 FAIL）**：CompileInfo 结构体/序列化字段中含 platform 参数名（coreNum/ubSize/blockSize 等）时，深入检查读写两端——结构字段本身不报，避免历史遗留死字段误报

**步骤3 — 豁免排除**：compileInfo 中的合法编译期配置字段（tilingKey 派生的编译期常量、socVersion 静态信息等非运行时可查询的环境参数）不报。

**步骤4 — 范围过滤与意见输出**：PR 模式仅报 diff 新增/修改的传递路径（存量未动代码按 methodology 范围外规则不进正文）；FAIL + confidence HIGH，fix_suggestion 给出「Tiling 内直接获取 + HOST-3 校验」的完整改写。
