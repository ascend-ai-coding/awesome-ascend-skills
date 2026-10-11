# OP_HOST UT 详细指南

## 概述

OP_HOST UT 包含两种测试类型：
- **Tiling测试**：验证Tiling策略、TilingKey、TilingData、Workspace
- **InferShape测试**：验证输出Shape和Dtype推导

---

## 前置准备

### 1. 检测 CompileInfo 类型

不同算子的 CompileInfo 类型和命名空间不同，编写 Tiling 测试前必须检测类型。

**检测方法**：

```bash
# 在 tiling 实现文件中查找 TilingParse<TYPE>
find op_host -name "*_tiling*.cpp" -exec grep -l "TilingParse" {} \;
grep -n "TilingParse" op_host/<arch>/<op>_tiling_<arch>.cpp
```

**典型代码行**：
```cpp
IMPL_OP_OPTILING(Mul).TilingParse<BroadcastCompileInfo>(context, compileInfo);
```

**类型对照表**：

| TYPE | 命名空间 | 头文件 | 示例算子 |
|------|---------|--------|---------|
| `BroadcastCompileInfo` | `Ops::Base::` | `atvoss/broadcast/broadcast_tiling.h` | Mul, Div, Sub |
| `<Op>CompileInfo` | `optiling::` | 算子tiling头文件 | Add, Abs |

**使用示例**：

```cpp
// BroadcastCompileInfo（广播类算子）
#include "atvoss/broadcast/broadcast_tiling.h"
Ops::Base::BroadcastCompileInfo compileInfo = {64, 245760};

// 自定义 CompileInfo（如 AddCompileInfo）
#include "../../../../op_host/arch35/add_tiling_arch35.h"
optiling::AddCompileInfo compileInfo = {64, 245760};
```

**常见错误**：

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `'BroadcastCompileInfo' is not a member of 'optiling'` | 命名空间错误 | 使用 `Ops::Base::` |
| `no matching function for call to 'TilingParse<...>'` | 类型不匹配 | 检测正确的 TYPE |

---

### 2. 确认 NodeAttrs 配置

> **这是 op_host UT 编写中最常见的错误来源。大部分"测试运行返回 GRAPH_FAILED"问题都与此相关。**

#### 问题表现
- 测试编译通过，但运行失败
- `infer_shape_func()` 返回 `4294967295` (GRAPH_FAILED)
- `tiling_func()` 返回 `GRAPH_FAILED`

#### 根本原因
大部分算子的 InferShape/Tiling 函数会调用 `context->GetAttr()` 读取算子属性。如果测试中未配置 `NodeAttrs`，这些函数会因为读取不到必需属性而返回失败。

#### 解决方案

**InferShape 测试必须包含 NodeAttrs**：

```cpp
auto holder = gert::InferShapeContextFaker()
                  .NodeIoNum(2, 1)
                  .IrInstanceNum({1, 1})
                  .InputShapes({&x1_shape, &x2_shape})
                  .OutputShapes({&output_shape})
                  .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                  .Build();
```

**Tiling 测试需要 NodeAttrs + NodeInputTd**：

```cpp
auto holder = gert::TilingContextFaker()
                  .SetOpType("BatchMatMulV3")
                  .NodeIoNum(2, 1)
                  .IrInstanceNum({1, 1})
                  .InputShapes({&x1_shape, &x2_shape})
                  .OutputShapes(output_shapes_ref)
                  .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                              {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                  .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                  .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                  .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                  .Build();
```

**如何确定需要哪些属性**：

1. 查看算子源码中的 `INFER_SHAPE_FUNC` 或 `TILING_FUNC` 宏
2. 搜索 `context->GetAttr()` 或 `GetAttrValue()` 调用
3. 参考同算子仓中类似算子的测试文件

**常见算子属性对照表**：

| 算子类型 | 常见属性 | 命名空间 |
|---------|---------|---------|
| MatMul系列 | adj_x1, adj_x2, offset_x, opImplMode | Ops::NN |
| Reduce系列 | axes, keepdims | Ops::Math |
| Activation系列 | 通常无特殊属性 | - |
| Conv系列 | pads, strides, dilations, groups | Ops::NN |

---

## 目录结构

```
<repo>/<category>/<op>/tests/ut/op_host/
├── CMakeLists.txt
├── test_<op>_infershape.cpp
├── test_<op>_tiling.cpp
└── arch35/                     # 芯片架构目录（如需要）
    └── test_<op>_tiling.cpp
```

---

## Tiling 测试

### 核心组件

```cpp
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

// 1. CompileInfo（必须先检测类型）
optiling::AddCompileInfo compileInfo = {64, 245760};

// 2. TilingContextPara
gert::TilingContextPara tilingContextPara("Add",
    { {{{1, 64}, {1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },  // 输入
    { {{{1, 64}, {1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },  // 输出
    &compileInfo);

// 3. ExecuteTestCase
ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
```

### 测试用例示例

```cpp
// 成功场景
TEST_F(AddTilingTest, test_tiling_fp16_001) {
    optiling::AddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Add",
        { {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },
        { {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 102, "...", {16777216});
}

// 失败场景
TEST_F(AddTilingTest, test_tiling_failed_dtype) {
    optiling::AddCompileInfo compileInfo = {64, 245760};
    gert::TilingContextPara tilingContextPara("Add",
        { {{{1, 64}, {1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND} },  // 不支持的dtype
        { {{{1, 64}, {1, 64}}, ge::DT_DOUBLE, ge::FORMAT_ND} },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0, "", {});
}
```

---

## InferShape 测试

### 核心组件

```cpp
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

// InferShapeContextPara
gert::InfershapeContextPara infershapeContextPara("Abs",
    { {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND} },  // 输入
    { {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND} });               // 输出（空shape待推导）

// ExecuteTestCase
std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
```

### 测试用例示例

```cpp
// 成功场景
TEST_F(AbsInferShapeTest, test_infershape_fp16) {
    gert::InfershapeContextPara infershapeContextPara("Abs",
        { {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND} });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 失败场景：不支持的dtype
TEST_F(AbsInferShapeTest, test_infershape_failed_dtype) {
    gert::InfershapeContextPara infershapeContextPara("Abs",
        { {{{4, 3, 4}, {4, 3, 4}}, ge::DT_DOUBLE, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_DOUBLE, ge::FORMAT_ND} });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
```

---

## 编译命令

```bash
# 编译 op_host UT
bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>'

# 编译并生成覆盖率
bash build.sh -u --ophost --ops='<op_name>' --soc='<soc_version>' --cov
```

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 测试运行返回 GRAPH_FAILED | 未配置 NodeAttrs | 查看源码中的 GetAttr 调用，添加必需属性 |
| CompileInfo 类型错误 | 命名空间不匹配 | 检测正确的 TYPE 和命名空间 |
| 编译找不到头文件 | include 路径不正确 | 使用相对路径或检查 CMakeLists.txt |