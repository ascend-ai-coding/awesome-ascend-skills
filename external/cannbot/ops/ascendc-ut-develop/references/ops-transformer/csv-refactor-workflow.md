# 重构流程详解

本文档提供完整的 Host UT CSV 格式重构步骤。

---

## 第一步：分析算子定义文件

从 `{算子目录}/op_host/{算子名}_def.cpp` 提取以下信息：

### 1.1 提取输入 Tensor 列表

```cpp
this->Input("query")
    .ParamType(REQUIRED)  // REQUIRED/DYNAMIC/OPTIONAL
    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
    .FormatList({ge::FORMAT_ND});
```

**记录每个输入 Tensor：**

- Tensor 名称
- ParamType（REQUIRED/DYNAMIC/OPTIONAL）
- 支持的 DataType 列表
- 支持的 Format 列表

### 1.2 提取输出 Tensor 列表

```cpp
this->Output("attention_out")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
    .FormatList({ge::FORMAT_ND});
```

**记录每个输出 Tensor：**

- Tensor 名称
- ParamType
- DataType 列表
- Format 列表

### 1.3 提取属性列表

```cpp
this->Attr("num_heads").AttrType(REQUIRED).Int(1);
this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
this->Attr("input_layout").AttrType(OPTIONAL).String("BSND");
this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Bool(false);
```

**记录每个属性：**

- 属性名称
- AttrType（REQUIRED/OPTIONAL）
- 类型（Int/Float/String/Bool）
- 默认值（如果有）

---

## 第二步：创建参数结构体

创建 `{算子目录}/tests/ut/op_host/{算子名}_param.h`。

### 2.0 理解基类结构

框架提供了 `HostUtParamBase` 基类（定义在 `op_host_csv_case_loader.h`），包含以下通用字段：

```cpp
struct HostUtParamBase {
    std::string case_name;              // 测试用例名称
    ge::graphStatus expectResult;       // 期望执行结果
    std::vector<uint32_t> inputInstance;  // 输入 Tensor 实例标记
    std::vector<uint32_t> outputInstance; // 输出 Tensor 实例标记
  
    HostUtParamBase(const csv_map& csvMap) {
        this->case_name = ReadMap(csvMap, "case_name");
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};
```

**关键点：**

- 不要重复定义 `case_name`、`expectResult`、`inputInstance`、`outputInstance`
- 算子的参数结构继承 `HostUtParamBase`
- `operator<<` 函数已在基类定义，无需重复

### 2.1 算子基础参数结构设计

```cpp
// 继承 HostUtParamBase，添加算子特定的属性字段
struct {{OpName}}HostUtParamBase : public HostUtParamBase {
    // ========== 算子属性（从 xxx_def.cpp 提取）==========
    int64_t num_heads = 0;
    float scale = 1.0f;
    std::string input_layout = "BSND";
    bool softmax_lse_flag = false;
    // ... 添加所有属性
  
    {{OpName}}HostUtParamBase(const csv_map& csvMap) : HostUtParamBase(csvMap) {
        // 读取属性
        num_heads = std::stoll(ReadMap(csvMap, "num_heads"));
        scale = std::stof(ReadMap(csvMap, "scale"));
        input_layout = ReadMap(csvMap, "input_layout");
        softmax_lse_flag = std::stoi(ReadMap(csvMap, "softmax_lse_flag"));
        // ... 添加所有属性读取
    }
};
```

### 2.2 Tiling 测试参数结构

```cpp
struct {{OpName}}TilingUtParam: public {{OpName}}HostUtParamBase {
    // ========== 输入 Tensor ==========
    std::vector<int> inputInstance;
    gert::TilingContextPara::TensorDescription query = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value = TD_DEFAULT;
    // ... 添加所有输入 Tensor
  
    // ========== 输出 Tensor ==========
    std::vector<int> outputInstance;
    gert::TilingContextPara::TensorDescription attention_out = TD_DEFAULT;
    // ... 添加所有输出 Tensor
  
    // ========== Tiling 期望结果 ==========
    uint64_t expectTilingKey = 0;
    std::string expectTilingDataHash;
  
    {{OpName}}TilingUtParam(const csv_map& csvMap): 
        {{OpName}}HostUtParamBase(csvMap) {
  
        // 读取输入 Tensor
        inputInstance.emplace_back(GetTensorGE(csvMap, 
            "query_shape", "query_dtype", "query_format", this->query));
        inputInstance.emplace_back(GetTensorGE(csvMap, 
            "key_shape", "key_dtype", "key_format", this->key));
        // ... 添加所有输入 Tensor
  
        // 读取输出 Tensor
        outputInstance.emplace_back(GetTensorGE(csvMap, 
            "attention_out_shape", "attention_out_dtype", "attention_out_format", 
            this->attention_out));
        // ... 添加所有输出 Tensor
  
        // 读取期望结果（可选）
        expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey"));
        expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
    }
};
```

### 2.3 InferShape 测试参数结构

```cpp
struct {{OpName}}InferShapeUtParam: public {{OpName}}HostUtParamBase {
    std::vector<int> inputInstance;
    std::vector<int> outputInstance;
  
    // ========== 输入 Tensor ==========
    gert::InfershapeContextPara::TensorDescription query = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key = ID_DEFAULT;
    // ... 添加所有输入 Tensor
  
    // ========== 输出 Tensor ==========
    gert::InfershapeContextPara::TensorDescription attention_out = ID_DEFAULT;
    // ... 添加所有输出 Tensor
  
    {{OpName}}InferShapeUtParam(const csv_map& csvMap): 
        {{OpName}}HostUtParamBase(csvMap) {
  
        // 同 Tiling 参数结构，使用 GetTensorGE 读取
        // ...
    }
};
```

### 2.4 InferDataType 测试参数结构

```cpp
struct {{OpName}}InferDTypeUtParam: public {{OpName}}HostUtParamBase {
    std::vector<int> inputInstance;
    std::vector<int> outputInstance;
  
    // ========== 输入 Tensor 数据类型 ==========
    ge::DataType query = ge::DT_UNDEFINED;
    ge::DataType key = ge::DT_UNDEFINED;
    // ... 添加所有输入 Tensor
  
    // ========== 输出 Tensor 数据类型 ==========
    ge::DataType attention_out = ge::DT_UNDEFINED;
    // ... 添加所有输出 Tensor
  
    {{OpName}}InferDTypeUtParam(const csv_map& csvMap): 
        {{OpName}}HostUtParamBase(csvMap) {
  
        // 使用 GetDataTypeGE 读取数据类型
        inputInstance.emplace_back(GetDataTypeGE(csvMap, "query_dtype", this->query));
        inputInstance.emplace_back(GetDataTypeGE(csvMap, "key_dtype", this->key));
        // ... 添加所有输入 Tensor
  
        outputInstance.emplace_back(GetDataTypeGE(csvMap, "attention_out_dtype", this->attention_out));
        // ... 添加所有输出 Tensor
    }
};
```

---

## 第三步：创建 CSV 测试用例文件

按照 [csv-format-spec.md](csv-format-spec.md) 的规范创建 CSV 文件。

### 3.1 设计列名

基于提取的算子定义，列名顺序建议：

```
通用字段 → 输入 Tensor → 输出 Tensor → 属性 → 期望结果 → 注释
```

### 3.2 填写测试用例

**注意，仅填写第一条预期成功的用例，迁移所有用例放在后续步骤**

根据测试场景填写参数：

- 成功场景：完整填写所有 REQUIRED Tensor 和属性
- Optional Tensor：根据 [csv-format-spec.md](csv-format-spec.md#tensor-列填写规则) 处理
- 失败场景：expectResult=FAILED，不填写期望输出

---

## 第四步：重构测试 C++ 文件

### 4.0 文件命名规范

**必须严格按照以下规范命名文件**：

| 测试类型      | 文件名格式                         | 示例                                                      |
| ------------- | ---------------------------------- | --------------------------------------------------------- |
| Tiling        | `test_{op}_tiling.cpp`           | `test_fused_infer_attention_score_tiling.cpp`           |
| InferShape    | `test_{op}_shape_infershape.cpp` | `test_fused_infer_attention_score_shape_infershape.cpp` |
| InferDataType | `test_{op}_dtype_infershape.cpp` | `test_fused_infer_attention_score_dtype_infershape.cpp` |

**重要提示**：

- InferShape 文件名必须包含 `_shape_` 关键字
- InferDataType 文件名必须包含 `_dtype_` 关键字
- CSV 文件名与 CPP 文件名相同，只是扩展名不同

### 4.1 通用结构

```cpp
#include <gtest/gtest.h>
#include "{算子名}_param.h"
#include "op_host_csv_case_loader.h"
// 其他必要头文件

namespace {{OpName}}UT {

class {{OpName}}Test : public testing::TestWithParam<{{OpName}}UtParam> {
protected:
    static void SetUpTestCase() { }
    static void TearDownTestCase() { }
};

TEST_P({{OpName}}Test, param) {
    auto param = GetParam();
    // 构建测试上下文
    // 执行测试
    // 验证结果
}

INSTANTIATE_TEST_SUITE_P(
    {{OpName}},
    {{OpName}}Test,
    testing::ValuesIn(GetCasesFromCsv<{{OpName}}UtParam>(
        ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<{{OpName}}UtParam>
);

} // namespace {{OpName}}UT
```

**头文件包含顺序（非常重要！）**

`<gtest/gtest.h>` **必须第一个包含**，在所有其他头文件之前。原因：

- `{算子名}_param.h` 包含了 `op_host_csv_case_loader.h`
- `op_host_csv_case_loader.h` 包含了 `csv_case_load_utils.h`
- `csv_case_load_utils.h` 中的 `PrintCaseInfoString()` 函数使用了 `testing::TestParamInfo` 类型
- 如果 `<gtest/gtest.h>` 没有先被包含，编译时会报错：`error: 'testing' does not name a type`

**错误示例**：
```cpp
// 错误！会导致编译失败
#include "xxx_param.h"  // 包含了 op_host_csv_case_loader.h
#include <gtest/gtest.h> // 太晚了！
```

**正确示例**：
```cpp
// 正确
#include <gtest/gtest.h>  // 必须第一个！
#include "xxx_param.h"
```

### 4.2 Tiling 测试实现

参考 [assets/csv-refactor/template_tiling.cpp](../../assets/csv-refactor/template_tiling.cpp)

关键点：

- 使用 `gert::TilingContextFaker` 构建上下文
- 添加 SocInfo（核数、UB大小等）
- 验证 tiling key 和 tiling data hash（可选）

**头文件相对路径注意事项**

测试文件位于 `tests/ut/op_host/arch{32,35}/` 目录下，需要正确计算到 `op_host/` 目录的相对路径：

| 测试文件位置 | 相对路径级数 | 示例 |
|-------------|-------------|------|
| `tests/ut/op_host/` | 1 级 | `#include "../../op_host/xxx_compile_info.h"` |
| `tests/ut/op_host/arch32/` | 3 级 | `#include "../../../op_host/xxx_compile_info.h"` |
| `tests/ut/op_host/arch35/` | 4 级 | `#include "../../../../op_host/xxx_compile_info.h"` |

**常见错误**：路径级数计算错误导致找不到头文件。建议使用 `find` 命令确认正确路径：
```bash
# 从测试文件目录计算到 op_host 目录的相对路径
find . -name "xxx_compile_info.h" -type f
```

### 4.3 InferShape 测试实现

参考 [assets/csv-refactor/template_infershape.cpp](../../assets/csv-refactor/template_infershape.cpp)

关键点：

- 使用 `gert::InfershapeContextPara` 构建上下文
- 通过 `inputInstance/outputInstance` 标记哪些 Tensor 存在
- 验证输出 Tensor 的 shape

### 4.4 InferDataType 测试实现

参考 [assets/csv-refactor/template_inferdtype.cpp](../../assets/csv-refactor/template_inferdtype.cpp)

关键点：

- 使用 `gert::InferDataTypeContextFaker` 构建上下文
- 只处理 dtype，不涉及 shape/format
- 验证输出 Tensor 的数据类型

---

## 第五步：编译验证第一条用例

**⚠️ 必须验证"编译 + 运行"！CSV 格式错误只能在运行时发现！**

```bash
# 推荐：一步完成编译 + 运行
bash build.sh --ophost_test --ops=${op_name} --soc=${soc_version} -j8

# 或者分步执行
bash build.sh --ophost_test --noexec --ops=${op_name} --soc=${soc_version} -j8
./build/tests/ut/framework_normal/op_host/transformer_op_host_ut --gtest_filter="*${OpName}*"
```

**验证检查项**：
- [ ] 编译通过（语法正确）
- [ ] 运行通过（CSV 格式正确）
- [ ] 第一条用例显示 PASS

**常见运行时错误**：
- `Row data does not match CSV header columns` - CSV 列数与表头不匹配
- `Failed to parse CSV` - CSV 格式错误（如空行、注释行）
- `case_name not found` - 必需字段缺失

**不允许和后面的编译验证步骤合并，写完第一条用例就马上编译+运行验证**

---

## 第六步：迁移所有用例

将**所有**旧用例迁移至新框架

注意：

- 新增用例时，**不得更改已迁移用例和 CSV 表头**
- 每 5 条一组，**确保编译运行通过后再迁移下一组**
- **所有旧用例都要完整迁移**，迁移中出现了框架不适配的情况，及时向用户汇报

## 自动化工具

对于复杂算子（输入 Tensor > 10），可以使用自动化脚本：

```bash
python scripts/generate_csv_template.py {算子名}_def.cpp
```

脚本输出：

- CSV 列名行
- 参数结构体框架代码

需要手动补充完整逻辑。

---

## 测试用例设计指南

### 基础测试用例模板

#### 成功场景测试

| 测试类型                | 测试要点                 | 示例                         |
| ----------------------- | ------------------------ | ---------------------------- |
| **最小参数集**    | 仅使用 REQUIRED Tensor   | `basic_required`           |
| **所有 Optional** | 启用所有 Optional Tensor | `all_optional`             |
| **dtype 组合**    | 测试支持的每种 dtype     | `fp16_test`, `bf16_test` |
| **format 组合**   | 测试支持的每种 format    | `nd_test`, `nchw_test`   |
| **典型 shape**    | 常见的实际使用 shape     | `bsnd_4_13_16_512`         |

#### 边界场景测试

| 测试类型               | 测试要点               | 示例                           |
| ---------------------- | ---------------------- | ------------------------------ |
| **空 Tensor**    | Optional Tensor 不使用 | `optional_empty`             |
| **最小 shape**   | 最小的有效 shape       | `min_shape_1_1_1`            |
| **大 shape**     | 大规模的 shape         | `large_shape_1024_1024_1024` |
| **非对齐 shape** | 不满足对齐要求的 shape | `non_aligned_7_13_511`       |

#### 失败场景测试

| 测试类型               | 测试要点             | 示例                 |
| ---------------------- | -------------------- | -------------------- |
| **参数缺失**     | 缺少 REQUIRED 参数   | `missing_required` |
| **类型不匹配**   | dtype 不在支持列表中 | `dtype_mismatch`   |
| **shape 不合法** | shape 不满足约束     | `invalid_shape`    |
| **属性值非法**   | 属性值不在合法范围   | `invalid_attr`     |

### 测试覆盖率要求

建议的最低覆盖率：

| 覆盖类型                 | 要求                         | 说明                       |
| ------------------------ | ---------------------------- | -------------------------- |
| **dtype 覆盖**     | 100%                         | 每种支持的 dtype 都要测试  |
| **format 覆盖**    | 80%+                         | 主要 format 要测试         |
| **ParamType 覆盖** | REQUIRED 100%, Optional 50%+ | Optional Tensor 的主要场景 |
| **边界场景**       | 3-5 个                       | 最小、最大、非对齐等       |
| **失败场景**       | 5-10 个                      | 主要失败原因               |

---

## 常见编译错误

### 错误 1：'testing' does not name a type

**错误信息**：
```
csv_case_load_utils.h:98:46: error: 'testing' does not name a type
   98 | inline std::string PrintCaseInfoString(const testing::TestParamInfo<T>& info)
```

**原因**：`<gtest/gtest.h>` 没有在 `{算子名}_param.h` 之前包含。

**解决方案**：确保测试文件中 `<gtest/gtest.h>` 是第一个包含的头文件：
```cpp
// 正确顺序
#include <gtest/gtest.h>  // 必须第一个！
#include "xxx_param.h"
```

### 错误 2：找不到头文件

**错误信息**：
```
fatal error: xxx_compile_info.h: No such file or directory
```

**原因**：头文件相对路径计算错误。

**解决方案**：
1. 使用 `find` 命令确认正确路径
2. 参考本文档 4.2 节的相对路径级数表

```bash
# 从测试文件目录确认头文件位置
find . -name "xxx_compile_info.h" -type f
```
