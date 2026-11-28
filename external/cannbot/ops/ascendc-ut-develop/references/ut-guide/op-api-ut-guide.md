# OP_API UT 详细指南

## 概述

测试 ACLNN 算子接口的正确性：
- GetWorkspaceSize 返回值
- 输入参数校验（dtype、shape、format、nullptr）
- 输出结果正确性

---

## 目录结构

```
<repo>/<category>/<op>/tests/ut/op_api/
├── CMakeLists.txt
├── test_aclnn_<op>.cpp          # 主接口测试
├── test_aclnn_<op>_out.cpp      # out变体测试（如有）
└── test_aclnn_<op>_inplace.cpp  # inplace变体测试（如有）
```

---

## 核心组件

### TensorDesc

```cpp
// 基本构造
auto tensor = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

// 链式调用
auto tensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
              .ValueRange(-2.0, 2.0)
              .Precision(0.0001, 0.0001);

// 非连续内存
auto tensor = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
```

### ScalarDesc

```cpp
auto alpha = ScalarDesc(1.0f);           // float
auto value = ScalarDesc(42);             // int
auto flag = ScalarDesc(true);            // bool
```

### OP_API_UT 宏

```cpp
// 单输入单输出
auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));

// 多输入带Scalar
auto ut = OP_API_UT(aclnnAdd, INPUT(self, other, alpha), OUTPUT(out));

// nullptr测试
auto ut = OP_API_UT(aclnnAbs, INPUT((aclTensor*)nullptr), OUTPUT(out));
```

---

## 测试用例示例

### 异常用例（最先编写）

```cpp
TEST_F(l2_abs_test, case_anullptr_input) {
    auto out = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAbs, INPUT((aclTensor*)nullptr), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_abs_test, case_invalid_dtype) {
    auto self = TensorDesc({3, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto out = TensorDesc(self);
    auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
```

### 正常用例

```cpp
TEST_F(l2_abs_test, case_abs_for_float_type) {
    auto self = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out = TensorDesc(self).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
```

---

## CMakeLists.txt

```cmake
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
```

---

## 常见返回值

| 返回值 | 说明 |
|-------|------|
| `ACL_SUCCESS` | 成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 参数为空指针 |
| `ACLNN_ERR_PARAM_INVALID` | 参数无效 |

---

## 编译命令

```bash
# 编译 op_api UT
bash build.sh -u --opapi --ops='<op_name>' --soc='<soc_version>'

# 编译并生成覆盖率
bash build.sh -u --opapi --ops='<op_name>' --soc='<soc_version>' --cov
```

---

## 测试用例编写顺序（TDD）

1. **异常用例**：nullptr 测试、无效 dtype 测试、shape 不匹配测试、超过 8 维测试
2. **正常用例**：所有支持的 dtype、所有支持的 format
3. **边界用例**：空 tensor、0 维 tensor

---

## Dtype 排列组合校验

### 概述

从算子定义文件（`xxx_def.cpp`）中提取输入 Tensor 支持的 dtype 列表，自动生成所有合法的 dtype 排列组合测试用例，确保：

- 所有支持的 dtype 组合都被覆盖
- 不支持的 dtype 组合被识别为异常用例
- 提升算子 dtype 分支的覆盖率

### 前置步骤

**1. 定位算子定义文件**

```bash
# 在算子目录下查找 def 文件
find ${op_path}/op_host -name "*_def.cpp"
```

**2. 提取 dtype 信息**

### Dtype 排列组合校验

#### DataType 形式

| 形式 | 语法 | 组合数 |
|------|------|--------|
| 数组 | `.DataType({...})` | 数组长度 |
| 固定 | `.DataTypeList({...})` | 不计入 |

**约束**：数组形式参数长度必须相同。

#### 校验流程

| 步骤 | 操作 | 命令/方法 | 通过条件 |
|------|------|---------|---------|
| 1 | 脚本提取 | `python scripts/extract_dtype_combinations.py ${def_file}` | 组合数 N > 0 |
| 2 | 补充用例 | 按模板补充 N 个 dtype 用例 | UT 文件用例数 M ≥ N |
| 3 | 覆盖率分析 | `lcov --list ops.info_filtered \| grep "dtype" \| grep ":0"` | 无未覆盖分支 |
| 4 | 实际测试 | `bash build.sh -u --opapi --ops=${op_name} --soc=${soc}` | N 个组合全部 PASS |
| 5 | 异常校验 | 构造 3 种非法 dtype 组合 | 返回 ACLNN_ERR_PARAM_INVALID |

**判定**：5 步全部通过 → dtype 校验完整。

**脚本输出**：
- `/tmp/${op_name}_dtype_combinations.json` - dtype 组合数据
- 测试用例模板（屏幕输出）- 直接补充到 UT 文件

#### 异常组合构造

| 异常类型 | 构造方法 | 预期结果 |
|---------|---------|---------|
| 跨位置组合 | query 用 DataType[0]，key 用 DataType[1] | ACLNN_ERR_PARAM_INVALID |
| 未定义 dtype | 使用 def 文件中未出现的 dtype（如 DOUBLE） | ACLNN_ERR_PARAM_INVALID |
| dtype 不一致 | 同一位置不同 Tensor 使用不同 dtype | ACLNN_ERR_PARAM_INVALID |

**示例**：

```cpp
// 跨位置组合
TEST_F(l2_flash_attention_test, case_invalid_cross_position) {
    auto query = TensorDesc({4, 13, 8192}, ACL_FLOAT16, ACL_FORMAT_ND);  // 位置 0
    auto key = TensorDesc({4, 13, 1024}, ACL_BF16, ACL_FORMAT_ND);       // 位置 1
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 未定义 dtype
TEST_F(l2_flash_attention_test, case_invalid_unknown_dtype) {
    auto query = TensorDesc({4, 13, 8192}, ACL_DOUBLE, ACL_FORMAT_ND);  // DOUBLE 未定义
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
```

#### 异常处理

| 异常情况 | 原因 | 处理 |
|---------|------|------|
| lcov 有 `:0` | 代码存在 def 未定义的 dtype 分支 | 分析代码补充用例 |
| 测试有 FAIL | def 定义了不支持的 dtype | 删除组合或修正 def |
| 异常组合未返回错误码 | 算子校验逻辑缺失 | 修复算子代码 |

---

### Dtype 映射表

| ge::DataType | aclDataType | CSV 写法 |
|--------------|-------------|----------|
| `ge::DT_FLOAT` | `ACL_FLOAT` | `FLOAT` |
| `ge::DT_FLOAT16` | `ACL_FLOAT16` | `FLOAT16` |
| `ge::DT_BF16` | `ACL_BF16` | `BF16` |
| `ge::DT_INT8` | `ACL_INT8` | `INT8` |
| `ge::DT_INT32` | `ACL_INT32` | `INT32` |
| `ge::DT_INT64` | `ACL_INT64` | `INT64` |
| `ge::DT_BOOL` | `ACL_BOOL` | `BOOL` |

**补充查找**：若所需 dtype 未在表中列出，可查看 CANN 安装路径下的枚举源文件获取完整定义：
- `ge::DataType` / `AscendC::DataType`：`${ASCEND_HOME_PATH}/asc/include/basic_api/kernel_type.h`
- `aclDataType`：`${ASCEND_HOME_PATH}/include/acl/acl_base_rt.h`

