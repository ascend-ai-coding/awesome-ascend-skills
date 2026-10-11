# add_example 算子 UT 测试

## 概述

单元测试（UT）验证 Tiling、InferShape 和 op_api 逻辑，在 CPU 环境运行，无需 NPU。

## 快速开始

```bash
./run.sh                    # 一键运行所有测试（op_host + op_api）
./run.sh --ophost           # 仅运行 op_host 测试
./run.sh --opapi            # 仅运行 op_api 测试
cd ../../ && bash build.sh --soc=ascend910b -u  # 从根目录运行
```

## 目录结构

```
tests/ut/
├── CMakeLists.txt              # CMake 构建配置
├── run.sh                      # 测试脚本（支持 op_host + op_api）
├── cmake/BuildGoogleTest.cmake # 自动编译 GTest
├── common/                     # 测试辅助代码
├── op_host/                    # op_host 测试用例
│   ├── test_add_example_tiling.cpp
│   └── test_add_example_infershape.cpp
└── op_api/                     # op_api 测试用例
    ├── test_add_example.cpp
    ├── test_aclnn_add_example.cpp
    └── test_utils.h
```

## 测试覆盖

### op_host 测试
- Tiling 逻辑：基础 shape / 多维度 / 边界条件
- InferShape 逻辑：广播规则 / shape 推导

### op_api 测试
- l0op::AddExample：正常场景 / 数据类型 / 边界场景 / 错误场景
- aclnnAddExample：参数校验 / workspace 计算 / dtype 检查

## 添加新测试

参考现有测试代码：
- op_host: `op_host/test_add_example_tiling.cpp`、`op_host/test_add_example_infershape.cpp`
- op_api: `op_api/test_add_example.cpp`、`op_api/test_aclnn_add_example.cpp`

## 常见问题

### 1. 编译失败：ABI 兼容性

**错误**：`undefined reference to ...`

**解决方案**：`./run.sh -c`

### 2. 找不到 Google Test

**解决方案**：确保网络连接（首次需下载 Google Test 1.14.0）

### 3. 环境变量未设置

**解决方案**：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

## 技术要点

**BuildGoogleTest.cmake**：自动编译 Google Test，确保 ABI 兼容性，下载 Google Test 1.14.0

**Context Faker**：模拟 Ascend C 运行时上下文
- `TilingContextFaker`：Tiling 阶段
- `InferShapeContextFaker`：InferShape 阶段

**Case Executor**：封装测试执行逻辑
- `TilingCaseExecutor`：Tiling 测试执行器
- `InferShapeCaseExecutor`：InferShape 测试执行器
