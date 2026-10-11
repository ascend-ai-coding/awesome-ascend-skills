# 算子调用示例开发指南

本文档提供算子调用示例（examples）的开发指南，说明 aclnn 调用示例和图模式调用示例的代码结构和开发要点。

## 1. 概述

算子调用示例是**用户面交付件**，放在 `examples/` 目录下，供用户参考如何调用算子。与 ST 测试不同，调用示例不包含精度比对和 Golden 计算，仅演示调用流程并打印结果。

```
{op_name}/examples/
├── test_aclnn_{op_name}.cpp     # aclnn 两段式 API 调用示例
└── test_geir_{op_name}.cpp      # 图模式 (GE IR) 调用示例
```

**与 ST 测试的区别**：

| 维度 | examples/ 调用示例 | tests/st/ ST 测试 |
|------|-------------------|------------------|
| 目的 | 展示调用方式，给用户参考 | 验证算子精度 |
| 精度比对 | 无 | 有 Golden + CompareResults |
| 测试用例 | 固定一组简单数据 | 多组用例覆盖各场景 |
| Mock 模式 | 不支持 | 支持 Mock/Real 切换 |
| 文档链接 | README "调用说明" 章节引用 | 不在 README 中引用 |

## 2. aclnn 调用示例

### 2.1 代码结构

```
test_aclnn_{op_name}.cpp
├── 宏定义 (CHECK_RET, LOG_PRINT)
├── GetShapeSize()           # 计算 shape 元素总数
├── PrintOutResult()         # 打印输出结果
├── Init()                   # ACL 初始化（设备、流）
├── CreateAclTensor()        # 创建 aclTensor（模板函数）
└── main()
    ├── 1. Init (device/stream)
    ├── 2. 构造输入输出 Tensor
    ├── 3. aclnn{OpName}GetWorkspaceSize()  # 第一段接口
    ├── 4. 申请 workspace
    ├── 5. aclnn{OpName}()                  # 第二段接口
    ├── 6. aclrtSynchronizeStream()
    ├── 7. PrintOutResult()
    ├── 8. 释放 aclTensor
    ├── 9. 释放 device 资源
    └── 10. aclFinalize()
```

### 2.2 替换要点

| 原值 | 替换为 | 说明 |
|------|--------|------|
| `add_example` | `{op_name}` | 文件名、注释中的算子名 |
| `aclnnAddExample` | `aclnn{OpName}` | aclnn API 函数名 |
| `aclnnAddExampleGetWorkspaceSize` | `aclnn{OpName}GetWorkspaceSize` | 第一段接口 |
| `selfX`, `selfY` | 根据算子输入参数名调整 | 输入变量名 |
| `selfXShape = {32, 4, 4, 4}` | 根据算子实际 shape 调整 | 输入数据 shape |
| `ACL_FLOAT` | 根据算子支持的数据类型调整 | 数据类型 |
| `aclnn_add_example.h` | `aclnn_{op_name}.h` | include 头文件 |

### 2.3 必须修改的部分

1. **输入输出构造**：根据算子接口定义调整输入数量、类型、shape
2. **aclnn 接口调用**：替换为实际的 `aclnn{OpName}GetWorkspaceSize` 和 `aclnn{OpName}`
3. **PrintOutResult**：根据输出含义调整打印格式
4. **include 头文件**：替换为实际的 aclnn 头文件

## 3. 图模式 (GE IR) 调用示例

### 3.1 代码结构

```
test_geir_{op_name}.cpp
├── 宏定义 (ADD_INPUT, ADD_CONST_INPUT, ADD_OUTPUT)
├── 辅助函数
│   ├── GetTime()                # 时间戳
│   ├── GetDataTypeSize()        # 数据类型字节数
│   ├── GenOnesData()            # 生成测试数据
│   └── WriteDataToFile()        # 写入 bin 文件
├── CreateOppInGraph()           # 创建单算子图
│   ├── op::{OpName}("op1")     # 实例化算子
│   ├── ADD_INPUT(...)           # 添加输入
│   └── ADD_OUTPUT(...)          # 添加输出
└── main()
    ├── 1. GEInitialize()
    ├── 2. CreateOppInGraph()     # 构图
    ├── 3. graph.SetInputs/SetOutputs
    ├── 4. Session 创建
    ├── 5. session->AddGraph()
    ├── 6. session->RunGraph()   # 执行
    ├── 7. 输出结果写入 bin
    └── 8. GEFinalize()
```

### 3.2 替换要点

| 原值 | 替换为 | 说明 |
|------|--------|------|
| `AddExample` | `{OpName}` | 算子类名 (PascalCase) |
| `add_example` | `{op_name}` | 文件名、变量名 |
| `experiment_ops.h` | 根据算子注册头文件调整 | 算子注册头文件 |
| `add_example_proto.h` | `{op_name}_proto.h` | 图模式 proto 头文件 |
| `set_input_x1`, `set_input_x2` | 根据算子输入参数名调整 | 输入设置方法 |
| `update_output_desc_y` | 根据算子输出参数名调整 | 输出设置方法 |
| `xShape = {32, 4, 4, 4}` | 根据算子实际 shape 调整 | 输入数据 shape |
| `DT_FLOAT` | 根据算子支持的数据类型调整 | 数据类型 |

### 3.3 必须修改的部分

1. **CreateOppInGraph()**：替换算子实例化、输入输出定义
2. **include 头文件**：替换 proto 头文件和算子注册头文件
3. **输入/输出宏调用**：根据算子接口调整 ADD_INPUT / ADD_OUTPUT 的参数

## 4. 编译和验证

examples 目录提供独立的 `run.sh` 脚本，同时也集成到根目录 `build.sh` 的 `-e/--example` 选项中。

### 4.1 通过 run.sh 独立运行

```bash
# 前置：编译并安装算子包
bash build.sh --soc=ascend910b -j8
./build/custom_opp_ubuntu_aarch64.run

# 运行 aclnn 调用示例（默认）
cd examples && ./run.sh

# 运行图模式调用示例
cd examples && ./run.sh --graph

# 清理构建目录
cd examples && ./run.sh --clean
```

### 4.2 通过 build.sh 集成运行

```bash
# 编译 + 安装 + 运行 aclnn 调用示例
bash build.sh -e --soc=ascend910b

# 编译 + 安装 + 运行图模式调用示例
bash build.sh -e --graph --soc=ascend910b
```

> `build.sh -e` 会自动安装算子包，然后调用 `examples/run.sh`。

### 4.3 预期输出

**aclnn 示例**：

```
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
...
```

**图模式示例**：

```
INFO - [XIR]: Session run ir compute graph success
INFO - [XIR]: Precision is ok
INFO - [XIR]: Finalize ir graph session success
```

### 4.4 编译构建配置说明

examples 目录文件结构：

| 文件 | 用途 | 说明 |
|------|------|------|
| `run.sh` | 统一执行入口 | 支持 `--eager`（默认）/ `--graph` / `--clean` |
| `CMakeLists.txt` | aclnn 调用示例编译 | 依赖 libascendcl + libnnopbase + libcust_opapi |
| `CMakeLists_geir.txt` | 图模式调用示例编译 | 依赖 libgraph + libge_runner + libgraph_base |

**aclnn 编译依赖**：
- 算子包已安装到 `${ASCEND_HOME_PATH}/opp/vendors/`
- 自动查找 vendors 下的自定义算子包头文件和库

**图模式编译依赖**：
- CANN toolkit 已安装
- GE 编译器头文件和 stub 库位于 `${ASCEND_HOME_PATH}/compiler/`

## 5. 参考资源

- **完整示例**：`references/add_example/examples/`
- **aclnn 接口说明**：`references/add_example/op_api/`
- **图模式适配**：`references/advanced-guide.md` → 图模式适配章节
- **算子调用方式**：`ops-math/docs/zh/invocation/op_invocation.md`（CMakeLists.txt 模板来源）
