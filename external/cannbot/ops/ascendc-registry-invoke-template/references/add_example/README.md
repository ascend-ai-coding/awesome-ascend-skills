# AddExample

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
 |Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
 |Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：完成加法计算。

- 计算公式：

$$
y = x1 + x2
$$

## 目录结构

```
add_example/
├── op_host/                    # Host 侧代码
│   ├── add_example_def.cpp     # 算子定义
│   ├── add_example_infershape.cpp  # 形状推导
│   ├── arch22/                 # arch22 架构 Tiling
│   └── arch35/                 # arch35 架构 Tiling
├── op_kernel/                  # Kernel 侧代码
│   ├── arch22/                 # arch22 架构实现
│   └── arch35/                 # arch35 架构实现
├── op_api/                     # ACLNN 接口
│   ├── aclnn_add_example.cpp   # L2 API 实现
│   ├── aclnn_add_example.h     # L2 API 头文件
│   ├── add_example.cpp         # L0 API 实现
│   └── add_example.h           # L0 API 头文件
├── tests/                      # 测试代码
│   ├── ut/                     # 单元测试
│   │   ├── op_host/            # Host 侧 UT
│   │   └── op_api/             # API 侧 UT
│   └── st/                     # 系统测试
├── CMakeLists.txt              # 构建配置
└── build.sh                    # 构建脚本
```

## 快速验证

```bash
# 1. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 编译算子
bash build.sh --soc=ascend910b -j8

# 3. 运行 UT 测试（编译后自动运行）
bash build.sh -u --soc=ascend910b

# 4. 运行 ST 测试（Real 模式，需 NPU；Mock 模式需进入 tests/st 目录运行）
bash build.sh -s --soc=ascend910b
```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | 待进行add_example计算的入参，公式中的x1。 | FLOAT、INT32 | ND |
| x2 | 输入 | 待进行add_example计算的入参，公式中的x2。 | FLOAT、INT32 | ND |
| y | 输出 | 待进行add_example计算的出参，公式中的y。 | FLOAT、INT32 | ND |

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn调用 | [ST 测试 (C++)](./tests/st/) | 参见[ST 测试说明](./tests/st/README.md)。系统测试通过 C++ 原生测试验证算子精度。 |