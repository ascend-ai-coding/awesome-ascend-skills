# MatmulBlazeExample

## 基于 blaze 库的普通 MatMul 算子脚手架

本工程是一个 **基于 blaze library 的普通 MatMul 算子的 registry-invoke 脚手架模板**，
基于 Blaze/tensor_api 实现 matmul 的 AIC 侧流水，使用 blaze library 内置的 `BlockEpilogueEmpty`（空后处理）。

本脚手架供 ops-registry-invoke 工作流在开发普通 MatMul 单算子时参考复用。
Tiling 采用简单 mock 策略（硬编码 baseM/N/K），不含真实 SWAT tiling engine，开发者按需替换为真实 tiling。

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend950|√|

## 功能说明

- 算子功能：普通矩阵乘法。

- 计算公式：

$$
C = A \times B
$$

其中 A=(M,K) BF16, B=(K,N) BF16, C=(M,N) BF16。

## 目录结构

```
matmul_blaze_example/
├── op_host/                             # Host 侧代码（自包含，不依赖 op_kernel）
│   ├── matmul_blaze_example_def.cpp              # 算子定义
│   ├── matmul_blaze_example_infershape.cpp       # 形状推导
│   ├── CMakeLists.txt
│   └── arch35/                          # arch35 架构 Tiling
│       ├── matmul_blaze_example_tiling.cpp        # Tiling 实现（mock stub）
│       ├── matmul_blaze_example_tiling_data.h     # TilingData 结构体（Host/Kernel 共享）
│       └── matmul_blaze_example_tiling_key.h      # TilingKey 定义
├── op_kernel/                           # Kernel 侧代码
│   ├── matmul_blaze_example_arch35.cpp   # Kernel 入口（blaze library GemmUniversal）
│   ├── custom_compile_options.ini       # opbuild 编译选项注入
│   ├── CMakeLists.txt
│   └── include/                         # Blaze/tensor_api 库（由 blaze skill 拉取）
│       ├── blaze/                       # [拉取] Blaze 库（从 ops-tensor 仓）
│       └── tensor_api/                  # [拉取] tensor_api 库（从 ops-tensor 仓）
├── op_api/                              # ACLNN 接口
│   ├── aclnn_matmul_blaze_example.h     # L2 API 头文件
│   ├── aclnn_matmul_blaze_example.cpp   # L2 API 实现
│   ├── matmul_blaze_example.h           # L0 API 头文件
│   └── matmul_blaze_example.cpp         # L0 API 实现
├── op_graph/                            # Graph mode proto
│   ├── CMakeLists.txt
│   └── matmul_blaze_example_proto.h
├── examples/                            # 算子调用示例
├── tests/                               # 测试
│   ├── st/                              # 系统测试
│   └── ut/                              # 单元测试
├── CMakeLists.txt                       # 构建配置
├── build.sh                             # 构建脚本
└── README.md                            # 本文档
```

### Blaze 库依赖拉取

编译前需通过 ascendc-blaze-best-practice skill 拉取 blaze/tensor_api 库：

```bash
cp -r ops-tensor/include/blaze op_kernel/include/
cp -r ops-tensor/include/tensor_api op_kernel/include/
```

详见 ascendc-blaze-best-practice skill 的 step1-setup.md。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|----------|------|---------|---------|
| a | 输入 | 左矩阵，shape=(M,K)。 | BF16 | ND |
| b | 输入 | 右矩阵，shape=(K,N)。 | BF16 | ND |
| c | 输出 | 矩阵乘结果，shape=(M,N)。 | BF16 | ND |

## 约束说明

- 仅支持 2D GEMM（无 batch 维度）
- M、N、K 必须 >= 1
- 仅支持 Ascend950 (DAV_3510) 架构
- a 的 K 维必须等于 b 的 K 维
- 当前 Tiling 为 mock 实现（硬编码 baseM/N/K=128, MAX_CORES=48），开发者按需替换为真实 tiling
