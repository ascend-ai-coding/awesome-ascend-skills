---
name: external-cannbot-ops-ascendc-registry-invoke-template
description: 完整自定义算子工程模板。通过提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考，帮助快速搭建并实现 registry-invoke
  方式的自定义算子工程。当需要创建完整自定义算子工程、参考标准目录结构、补齐 UT/ST、适配多芯片架构或查找工程样例时使用。
original-name: ascendc-registry-invoke-template
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# Ascend C 自定义算子增强

知识库类技能，为算子开发者提供高级开发技术和完整示例参考。

## 核心能力

- **完整示例参考** - 提供完整的、独立自包含的 AddExample 算子实现（ascend910b/ascend910_93/ascend950等多芯片架构支持、UT/ST完整测试用例）
- **高级开发技巧** - Tiling模板编程、多架构隔离
- **开发指南精要** - 从基础到进阶的开发流程和最佳实践

## 使用场景

| 场景 | 推荐资源 |
|------|----------|
| 首次开发算子 | `references/basic-guide.md` + `references/add_example/` |
| 实现复杂算子 | `references/advanced-guide.md` |
| 多架构适配 | `references/advanced-guide.md` → 代际隔离章节 |
| Tiling模板编程 | `references/advanced-guide.md` → Tiling模板编程章节 |
| 编译部署 | `references/build-deploy-guide.md` |
| 编写UT测试 | `references/basic-guide.md` → UT验证章节 |
| 编写ST测试 | `references/st-test-guide.md` + `references/add_example/tests/st/` |
| ACLNN 接口开发 | `references/add_example/op_api/`（代码注释自表达，见下方说明） |
| 算子调用示例 | `references/example-guide.md` + `references/add_example/examples/` |
| 查阅完整代码/手写AscendC | `references/add_example/` |
| 多芯片开发 | `references/npu-arch-adapt.md` |
| SIMT 算子开发 | `references/simt-engineering-guide.md` |

#### ACLNN 接口开发指南

基于 `add_example/op_api/` 创建新算子：

1. **文件命名**：`aclnn_{op}.h/cpp`（L2 API）、`{op}.h/cpp`（L0 API）
2. **L2 API 核心流程**：CREATE_EXECUTOR → CheckParams → Contiguous → l0op::{Op} → ViewCopy → GetWorkspaceSize
3. **L0 API 核心流程**：InferShape → IsAiCoreSupport → AllocTensor → {Op}AiCore
4. **关键修改点**：
   - 数据类型支持列表（根据算子需求调整）
   - 形状推导规则（InferShape）
   - Kernel 名称（ADD_TO_LAUNCHER_LIST_AICORE）

## 资源导航

### references/add_example/

完整的 AddExample 算子实现，包含：

```
add_example/
├── build.sh            # 构建脚本
├── CMakeLists.txt      # CMake配置
├── op_host/            # Host侧实现
│   ├── add_example_def.cpp       # 算子定义
│   ├── add_example_infershape.cpp # Shape推导
│   ├── arch22/                    # Ascend910B Tiling
│   └── arch35/                    # Ascend950 Tiling
├── op_kernel/          # Kernel侧实现
│   ├── add_example_arch22.cpp    # Ascend910B Kernel
│   ├── add_example_arch35.cpp    # Ascend950 Kernel
│   ├── arch22/                    # Ascend910B 实现头文件
│   └── arch35/                    # Ascend950 实现头文件
├── op_api/             # ACLNN接口
│   ├── aclnn_add_example.cpp/h   # L2 API（对外接口）
│   └── add_example.cpp/h         # L0 API（内部实现）
├── op_graph/           # 图模式适配
├── examples/           # 用户调用示例
│   ├── test_aclnn_add_example.cpp  # aclnn两段式调用示例
│   └── test_geir_add_example.cpp   # 图模式(GE IR)调用示例
└── tests/              # UT/ST测试
    ├── ut/             # 单元测试
    │   ├── op_host/    # Host侧UT
    │   └── op_api/     # API侧UT
    └── st/             # 系统测试
```

### references/matmul_blaze_example/

完整的 基于blaze的matmul类 算子实现，包含：

```
matmul_blaze_example/
├── build.sh            # 构建脚本
├── CMakeLists.txt      # CMake配置
├── op_host/            # Host侧实现
│   ├── quant_matmul_gelu_example_def.cpp        # 算子定义
│   ├── quant_matmul_gelu_example_infershape.cpp # Shape推导
│   └── arch35/                                  # Ascend950 Tiling
├── op_kernel/          # Kernel侧实现
│   ├── quant_matmul_gelu_example_arch35.cpp   # Ascend950 Kernel
│   └── arch35/                                # Ascend950 实现头文件
├── op_api/             # ACLNN接口
│   ├── quant_matmul_gelu_example.cpp/h   # L2 API（对外接口）
│   └── quant_matmul_gelu_example.cpp/h   # L0 API（内部实现）
├── op_graph/           # 图模式适配
├── examples/           # 用户调用示例
│   ├── test_aclnn_quant_matmul_gelu_example.cpp  # aclnn两段式调用示例
└── tests/              # UT/ST测试
    ├── ut/             # 单元测试
    │   ├── op_host/    # Host侧UT
    │   └── op_api/     # API侧UT
    └── st/             # 系统测试
```

### references/basic-guide.md

基础开发指南精要：
- 工程创建与目录结构
- 算子定义（OpDef）
- Tiling实现基础
- Kernel实现基础
- 编译部署流程
- UT验证方法

### references/advanced-guide.md

高级开发指南精要：
- 核心概念与术语表
- 算子原型高级配置
- Tiling模板编程
- 多硬件平台差异化注册
- 代际隔离（arch22/arch35）
- 图模式适配
- aclnn接口配置
- 常见问题与约束条件

### references/build-deploy-guide.md

编译部署指南，包含：
- build.sh 脚本使用
- CMakeLists.txt 配置
- 构建产物说明
- 安装/卸载/升级流程
- 常见问题

### references/st-test-guide.md

ST 测试开发指南，包含：
- 目录结构规范
- 完整代码模板（基于 add_example 样例）
- Mock/Real 模式切换
- CPU Golden 实现
- 精度比对函数
- 开发流程与完成标准

### references/npu-arch-adapt.md

多芯片开发指南，原则先行 + 适配清单对照：
- 核心原则（同架构复用、构建系统与代码的分工、适配范围）
- 适配清单（_def.cpp、CMakeLists、build.sh 等）
- 编译配置两种模式（列表声明 / 条件分支）
- 常见陷阱与预防

## 快速参考

### 算子开发流程

```
算子设计 → 算子定义 → Tiling实现 → Kernel实现 → 测试验证 → 编译部署
```

### 关键交付件

| 阶段 | 交付件 | 位置 |
|------|--------|------|
| 算子定义 | `{op}_def.cpp` | `op_host/` |
| Tiling | `{op}_tiling.cpp` | `op_host/arch{32,35}/` |
| TilingData | `{op}_tiling_data.h` | `op_kernel/arch{32,35}/` |
| TilingKey | `{op}_tiling_key.h` | `op_kernel/arch{32,35}/` |
| Kernel | `{op}.cpp` / `{op}_apt.cpp` | `op_kernel/` |
| L0 API | `{op}.cpp` / `{op}.h` | `op_api/` |
| L2 API | `aclnn_{op}.cpp` / `aclnn_{op}.h` | `op_api/` |
| 图模式 | `{op}_proto.h` | `op_graph/` |
| 调用示例 | `test_aclnn_{op}.cpp` / `test_geir_{op}.cpp` | `examples/` |
| UT测试 | `test_{op}_*.cpp` | `tests/ut/` |

### 芯片架构映射
参见skill：`npu-arch`

### 核函数参数顺序（固定）

```c++
__global__ __aicore__ void {op_name}(
    GM_ADDR input1,   // 输入参数
    GM_ADDR input2,
    GM_ADDR output,   // 输出参数
    GM_ADDR workspace, // workspace（固定）
    GM_ADDR tiling     // tiling（固定）
)
```

## 相关技能

- `ascendc-tiling-design` - Tiling 设计指南
- `ascendc-api-best-practices` - API使用最佳实践
- `npu-arch` - NPU架构知识
