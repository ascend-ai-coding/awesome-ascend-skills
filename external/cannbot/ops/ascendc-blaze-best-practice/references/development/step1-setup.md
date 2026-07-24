# Step 1: 工程初始化

> **定位**：Agent 拿到算子需求后第一件事做的事——搭建工程目录、拉取依赖、配置构建。

---

## §1 工程目录结构

Blaze matmul 类算子工程采用以下标准目录结构：

```
<your_op>/
├── test_<your_op>.cpp               # Launcher（host 端入口：tiling + 内存管理 + kernel 启动）
├── CMakeLists.txt                   # 构建配置
├── op_kernel/                       # Kernel 层（device 端）
│   ├── <your_op>_kernel.h           # Kernel 模板函数（组装类型链 + 调用组件）
│   ├── <your_op>_kernel.cpp         # Wrapper（extern "C" 包装，供 launcher 调用）
│   └── include/
│       ├── blaze/                   # [拉取] Blaze 库（从 ops-tensor 仓）
│       ├── tensor_api/              # [拉取] tensor_api（从 ops-tensor 仓）
│       └── blaze_custom/            # [可选] Grouped/Fusion/自定义扩展场景才拷贝；MX C+V 需 bridge Kernel/Epilogue
└── op_tiling/                       # Tiling 层（host 端）
    ├── <your_op>_tiling_data.h      # TilingData POD 结构体（host-device 数据交换）
    └── <your_op>_tiling.h           # Tiling 引擎（包含常量 / helper / 平台查询）
```

| 目录 | 职责 | 说明 |
|------|------|------|
| 顶层 | Launcher + 构建 | `test_<your_op>.cpp` 是 host 端入口，`CMakeLists.txt` 配置编译 |
| `op_kernel/` | Kernel 层 | kernel 模板函数 + Blaze 库依赖 + 自定义扩展 |
| `op_kernel/include/blaze/` | Blaze 库 | 从 ops-tensor 仓拉取，提供 BlockMmad/BlockScheduler/Kernel 组件 |
| `op_kernel/include/tensor_api/` | tensor_api | 从 ops-tensor 仓拉取，Blaze 的底层依赖 |
| `op_kernel/include/blaze_custom/` | 自定义扩展 | 仅 Grouped MatMul、C+V 融合或标准 Blaze 组件无法满足需求时使用 |
| `assets/op_tiling/` | Tiling 层 | TilingData 结构体 + Tiling 引擎 |

---

## §2 拉取 blaze 库

ops-tensor 仓由 plugin init.sh 自动克隆到 `<plugin-root>/ops-tensor/`。手动开发时执行：

```bash
cp -r ops-tensor/include/blaze op_kernel/include/
cp -r ops-tensor/include/tensor_api op_kernel/include/
```

编译时需将以下路径同时加入 include path：
- `op_kernel/include/tensor_api/include`
- `op_kernel/include/tensor_api`
- `op_kernel/include/blaze`

---

## §3 可选：拷贝 blaze_custom 扩展模块

普通 MatMul 单算子和纯 MX 量化 MatMul 默认不拷贝 blaze_custom，仅使用 `op_kernel/include/blaze/` 与 `op_kernel/include/tensor_api/`。

当使用 Grouped MatMul、C+V 融合或自定义 Block/Scheduler/Epilogue 扩展时，才从 skill 拷贝自定义模块：

```bash
cp -r <skill-path>/blaze_custom op_kernel/include/
```

blaze_custom 包含 5 个子目录：
- `kernel/`（4 个文件）：MatmulKernel、MatmulKernelFused、GroupMatmulKernel、MatmulKernelMxFused
- `block/`：BlockMmad SWAT 主模板 + Scheduler
- `policy/`（1 个文件）：DispatchPolicy 定义
- `epilogue/`（3 个文件）：RegBase/MemBase Epilogue + CV 同步常量
- `utils/`（6 个文件）：布局工具、常量、通用工具

按需裁剪：仅拷贝当前场景所需的模块。模块能力详见 `references/modules/blaze-custom/` 目录。

**MX C+V 依赖规则**：MXFP8/MXFP4 MatMul + Vector Epilogue 是受控组合态，需要同时具备：

- blaze library 的 MX `BlockMmad` / `BlockSchedulerQuantBatchMatmulV3` 头文件；
- blaze_custom 的 `kernel/matmul_kernel_mx_fused.h`、`epilogue/cv_sync_constants.h` 和自定义 Epilogue；
- `assets/op_tiling/mx/` 下的 `QuantMatmulTilingData` 与 `QuantMatmulTilingSwat`。

**混用规则**：默认禁止 blaze_custom 模块和 blaze 库模块在同一 kernel 入口函数中任意混用。普通 MatMul 单算子和纯 MX 量化 MatMul 均使用 blaze library 全套组件；普通 C+V 和 Grouped C+V 使用 blaze_custom；唯一受控例外是 `Kernel::MxMatmulKernelFused` 桥接 blaze library MX Block/Scheduler 与自定义 Epilogue。详见 `references/development/step2-kernel-design.md` §2。

---

## §4 CMake 骨架

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_matmul LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MATMUL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include/blaze
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include/tensor_api
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include/tensor_api/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set(ASCEND_INCLUDE_DIRS
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/asc
    ${ASCEND_DIR}/asc/include
    ${ASCEND_DIR}/compiler/asc/include
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_DIR}/compiler/tikcpp/include
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_DIR}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_DIR}/compiler/ascendc/impl/aicore/basic_api
)

add_executable(${PROJECT_NAME} test_${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${MATMUL_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${ASCEND_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    $<$<COMPILE_LANGUAGE:ASC>:-w>
    $<$<COMPILE_LANGUAGE:ASC>:-O3>
)
target_link_libraries(${PROJECT_NAME} PRIVATE m dl platform tiling_api register)
```

关键点：
- 语言设置为 `ASC`（非 `CXX`），这是 Ascend C 编译器的要求
- `--npu-arch=dav-3510` 指定目标架构为 Ascend 950
- C++17 标准是必须的
- `ASCEND_INCLUDE_DIRS` 建议从最小可编译集合起步，避免默认复制大而全的 SDK 头路径
- 普通 MatMul 单算子和 MX 量化 MatMul 的 include path 必须包含 `op_kernel/include/blaze` 与 tensor_api 两级路径
- 基础 MatMul 场景下，SDK include 应保持最小可编译集合，避免引入无关头路径
- 基础 MatMul 的 tiling 资产建议使用带 `blaze_` 或 `basic_` 前缀的文件名，避免与 SDK `matmul_tiling*.h` 同名冲突
- 仅 Grouped/Fusion/自定义扩展场景需要额外加入 `op_kernel/include/blaze_custom`；MX C+V 还必须保留 `op_kernel/include/blaze` 和 tensor_api 路径，因为 Block/Scheduler 来自 blaze library

---

**下一步**：→ `references/development/step2-kernel-design.md`（定义 Kernel 入口函数）

**可选参考**：→ `references/fundamentals/blaze-framework-overview.md`（如需理解 NPU 执行模型和 Blaze 架构）
