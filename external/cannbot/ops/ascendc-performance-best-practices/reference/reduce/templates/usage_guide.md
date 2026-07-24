# 模板使用指南

本文件说明如何将 `templates/` 下的 SoftmaxV2 Kernel 模板集成到实际算子项目中。

## 0. 模板文件格式与使用约定

`templates/` 下的所有源码模板均以 `.template` 为后缀（如 `softmax_v2_base.template`）。这些是**参考模板**，不是可直接编译的源码。

**使用时必须将 `.template` 重命名/复制为对应的 `.h`（或 `.cpp`）文件**，否则最终工程里的 `#include` 引用的是 `.h`，代码不易读也无法编译。转换规则很简单，去掉 `.template` 后缀并按文件头 `\file` 注释中标注的名字落地：

| 模板文件 | 转换后落地文件 |
|----------|---------------|
| `dav310/softmax_v2_base.template` | `dav310/softmax_v2_base.h` |
| `dav310/softmax_v2_ar_small_r.template` | `dav310/softmax_v2_ar_small_r.h` |
| `dav310/softmax_v2_ar_full_load.template` | `dav310/softmax_v2_ar_full_load.h` |
| `dav310/softmax_v2_ar_recompute.template` | `dav310/softmax_v2_ar_recompute.h` |
| `dav310/softmax_v2_ara_full_load.template` | `dav310/softmax_v2_ara_full_load.h` |
| `dav310/softmax_v2_ara_recompute.template` | `dav310/softmax_v2_ara_recompute.h` |
| `dav310/softmax_v2_ara_online.template` | `dav310/softmax_v2_ara_online.h` |
| `dav310/kernel_utils.template` | `dav310/kernel_utils.h` |
| `softmax_v2_tiling_data.template` | `softmax_v2_tiling_data.h` |
| `softmax_v2_tiling.template` | `softmax_v2_tiling.h` |

转换命令示例（保留目录结构）：

```bash
# 在 templates/ 目录下，把所有 .template 复制成同名 .h 到工程目录
find . -name '*.template' | while read f; do
  cp "$f" "${f%.template}.h"
done
```

> **架构目录说明**：Kernel 模板位于 `dav310/` 子目录（对应目标架构 `dav-3510`/Ascend950），`kernel_utils.template` 也在 `dav310/` 下，与其它 Kernel 模板同级。

## 1. 文件清单

### Kernel 文件（7 个，均在 `dav310/` 下）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `dav310/softmax_v2_base.template` | `softmax_v2_base.h` | 公共基类 `SoftmaxV2OpsBase`：CastTrait、VF 工具、CopyIn/Out、NlastReduceSum、UpdateCache 等 |
| `dav310/softmax_v2_ar_small_r.template` | `softmax_v2_ar_small_r.h` | AR-SmallR：`SoftmaxV2ArSmallR<Tx,Ty>`，R 极小转置向量化 |
| `dav310/softmax_v2_ar_full_load.template` | `softmax_v2_ar_full_load.h` | AR-FullLoad：`SoftmaxV2AR<T_in,T_out>`，R 全量载入 UB |
| `dav310/softmax_v2_ar_recompute.template` | `softmax_v2_ar_recompute.h` | AR-Recompute：`SoftmaxV2ArRecompute<Tx,Ty>`，R 超 UB 三阶段重读 |
| `dav310/softmax_v2_ara_full_load.template` | `softmax_v2_ara_full_load.h` | ARA-FullLoad：`SoftmaxV2ARA<T1,T2>`，3D R 全量载入 |
| `dav310/softmax_v2_ara_recompute.template` | `softmax_v2_ara_recompute.h` | ARA-Recompute：`SoftmaxV2ARARecompute<T1,T2>`，3D R 超 UB |
| `dav310/softmax_v2_ara_online.template` | `softmax_v2_ara_online.h` | ARA-Online：`SoftmaxV2ARAOnline<T1,T2>`，3D 在线 max/sum |

### 公共依赖（3 个）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `softmax_v2_tiling_data.template` | `softmax_v2_tiling_data.h` | 六套 TilingData 结构体定义（packed），host/device 共用 |
| `softmax_v2_tiling.template` | `softmax_v2_tiling.h` | 六个独立 Host tiling 计算函数 |
| `dav310/kernel_utils.template` | `dav310/kernel_utils.h` | Device 侧工具模板（CeilDiv、CeilAlign、FloorDiv、Aligned 等） |

## 2. 六类可独立选择的模板

每个模板可独立选择、独立包含和独立实例化，不使用 TilingKey、templateCode 或统一模板分发器。

| 模板 | 类名 | TilingData | Tiling 函数 | 构造方式 |
|------|------|-----------|------------|---------|
| AR-SmallR | `SoftmaxV2ArSmallR` | `SoftmaxV2ArSmallRTilingData` | `TilingArSmallR` | AR 系：构造传 `TPipe*` |
| AR-FullLoad | `SoftmaxV2AR` | `SoftmaxV2ARTilingData` | `TilingArFullLoad` | AR 系：构造传 `TPipe*` |
| AR-Recompute | `SoftmaxV2ArRecompute` | `SoftmaxV2ArRecomputeTilingData` | `TilingArRecompute` | AR 系：构造传 `TPipe*` |
| ARA-FullLoad | `SoftmaxV2ARA` | `SoftmaxV2ARATilingData` | `TilingAraFullLoad` | ARA 系：构造传 TilingData* |
| ARA-Recompute | `SoftmaxV2ARARecompute` | `SoftmaxV2ARARecomputeTilingData` | `TilingAraRecompute` | ARA 系：构造传 TilingData* |
| ARA-Online | `SoftmaxV2ARAOnline` | `SoftmaxV2ARAOnlineTilingData` | `TilingAraOnline` | ARA 系：构造传 TilingData* |

## 3. 每个模板的最小依赖集合

> 下面的依赖关系均以**转换后的 `.h`** 描述。集成前请先按第 0 节把 `.template` 落地为 `.h`。

```
具体模板（如 dav310/softmax_v2_ar_small_r.h）
├── dav310/softmax_v2_base.h（按实际依赖）
│   └── dav310/kernel_utils.h（同目录）
├── softmax_v2_tiling_data.h
└── CANN SDK headers（kernel_tiling/kernel_tiling.h, kernel_operator.h, op_kernel/platform_util.h）
```

使用者不需要包含其他不相关模板。每个模板显式选择，不依赖模板注册表或自动路由。

## 4. 如何显式选择一个模板

在 kernel 入口文件中 `#include` 需要的模板头文件和 TilingData 头文件：

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ar_full_load.h"  // 只包含需要的模板
```

## 5. 如何构造对应的 TilingData

在 Host 侧使用 `softmax_v2_tiling.h` 中的对应函数：

```cpp
#include "softmax_v2_tiling.h"

softmax_tiling::PlatformParam plat;
// 运行时查询平台参数
plat.ubSize = static_cast<int64_t>(ubSize);
plat.numBlocks = static_cast<int64_t>(coreNumAiv);

softmax_tiling::CaseShape shape;
shape.a1 = a1; shape.r = r; shape.a0 = a0; shape.dtypeCode = 0; // 0=FP32, 1=FP16, 2=BF16

SoftmaxV2ARTilingData td;
int64_t blockDim = softmax_tiling::TilingArFullLoad(shape, plat, td);
```

## 6. 如何实例化类并调用 Process

### AR 系模板（构造传 `TPipe*`，Init 传 TilingData 指针）

```cpp
TPipe pipe;
SoftmaxV2AR<float, float> op(&pipe);       // 构造传 TPipe*
op.Init(x, y, &tiling);                     // Init 传 TilingData*
op.Process();
```

### ARA 系模板（构造传 TilingData 指针，Init 传 `TPipe*`）

```cpp
TPipe pipe;
SoftmaxV2ARA<float, float> op(&tiling);     // 构造传 TilingData*
op.Init(x, y, &pipe);                        // Init 传 TPipe*
op.Process();
```

> **注意**：AR 和 ARA 系的构造/Init 参数顺序不同，不能混用。

## 7. FP32、FP16、BF16 实例化方法

```cpp
// FP32
SoftmaxV2AR<float, float> op_fp32(&pipe);
// FP16
SoftmaxV2AR<half, half> op_fp16(&pipe);
// BF16
SoftmaxV2AR<bfloat16_t, bfloat16_t> op_bf16(&pipe);
```

输入输出 dtype 必须一致（`Tx == Ty`）。内部计算始终在 FP32 进行。

## 8. MicroAPI、`__vector__`、CANN SDK include 和目标架构要求

- **MicroAPI**：所有模板使用 `AscendC::MicroAPI::RegTensor`、`MaskReg` 等 VF 类型，需 `__VEC_SCOPE__` 作用域。
- **`__vector__`**：Kernel 入口需标注 `__global__ __aicore__ __vector__`，纯 AIV/Vector 核。
- **CANN SDK include**：
  - `kernel_tiling/kernel_tiling.h` — TilingData 基础设施
  - `kernel_operator.h` — AscendC 高层 API 和 MicroAPI
  - `op_kernel/platform_util.h` — 平台工具
- **目标架构**：`dav-3510`（Ascend950），VRegSize=256B，VL_FP32=64。

## 9. 构建方式

### 方式一：ASC 语言单 .asc 文件（host + kernel 同一翻译单元）

参考源工程 `softmax.asc`，host main 与 kernel 入口写在同一个 `.asc` 文件中，一趟 ASC 编译，`<<<blockDim, nullptr, stream>>>` 直调。

关键 CMakeLists.txt 配置：

```cmake
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "")
find_package(ASC REQUIRED)
project(softmax LANGUAGES ASC CXX)

add_executable(softmax softmax.asc)
target_include_directories(softmax PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)
target_compile_options(softmax PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:-dc>
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_link_options(softmax PRIVATE
    --npu-arch=${CMAKE_ASC_ARCHITECTURES}
)
```

### 方式二：`ascendc_library(kernellaunch)` 框架

```cmake
set(ASCEND_KERNEL_LAUNCH_ONLY ON)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(ascendc_kernels_npu STATIC op_kernel/softmax_kernel.cpp)
ascendc_include_directories(ascendc_kernels_npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
```

Host 侧通过 `ACLRT_LAUNCH_KERNEL` 调用。

## 10. 构建注意事项

### `-dc` 和 device link `--npu-arch`

dav310 头文件使用 `AscendC::MicroAPI::RegTensor` 等 reg_compute 类型，仅在 device arch 下声明。单 .asc 编译时需：

1. **`-dc`**：生成可重定位 device 代码。
2. **`--npu-arch=dav-3510`**：device 编译和链接阶段均需显式指定架构，否则默认落到 `dav-m100` 导致 device 符号无法解析。
3. **`__ASC_NPU_HOST__` 守卫**：host pass 需打开 `__ASC_NPU_HOST__` 使 reg_compute 类型在 host pass 可见；device pass 不打开，避免污染向量 intrinsic 类型映射。源码内用 `#ifndef __NPU_ARCH__` / `#define __ASC_NPU_HOST__` 守卫实现按 pass 限定。

### VF fusion

`--cce-simd-vf-fusion` 选项控制 VF 融合模式（`true`/`false`），按需开启。

### 两种方式的选择

- **单 .asc**：适合快速验证和直调场景，host+kernel 同一翻译单元。
- **`ascendc_library`**：适合正式算子工程，自动生成 `aclrtlaunch_*.h`。

> 两种方式均可使用，不存在"禁止使用 `add_executable + .asc`"的限制。关键是正确配置 `-dc`、`--npu-arch` 和 include 路径。

## 11. 参考模板与完整可运行工程之间的边界

`templates/` 目录提供的是**参考模板**：

- **包含**：7 个 Kernel 模板（`dav310/*.template`）、公共 TilingData 定义、Host tiling 参考实现、`dav310/kernel_utils.template`、文档。集成前按第 0 节将 `.template` 转成 `.h`/`.cpp`。
- **不包含**：完整的 CMakeLists.txt、`.asc` 入口文件、数据生成/验证脚本、input/output I/O 工具。
- 完整可运行工程参考：`softmaxNewKernel/` 源工程（含 `softmax.asc`、`CMakeLists.txt`、`data_utils.h`）。

集成时需自行编写：
1. Kernel 入口（`.asc` 或 `.cpp`），按本文档第 4-6 节的方式实例化模板。
2. Host main 或 op_host，调用 `softmax_v2_tiling.h` 中的 tiling 函数。
3. CMakeLists.txt，按本文档第 9 节的方式配置构建。
4. 数据 I/O 工具（如 `data_utils.h`）。

## 12. 常见编译问题

### 问题 1：MicroAPI 类型找不到

**现象**：`error: no template named 'RegTensor'` 或 `error: 'MicroAPI' has not been declared`

**根因**：reg_compute 类型仅在 device arch 下声明，host pass 不可见。

**解决**：在 include 之前添加 `__ASC_NPU_HOST__` 守卫（见第 10 节），或使用 `ascendc_library(kernellaunch)` 框架。

### 问题 2：device 链接符号未定义

**现象**：链接阶段报 device 符号无法解析。

**根因**：`--npu-arch` 未在链接阶段指定，默认架构不匹配。

**解决**：`target_link_options` 中添加 `--npu-arch=dav-3510`。

### 问题 3：compile_commands.json 不存在

**现象**：`error: No such file or directory: 'compile_commands.json'`

**解决**：`export CMAKE_EXPORT_COMPILE_COMMANDS=ON` 后在同一 shell 执行 cmake 和 make。
