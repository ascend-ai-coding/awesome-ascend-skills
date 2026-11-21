/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

# 模板使用指南

本文件说明如何将 `templates/` 下的 Gather Kernel 模板集成到实际算子项目中。

## 0. 模板文件格式与使用约定

`templates/` 下的所有源码模板均以 `.template` 为后缀（如 `gather_simt.template`）。这些是**参考模板**，不是可直接编译的源码。

**使用时必须将 `.template` 重命名/复制为对应的 `.h` 文件**，否则最终工程里的 `#include` 引用的是 `.h`，代码不易读也无法编译。转换规则：去掉 `.template` 后缀并按文件头 `\file` 注释中标注的名字落地：

| 模板文件 | 转换后落地文件 |
|----------|---------------|
| `dav310/gather_simt.template` | `dav310/gather_simt.h` |
| `dav310/gather_simd_last_gather.template` | `dav310/gather_simd_last_gather.h` |
| `dav310/gather_simd_row_copy.template` | `dav310/gather_simd_row_copy.h` |
| `dav310/gather_ga_all_load.template` | `dav310/gather_ga_all_load.h` |
| `dav310/gather_full_load.template` | `dav310/gather_full_load.h` |
| `dav310/gather_simt_two_dim.template` | `dav310/gather_simt_two_dim.h` |
| `dav310/gather_elements_v2_transpose.template` | `dav310/gather_elements_v2_transpose.h` |
| `dav310/gather_elements_v2_scalar.template` | `dav310/gather_elements_v2_scalar.h` |
| `dav310/gather_elements_v2_last_dim.template` | `dav310/gather_elements_v2_last_dim.h` |
| `dav310/gather_elements_v2_common.template` | `dav310/gather_elements_v2_common.h` |
| `gather_elements_v2_last_dim_tiling.template` | `gather_elements_v2_last_dim_tiling.h` |
| `gather_elements_v2_tiling.template` | `gather_elements_v2_tiling.cpp`（Host tiling 注册，转 .cpp 使用） |
| `gather_elements_v2_route.template` | `gather_elements_v2_route.h` |
| `gather_tiling_data.template` | `gather_tiling_data.h` |
| `gather_tiling.template` | `gather_tiling.h` |

转换命令示例（保留目录结构）：

```bash
# .template → .h（绝大多数模板）
find . -name '*.template' ! -name 'gather_elements_v2_tiling.template' | while read f; do
  cp "$f" "${f%.template}.h"
done
# gather_elements_v2_tiling.template → .cpp（含 IMPL_OP_OPTILING 源文件）
cp gather_elements_v2_tiling.template gather_elements_v2_tiling.cpp
```

> **架构目录说明**：Kernel 模板位于 `dav310/` 子目录（对应目标架构 `dav-3510`/Ascend950），与 Softmax 等其它族模板同级。

## 1. 文件清单

### Kernel 模板（6 个，均在 `dav310/` 下）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `dav310/gather_simt.template` | `gather_simt.h` | SIMT 标量并行：`GatherSimt<X_T, INDICES_T, INDEX_SIZE_T>`，2048 线程直接 GM→GM，通用兜底 |
| `dav310/gather_simd_last_gather.template` | `gather_simd_last_gather.h` | SIMD 末轴 Gather：`GatherSimdLastGather<X_T, INDICES_T, NEG_INDICE_SUPPORT>`，Gather 向量化 |
| `dav310/gather_simd_row_copy.template` | `gather_simd_row_copy.h` | SIMD 行拷贝：`GatherSimdRowCopy<INDICES_T>`，DataCopyPad 逐行搬 + index 滑窗缓存 |
| `dav310/gather_ga_all_load.template` | `gather_ga_all_load.h` | GA 全载：`GatherGaAllLoad<INDICES_T, NIS>`，x GA 块全载 UB + Gather |
| `dav310/gather_full_load.template` | `gather_full_load.h` | Full-Load：`GatherFullLoad<T, INDICES_T, U>`，GA 全载 UB + SIMT 从 UB 取值 |
| `dav310/gather_simt_two_dim.template` | `gather_simt_two_dim.h` | SIMT 两维：`GatherSimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>`，2D 特化（1 次 fast-div） |

### 公共依赖（2 个，host/device 共用）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `gather_tiling_data.template` | `gather_tiling_data.h` | 六套 TilingData 结构体定义（packed）：`GatherTilingData` / `GatherLastTilingData` / `GatherGaAllLoadTilingData` / `GatherSimdTwoDimTilingData` / `GatherSimtTilingData` / `GatherTilingDataEmptyInput` |
| `gather_tiling.template` | `gather_tiling.h` | Host 侧 tiling 决策树与切分函数：`GatherSelectTiling()` 门控 + 各模板 tiling 函数 |

### GatherElementsV2 Kernel 模板（3 个，均在 `dav310/` 下）

> GatherElementsV2（torch.gather 语义，index 与 output 同形，PER_ELEMENT）的模板，与上述 GatherV2（PER_SLICE）模板独立。详见各模板 `.md` 文档。

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `dav310/gather_elements_v2_transpose.template` | `gather_elements_v2_transpose.h` | Transpose + Gather：`GatherElementsV2TransposeKernel<T_X, T_IDX>`，dim!=last 时 kernel 内转置 + Gather 指令向量化 |
| `dav310/gather_elements_v2_scalar.template` | `gather_elements_v2_scalar.h` | Scalar 兜底：`GatherElementsV2ScalarKernel<T_X, T_IDX>`，UB 不够 Transpose 时逐元素标量 gather |
| `dav310/gather_elements_v2_last_dim.template` | `gather_elements_v2_last_dim.h` | LastDim：`GatherElementsV2LastDim<T_DATA, T_IDX>`，dim==last 时 x 行全载 UB + Gather 指令 |
| `dav310/gather_elements_v2_common.template` | `gather_elements_v2_common.h` | 公共基类 `GatherElementsV2KernelBase`：合轴参数 + 核间分组 + 同步原语 |

### GatherElementsV2 Host 路由与 tiling（3 个，在 `templates/` 根目录）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `gather_elements_v2_route.template` | `gather_elements_v2_route.h` | API/Host 上游路由：区分末轴、物理 Transpose、kernel 内 Transpose、Scalar/SIMT；执行能力与成本门控 |
| `gather_elements_v2_tiling.template` | `gather_elements_v2_tiling.cpp` | Host tiling 主入口：`GatherElementsV2Tiling` 类（scalar/transpose 切分）+ `TilingGatherElementsV2()` 决策树 + `IMPL_OP_OPTILING` 注册 |
| `gather_elements_v2_last_dim_tiling.template` | `gather_elements_v2_last_dim_tiling.h` | Host tiling last_dim 路径：`GatherElementsV2LastDimTiling` 类（合轴 + UB 切分 + scalarMode 回退） |

## 2. 可独立选择的模板

GatherV2（PER_SLICE）模板可以独立选择。GatherElementsV2（PER_ELEMENT）的三个 device 模板必须由 Host 路由统一选择；尤其不能仅凭 `dim!=last` 直接实例化内部 Transpose。

PER_ELEMENT small-GD tiled full-load 当前是已验证的实验性实现模式，不是可直接复制的通用模板。集成前先读 [gather_elements_small_gd_tiled.md](gather_elements_small_gd_tiled.md)，参数化 `Gx/Gi`，并完成代表 shape A/B 与全用例回归；禁止复制固定 gatherDim specialization 后放宽 Host 门控。

| 模板 | 类名 | TilingData | 构造方式 |
|------|------|-----------|---------|
| SIMT 标量并行 | `gatherv2::GatherSimt` | `GatherTilingData` | 无参构造，`Init(x, indices, y, &tiling)` |
| SIMD 末轴 Gather | `gatherv2::GatherSimdLastGather` | `GatherLastTilingData` | 构造传 `TPipe*`，`Init(x, indices, y, &tiling)` |

### 最广适用的路径：SIMT

**无任何 shape/连续性假设**，所有 Gather 场景都能跑（含散点读 torch.gather dim=0）。当不确定选哪个、或 SIMD 末轴/行拷贝/GA 全载的门控不命中时，**优先选 SIMT**。它也是唯一一个不需要 UB 装载 x 的路径，UB 压力最小。

### GatherElementsV2 必须成套接入

接入 GatherElementsV2 时，至少同时复制 `gather_elements_v2_route.template`、`gather_elements_v2_tiling.template` 和选中的 device 模板。调用顺序是：

1. API/Host 用 `SelectGatherElementsV2Route()` 判断末轴、物理 Transpose、内部 Transpose或 Scalar/SIMT。
2. 只有返回 `INTERNAL_TRANSPOSE` 时，才让 `gather_elements_v2_tiling.cpp` 生成 TILING_KEY=1。
3. 物理 Transpose 是三个独立 Transpose 加一个末轴 Gather，不使用内部 Transpose device 模板。
4. 当前内部 Transpose 模板仅支持 2B/4B 数据；int8（例如 case6）必须走物理 Transpose 或 Scalar/SIMT。
5. 对 dim!=last 的 small-GD 场景，在回退 Scalar/SIMT 前评估 tiled full-load；Scalar 与 SIMT 必须同口径 A/B，Transpose UB 不够不是 Scalar 性能占优的证据。

`gather_elements_v2_tiling.template` 与 `dav310/gather_elements_v2_transpose.template` 对应 ops-nn 的内部实现，不包含 API 外层的 `IfUseGatherElementsV2`/`MemCheck`。漏复制 route 模板会把“内部 tiling 可生成”误当成“全局路径最优”。

## 3. 每个模板的最小依赖集合

> 依赖关系以**转换后的 `.h`** 描述。集成前请先按第 0 节把 `.template` 落地为 `.h`。

```
dav310/gather_simt.h
├── gather_tiling_data.h
├── kernel_tiling/kernel_tiling.h          CANN SDK：TilingData 基基础设施
├── kernel_operator.h                      CANN SDK：AscendC 高层 API
└── simt_api/asc_simt.h                    CANN SDK：Simt::UintDiv / asc_vf_call / GetUintDivMagicAndShift

dav310/gather_simd_last_gather.h
├── gather_tiling_data.h
├── kernel_operator.h                      CANN SDK：AscendC + Reg API
├── op_kernel/platform_util.h              CANN SDK：Ops::Base::GetVRegSize 等
└── reg_compute/kernel_reg_compute_datacopy_intf.h  CANN SDK：DataCopyGatherImpl
```

## 4. 如何显式选择一个模板

在 kernel 入口文件中 `#include` 需要的模板头文件和 TilingData 头文件：

```cpp
#include "gather_tiling_data.h"
#include "dav310/gather_simt.h"   // 只包含需要的模板
```

## 5. 如何实例化并调用

### SIMT 模板（无参构造，Init 传 TilingData 指针）

```cpp
gatherv2::GatherSimt<float, int32_t, uint32_t> op;
op.Init(x, indices, y, &tiling);
op.Process();
```

模板参数 `INDEX_SIZE_T` 取 `uint32_t`（元素总数 < 4G）或 `uint64_t`（超大张量）。`uint64_t` 时寄存器压力增大，建议 `threadNum` 降到 1024（见 [simt_gather.md](simt_gather.md) 线程数选择）。

### SIMD 末轴 Gather 模板（构造传 `TPipe*`）

```cpp
TPipe pipe;
gatherv2::GatherSimdLastGather<float, int32_t, true> op(&pipe);  // true = 支持负索引
op.Init(x, indices, y, &tiling);
op.Process();
```

## 6. dtype 实例化

```cpp
// FP32
gatherv2::GatherSimt<float, int32_t, uint32_t> op_fp32;
// FP16
gatherv2::GatherSimt<half, int32_t, uint32_t> op_fp16;
// int8 数据 + int32 索引（SIMD 末轴会自动把 int8 提升为 int16 对齐 32B burst）
gatherv2::GatherSimdLastGather<int8_t, int32_t, false> op_i8(&pipe);
```

输入输出 dtype 必须一致。索引 dtype 支持 `int8/int16/int32/int64`，SIMD 末轴内部统一提升为 `int16/int32` 参与 VF 比较/加法。

## 7. Reg 命名空间、Kernel 入口与 CANN SDK include

- **主命名空间**：SIMD 末轴模板统一使用 `AscendC::Reg` 下的 `Gather` / `DataCopy` / `Adds` / `CompareScalar` 等 VF 指令，不在模板中自行声明 `MicroAPI` 别名。引用类型 `RegTensor<T>`、`MaskReg` 来自 `using AscendC::Reg::RegTensor` 等声明。
- **`__VEC_SCOPE__`**：所有 Reg VF 调用需包在 `__VEC_SCOPE__ { ... }` 作用域内。
- **Kernel 入口**：复用目标工程同构成功实现的入口和实际编译模式；`.cpp -xasc` 直调不能仅因使用 Reg API 就自行增加 `__vector__` 或 `-dc`。
- **CANN SDK include**：
  - `kernel_tiling/kernel_tiling.h` — TilingData 基础设施
  - `kernel_operator.h` — AscendC 高层 API 和 Reg API
  - `simt_api/asc_simt.h` — SIMT 线程 API（仅 SIMT 模板需要）
  - `op_kernel/platform_util.h` — 平台工具
  - `reg_compute/kernel_reg_compute_datacopy_intf.h` — DataCopyGatherImpl（仅 SIMD 末轴需要）
- **目标架构**：`dav-3510`（Ascend950），VRegSize=256B。

## 8. 构建方式

### 方式一：bisheng / 单 .asc 文件（host + kernel 同一翻译单元）

参考源工程 `gather_v2.asc`，host main 与 kernel 入口写在同一 `.asc` 文件中，一趟 ASC 编译，`<<<blockDim, nullptr, stream>>>` 直调。

```bash
bisheng --npu-arch=dav-3510 -xasc -dc gather_v2.asc -o gather_v2
```

关键编译选项：

| 选项 | 作用 |
|------|------|
| `--npu-arch=dav-3510` | 指定目标架构（Ascend950），device 编译与链接阶段均需指定 |
| `-xasc` | 按 ASC 语言编译（host + device 同翻译单元） |
| `-dc` | 生成可重定位 device 代码（reg_compute / simt_api 类型仅在 device arch 下声明） |

### 方式二：`ascendc_library(kernellaunch)` 框架

```cmake
set(ASCEND_KERNEL_LAUNCH_ONLY ON)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(ascendc_kernels_npu STATIC op_kernel/gather_kernel.cpp)
ascendc_include_directories(ascendc_kernels_npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
target_compile_options(ascendc_kernels_npu PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:-dc>
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)
target_link_options(ascendc_kernels_npu PRIVATE --npu-arch=dav-3510)
```

Host 侧通过 `ACLRT_LAUNCH_KERNEL` 调用。

## 9. 常见编译问题

### 问题 1：`AscendC::Reg` 或 `RegTensor` 未声明

**现象**：`error: namespace "AscendC::Reg" has not been declared` 或 `error: no template named 'RegTensor'`。

**根因**：实际目标架构、include 链或 host/device pass 与模板要求不一致。

**解决**：
1. 保存并检查同工程真实编译命令，确认 `--npu-arch=dav-3510` 实际生效。
2. 确保已 `#include "kernel_operator.h"`，并只读检查它是否按当前架构引入 reg_compute 接口。
3. 对照同工程成功实现处理 host/device pass；不得为消除报错而自行修改 SDK 或无条件定义命名空间别名。

### 问题 2：按值传递 TilingData 结构体参数报错

**现象**：kernel 入口 `void gather_kernel(GM_ADDR x, ..., GatherLastTilingData tiling)`（结构体**按值**传入）编译报 `__vector__` 相关错误或参数传递异常。

**根因**：device 函数按值接收聚合结构体参数时，需 `__vector__` 属性标注入口，使参数走向量核 calling convention；否则默认 calling convention 与结构体 by-value 不兼容。

**解决**：
1. 入口标注 `__global__ __aicore__ __vector__`。
2. 或改为传指针 `const GatherLastTilingData *tiling` + device 侧 `CopyTiling` / `GET_TILING_DATA_WITH_STRUCT` 从 GM 加载（SIMT 模板 Init 即用指针，无此问题）。

### 问题 3：device 链接符号未定义

**现象**：链接阶段报 `Simt::UintDiv` / `asc_vf_call` / `DataCopyGatherImpl` 等 device 符号无法解析。

**根因**：`--npu-arch` 未在链接阶段指定，默认架构不匹配（落到 `dav-m100`）。

**解决**：`target_link_options` 中添加 `--npu-arch=dav-3510`，或 bisheng 命令行链接阶段同样指定。

### 问题 4：SIMT LAUNCH_BOUND 过高导致 register spill

**现象**：`uint64_t` 索引 + `LAUNCH_BOUND(2048)` 编译通过但性能回退（register spill 到 local memory）。

**根因**：2048 线程时每线程仅 16 寄存器，`uint64_t` + 两级 fast-div 寄存器压力过大。

**解决**：`INDEX_SIZE_T=uint64_t` 时把 `threadNum` 降到 1024（每线程 32 寄存器）。详见 [simt_gather.md](simt_gather.md) 线程数选择策略。

## 10. 参考模板与完整可运行工程之间的边界

`templates/` 目录提供的是**参考模板**：

- **包含**：Gather/GatherElementsV2 Kernel 模板、TilingData 定义、Host 路由与 tiling 参考实现、模板说明文档。集成前按第 0 节转换 `.template`；GatherElementsV2 必须保留上游 route 门控。
- **不包含**：完整 CMakeLists.txt、`.asc` 入口文件、数据生成/验证脚本。

完整可运行工程参考：`asc-devkit/examples/03_simt_api/.../general_gather/gather_v2.asc`（含 host main、`<<<...>>>` 直调、golden 验证）。
