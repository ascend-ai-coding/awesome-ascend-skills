# 自定义算子 PyTorch 绑定构建指南（torch_aclnn_helper.h + EXEC_NPU_CMD 标准方式）

## 概述

**本指南规定的是构建 PyTorch 到 ACLNN 绑定的唯一标准方式**：使用 `torch_aclnn_helper.h` 头文件中的 `EXEC_NPU_CMD` 宏。**禁止**使用手动调用 aclnn C API（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）、`import ascend_kernel`、Pybind 直接封装等方式。

## 适用场景

- 算子已编译为 CANN OPP 包并安装（`vendors/<vendor_name>/` 结构）
- 算子有 `op_api/lib/libcust_opapi.so`（包含 aclnn 两段式接口符号）
- 算子**未**注册到 `torch.ops.npu`（即 `torch_npu` 内置绑定中不包含该算子）
- 需要通过 Python 调用算子进行精度测试

## 核心文件：torch_aclnn_helper.h

### 文件位置

`torch_aclnn_helper.h` 位于 ascend-kernel 项目的 `csrc/utils/` 目录，是 CANN 官方提供的 PyTorch-ACLNN 桥接层。**MUST** 将此文件复制到绑定项目的源码目录中。

### 核心能力

`torch_aclnn_helper.h` 提供以下能力：

| 能力 | 实现方式 | 说明 |
|------|---------|------|
| 动态库加载 | `GetOpApiFuncAddr()` | 先查 `libcust_opapi.so`，再查 `libopapi.so` |
| 类型转换 | `ConvertTypes()` 模板 | `at::Tensor` → `aclTensor*`，`at::Scalar` → `aclScalar*` 等 |
| 两段式调用 | `EXEC_NPU_CMD` 宏 | 自动完成 GetWorkspaceSize + Execute + 释放 |
| 内存管理 | `InitHugeMem`/`ReleaseHugeMem` | 自动管理大页内存 |
| Stream 管理 | `c10_npu::getCurrentNPUStream()` | 自动获取当前 NPU stream |
| 错误处理 | `TORCH_CHECK` | 自动检查返回值并报错 |

### EXEC_NPU_CMD 宏详解

```cpp
EXEC_NPU_CMD(aclnn_api, param1, param2, ..., output1, output2, ...)
```

**宏展开后的执行流程**：

```
1. GetOpApiFuncAddr("aclnnXxxGetWorkspaceSize")  → 动态加载 GetWorkspaceSize 函数
2. GetOpApiFuncAddr("aclnnXxx")                   → 动态加载 Execute 函数
3. InitHugeMemThreadLocal(nullptr, false)          → 初始化大页内存
4. ConvertTypes(param1, param2, ..., &workspace_size, &executor)  → 自动类型转换
5. call(getWorkspaceSizeFunc, converted_params)    → 调用 GetWorkspaceSize
6. TORCH_CHECK(workspace_status == 0, ...)         → 检查返回值
7. at::empty({workspace_size}, ...)                → 分配 workspace
8. opApiFunc(workspace_addr, workspace_size, executor, acl_stream)  → 执行算子
9. ReleaseConvertTypes(converted_params)           → 释放所有 aclTensor*/aclScalar* 等
10. ReleaseHugeMem(nullptr, false)                 → 释放大页内存
```

### ConvertType 类型转换体系

| PyTorch 类型 | ACL 类型 | 转换行为 |
|-------------|---------|---------|
| `at::Tensor` | `aclTensor*` | 自动处理 sizes/strides/dtype/format/storage/data_ptr |
| `c10::optional<at::Tensor>` | `aclTensor*` (nullable) | 有值转、无值返 nullptr |
| `at::Scalar` | `aclScalar*` | 支持 Double/Long/Bool/ComplexDouble |
| `at::IntArrayRef` | `aclIntArray*` | 直接转换 |
| `at::ArrayRef<bool>` | `aclBoolArray*` | 直接转换 |
| `at::TensorList` | `aclTensorList*` | 逐元素转换后创建列表 |
| `c10::optional<at::IntArrayRef>` | `aclIntArray*` (nullable) | 有值转、无值返 nullptr |
| `c10::optional<at::Scalar>` | `aclScalar*` (nullable) | 有值转、无值返 nullptr |
| `at::ScalarType` | `aclDataType` | 查表 `kATenScalarTypeToAclDataTypeTable` |
| 基本类型 (int64_t, double, bool) | 原样传递 | 恒等转换 |

**dtype 映射表**（`kATenScalarTypeToAclDataTypeTable`）：

| at::ScalarType | aclDataType |
|---------------|-------------|
| Byte | ACL_UINT8 |
| Char | ACL_INT8 |
| Short | ACL_INT16 |
| Int | ACL_INT32 |
| Long | ACL_INT64 |
| Half | ACL_FLOAT16 |
| Float | ACL_FLOAT |
| Double | ACL_DOUBLE |
| Bool | ACL_BOOL |
| BFloat16 | ACL_BF16 |
| ComplexFloat | ACL_COMPLEX64 |
| ComplexDouble | ACL_COMPLEX128 |

### 动态库加载机制

```cpp
GetOpApiFuncAddr("aclnnTopKTopPSampleV2")
  → 先 dlopen + dlsym libcust_opapi.so（自定义算子包）
  → 再 dlopen + dlsym libopapi.so（CANN 标准库）
```

**关键**：`EXEC_NPU_CMD` 通过 `dlsym` 动态查找 aclnn 符号，**不需要**编译时链接 `libcust_opapi.so`。但运行时必须确保 `LD_LIBRARY_PATH` 包含 `libcust_opapi.so` 所在路径。

## 完整流程

### Step 1：从算子源码提取接口信息

**必须读取的文件**：

| 文件 | 提取内容 |
|------|---------|
| `op_host/op_api/aclnn_<op_name>.h` | aclnn C 接口签名（函数名、参数类型、参数顺序） |
| `op_host/op_api/aclnn_<op_name>.cpp` | 参数校验逻辑（dtype 约束、shape 约束、属性默认值） |
| `op_host/op_api/<op_name>.h` | L0 接口签名（输出 shape 和 dtype 推导） |
| `op_host/op_api/<op_name>.cpp` | 输出 tensor 的 AllocTensor 逻辑（确认输出 dtype/shape） |
| `op_host/config/ascend950/<op_name>_binary.json` | 算子注册信息（opInterface 名、输入输出定义） |

**关键提取项**：

1. **aclnn 函数名**：如 `aclnnTopKTopPSampleV2GetWorkspaceSize` / `aclnnTopKTopPSampleV2`
2. **输入参数**：每个参数的 `const aclTensor*` / `int64_t` / `double` / `bool` 类型
3. **输出参数**：`aclTensor*` 指针（非 const），数量和含义
4. **属性参数**：`int64_t` / `double` / `bool` 类型的算子属性
5. **输出 dtype 推导**：从 L0 的 `AllocTensor` 确认输出 dtype（可能与输入不同，如量化输出 int8）
6. **输出 shape 推导**：从 L0 的 `AllocTensor` 确认输出 shape

### Step 2：创建 PyTorch 绑定文件

#### 2.1 目录结构

```
<op_name>/pytorch/
├── CMakeLists.txt              ← 编译配置
├── torch_aclnn_helper.h        ← CANN 标准桥接头文件（从 ascend-kernel/csrc/utils/ 复制）
├── <op_name>.cpp               ← PyTorch 绑定（EXEC_NPU_CMD + TORCH_LIBRARY 注册）
├── test_<op_name>.py           ← Python 验证脚本
└── build/                      ← 构建产物（自动生成）
    └── libcustom_ops.so
```

**重要**：
- 绑定文件**必须**使用 `.cpp` 扩展名，**禁止**使用 `.asc`（ASC 编译器在纯 C++ 绑定场景下不适用）
- `torch_aclnn_helper.h` **必须**从 ascend-kernel 项目的 `csrc/utils/` 目录复制到绑定项目源码目录

#### 2.2 `.cpp` 文件模板（EXEC_NPU_CMD 标准方式）

```cpp
#include <torch/library.h>
#include "torch_aclnn_helper.h"

using namespace at;

namespace ascend_kernel {

// 输出 shape 计算函数（根据算子逻辑实现）
c10::SmallVector<int64_t, 3> <op_name>_npu_output_size(
    const at::Tensor &self, ...)
{
    // 根据 L0 AllocTensor 逻辑计算输出 shape
    ...
}

// Python 调用函数 — 使用 EXEC_NPU_CMD
std::tuple<at::Tensor, ...> <op_name>(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& gamma_opt,
    ...其他输入,
    int64_t num_groups,
    double eps,
    bool activate_silu)
{
    // 1. 参数校验
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");

    // 2. contiguous 化（EXEC_NPU_CMD 内部的 ConvertType 要求 tensor contiguous）
    at::Tensor x_contiguous = x.contiguous();
    ...

    // 3. 分配输出 tensor（使用 torch_npu 原生 API）
    //    at_npu::native::empty_with_format 保留 NPU format 信息
    auto output_size = <op_name>_npu_output_size(x_contiguous, ...);
    at::Tensor out = at_npu::native::empty_with_format(
        output_size, x.options().dtype(at::kChar),  // 量化输出 int8 示例
        at_npu::native::get_npu_format(x));
    at::Tensor mean_out = at_npu::native::empty_with_format(
        {N, num_groups}, x.options(),
        at_npu::native::get_npu_format(x));
    ...

    // 4. EXEC_NPU_CMD 一行调用（自动处理类型转换 + workspace + 执行 + 释放）
    //    参数顺序与 aclnn<OpName>GetWorkspaceSize 的参数顺序完全一致
    //    输出 tensor 放在最后（在属性参数之后）
    EXEC_NPU_CMD(aclnn<OpName>,
        x_contiguous, ...,           // 输入 tensor
        num_groups, eps, activate_silu,  // 属性参数
        out, mean_out, ...);          // 输出 tensor

    return std::make_tuple(out, mean_out, ...);
}

}  // namespace ascend_kernel

// 注册到 PyTorch — 使用 npu 命名空间 + PrivateUse1 后端
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("<op_name>(Tensor x, Tensor? gamma, ..., int num_groups, float eps=1e-5, bool activate_silu=True) -> (Tensor, Tensor, ...)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("<op_name>", TORCH_FN(ascend_kernel::<op_name>));
}
```

**关键要点**：

1. **`#include "torch_aclnn_helper.h"`**：这一个 include 即可，不需要 `#include <acl/acl.h>` 或 `#include <aclnn/acl_meta.h>`（helper 内部已包含）
2. **`EXEC_NPU_CMD(aclnnXxx, ...)`**：参数顺序与 `aclnnXxxGetWorkspaceSize` 的参数顺序**完全一致**，不需要手动添加 `workspace_size_addr` 和 `executor_addr`（宏内部自动追加）
3. **输出 tensor 预分配**：使用 `at_npu::native::empty_with_format` 保留 NPU format 信息，**不要**使用 `at::empty`（可能丢失 format）
4. **`TORCH_LIBRARY_FRAGMENT(npu, m)`**：使用 `npu` 命名空间（不是 `custom_ops`），与 `torch_npu` 内置算子保持一致
5. **`TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`**：绑定到 NPU 设备（PrivateUse1 = NPU）

#### 2.3 CMakeLists.txt 模板

```cmake
cmake_minimum_required(VERSION 3.16)
project(<op_name>_pytorch LANGUAGES CXX)

# 纯 C++ 编译，不使用 ASC 编译器
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Torch cmake 路径
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

# torch_npu 路径
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS
    ${TORCH_NPU_PATH}/include
    ${TORCH_NPU_PATH}/include/third_party/acl/inc
)

# CANN 头文件路径
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
set(CANN_INCLUDE_DIRS
    ${ASCEND_HOME}/aarch64-linux/include
    ${ASCEND_HOME}/aarch64-linux/include/aclnn
)

add_library(custom_ops SHARED
    <op_name>.cpp
)

target_include_directories(custom_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}          # 包含 torch_aclnn_helper.h
)

target_link_libraries(custom_ops PRIVATE
    torch_npu
    Python3::Python
    ${TORCH_LIBRARIES}
    dl                                   # dlsym 需要 libdl
)

target_compile_options(custom_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    -D_GLIBCXX_USE_CXX11_ABI=1          # 必须与 torch_npu ABI 一致
)
```

**关键配置说明**：

| 配置项 | 说明 | 踩坑经验 |
|--------|------|---------|
| `LANGUAGES CXX` | 纯 C++ 编译 | **禁止**使用 `ASC`（ASC 编译器报 `NPU arch not supported`） |
| `find_package(Torch)` | 通过 Python 动态获取 cmake_prefix_path | **必须**，不能硬编码 |
| `TORCH_NPU_INCLUDE_DIRS` | torch_npu 的 include 路径 | 包含 `third_party/acl/inc` 等 |
| `ASCEND_HOME` | `$ENV{ASCEND_HOME_PATH}` | CANN 头文件路径 |
| `target_link_libraries` | `torch_npu`, `dl` | **必须链接 `dl`**（dlsym 需要），**不需要**链接 `libcust_opapi.so`（EXEC_NPU_CMD 运行时动态加载） |
| `-D_GLIBCXX_USE_CXX11_ABI=1` | C++11 ABI | **必须与 torch_npu ABI 一致** |
| `${CMAKE_CURRENT_SOURCE_DIR}` | 包含 `torch_aclnn_helper.h` | helper 文件必须在 include 路径中 |

#### 2.4 Python 验证脚本模板

```python
#!/usr/bin/env python3
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'  # 指定空闲 NPU 设备，避免设备被占用卡住

import torch
import torch_npu

# 加载自定义算子绑定
torch.ops.load_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "libcustom_ops.so"))

# 调用算子 — 使用 torch.ops.npu 命名空间（与 TORCH_LIBRARY_FRAGMENT(npu) 对应）
out, mean_out, rstd_out = torch.ops.npu.<op_name>(
    x, gamma, beta, quant_scale,
    num_groups=8, eps=1e-5, activate_silu=True
)
```

### Step 3：构建

```bash
# 环境配置
source <cann_path>/set_env.sh
conda activate <env_name>

# 设置运行时库搜索路径（EXEC_NPU_CMD 运行时需要 dlopen libcust_opapi.so）
export LD_LIBRARY_PATH=<vendors_path>/<vendor_name>/op_api/lib:$LD_LIBRARY_PATH

# 构建（纯 C++ 编译，不使用 ASC 编译器）
cd <op_name>/pytorch
rm -rf build && mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### Step 4：验证构建产物

| 检查项 | 命令 | 预期结果 |
|--------|------|---------|
| `.so` 文件存在 | `ls build/libcustom_ops.so` | 文件存在 |
| Python 可加载 | `python3 -c "import torch; torch.ops.load_library('build/libcustom_ops.so'); print('OK')"` | 输出 `OK` |
| 算子可调用 | 运行验证脚本 | 输出正确的 shape/dtype |

**注意**：使用 `EXEC_NPU_CMD` 时，`nm -D` 不会显示 aclnn 符号（因为是运行时 `dlsym` 动态加载），这是正常行为。

### Step 5：在精度测试中集成

精度测试脚本中使用 `torch.ops.npu.<op_name>(...)` 调用算子：

```python
import os
os.environ['ASCEND_CUSTOM_OPP_PATH'] = '<vendors_path>'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

import torch
import torch_npu

# 加载自定义算子绑定
torch.ops.load_library("<path_to>/libcustom_ops.so")

# 精度测试中调用 — 使用 torch.ops.npu 命名空间
def npu_call(x, gamma, beta, quant_scale, num_groups, eps, activate_silu):
    out, mean_out, rstd_out = torch.ops.npu.<op_name>(
        x, gamma, beta, quant_scale,
        num_groups=num_groups, eps=eps, activate_silu=activate_silu
    )
    return out  # 或返回全部输出

def cpu_reference(x, gamma, beta, quant_scale, num_groups, eps, activate_silu):
    # 手写 CPU 参考实现
    ...
```

## 技术难点与解决方案

### 难点 1：CANN OPP 包不等于 PyTorch 可调用

**现象**：算子已通过 `cann-ops-nn-custom_linux-aarch64.run` 安装到 `vendors/` 目录，但 Python 中 `torch.ops.npu.<op_name>` 不存在。

**根因**：CANN OPP 包只注册算子到 CANN 运行时（tiling + kernel binary），不包含 PyTorch C++ Extension 绑定层。`torch.ops.npu` 中的算子来自 `torch_npu` 包的内置 C++ 封装。

**解决方案**：构建独立的 PyTorch C++ Extension `.so`，通过 `TORCH_LIBRARY_FRAGMENT(npu)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1)` 注册算子。

### 难点 2：运行时链接 libcust_opapi.so

**现象**：编译通过但运行时 `EXEC_NPU_CMD` 报 `aclnnXxx or aclnnXxxGetWorkspaceSize not in libopapi.so`。

**根因**：`EXEC_NPU_CMD` 先从 `libcust_opapi.so` 查找符号，找不到再从 `libopapi.so` 查找。如果两个库都找不到，说明 `LD_LIBRARY_PATH` 未包含 `libcust_opapi.so` 所在路径。

**解决方案**：

```bash
export LD_LIBRARY_PATH=/path/to/vendors/<vendor>/op_api/lib:$LD_LIBRARY_PATH
```

### 难点 3：环境变量 ASCEND_CUSTOM_OPP_PATH

**现象**：aclnn 调用返回错误码（非 0），算子执行失败。

**根因**：CANN 运行时需要通过 `ASCEND_CUSTOM_OPP_PATH` 找到算子的 tiling 和 kernel binary。

**解决方案**：
```bash
export ASCEND_CUSTOM_OPP_PATH=/path/to/vendors
```

### 难点 4：NPU 设备被占用

**现象**：Python 脚本在创建 NPU tensor 时卡住不动，无报错。

**根因**：多个进程争用同一 NPU 设备。

**解决方案**：
```bash
export ASCEND_RT_VISIBLE_DEVICES=N  # 指定空闲 NPU 设备编号
```

### 难点 5：C++ ABI 不匹配

**现象**：加载 `.so` 时报 `undefined symbol` 或 `GLIBCXX_*` 相关错误。

**根因**：编译时使用的 C++ ABI 与 `torch_npu` 不一致。

**解决方案**：CMakeLists.txt 中添加 `-D_GLIBCXX_USE_CXX11_ABI=1`（与 torch_npu 保持一致）。

## 常见编译/运行错误速查

| 错误 | 根因 | 修复 |
|------|------|------|
| `unknown type name 'aclTensor'` | 未包含 `torch_aclnn_helper.h` | `#include "torch_aclnn_helper.h"` |
| `use of undeclared identifier 'ACL_BF16'` | 旧版头文件 | `torch_aclnn_helper.h` 已包含，无需额外处理 |
| `NPU arch is not supported!!!` | 使用了 ASC 编译器 | 改为纯 C++ 编译（`LANGUAGES CXX`），文件扩展名 `.cpp` |
| `Cannot determine link language for target` | 文件扩展名 `.asc` 但无 ASC 编译器 | 改为 `.cpp` |
| `aclnnXxx not in libopapi.so` | 运行时找不到 `libcust_opapi.so` | `export LD_LIBRARY_PATH=<op_api/lib>:$LD_LIBRARY_PATH` |
| `aclnnXxxGetWorkspaceSize failed, ret=XXX` | CANN 运行时找不到算子 tiling/kernel | `export ASCEND_CUSTOM_OPP_PATH=<vendors_path>` |
| NPU tensor 创建卡住 | NPU 设备被占用 | `export ASCEND_RT_VISIBLE_DEVICES=N` |
| `undefined symbol: _ZN...` | C++ ABI 不匹配 | 添加 `-D_GLIBCXX_USE_CXX11_ABI=1` |
| `no member named 'is_npu'` | `is_npu()` 不是 PyTorch 标准 API | 改用 `x.device().type() == c10::DeviceType::PrivateUse1` |

## 禁止事项

- ❌ **禁止**使用 PyTorch 原生同名接口（如 `torch.gather`、`torch.acosh` 等）替代 EXEC_NPU_CMD 绑定，**所有迁移算子都必须构建绑定**（确保走 950 迁移后路径）
- ❌ **禁止**使用手动 aclnn C API 调用（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`），**必须**使用 `EXEC_NPU_CMD`
- ❌ **禁止**手动实现 `make_acl_tensor` / `dtype_to_acl` / `destroy_acl_tensor`，**必须**使用 `torch_aclnn_helper.h` 的 `ConvertType` 体系
- ❌ **禁止**使用 `.asc` 文件扩展名，**必须**使用 `.cpp`
- ❌ **禁止**使用 ASC 编译器编译绑定文件，**必须**使用纯 C++ 编译
- ❌ **禁止**使用 `TORCH_LIBRARY(custom_ops, m)` 注册，**必须**使用 `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)`
- ❌ **禁止**编译时链接 `libcust_opapi.so`，`EXEC_NPU_CMD` 运行时动态加载
- ❌ **禁止**使用 `import ascend_kernel` 方式加载算子，**必须**使用 `torch.ops.load_library()` + `torch.ops.npu.<op_name>()`
- ❌ **禁止**在精度测试脚本中不设置 `ASCEND_RT_VISIBLE_DEVICES`
