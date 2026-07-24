# Step 3: 编写 Launcher

> **定位**：Launcher 编写完整参考。CMake 和 run.sh 配置见 `references/development/step1-setup.md` §4-§5。

---

## §1 Launcher 职责

Launcher 是 host 端 C++ 程序，承担以下职责：

1. **Tiling 计算**：调用 Tiling 引擎生成 `TilingData` 结构体
2. **内存管理**：分配 host/device 缓冲区，管理生命周期
3. **Kernel 启动**：通过 `<<<gridDim, nullptr, stream>>>` 语法启动 device 端 kernel
4. **结果输出**：D2H 回拷后将计算结果写入文件

### 三种 Kernel 模式

| 模式 | 修饰符 | 典型场景 |
|------|--------|---------|
| 纯 AIC（Cube） | `__cube__` | 基础 matmul、MX 量化 matmul |
| 混合 AIC/AIV | `__mix__(aicCount, aivCount)` | matmul + vector 后融合 |
| Blaze 库直调 | `__cube__` | 复用 Blaze GemmUniversal |

---

## §2 ACL 会话管理

```cpp
constexpr int32_t deviceId = 0;
aclrtContext context = nullptr;
aclrtStream stream = nullptr;

ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(deviceId));
ACL_CHECK(aclrtCreateContext(&context, deviceId));
ACL_CHECK(aclrtCreateStream(&stream));
```

- 退出时显式调用 `aclrtDestroyStream`、`aclrtDestroyContext`、`aclrtResetDevice` 和 `aclFinalize`
- `deviceId` 通常固定为 `0`，多卡场景通过 `npu-smi info` 确认可用设备
- 运行 launcher 前必须 source CANN 环境变量

---

## §3 内存管理

### Host 内存

```cpp
uint16_t* hA = nullptr;
CHECK_COND(aclrtMallocHost((void**)&hA, sizeA) == ACL_SUCCESS, "alloc host A failed.");
std::unique_ptr<void, aclError (*)(void*)> hostA(hA, aclrtFreeHost);
```

### Device 内存

```cpp
GM_ADDR dA = nullptr;
CHECK_COND(aclrtMalloc((void**)&dA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY) == ACL_SUCCESS, "alloc device A failed.");
std::unique_ptr<void, aclError (*)(void*)> deviceA(dA, aclrtFree);
```

### 缓冲区大小计算

**ND 格式**：

```cpp
uint64_t sizeA = m * k * ELEM_BYTES;
```

**NZ 格式**（物理排列与 ND 不同，需按物理维度计算）：

```cpp
static uint64_t CalcNzSize(uint64_t dim0, uint64_t dim1, uint64_t c0, uint64_t elemBytes)
{
    uint64_t dim0Blocks = (dim0 + 15) / 16;
    uint64_t dim1Blocks = (dim1 + c0 - 1) / c0;
    return dim1Blocks * dim0Blocks * 16 * c0 * elemBytes;
}

constexpr uint64_t ELEM_BYTES = sizeof(MyDType);  // bf16/fp16: 2, fp32: 4, fp8/int8: 1
constexpr uint64_t C0 = 32 / ELEM_BYTES;          // bf16/fp16: 16, fp8/int8: 32

uint64_t sizeA = aIsNz
    ? (transA ? CalcNzSize(k, m, C0, ELEM_BYTES) : CalcNzSize(m, k, C0, ELEM_BYTES))
    : m * k * ELEM_BYTES;
```

**关键原则**：
- 所有缓冲区必须用 `std::unique_ptr` 包装
- Host 内存用 `aclrtFreeHost` 释放，Device 内存用 `aclrtFree` 释放
- Device 内存分配使用 `ACL_MEM_MALLOC_HUGE_ONLY`
- NZ 格式的 size 计算必须传入 `elemBytes` 参数，不要硬编码

---

## §4 文件 I/O

### 输入读取

```cpp
ExampleIoPaths paths = GetExampleIoPaths();

CHECK_COND(ReadExactFile(paths.inputDir + "/input_a.bin", hA, sizeA), "read A failed.");
CHECK_COND(ReadExactFile(paths.inputDir + "/input_b.bin", hB, sizeB), "read B failed.");
```

### 输出写入

```cpp
CHECK_COND(WriteFile(paths.outputDir + "/output.bin", hC, sizeC), "write output failed.");
```

### 目录约定

```
<executable_dir>/
├── input/          # 输入 .bin 文件
│   ├── input_a.bin
│   ├── input_b.bin
│   ├── input_d.bin       # epilogue 额外输入
│   ├── input_scaleA.bin  # MX 量化 Scale
│   └── input_scaleB.bin
└── output/         # 输出 .bin 文件
    └── output.bin（或 npu_out.bin）
```

---

## §5 Layout Dispatch

### conditional_t + 扁平分发（推荐模式）

**Kernel 签名**（含 CubeFormat 模板参数）：

```cpp
template <bool TransA, bool TransB, CubeFormat FormatA, CubeFormat FormatB>
__global__ __aicore__ __cube__ void my_kernel(...)
{
    using LayoutA = AscendC::Std::conditional_t<
        (FormatA == CubeFormat::NZ),
        AscendC::Std::conditional_t<TransA, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    // LayoutB 同理
}
```

**Host 端分发**（16 种组合）：

```cpp
CubeFormat fA = (aLayout == "nz") ? CubeFormat::NZ : CubeFormat::ND;
CubeFormat fB = (bLayout == "nz") ? CubeFormat::NZ : CubeFormat::ND;

if      (!transA && !transB && fA == CubeFormat::ND && fB == CubeFormat::ND)
    my_kernel<false, false, CubeFormat::ND, CubeFormat::ND><<<...>>>(...);
else if (!transA && !transB && fA == CubeFormat::ND && fB == CubeFormat::NZ)
    my_kernel<false, false, CubeFormat::ND, CubeFormat::NZ><<<...>>>(...);
// ... 共 16 行，transA/transB 各 2 值 × fA/fB 各 2 值
else
    my_kernel<true, true, CubeFormat::NZ, CubeFormat::NZ><<<...>>>(...);
```

### 简化分发（固定 layout / 固定 trans）

仅支持 ND 输入时，只需分发 transA/transB（4 种组合）：

```cpp
if      (!transA && !transB)
    my_kernel<false, false><<<...>>>(...);
else if (!transA &&  transB)
    my_kernel<false, true><<<...>>>(...);
else if ( transA && !transB)
    my_kernel<true, false><<<...>>>(...);
else
    my_kernel<true, true><<<...>>>(...);
```

### 普通 MatMul 分发范围

普通开发模板按用户需求固定 dtype，不默认生成 fp16/bf16/fp32 runtime dispatch；Launcher 只分发：

```
transA × transB × formatA × formatB = 16 种组合
```

推荐使用分层函数组织：

```
LaunchByTransA
  → LaunchByTransB
    → LaunchByALayout
      → LaunchByBLayout
```

具体示例见 `references/scenarios/basic-matmul-development.md` §4。

### MX 量化分发

MX 量化交付工程通常固定为 MXFP8 或 MXFP4；除非用户明确要求通用 demo，不默认生成 MXFP8/MXFP4 runtime dispatch。若需要同时支持两类 dtype，可在 host 端显式分发到不同模板实例。

```cpp
if (dtype == "mxfp4") {
    my_quant_kernel<transA, transB, DT_FLOAT4_E2M1, DT_FLOAT4_E2M1><<<...>>>(...);
} else {
    my_quant_kernel<transA, transB, DT_FLOAT8_E4M3FN, DT_FLOAT8_E4M3FN><<<...>>>(...);
}
```

Layout 格式的详细定义和 ND/NZ/ZN 转换方法，详见 `references/fundamentals/blaze-matmul-layout.md`。

---

## §6 Kernel Launch

```cpp
kernelName<templateArgs>
    <<<gridDim, nullptr, stream>>>(kernelArgs...);
```

- `gridDim`：启动核数，取 `tilingData.usedCoreNum`
- 第二个参数：保留，固定传 `nullptr`
- `stream`：从 `AclRtSession` 获取的 ACL stream
- 模板参数在 launch 时显式指定，不要依赖模板参数推导

---

## §7 开发验证流程

Launcher 本身只负责到 D2H 回拷 + 写输出文件。

### 验证步骤

1. **编译**：无错误、无警告
2. **运行**：不崩溃（无 segfault、无 hang）
3. **精度**：外部脚本与 CPU golden 对比（非 launcher 职责）

### 精度标准

| 数据类型 | rtol | atol |
|---------|------|------|
| fp16 | 1e-3 | 1e-3 |
| bf16 | 2e-2 | 2e-2 |
| fp32 | 1e-4 | 1e-4 |
| mxfp8 | 1e-2 | 1e-2 |
| mxfp4 | 5e-2 | 5e-2 |

fp32 建议从 `1e-4/1e-4` 作为工程验收阈值开始；如需更严精度，再单独分析 HF32、累加路径和硬件误差。

### 常见调试手段

- 检查 TilingData 字段是否合理（baseM/baseN/baseK 是否为 16 的倍数）
- 检查 buffer 大小是否正确（NZ 格式需按物理维度计算）
- 检查 kernel launch 参数（gridDim 是否等于 usedCoreNum）

---

## §8 Include 文件组织

Launcher 的 include 按 4 层分组：

| 层 | 来源 | 路径前缀 | 职责 |
|----|------|---------|------|
| 标准库 | C++ STL | `<...>` | 基础类型、容器、异常 |
| SDK | CANN Toolkit | `"acl/..."`, `"kernel_operator.h"` | ACL 运行时 + AscendC 设备 API |
| 项目 Host 侧 | 工程 `assets/op_tiling/` | `"op_tiling/..."` | tiling 引擎 |
| 项目 Kernel 侧 | 工程 `op_kernel/` | `"op_kernel/..."` | kernel 模板、blaze 组件 |

### 按需 include 决策表

| 场景 | Host 侧差异 | Kernel 侧差异 |
|------|------------|-------------|
| 普通 MatMul 单算子 | Basic MatMul TilingData / Tiling 引擎 | `op_kernel/<op>_kernel.h` 内部 include `blaze/gemm/...`；基础场景使用最小 SDK include 集 |
| MX 量化单算子 | `assets/op_tiling/quant_matmul_mx_tiling.h` | `op_kernel/<op>_kernel.h` 内部 include `blaze/gemm/...`；默认先用最小 SDK include 集，缺失再最小回补 |
| 普通 CV 融合 | fusion tiling + epilogue 额外输入 | `blaze_custom` fusion kernel + `assets/blaze_custom/epilogue/*.h` |
| MX CV 融合 | `QuantMatmulTilingData` + ScaleA/ScaleB + epilogue 额外输入 | `MxMatmulKernelFused` + blaze library MX Block/Scheduler + 自定义 Epilogue |
| Grouped MatMul | grouped tiling | `blaze_custom` group kernel |
| Grouped CV 融合 | grouped tiling + groupList + epilogue 额外输入 | `GroupMatmulKernel<..., Epilogue>` + context-based Epilogue |

---

## §9 完整 Launcher 模板

以下是普通 MatMul 单算子最小化 Launcher 骨架。Kernel 侧完整 blaze library 组装代码见 `references/scenarios/basic-matmul-development.md` §3，Launcher 只负责 tiling、内存、分发和启动。

```cpp
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "acl/acl.h"
#include "kernel_operator.h"

// ---- 项目：Host 侧 ----
#include "op_tiling/matmul/blaze_matmul_tiling.h"
#include "op_tiling/matmul/blaze_matmul_tiling_data.h"

// ---- 项目：Kernel 侧 ----
#include "op_kernel/my_matmul_kernel.h"          // 内部使用 blaze/gemm/... 组装

extern "C" void my_matmul_launch(
    aclrtStream stream,
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dBias, GM_ADDR dC,
    const MatmulTilingData tilingData,
    bool transA, bool transB,
    CubeFormat formatA, CubeFormat formatB);

int main(int argc, char* argv[])
{
    try {
        // 参数解析：普通开发模板固定 dtype，仅解析 shape/layout/trans/bias
        uint64_t m = ParsePositiveUint64(argv[1], "m");
        uint64_t k = ParsePositiveUint64(argv[2], "k");
        uint64_t n = ParsePositiveUint64(argv[3], "n");
        bool transA = ParseBool(argv[4]);
        bool transB = ParseBool(argv[5]);
        CubeFormat formatA = ParseFormat(argv[6]);  // "nd" / "nz"
        CubeFormat formatB = ParseFormat(argv[7]);
        bool hasBias = ParseBool(argv[8]);

        // [MODIFY] dtype 由用户需求固定。fp16/bf16=2，fp32=4；NZ C0 分别为 16/8。
        constexpr uint64_t ELEM_BYTES = sizeof(uint16_t);
        constexpr uint64_t C0 = 32 / ELEM_BYTES;

        uint64_t sizeA = CalcInputSize(m, k, transA, formatA, C0, ELEM_BYTES);
        uint64_t sizeB = CalcInputSize(k, n, transB, formatB, C0, ELEM_BYTES);
        uint64_t sizeBias = hasBias ? n * sizeof(float) : 0;
        uint64_t sizeC = m * n * ELEM_BYTES;

        MatmulTilingData tilingData;
        MatmulTilingSwat tilingEngine;
        tilingEngine.GetTilingData(m, n, k, ELEM_BYTES, tilingData, transA, transB,
            formatA == CubeFormat::NZ, formatB == CubeFormat::NZ, hasBias);

        aclrtContext context = nullptr;
        aclrtStream stream = nullptr;
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(0));
        ACL_CHECK(aclrtCreateContext(&context, 0));
        ACL_CHECK(aclrtCreateStream(&stream));

        // 内存分配 + 文件读取 + H2D
        // ... 见 §3 内存管理、§4 文件 I/O

        my_matmul_launch(stream, dA, dB, dBias, dC, tilingData, transA, transB, formatA, formatB);

        // D2H + 同步 + 写输出
        // ... 见 §7 验证流程
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtDestroyContext(context));
        ACL_CHECK(aclrtResetDevice(0));
        ACL_CHECK(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
```

MX 量化单算子 Launcher 仍按 `references/scenarios/mx-matmul-development.md` 使用 `QuantMatmulTilingData` 和 Scale 输入。Grouped 场景使用 `totalM` 调用对应非 grouped tiling，并额外准备 `groupList/groupNum` 作为 grouped kernel 参数；它们不写入 tiling data。MX C+V 融合转到 `references/scenarios/fusion-matmul-development.md`：host 侧仍使用 `QuantMatmulTilingSwat`，但需要同时准备 A/B/ScaleA/ScaleB、epilogue 额外输入和输出，kernel wrapper 使用 `__mix__(1, 2)` 路径并按 transA/transB 分发。`host_utils` 目录不是本 skill 的一部分，launcher 所需 helper 由工程内自行实现或直接内联。

---

## §10 构建与运行

CMake 配置详见 `references/development/step1-setup.md` §4。

---

**按需查阅**：
- `references/fundamentals/blaze-matmul-layout.md`（Layout 格式详情：ND/NZ/ZN 定义、数据生成、LayoutPtn 选择）
