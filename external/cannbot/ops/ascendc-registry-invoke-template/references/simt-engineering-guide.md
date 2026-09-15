# SIMT 工程开发差异指南

> SIMD/SIMT 共享的通用工程规范（OpDef、注册流程、目录结构、CMake、编译流程等）请参考 `ascendc-registry-invoke-template` skill。本文件仅记录 **SIMT 独有的工程差异点**。

---

## SIMT Kernel 开发独有规范

### SIMT 专属约束（6 条）

1. `__simt_vf__` 修饰的函数内所调用的自定义子函数必须带 `__simt_callee__` 修饰（SIMD 对应 `__simd_callee__`，不可混用）
2. 线程数必须是 `constexpr` 编译期常量，`LAUNCH_BOUND(N)` 与 `Simt::Dim3(N)` 必须使用同一个常量
3. DCache 一致性：Scalar 单元写 GM 后需 `DataCacheCleanAndInvalid` 刷新（SIMD 无此问题）
4. 纯 SIMT 算子直接使用 `GM_ADDR` 参数，避免 GlobalTensor 中转（SIMD 需要 GlobalTensor）
5. 禁止在 `Simt::Dim3(...)` 中使用从 tilingData 读取的变量（必须是 constexpr 或字面量）
6. VF 参数 size 控制在 28×32bit 以内，超出则将不常用参数存到 UB buffer 传入 `__ubuf__` 指针

### SIMT kernel 模板

```cpp
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "{op_name}_tiling_data.h"
#include "{op_name}_tiling_key.h"

namespace Ns{OpName} {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM)
inline void Op{OpName}Simt(...) {
    // SIMT kernel 实现
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const {OpName}TilingData* tilingData) {
    AscendC::Simt::VF_CALL<Op{OpName}Simt<T>>(AscendC::Simt::Dim3(THREAD_NUM), ...);
}

} // namespace Ns{OpName}
```

---

## SIMT Tiling 开发独有规范

### VF 线程数不在 tiling 侧设置

- 线程数是 kernel 侧的编译期常量（`constexpr`）
- TilingData 结构体中**禁止包含** `threadNum`/`blockDimX`/`blockDimY` 等线程数相关字段

### LocalMemory 设置（SIMT 独有）

SIMT 算子需要为 DCache 保留空间，通过 `SetLocalMemorySize` 将 UB 可用空间设置为 `ubsize - DCACHE_SIZE`：

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;
uint64_t ubsize = 256 * 1024;
context->SetLocalMemorySize(ubsize - DCACHE_SIZE);
```

### Workspace 设置（多核交换数据时）

```cpp
context->SetScheduleMode(1);  // 同步模式（SIMT 多核数据交换必须）
size_t workspaceSize = ...;
context->SetWorkspaceSize(workspaceSize);
```

---

## SIMT Include 路径规范

### op_kernel/*.cpp（SIMT 独有路径差异）

```cpp
#include "arch35/{op_name}_simt.h"  // 必须包含，会级联包含 tiling_data.h 和 tiling_key.h
```

- 必须使用 `"arch35/xxx.h"` 格式
- 禁止使用 `"xxx.h"` 格式（编译找不到）

### op_kernel/arch35/*_simt.h

```cpp
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"  // SIMT 独有，必须在 namespace 外部
#include "{op_name}_tiling_data.h"
#include "{op_name}_tiling_key.h"
```

- `simt_api/asc_simt.h` 必须在 namespace 外部包含（SIMD 无此头文件）
- 同目录下的 `.h` 文件直接使用文件名

### 常见错误

| 错误 | 正确 |
|------|------|
| `#include "trilu_simt.h"` | `#include "arch35/trilu_simt.h"` |
| `simt_api/asc_simt.h` 在 namespace 内 | `simt_api/asc_simt.h` 在 namespace 外 |

---

## SIMT 精度调试（独有方式）

| 位置 | 方法 | 代码 |
|------|------|------|
| SIMT VF 内部 | `AscendC::Simt::printf` | `AscendC::Simt::printf("simt value %ld", value);` |
| SIMT VF 外部 | `PRINTF` 宏 | `PRINTF("simt value %ld", value);` |

> Tiling 侧的 `std::cout` 与 SIMD 相同，不在本文件范围。

---

## SIMT Linter 专属规则

以下规则仅适用于 SIMT kernel 代码，与 SIMD 通用规则不同：

| 规则 | 检查内容 | 级别 |
|------|----------|------|
| R06 | SIMT kernel 属性（`__simt_vf__` `__aicore__` LAUNCH_BOUND） | FAIL |
| R07 | 必须使用 VF_CALL（禁止 cce_parallel） | FAIL |
| R12 | 线程遍历（GetThreadIdx + GetThreadNum） | WARN |
| R14 | LAUNCH_BOUND 范围（32~2048，建议 32 倍数） | FAIL |
| R16 | 禁止项（cce_parallel、参数传结构体、VF参数超28×32bit） | FAIL |
| R17 | `__simt_callee__` 属性（SIMD callee 不可用于 SIMT VF） | FAIL |
| R18 | 函数体行数（单函数不超过 50 行） | FAIL |
| R19 | Dim3 编译期常量（禁止从 tiling 获取） | FAIL |

> SIMD/SIMT 共享的通用 Linter 规则（目录结构、include、OpDef 等）请参考 `ascendc-registry-invoke-template` skill。