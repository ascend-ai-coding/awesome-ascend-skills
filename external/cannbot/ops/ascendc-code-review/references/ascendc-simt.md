# SIMT code review

```
<适用>
语言: C++
侧别: Kernel
默认启用: true

领域: true
触发: Simt, simt_api/asc_simt.h
</适用>
```

## 快速索引

| 规范编号 | 规范名称 | 类别 | 严重级别 | 适用范围 |
|---------|---------|------|---------|---------|
| SIMT-1 | 禁止使用 Simt::GetThreadNum | API转换 | 高 | [适用: Kernel] |
| SIMT-2 | 禁止使用 Simt::GetThreadIdx | API转换 | 高 | [适用: Kernel] |
| SIMT-3 | 禁止使用 Simt::GetBlockIdx | API转换 | 高 | [适用: Kernel] |
| SIMT-4 | 禁止使用 Simt::GetBlockNum | API转换 | 高 | [适用: Kernel] |
| SIMT-5 | 禁止使用 Simt::VF_CALL/Dim3 | API转换 | 高 | [适用: Kernel] |
| SIMT-6 | 禁止使用 Simt::Min/Max（float） | API转换 | 高 | [适用: Kernel] |
| SIMT-7 | 禁止使用 Simt::Min/Max（int） | API转换 | 高 | [适用: Kernel] |
| SIMT-8 | 禁止使用 Simt::Floor/Ceil | API转换 | 高 | [适用: Kernel] |
| SIMT-9 | 禁止使用 Simt::Abs | API转换 | 高 | [适用: Kernel] |
| SIMT-10 | 禁止使用 Simt::AtomicAdd/AtomicSub | API转换 | 高 | [适用: Kernel] |
| SIMT-11 | 禁止使用 Simt::ThreadBarrier | API转换 | 高 | [适用: Kernel] |
| SIMT-12 | 禁止使用 Simt::WarpShflSync | API转换 | 中 | [适用: Kernel] |
| SIMT-13 | 禁止使用 Simt::IsNan | API转换 | 高 | [适用: Kernel] |
| SIMT-14 | 禁止使用 Simt::IsFinite | API转换 | 高 | [适用: Kernel] |
| SIMT-15 | 禁止使用 Simt::IsInf | API转换 | 高 | [适用: Kernel] |
| SIMT-16 | simt_api/asc_simt.h 头文件必须包含在 namespace 外部 | 头文件 | 高 | [适用: Kernel] |
| SIMT-17 | 禁止定义与 C 风格 API 变量同名的局部变量 | 变量 | 高 | [适用: Kernel] |
| SIMT-18 | SIMT VF 入口函数参数 size 控制在 28x32bit 以内 | API限制 | 中 | [适用: Kernel] |
| SIMT-19 | 必须使用 asc_vf_call 启动 SIMT 函数，禁止 cce::parallel_for | API使用 | 高 | [适用: Kernel] |
| SIMT-20 | SIMT 函数使用 UB 时须在 tiling 侧调用 SetLocalMemorySize | 内存管理 | 高 | [适用: Kernel] |
| SIMT-21 | LAUNCH_BOUND 线程数必须为 constexpr 常量 | 常量 | 高 | [适用: Kernel] |
| SIMT-22 | asc_vf_call 调用的函数必须带 __simt_vf__ 修饰 | API使用 | 高 | [适用: Kernel] |
| SIMT-23 | __simt_vf__ 内调用的子函数必须带 __simt_callee__ 修饰 | API使用 | 高 | [适用: Kernel] |

---

## SIMT-1: 禁止使用 Simt::GetThreadNum

**严重级别**：高

### 问题描述

`Simt::GetThreadNum()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `blockDim.x/y/z`。

不带 `Simt::` 前缀的 `GetThreadNum()` 是 Ascend C API，无需转换。

**维度对应规则**：
- `GetThreadNum<0>()` → `blockDim.x` （x 维度）
- `GetThreadNum<1>()` → `blockDim.y` （y 维度）
- `GetThreadNum<2>()` → `blockDim.z` （z 维度）

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t threadNum = AscendC::Simt::GetThreadNum<0>();
uint32_t threadNum = Simt::GetThreadNum();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（根据维度选择）
uint32_t threadNumX = blockDim.x;  // x维度
uint32_t threadNumY = blockDim.y;  // y维度
uint32_t threadNumZ = blockDim.z;  // z维度

// ✅ 正确：不带 Simt:: 前缀的 GetThreadNum 无需转换
uint32_t threadNum = GetThreadNum();  // Ascend C API，保持原样
```

### 注意事项

只检视实际代码，排除注释中的 API。转换时需根据模板参数 `<维度>` 选择正确的成员（.x/.y/.z）。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 C 风格替代 API 拼写与转换规则与本条一致；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "GetThreadNum" <file>
```

**步骤2 — 判定**：命中的是带 `Simt::` 前缀的 C++ 风格调用即 FAIL；不带 `Simt::` 前缀的 Ascend C 调用 PASS。

---

## SIMT-2: 禁止使用 Simt::GetThreadIdx

**严重级别**：高

### 问题描述

`Simt::GetThreadIdx()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `threadIdx.x/y/z`。

不带 `Simt::` 前缀的 `GetThreadIdx()` 是 Ascend C API，无需转换。

**维度对应规则**：
- `GetThreadIdx<0>()` → `threadIdx.x` （x 维度）
- `GetThreadIdx<1>()` → `threadIdx.y` （y 维度）
- `GetThreadIdx<2>()` → `threadIdx.z` （z 维度）

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
uint32_t threadIdx = Simt::GetThreadIdx();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（根据维度选择）
uint32_t tidX = threadIdx.x;  // x维度
uint32_t tidY = threadIdx.y;  // y维度
uint32_t tidZ = threadIdx.z;  // z维度

// ✅ 正确：不带 Simt:: 前缀的 GetThreadIdx 无需转换
uint32_t threadIdx = GetThreadIdx();  // Ascend C API，保持原样
```

### 注意事项

只检视实际代码，排除注释中的 API。转换时需根据模板参数 `<维度>` 选择正确的成员（.x/.y/.z）。避免变量名与 `threadIdx` 冲突（详见 SIMT-17），建议使用 `tid` 或 `thread_id`。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 C 风格替代 API 拼写与转换规则与本条一致；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "GetThreadIdx" <file>
```

**步骤2 — 判定**：命中的是带 `Simt::` 前缀的 C++ 风格调用即 FAIL；不带 `Simt::` 前缀的 Ascend C 调用 PASS。

---

## SIMT-3: 禁止使用 Simt::GetBlockIdx

**严重级别**：高

### 问题描述

`Simt::GetBlockIdx()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `blockIdx.x`。

不带 `Simt::` 前缀的 `GetBlockIdx()` 是 Ascend C API，无需转换。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t blockIdx = Simt::GetBlockIdx();
uint32_t blockIdx = AscendC::Simt::GetBlockIdx();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（Simt::GetBlockIdx 转换后）
uint32_t bid = blockIdx.x;

// ✅ 正确：不带 Simt:: 前缀的 GetBlockIdx 无需转换
uint32_t blockIdx = GetBlockIdx();  // Ascend C API，保持原样
```

### 注意事项

只检视实际代码，排除注释中的 API。避免变量名与 `blockIdx` 冲突（详见 SIMT-17），建议使用 `bid` 或 `block_id`。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 C 风格替代 API 拼写与转换规则与本条一致；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "GetBlockIdx" <file>
```

**步骤2 — 判定**：命中的是带 `Simt::` 前缀的 C++ 风格调用即 FAIL；不带 `Simt::` 前缀的 Ascend C 调用 PASS。

---

## SIMT-4: 禁止使用 Simt::GetBlockNum

**严重级别**：高

### 问题描述

`Simt::GetBlockNum()` 是 C++ 风格 API（带 `Simt::` 前缀），应转换为 C 风格 API `gridDim.x/y/z`。

不带 `Simt::` 前缀的 `GetBlockNum()` 是 Ascend C API，无需转换。

**维度对应规则**：
- `GetBlockNum<0>()` 或 `GetBlockNum()` → `gridDim.x` （x 维度，默认）
- `GetBlockNum<1>()` → `gridDim.y` （y 维度）
- `GetBlockNum<2>()` → `gridDim.z` （z 维度）

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（带 Simt:: 前缀）
uint32_t blockNum = AscendC::Simt::GetBlockNum();
uint32_t blockNum = Simt::GetBlockNum();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API（根据维度选择）
uint32_t gridDimX = gridDim.x;  // x维度
uint32_t gridDimY = gridDim.y;  // y维度
uint32_t gridDimZ = gridDim.z;  // z维度

// ✅ 正确：不带 Simt:: 前缀的 GetBlockNum 无需转换
uint32_t blockNum = GetBlockNum();  // Ascend C API，保持原样
```

### 注意事项

只检视实际代码，排除注释中的 API。转换时需根据模板参数 `<维度>` 选择正确的成员（.x/.y/.z）。避免变量名与 `gridDim`、`blockDim` 冲突（详见 SIMT-17），建议使用 `grid_size` 或 `block_size`。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 C 风格替代 API 拼写与转换规则与本条一致；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "GetBlockNum" <file>
```

**步骤2 — 判定**：命中的是带 `Simt::` 前缀的 C++ 风格调用即 FAIL；不带 `Simt::` 前缀的 Ascend C 调用 PASS。

---

## SIMT-5: 禁止使用 Simt::VF_CALL/Dim3

**严重级别**：高

### 问题描述

`Simt::VF_CALL` 和 `Simt::Dim3` 是 C++ 风格 API，应转换为 C 风格 API `asc_vf_call` 和 C 风格类型 `dim3`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
Simt::VF_CALL<KernelClass>(Simt::Dim3{32, 1, 1}, Simt::Dim3{1, 1, 1}, ...);
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API 和 C 风格类型
asc_vf_call<KernelClass>(dim3{32, 1, 1}, dim3{1, 1, 1}, ...);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `asc_vf_call` 与 `dim3` 的替代写法；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "VF_CALL\|Simt::Dim3" <file>
```

---

## SIMT-6: 禁止使用 Simt::Min/Max（float）

**严重级别**：高

### 问题描述

`Simt::Min` 和 `Simt::Max` 用于 float 类型时，应转换为标准数学函数 `fminf` 和 `fmaxf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（float 类型）
float minVal = Simt::Min(a, b);
float maxVal = Simt::Max(a, b);
```

### 正确示例

```cpp
// ✅ 正确：使用标准数学函数（float 类型）
float minVal = fminf(a, b);
float maxVal = fmaxf(a, b);
```

### 注意事项

需区分参数类型：float 类型使用 `fminf/fmaxf`，int 类型使用 `min/max`（详见 SIMT-7）。只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `fminf/fmaxf` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "Simt::Min\|Simt::Max" <file>
```

**步骤2 — 判定**：命中后确认参数类型为 float 时判本条；int 类型转判 SIMT-7。

---

## SIMT-7: 禁止使用 Simt::Min/Max（int）

**严重级别**：高

### 问题描述

`Simt::Min` 和 `Simt::Max` 用于 int/int64_t 类型时，应转换为标准函数 `min` 和 `max`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API（int 类型）
int minVal = Simt::Min(a, b);
int64_t maxVal = Simt::Max(a, b);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数（int 类型）
int minVal = min(a, b);
int64_t maxVal = max(a, b);
```

### 注意事项

需根据参数类型选择正确的替代函数：int 类型使用 `min/max`，float 类型使用 `fminf/fmaxf`（详见 SIMT-6）。只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `min/max` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "Simt::Min\|Simt::Max" <file>
```

**步骤2 — 判定**：命中后确认参数类型为 int/int64_t 时判本条；float 类型转判 SIMT-6。

---

## SIMT-8: 禁止使用 Simt::Floor/Ceil

**严重级别**：高

### 问题描述

`Simt::Floor` 和 `Simt::Ceil` 用于 float 类型时，应转换为标准数学函数 `floorf` 和 `ceilf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
float floorVal = Simt::Floor(value);
float ceilVal = Simt::Ceil(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准数学函数
float floorVal = floorf(value);
float ceilVal = ceilf(value);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `floorf/ceilf` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "Simt::Floor\|Simt::Ceil" <file>
```

---

## SIMT-9: 禁止使用 Simt::Abs

**严重级别**：高

### 问题描述

`Simt::Abs` 用于 float 类型时，应转换为标准数学函数 `fabsf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
float absVal = Simt::Abs(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准数学函数
float absVal = fabsf(value);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `fabsf` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "Simt::Abs" <file>
```

---

## SIMT-10: 禁止使用 Simt::AtomicAdd/AtomicSub

**严重级别**：高

### 问题描述

`Simt::AtomicAdd` 和 `Simt::AtomicSub` 是 C++ 风格 API，应转换为 C 风格 API `asc_atomic_add` 和 `asc_atomic_sub`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
Simt::AtomicAdd<int64_t>(&counter, value);
Simt::AtomicSub<int>(&counter, value);
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API
asc_atomic_add(&counter, value);
asc_atomic_sub(&counter, value);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `asc_atomic_add/asc_atomic_sub` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "AtomicAdd\|AtomicSub" <file>
```

**步骤2 — 判定**：命中的是带 `Simt::` 前缀的 C++ 风格调用即 FAIL。

---

## SIMT-11: 禁止使用 Simt::ThreadBarrier

**严重级别**：高

### 问题描述

`Simt::ThreadBarrier()` 是 C++ 风格 API，应转换为 C 风格同步函数 `asc_syncthreads()`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
Simt::ThreadBarrier();
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格同步函数
asc_syncthreads();
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `asc_syncthreads` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "ThreadBarrier" <file>
```

---

## SIMT-12: 禁止使用 Simt::WarpShflSync

**严重级别**：中

### 问题描述

`Simt::WarpShflSync` 是 C++ 风格 API，应转换为 C 风格 API `asc_shfl`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
int value = Simt::WarpShflSync(mask, var, lane);
```

### 正确示例

```cpp
// ✅ 正确：使用 C 风格 API
int value = asc_shfl(mask, var, lane);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `asc_shfl` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "WarpShflSync" <file>
```

---

## SIMT-13: 禁止使用 Simt::IsNan

**严重级别**：高

### 问题描述

`Simt::IsNan` 是 C++ 风格 API，应转换为标准函数 `isnan`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
bool result = Simt::IsNan(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数
bool result = isnan(value);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `isnan` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "IsNan" <file>
```

---

## SIMT-14: 禁止使用 Simt::IsFinite

**严重级别**：高

### 问题描述

`Simt::IsFinite` 是 C++ 风格 API，应转换为标准函数 `isfinite`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
bool result = Simt::IsFinite(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数
bool result = isfinite(value);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `isfinite` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "IsFinite" <file>
```

---

## SIMT-15: 禁止使用 Simt::IsInf

**严重级别**：高

### 问题描述

`Simt::IsInf` 是 C++ 风格 API，应转换为标准函数 `isinf`。

### 错误示例

```cpp
// ❌ 禁止：使用 C++ 风格 API
bool result = Simt::IsInf(value);
```

### 正确示例

```cpp
// ✅ 正确：使用标准函数
bool result = isinf(value);
```

### 注意事项

只检视实际代码，排除注释中的 API。

### 专属检视方法

**步骤0 — 核对 API**：先通过 `/ascendc-docs-search` 查阅 `$ASC_DEVKIT_DIR` 中 `simt_api/asc_simt.h`，确认 `isinf` 替代规则；头文件与条例有差异时以最新头文件为准并记录差异。

**步骤1 — grep 定位**：

```bash
grep -n "IsInf" <file>
```

---

## SIMT-16: simt_api/asc_simt.h 头文件必须包含在 namespace 外部

**严重级别**：高

### 问题描述

使用 SIMT C 风格 API 时，需要包含头文件 `simt_api/asc_simt.h`，且必须放在 namespace 外部（与 `kernel_operator.h` 同级）。禁止在 namespace 内部或 `using namespace AscendC;` 之后包含该头文件。

### 错误示例

```cpp
// ❌ 错误：头文件在 namespace 内部
namespace MyKernel {
using namespace AscendC;
#include "simt_api/asc_simt.h"  // 错误位置
// ...
}
```

### 正确示例

```cpp
// ✅ 正确：头文件在 namespace 外部
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"  // 必须在此位置

namespace MyKernel {
using namespace AscendC;
// ...
}
```

### 注意事项

头文件位置错误会导致符号未定义或编译失败。只检视实际代码，排除注释中的内容。

### 专属检视方法

**步骤1 — 定位头文件包含**：

```bash
# 检查头文件是否包含
grep -n "simt_api/asc_simt.h" <file>

# 检查头文件位置（必须在 namespace 外部）
grep -B5 -A5 "namespace" <file> | grep "simt_api"
```

**步骤2 — 判定**：`simt_api/asc_simt.h` 出现在 `namespace` 声明之后即 FAIL。

---

## SIMT-17: 禁止定义与 C 风格 API 变量同名的局部变量

**严重级别**：高

### 问题描述

转换到 C 风格 API 后，`threadIdx`、`blockIdx`、`blockDim`、`gridDim` 为内置变量名。局部变量不能与其重名，否则产生命名冲突或遮蔽内置语义。

### 错误示例

```cpp
// ❌ 错误：变量名与 C 风格 API 变量冲突
uint32_t threadIdx = 0;  // 与 threadIdx.x 冲突
uint32_t blockIdx = 0;   // 与 blockIdx.x 冲突
uint32_t blockDim = 0;   // 与 blockDim.x 冲突
uint32_t gridDim = 0;    // 与 gridDim.x 冲突
```

### 正确示例

```cpp
// ✅ 正确：使用其他变量名
uint32_t tid = 0;        // thread id
uint32_t bid = 0;        // block id
uint32_t block_size = blockDim.x;  // 明确使用 .x 成员
uint32_t grid_size = gridDim.x;    // 明确使用 .x 成员
```

### 注意事项

建议改用 `tid`/`thread_id`、`bid`/`block_id`、`grid_size`/`block_size` 等不与内置变量冲突的命名。只检视实际代码，排除注释中的内容。

### 专属检视方法

**步骤1 — 冲突检测**：

```bash
grep -n "uint32_t threadIdx\|uint32_t blockIdx\|uint32_t blockDim\|uint32_t gridDim" <file>
```

**步骤2 — 判定**：检测到同名局部变量定义即 FAIL，修复方式见正确示例。

---

## SIMT-18: SIMT VF 入口函数参数 size 控制在 28x32bit 以内

**严重级别**：中

### 问题描述

SIMT VF 入口函数的参数 size 控制在 **28x32bit（112 字节）** 以内。超过该限制会导致编译或运行失败。

**优化原则**（超限时的处理顺序）：
1. VF 参数能用模板参数传递的，尽可能用模板参数传递
2. VF 参数能在 SIMT VF 内部计算得到的，就不要作为参数传递
3. 若参数 size 仍超过 28x32bit：
   - 将 VF 内部频繁使用的参数作为入参
   - 其余参数存到 UB 的 buffer 中
   - 将 buffer 地址（`__ubuf__` 指针）作为入参

### 错误示例

```cpp
// ❌ 错误：入口函数参数过多，超过 28x32bit（112 字节）
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t a, uint64_t b, __gm__ float* x, __gm__ float* y, __gm__ float* z,
    uint64_t c1, uint64_t c2, uint64_t c3, uint64_t c4, uint64_t c5,
    uint64_t c6, uint64_t c7, uint64_t c8, uint64_t c9, uint64_t c10,
    uint64_t c11, uint64_t c12)  // 累计超过 112 字节
{
    // ...
}
```

### 正确示例

```cpp
// ✅ 正确：常用参数直接传，不常用的放 UB
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t count,        // 8 字节
    __gm__ T* x,           // 8 字节
    __gm__ T* y,           // 8 字节
    __ubuf__ int64_t* ub   // 8 字节 - UB 指针传其余参数
) {
    int64_t dimSize = ub[0];      // 从 UB 读取
    int64_t stride = ub[1];
    // ...
}
```

### 注意事项

| 类型 | 占用字节数 |
|------|-----------------|
| int32_t / uint32_t / float | 4 |
| int64_t / uint64_t / double | 8 |
| `__gm__ T*` / `__ubuf__ T*` | 8 |
| half / bfloat16_t | 2 |
| bool | 1 |

### 专属检视方法

**步骤1 — 定位入口函数**：

```bash
grep -n "__simt_vf__" <file>
```

**步骤2 — 计算参数 size**：按问题描述中的类型占用表逐参数累加。

**步骤3 — 判定**：参数总 size 大于 112 字节则记录 review 意见，修复方式参考问题描述优化原则。

---

## SIMT-19: 必须使用 asc_vf_call 启动 SIMT 函数，禁止 cce::parallel_for

**严重级别**：高

### 问题描述

启动 SIMT VF 函数必须使用 `AscendC::Simt::asc_vf_call`（C 风格 `asc_vf_call`），禁止使用 `cce::parallel_for` 启动 SIMT 函数。

### 错误示例

```cpp
// ❌ 错误：使用 cce::parallel_for 调用
cce::parallel_for(threadNums, [=, *this]() LAUNCH_BOUND(BOUND_THREAD_NUM) {simt_func(__VA_ARGS__);});
```

### 正确示例

```cpp
// ✅ 正确：使用 asc_vf_call 调用
asc_vf_call<XXXVF>(para1, para2, ...);
```

### 注意事项

只检视实际代码，排除注释中的内容。

### 专属检视方法

**步骤1 — 定位启动方式**：

```bash
grep -n "cce::parallel_for" <file>
```

**步骤2 — 判定**：SIMT kernel 文件中存在 `cce::parallel_for` 即 FAIL，修复方式参考正确示例。

---

## SIMT-20: SIMT 函数使用 UB 时须在 tiling 侧调用 SetLocalMemorySize

**严重级别**：高

### 问题描述

在 SIMT 函数内部直接使用 UB（`TBuf<T>`）时，必须在 tiling 侧通过 `context->SetLocalMemorySize(...)` 正确设置 UB 大小，否则 SIMT 函数访问 UB 会越界或非法。

### 错误示例

```cpp
// ❌ 错误：tiling 侧未调用 SetLocalMemorySize 设置 UB 大小
// 但 SIMT kernel 中直接使用了 UB：
TBuf<TPosition::VECCALC> ubTensor = GetTPLBuffer<TPosition::VECCALC>(...);
```

### 正确示例

```cpp
// ✅ 正确：在 tiling 侧调用 SetLocalMemorySize 设置 UB
context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
```

### 注意事项

本条检查跨侧联动：SIMT kernel 侧发现 UB 使用后，需回查 tiling 侧（`op_host/`）是否调用 `SetLocalMemorySize`。

### 专属检视方法

**步骤1 — 定位 kernel 侧 UB 使用**：

```bash
grep -n "TBuf<TPosition::VECCALC>\|TBuf<>" <file>
```

**步骤2 — 判定分支**：kernel 中无 UB 使用，则检查结束；有使用则继续步骤3。

**步骤3 — 定位 tiling 侧设置**：

```bash
grep -n "SetLocalMemorySize" <operator_dir>/op_host/
```

**步骤4 — 判定**：未查找到 `SetLocalMemorySize` 即 FAIL，修复方式参考正确示例。

---

## SIMT-21: LAUNCH_BOUND 线程数必须为 constexpr 常量

**严重级别**：高

### 问题描述

`LAUNCH_BOUND(THREAD_NUM)` 中的线程数 `THREAD_NUM` 必须定义为 `constexpr` 常量。使用非常量表达式会导致编译失败或线程绑定异常。

### 错误示例

```cpp
// ❌ 错误：未使用 constexpr 定义线程数
uint32_t THREAD_NUM = 1024;
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void TestVF() {}
```

### 正确示例

```cpp
// ✅ 正确：使用 constexpr 定义线程数
constexpr uint32_t THREAD_NUM = 1024;
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void TestVF() {}
```

### 注意事项

只检视实际代码，排除注释中的内容。

### 专属检视方法

**步骤1 — 定位 LAUNCH_BOUND**：

```bash
grep -n "LAUNCH_BOUND\|__launch_bounds__" <file>
```

**步骤2 — 判定**：`LAUNCH_BOUND` 后 `()` 中为字面常量即 PASS；为变量 `THREAD_NUM` 时继续步骤3。

**步骤3 — 查变量定义**：

```bash
grep -n "${THREAD_NUM}" <file>
```

**步骤4 — 判定**：变量定义缺少 `constexpr` 关键字即 FAIL，修复方式参考正确示例。

---

## SIMT-22: asc_vf_call 调用的函数必须带 __simt_vf__ 修饰

**严重级别**：高

### 问题描述

`AscendC::Simt::asc_vf_call`（C 风格 `asc_vf_call`）所调用的函数，其声明和定义中必须带有 `__simt_vf__` 修饰符。

### 错误示例

```cpp
// ❌ 错误：未使用 __simt_vf__ 标识 asc_vf_call 调用的函数
inline void TestVF(__gm__ float* param1, __gm__ float* param2, __gm__ float* param3)
{
    // TODO
}
asc_vf_call<TestVF>(dim3{1024}, param1, param2, param3);
```

### 正确示例

```cpp
// ✅ 正确：使用 __simt_vf__ 标识 asc_vf_call 调用的函数
__simt_vf__ inline void TestVF(__gm__ float* param1, __gm__ float* param2, __gm__ float* param3)
{
    // TODO
}
asc_vf_call<TestVF>(dim3{1024}, param1, param2, param3);
```

### 注意事项

只检视实际代码，排除注释中的内容。

### 专属检视方法

**步骤1 — 定位 asc_vf_call 调用**：

```bash
grep -n "asc_vf_call" <file>
```

**步骤2 — 提取函数名**：取 `asc_vf_call` 后 `<>` 中的 SIMT VF 入口函数名 SIMT_FUNC。

**步骤3 — 查函数定义**：

```bash
grep -n "${SIMT_FUNC}" <file>
```

**步骤4 — 判定**：函数定义中无 `__simt_vf__` 修饰符即 FAIL，修复方式参考正确示例。

---

## SIMT-23: __simt_vf__ 内调用的子函数必须带 __simt_callee__ 修饰

**严重级别**：高

### 问题描述

`__simt_vf__` 修饰的函数内所调用的自定义子函数，必须带有 `__simt_callee__` 修饰符。缺失会导致 SIMT VF 调用链链接错误。

### 错误示例

```cpp
// ❌ 错误：未使用 __simt_callee__ 标识 SIMT VF 内部调用的函数
inline void TestFunc()
{
    // TODO
}
__simt_vf__ inline void TestVF(__gm__ float* param1, __gm__ float* param2, __gm__ float* param3)
{
    TestFunc();
}
asc_vf_call<TestVF>(dim3{1024}, param1, param2, param3);
```

### 正确示例

```cpp
// ✅ 正确：使用 __simt_callee__ 标识 SIMT VF 内部调用的函数
__simt_callee__ inline void TestFunc()
{
    // TODO
}
__simt_vf__ inline void TestVF(__gm__ float* param1, __gm__ float* param2, __gm__ float* param3)
{
    TestFunc();
}
asc_vf_call<TestVF>(dim3{1024}, param1, param2, param3);
```

### 注意事项

只检视实际代码，排除注释中的内容。

### 专属检视方法

**步骤1 — 定位入口函数**：

```bash
grep -n "__simt_vf__" <file>
```

**步骤2 — 提取被调函数名**：获取 `__simt_vf__` 函数体内调用的自定义函数名 CalleeFunc。

**步骤3 — 查被调函数定义**：

```bash
grep -n "${CalleeFunc}" <file>
```

**步骤4 — 判定**：被调函数定义中无 `__simt_callee__` 修饰符即 FAIL，修复方式参考正确示例。
