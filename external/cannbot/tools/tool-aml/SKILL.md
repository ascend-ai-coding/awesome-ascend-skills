---
name: external-cannbot-tools-tool-aml
description: 为 oam-tools-diag 的 AICore STL 诊断模块（aicore_stl_detect）新增算子的工具。当用户提供形态（Ascend950/Ascend960DT）、算子类型（aic/aiv/mix）、算子名、aic/aiv
  kernel cce 代码，以及每个输入的 size/填充方式/dtype 和每个输出每 block 的 size 时，自动生成 ops/<算子名>/ 下的 .cpp（host
  规格+下发+注册）、.asc（kernel 下发包装）与 aic_/aiv_*.cce 文件，并完成 host so 编译与注册校验。触发场景：为 aicore_stl_detect
  模块新增、修改或校验 STL 算子。
license: CANN-2.0
original-name: tool-aml
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-14'
synced-commit: 750045df251d031bf708de9dc1614c2202eed6dd
---

# tool-aml：AICore STL 新增算子生成器

在 `src/aml/aml/aml_diagnose/aicore_stl_detect/<形态>/ops/` 下新增一个自包含算子目录。算子规格(MakeOpDesc)、下发(Launch)、静态注册都在该目录的 `.cpp` 内；kernel 下发函数在同目录 `.asc`；device 核函数在 `.cce`。CMake 用 `ops/*/*.cpp` 与 `ops/*/*.asc` 的 GLOB 自动收集，**无需改动任何公共代码或 CMakeLists**。

## 必须遵守

- 生成前必须收集齐全参数，缺一项必须先向用户确认，不得臆造。
- 目录名、算子名、各文件名、`.cpp`/`.asc` 内符号名必须一致推导，不允许出现半成品。
- kernel 侧 cce 代码由用户提供；本 skill 只负责包装与生成 host/下发骨架，**不要自行编造 device 核函数内部实现**。
- 生成后必须完成 host so 编译与注册校验；dav-9201/dav-3510 kernel `.o` 编译需在目标 CANN 包环境验证。
- 输入/输出均为“字节”语义：输入 size、输出每 block size 都以字节给出。

## 一、必须先收集的输入

1. **形态**：`950`(Ascend950 / dav-3510) 或 `960DT`(Ascend960DT / dav-9201)。
2. **算子类型**：`aic` / `aiv` / `mix`。**由类型决定需要用户提供的 cce 类型与数量**：
   - `aic` → 仅需 **1 个 aic kernel cce**；
   - `aiv` → 仅需 **1 个 aiv kernel cce**；
   - `mix` → 需 **aic + aiv 两个 cce**(AIC 端与 AIV 端各一个 device 函数)。
3. **算子名**：如 `cube_covery`、`all_method_sync`。目录名 = `<类型>_<算子名>`(见命名规则)。
4. **输入列表**：每个输入声明
   - `size`：字节数；
   - `填充方式`：`fixed`(固定值) / `sequential`(递增序列) / `normal`(正态分布) / `file`(从文件读取)；
   - `dtype`：如 `fp16`/`fp32`/`fp8_e4m3`/`fp8_e5m2`/`fp4_e2m1`/`uint8`；**填充方式为 `file` 时可不声明**。映射到 `aclDataType`：`fp32→ACL_FLOAT`、`fp16→ACL_FLOAT16`、`fp8_e4m3→ACL_FLOAT8_E4M3FN`、`fp8_e5m2→ACL_FLOAT8_E5M2`、`fp4_e2m1→ACL_FLOAT4_E2M1`(半字节)、`uint8→ACL_UINT8`；
   - 各填充方式的附加参数：`fixed`→值；`sequential`→起始值/步长；`normal`→mean/stddev/可选 clamp 区间；`file`→文件名。
5. **输出列表**：每个输出声明 **每 block 的 size(字节)**（只需每块大小；总大小 = `blockDim × 每块 size`）。**完全以用户输入为准**，不由 skill 推断。
6. **下发标量参数**（如 `l2Read`/`l2Write`/`burst_len`/`l2Ctrl` 等）及下发方式（单次 / 循环多次）。

如果用户只给了 `.cce` 而没给上述规格，必须先解析并回问确认。

## 二、命名与目录规则

设算子名 `NAME`、类型 `TYPE ∈ {aic, aiv, mix}`、形态目录 `FORM = ascend950_aicore_stl | ascend960dt_aicore_stl`。

| 产物 | 路径 |
|---|---|
| 算子目录 | `<FORM>/ops/<TYPE>_<NAME>/` |
| host 实现 | `<TYPE>_<NAME>.cpp` (目录同名) |
| 下发包装 | `<TYPE>_<NAME>.asc` (目录同名) |
| AIC 核函数 | `aic_<NAME>.cce`（固定以 `aic_` 开头；仅 `TYPE==aic` 或 `mix`) |
| AIV 核函数 | `aiv_<NAME>.cce` (仅 `TYPE==aiv` 或 `mix`) |

> cce 文件名固定以 `aic_` / `aiv_` 开头（不按用户文件名），与角色一一对应。

占位符约定：`<PascalName>` = 目录名的 PascalCase（`mix_all_method_sync` → `MixAllMethodSync`），模板中函数/类型/变量名统一用它。
被 `.cpp`/`.asc` 引用的符号：
- `desc.name` = `"<TYPE>_<NAME>"`（与目录同名）；
- kernel 下发导出函数 = `AmlStlLaunch<PascalName>`，如 `AmlStlLaunchMixAllMethodSync`；
- `__global__` 包装名 = `<TYPE>_<NAME>_impl`，如 `mix_all_method_sync_impl`；
- device 函数名 = 用户 cce 里的函数名，需与 `<asc>` 内 `#include "aic_<NAME>.cce"` 一致。

## 三、形态差异表（生成时替换）

| 项 | 950 | 960DT |
|---|---|---|
| 形态目录 | `ascend950_aicore_stl` | `ascend960dt_aicore_stl` |
| namespace | `AmlAicoreStl::ascend950` | `AmlAicoreStl::ascend960dt` |
| NPU arch | `dav-3510` | `dav-9201` |
| kernel so | `libaml_stl_kernel_ascend950.so` | `libaml_stl_kernel_ascend960dt.so` |
| MIX 比例 | 1 AIC : 2 AIV，`__mix__(1, 2)` + `KERNEL_TYPE_MIX_AIC_1_2` | 1 AIC : 1 AIV，`__mix__(1, 1)` + `KERNEL_TYPE_MIX_AIC_1_1` |
| `UB_SIZE` | `0x40000` (256KB) | `0x60000` (384KB) |
| `L1_SIZE`/`L0C_SIZE` | `0x80000`/`0x40000` | `0x80000`/`0x40000` |

分组枚举：`aic`→`StlGroup::STL_GROUP_AIC`，`aiv`→`STL_GROUP_AIV`，`mix`→`STL_GROUP_MIX`。
kernel 类型宏：`aic`→`KERNEL_TYPE_AIC_ONLY`(`__global__ __aicore__`)，`aiv`→`KERNEL_TYPE_AIV_ONLY`(`__global__ __aicore__`)，`mix`→`KERNEL_TYPE_MIX_AIC_1_1`/`_1_2`(`__global__ __mix__(1,1/1,2)`)。

## 四、生成步骤

1. 建目录 `mkdir -p <FORM>/ops/<TYPE>_<NAME>`。
2. 写入用户提供的 cce（按类型 1 或 2 个）。
3. 按下方模板生成 `<TYPE>_<NAME>.cpp` 与 `<TYPE>_<NAME>.asc`，替换所有 `<...>` 占位符。
4. 运行 `check_stl_op.py` 做基础语法 + 越界自检，并把问题与修改方案反馈用户（见第七节）。
5. 编译 host so + 注册校验（见第六节）。
6. 记录 kernel `.o` 需在目标 CANN 包环境用 bisheng 验证。

### 4.1 host 实现 `.cpp` 模板

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * <保留仓库标准 Apache-2.0 头>
 */
// <TYPE>_<NAME> 算子 host 侧实现(<形态名> / <npu-arch>):
// 输入/输出规格(MakeOpDesc)与下发(Launch)自包含于本算子目录,静态初始化期注册。
// kernel 下发函数(AmlStlLaunch<PascalName>)由同目录 .asc 编译链入 kernel so。
#include "aml_aicore_stl_op.h"

#include <cstdint>
#include <vector>

#include "aml/ascend_ml.h"
#include "log/aml_log.h"

namespace {
using namespace AmlAicoreStl;
using namespace AmlAicoreStl::<role>;   // ascend950 或 ascend960dt

// kernel so 导出的下发函数签名，须与 .asc 中 extern "C" 函数完全一致。
using AmlStl<PascalName>Func = int32_t (*)(/* 入参:各输入地址、输出地址、标量参数、*/ uint32_t blockDim,
                                           aclrtStream stream);

// ---- 每个输入一个 Make<NAME>Input<X>() ----
StlInputSpec Make<NAME>Input0()
{
    StlInputSpec spec;
    spec.dtype = <ACL_DTYPE>;               // 见 dtype 映射
    spec.shape = {<...>};                   // 元素数乘积 × dtype 字节 = size
    // 按填充方式二选一填写：
    spec.mode = StlDataMode::STL_DATA_FIXED; spec.fixedValue = <v>;         // fixed
    // spec.mode = StlDataMode::STL_DATA_SEQUENTIAL; spec.seqStart = <s>; spec.seqStep = <d>;  // sequential
    // spec.mode = StlDataMode::STL_DATA_NORMAL; spec.normalMean=<>; spec.normalStddev=<>; ... // normal
    // spec.mode = StlDataMode::STL_DATA_FILE; spec.fileName = "<file.bin>"; // file(无需 dtype)
    return spec;
}

StlOpDesc Make<PascalName>Op(uint32_t blockDim)
{
    StlOpDesc desc;
    desc.name = "<TYPE>_<NAME>";
    desc.group = StlOpGroup::STL_GROUP_AIC;   // aic→AIC / aiv→AIV / mix→MIX（大写）
    desc.inputs = {Make<NAME>Input0() /*, ...*/};
    desc.outputs = {
        {static_cast<size_t>(blockDim) * <每块字节>},   // 每个输出一项；单核算子可写固定字节
    };
    desc.blockDim = blockDim;                  // 固定单核时改为 1
    return desc;
}

int Launch<PascalName>(const StlOpDesc &desc, const std::vector<void *> &inputDevs,
                        const std::vector<void *> &outputDevs, aclrtStream stream)
{
    if (/* 输入/输出非空校验 */ false) {
        AML_LOGE("<TYPE>_<NAME> invalid args.");
        return AML_ERROR_INVALID_PARAM;
    }
    auto func = reinterpret_cast<AmlStl<PascalName>Func>(ResolveKernelSymbol("AmlStlLaunch<PascalName>"));
    if (func == nullptr) {
        AML_LOGE("<TYPE>_<NAME> kernel symbol unavailable (kernel so not loaded).");
        return AML_ERROR_LIB_LOAD;
    }
    int32_t ret = func(reinterpret_cast<uint64_t>(inputDevs[0]), /* ... */ reinterpret_cast<uint64_t>(outputDevs[0]),
                       /* 标量参数 */ desc.blockDim, stream);
    if (ret != 0) {
        AML_LOGE("<TYPE>_<NAME> launch failed %d, possible software issues.", ret);
        return AML_ERROR_CASE_PROCESS;
    }
    return AML_SUCCESS;
}

const StlOpEntry g_<PascalName>Entry = {
    "<TYPE>_<NAME>",
    StlOpGroup::STL_GROUP_AIC,   // aic→AIC / aiv→AIV / mix→MIX（大写）
    Make<PascalName>Op,
    Launch<PascalName>,
};

struct StlOpRegistrar {
    StlOpRegistrar() { AmlAicoreStl::<namespace>::RegisterStlOp(g_<PascalName>Entry); }
};
static StlOpRegistrar g_<PascalName>Registrar;
}  // namespace
```

> 直接参考现有算子作为范本：
> - aic：`ascend960dt_aicore_stl/ops/aic_cube_covery/aic_cube_covery.cpp`
> - aiv：`ascend960dt_aicore_stl/ops/aiv_nddma_covery/aiv_nddma_covery.cpp`
> - mix：`ascend960dt_aicore_stl/ops/mix_all_method_sync/mix_all_method_sync.cpp`

### 五、下发包装 `<TYPE>_<NAME>.asc` 模板

```cpp
<Apache license header>

// AICore STL kernel host 下发:<TYPE>_<NAME> (<形态> / <arch>)
#include <cstdint>
#include "kernel_operator.h"
#include "acl/acl.h"
#include "../ops_include/mem_info.h"
#include "aic_<NAME>.cce"   // 仅 aic / mix
#include "aiv_<NAME>.cce"   // 仅 aiv / mix

// aic/aiv：__global__ __aicore__ void <TYPE>_<NAME>_impl(__gm__ void *src, __gm__ void *dst, <scalars>)
// mix：      __global__ __mix__(1, 1) void <TYPE>_<NAME>_impl(...)
__global__ <qualifiers> void <TYPE>_<NAME>_impl(__gm__ void *src, __gm__ void *dst /*, ...*/)
{
    KERNEL_TASK_TYPE_DEFAULT(<KERNEL_TYPE>);
    // aic: if ASCEND_IS_AIC { aic_<NAME>(...); }
    // aiv: if ASCEND_IS_AIV { aiv_<NAME>(...); }
    // mix: AIC 端调 aic_<NAME>(...dstB...)，AIV 端调 aiv_<NAME>(...dstD...)
}

extern "C" int32_t AmlStlLaunch<PascalName>(uint64_t srcAddr, uint64_t dstAddr /*, ...*/,
                                            uint32_t blockDim, aclrtStream stream)
{
    if (/* 地址/blockDim 非零校验 */ false) { return -1; }
    auto *src = reinterpret_cast<__gm__ void *>(srcAddr);
    auto *dst = reinterpret_cast<__gm__ void *>(dstAddr);
    <TYPE>_<NAME>_impl<<<blockDim, nullptr, stream>>>(src, dst /*, ...*/);
    return 0;
}
```

> 注意：`__mix__` 包装里 `AIC`/`AIV` 分支各自调用对应端 device 函数；mix 单核下发时 `blockDim` 仍传实际核数。

## 六、校验

1. **host so 编译**（在已配置的构建目录）：
   ```bash
   cmake --build <build-dir> --target aml_stl_ascend960dt   # 或 aml_stl_ascend950
   ```
2. **注册校验**：dlopen 形态 so，取 `AmlStlGetPlatformOps`，确认新算子在该 group 的算子列表中且 `outputs[].byteSize` 与预期一致（`blockDim × 每块 size`）。
3. **命名一致性自检**：`desc.name`、`AmlStlLaunch*` 导出符号、`#include` 的 cce 名、`ResolveKernelSymbol` 字符串四者一致。
4. **kernel `.o` 编译**：`.asc`+`.cce` 由 bisheng 按 `--npu-arch=<arch>` 编译，须在装有对应 CANN 包的环境验证（本地缺 dav-9201 头文件时无法本地编译，只能静态检查签名）。

## 七、生成后检查（基础语法 + 越界，必做）

生成完成后**必须**运行自检脚本，并用中文把**发现的问题 + 具体位置 + 修改方案**反馈给用户（不能只回“通过”）：

```bash
python3 scripts/check_stl_op.py --op-dir <FORM>/ops/<TYPE>_<NAME>
```

脚本检查（对照同形态兄弟算子，而非硬编码形态签名）：

- **基础语法**：括号/花括号/方括号配对。
- **命名一致性**：`.cpp` 的 `desc.name`、`ResolveKernelSymbol(...)`、`.asc` 的 `AmlStlLaunch<Name>`、`#include` 的 cce 名。
- **内在函数参数个数**：`copy_gm_to_cbuf_align_v2` / `copy_matrix_cc_to_gm` / `copy_matrix_cc_to_ub` / `copy_ubuf_to_gm_align_v2` 的参数个数须与同形态其它算子一致（950 与 960DT 签名有差异，如 `copy_matrix_cc_to_ub` 两端都是 26 个参数但语义不同）。
- **禁用模式**：960DT 上出现 `get_subblockid()`/`subblock`（1 AIC:1 AIV 无 subblock）。
- **越界**：host `desc.outputs[].byteSize / blockDim`（每 block 字节）须等于 kernel 里 `block_idx * <expr>` 的块偏移；输出字节数须为 4 的倍数。**这是历史 OOB 写 bug 的主因**（host 按 256KB/block 声明而 kernel 按 `UB_SIZE` 写）。
- **传输量估算**：`copy_ubuf_to_gm_align_v2` 的 `nBurst × burstLen` 是否超过每块输出（参考修复：`0x200 → 0x180`）。

报告格式（每个问题一段）：
1. 问题（含文件/函数名）；
2. 严重级别（ERROR/WARN）；
3. **修改方案**（给出建议代码或应改的常量/表达式）。

脚本级别说明：`[ERROR]` 必须修复后重跑；`[WARN]` 需人工确认；`[INFO]` 为需留意但可能故意（如注册名与目录名不同）。

## 八、提交前检查

- `git status --short` 只应出现新算子目录内的文件。
- 不修改任何公共头、注册表、CMakeLists（GLOB 自动收）。
- dtype/填充方式/输出块大小与用户声明逐项对齐。
- mix 算子确认 `__mix__(1,1)`(960DT) / `__mix__(1,2)`(950) 与 `KERNEL_TYPE_MIX_AIC_1_1`/`_1_2` 匹配。
