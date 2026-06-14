# L1 迁移实施详细指南

## 改动 1：_def.cpp 独立配置

950 使用独立 `OpAICoreConfig`，通过 `opFile` 切换到 RegBase kernel：

```cpp
OpAICoreConfig regbaseCfg;
regbaseCfg.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("opFile.value", "算子名_apt");  // 指向 RegBase kernel
this->AICore().AddConfig("ascend950", regbaseCfg);
```

如果 950 需要扩展数据类型（如新增 FP8/HiFloat8），在独立配置中添加：

```cpp
regbaseCfg.Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
               ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
    .Format({...});
```

**配置层共性特征**：

| 共性特征 | 说明 |
|---------|------|
| 独立 OpAICoreConfig | 950 使用独立的 `OpAICoreConfig` |
| opFile.value 设置 | 设置 `opFile.value = "算子名_apt"` |
| DynamicCompileStaticFlag | 950 配置普遍开启动态编译静态化 |
| DynamicRankSupportFlag | 950 配置普遍开启动态 Rank 支持 |
| DynamicShapeSupportFlag | 950 配置普遍开启动态 Shape 支持 |

### 配置层迁移模式

#### 模式 A：独立配置 + opFile 切换

950 使用独立 `OpAICoreConfig`，通过 `opFile` 切换到 RegBase kernel 入口文件。

```cpp
auto &config910b = this->AICore().AddConfig("ascend910b");
config910b.SetOpFile("moe_init_routing");

auto &config950 = this->AICore().AddConfig("ascend950");
config950.SetOpFile("moe_init_routing_apt");
```

**适用算子**：GroupNormSwish、SwigluQuant、MoeInitRouting 等

#### 模式 B：独立配置 + 扩展数据类型

950 使用独立配置，同时扩展 FP8/HiFloat8/INT8 等量化数据类型。

```cpp
auto &config950 = this->AICore().AddConfig("ascend950");
config950.SetOpFile("kv_rms_norm_rope_cache_apt");
config950.Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8,
               ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN})
    .Format({...});
```

**适用算子**：DequantSwigluQuant（6→76 组数据类型）、KvRmsnormRopeCache（8→26 组数据类型）

## 改动 2：创建 _apt.cpp

```cpp
// 算子名_apt.cpp — RegBase kernel 入口
#include "arch35/算子名_impl.hpp"    // ← 仅 include 路径变更
#include "arch35/算子名_bf16.hpp"
#include "arch35/算子名_single.hpp"

extern "C" __global__ __aicore__ void 算子名(GM_ADDR input_gm, GM_ADDR output_gm,
                                              GM_ADDR workspace, GM_ADDR tiling) {
    // 与 910b 版本相同的逻辑
}
```

**关键差异**：
- include 路径从根目录改为 `arch35/`
- Tiling 传参可能从 `tiling` 改为 `tempTilingGm`（解析后的结构体）

## 改动 3：创建 arch35/ 目录

将根目录下的实现头文件复制到 `arch35/`，做以下微调：

| 调整项 | 910b 版本 | 950 arch35/ 版本 |
|--------|-----------|-----------------|
| BF16 条件编译 | `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` | 直接支持，移除条件编译 |
| BF16 架构保护 | `#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 \|\| __NPU_ARCH__ == 3113))` | 移除保护 |
| Tiling 传参 | `op.Init(..., tiling)` — 传原始 tiling 指针 | `op.Init(..., tempTilingGm)` — 传解析后的结构体 |

**BF16 条件编译移除示例**：

```cpp
// 910b 版本：
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
  BF16 路径
#endif

// 950 arch35/ 版本：
BF16 路径（直接支持，无需条件编译）
```

**BF16 架构保护移除示例**：

```cpp
// 910b 版本：
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
#endif

// 950 arch35/ 版本：
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
```

## 改动 4：CMakeLists.txt

```cmake
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE 算子名 ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
```

**注意**：如果 `op_host/CMakeLists.txt` 中已有 ascend950 的 `add_ops_compile_options` 分支，直接复用，不要重复添加。

## 改动 5：config/ascend950/

从 `config/ascend910b/` 复制两个文件：

```bash
mkdir -p op_host/config/ascend950
cp op_host/config/ascend910b/<算子名>_binary.json op_host/config/ascend950/
cp op_host/config/ascend910b/<算子名>_simplified_key.ini op_host/config/ascend950/
```

**binary.json**：定义算子的二进制 kernel 编译信息（输入输出数据类型、格式等）。
**simplified_key.ini**：定义 opc 工具编译时的 `--simplified_key_mode` 选项值。

---

## 编译环境注意事项

### bisheng 脚本 shebang 问题

950 编译时，构建系统会在 `build/gen_bisheng_dir/` 下自动生成 `bisheng` 脚本：

```bash
ccache_args="/usr/bin/ccache <cann_path>/bin/bisheng"
args=$@
eval "${ccache_args} $args"
```

**问题**：该脚本缺少 `#!/bin/bash` shebang 行，导致 `Exec format error`。

**修复**：在首行添加 `#!/bin/bash`：

```bash
#!/bin/bash
ccache_args="/usr/bin/ccache <cann_path>/bin/bisheng"
args=$@
eval "${ccache_args} $args"
```

**注意**：每次 `rm -rf ./build` 后需重新修复，因为该文件是自动生成的。

### 编译命令格式

```bash
source <cann_path>/set_env.sh
cd ops-transformer
rm -rf ./build
bash build.sh --pkg --ops=<算子名> --soc=ascend950
```

**注意**：`--soc` 是双横线，不是 `-soc`。

### 日志落盘与错误定位

```bash
bash build.sh --pkg --ops=<算子名> --soc=ascend950 2>&1 | tee build.log
# 搜索错误
grep -n "[Ee]rror" build.log
```

重点关注：
- `[ERROR] TBE` — TBE 编译器错误（通常是 kernel 编译问题）
- `OSError: [Errno 8]` — bisheng 脚本 shebang 问题
- `gmake: *** [Makefile:156: all] Error 2` — 顶层编译失败，需查看具体子错误
- `Error 137` — 进程被 kill（通常是 OOM）

---

## L4 升级信号指引

当 L1 迁移不足以覆盖 950 适配需求时，需升级到 L4（Tiling 适配 + 数据类型扩展）。以下为判断信号：

| 升级信号 | 判断依据 | 说明 |
|---------|---------|------|
| Tiling 需要 `IsRegbaseSocVersion` 判断 | Tiling 中需根据芯片架构动态调整 UB 可用空间 | 950 SIMT DCache 复用 UB，必须预留 40KB |
| UB 预留不足 | 当前预留值 < `SIMT_UB_SIZE_BYTE`(40960) | 使用 `UB_REVERSE`/`UB_RESERVED_BYTE` 等自定义常量（如 1024）均不合规 |
| 950 需要扩展数据类型 | 需新增 FP8/HiFloat8/INT8 量化支持 | 量化类算子（如 DequantSwigluQuant、KvRmsnormRopeCache）数据类型组合大幅扩展 |
| Tiling 结构需要重构 | 950 使用全新 Tiling Key 体系 | 如 MoeGatingTopKSoftmax 使用 `TILING_KEY_IS(10000)`/`TILING_KEY_IS(20000)` 等新 Key |
| 涉及 Subnormal 处理 | 算子使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal 等 API | 351x 默认不支持 Subnormal，需通过 Config 配置软仿 |

**决策流程**：

```
算子是否使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt？
  └─ 是 → L4（需 Subnormal Config 配置）
  └─ 否 → UB 使用量是否需要预留 SIMT 空间？
        └─ 是 → L4（需 IsRegbaseSocVersion + SIMT_UB_SIZE_BYTE）
        └─ 否 → 950 是否需要扩展数据类型？
              └─ 是 → L4（需独立配置 + 数据类型扩展）
              └─ 否 → Tiling 结构是否需要重构？
                    └─ 是 → L4（需全新 Tiling Key 体系）
                    └─ 否 → 维持 L1
```

---

## UB 预留规范

### 950 平台 UB 预留要求

950 平台 SIMT DCache 复用 UB 空间，**必须**预留 `SIMT_UB_SIZE_BYTE = 40960`（40KB）。未预留或预留不足将导致 UB 越界风险。

### 各算子 UB 预留合规状态

| 算子 | Tiling 文件 | IsRegbaseSocVersion 判断 | SIMT_UB_SIZE_BYTE 预留 | 改造状态 | 合规性 |
|------|------------|------------------------|----------------------|---------|--------|
| MoeInitRouting | `moe_init_routing_tiling.cpp` | ✓ 有 | ✓ `ubSizePlatForm - SIMT_UB_SIZE_BYTE` | 已完成 | ✓ 合规 |
| MoeGatingTopKSoftmax | `moe_gating_top_k_softmax_tiling_arch35.cpp` | ✓ 有 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| DequantSwigluQuant | `dequant_swiglu_quant_tiling_arch35.cpp` | ✓ 有 | ✗ 使用 `UB_REVERSE`(1KB) | 部分改造 | △ 需确认 |
| KvRmsnormRopeCache | `kv_rms_norm_rope_cache_base_tiling.cpp` | ✓ 有 | ✗ 使用 `UB_RESERVED_BYTE`(1KB) | 部分改造 | △ 需确认 |
| GroupNormSwish | `group_norm_swish_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| MoeComputeExpertTokens | `moe_compute_expert_tokens_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| SwigluQuant | `swi_glu_quant_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| MoeFinalizeRouting | `moe_finalize_routing_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |
| InterleaveRope | `interleave_rope_tiling.cpp` | ✗ 无 | ✗ 未预留 | 未改造 | ✗ 不合规 |

### UB 空间使用档案

| 算子 | UB 总容量(950) | SIMT 预留 | 算子可用 UB | 主要 UB 用途 |
|------|-------------|----------|-----------|------------|
| MoeInitRouting | 256KB | 40KB | 216KB | 排序缓冲+Gather 缓冲+SrcToDst 缓冲 |
| MoeGatingTopKSoftmax | 256KB | 0KB ✗ | 256KB | TopK 排序+Softmax 缓冲 |
| DequantSwigluQuant | 256KB | UB_REVERSE | 256KB-UB_REVERSE | 反量化+SwiGLU+量化多缓冲 |
| KvRmsnormRopeCache | 256KB | UB_RESERVED_BYTE | 256KB-UB_RESERVED | RMSNorm+RoPE+KVCache 多缓冲 |
| GroupNormSwish | 256KB | 0KB ✗ | 256KB | GroupNorm 均值/方差+Swish 缓冲 |
| MoeComputeExpertTokens | 256KB | 0KB ✗ | 256KB | 排序计数缓冲 |
| SwigluQuant | 256KB | 0KB ✗ | 256KB | SwiGLU+量化多缓冲 |
| MoeFinalizeRouting | 256KB | 0KB ✗ | 256KB | 融合乘加多缓冲 |
| InterleaveRope | 256KB | 0KB ✗ | 256KB | RoPE 交错乘加缓冲 |

### 统一 UB 预留模板代码

```cpp
namespace Ops { namespace Common {
    constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;

    inline uint64_t GetAvailableUbSize(platform_ascendc::PlatformAscendC& platform,
                                        bool isRegbase) {
        uint64_t ubSizePlatForm;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        if (isRegbase) {
            ubSizePlatForm -= SIMT_UB_SIZE_BYTE;
        }
        return ubSizePlatForm;
    }
}}
```

**Tiling 中标准用法**：

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

---

## Kernel 入口层迁移模式

### 模式 A：直接复用根目录实现

arch35/ 代码与根目录代码逻辑完全一致，仅 include 路径不同。

```cpp
#include "arch35/group_norm_swish_base.h"
```

**适用算子**：GroupNormSwish

### 模式 B：编译时宏切换（SIMT 优化）

950 使用 SIMT 替代传统 Scatter/Gather，通过 `__NPU_ARCH__` 宏在同一 `_apt.cpp` 中切换实现路径。

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_src_to_dst_simt_op.h"
#endif
#include "arch35/moe_src_to_dst_op.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
#endif
```

**适用算子**：MoeInitRouting

### 模式 C：溢出模式控制 + Register-based 重写

950 使用 MicroAPI 重写核心计算路径，并通过 SPR 寄存器控制溢出模式。

```cpp
#define GLOBAL_OVERFLOW_MODE_CTRL 60

#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif

    // ... 核心计算（使用 MicroAPI Register-based 实现）...

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
```

**适用算子**：KvRmsnormRopeCache

### 模式切换机制汇总

| 机制 | 使用算子 | 实现方式 | 适用场景 |
|------|---------|---------|---------|
| 编译时宏切换 | MoeInitRouting, KvRmsnormRopeCache | `#if (__NPU_ARCH__ == 3510)` | 同一 _apt.cpp 中选择不同实现路径 |
| opFile 切换 | 所有深度迁移算子 | _def.cpp 中 `opFile.value = "xxx_apt"` | 编译时选择不同的 kernel 入口文件 |
| Tiling 运行时切换 | MoeInitRouting, DequantSwigluQuant, KvRmsnormRopeCache | `IsRegbaseSocVersion()` | 运行时根据芯片平台调整 Tiling 参数 |

---

## Tiling 层迁移模式

### 模式 A：UB 容量 + SIMT 预留

950 Tiling 中获取 UB 容量后，扣除 SIMT DCache 预留空间。

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

### 模式 B：架构判断 + Tiling Key 分流

950 使用全新 Tiling Key 体系，通过 `IsRegbaseSocVersion` 判断是否进入 950 Tiling 路径。

```cpp
if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    return false;
}
// 950 使用 TILING_KEY_IS(10000) / TILING_KEY_IS(20000) 等新 Key
```

### 模式 C：自定义预留常量（需确认合规性）

使用自定义常量预留 UB 空间，但预留值可能不足。

```cpp
// DequantSwigluQuant — 使用 UB_REVERSE = 1024（仅 1KB，远小于 40KB）
constexpr int64_t UB_REVERSE = 1024;
int64_t ubAvailable = ubSize_ - UB_REVERSE - 1 * BLOCK_SIZE;

// KvRmsnormRopeCache — 使用 UB_RESERVED_BYTE = 1024（仅 1KB，远小于 40KB）
static constexpr int64_t UB_RESERVED_BYTE = 1024;
int64_t ubFlexible_ = ubSize_ - UB_RESERVED_BYTE - ...;
```

**⚠️ 风险提示**：`UB_REVERSE` 和 `UB_RESERVED_BYTE` 均为 1024（1KB），远小于 `SIMT_UB_SIZE_BYTE = 40960`（40KB）。如果 950 平台上 SIMT DCache 确实复用 UB 空间，这两个算子存在 UB 越界风险。**必须替换为标准 `SIMT_UB_SIZE_BYTE = 40960`**。

### Tiling 层共性特征

| 共性特征 | 说明 |
|---------|------|
| IsRegbaseSocVersion() | 深度迁移算子的 Tiling 中普遍使用此函数判断架构 |
| UB 空间预留 | 950 上需预留 SIMT_UB_SIZE_BYTE（40KB） |
| 排序粒度调整 | 涉及排序的算子（MoE 系列），mrgSortListMaxElement 从 1024→2048 |

---

## Subnormal 处理

### 351x 默认不支持 Subnormal

SubNormal 浮点数指的是指数位全为 0、尾数不为 0 的浮点数，用于表示比最小正常数更小的值，避免"下溢为 0"。351x 版本默认不支持 Subnormal，Subnormal 浮点数在计算中被视为 0。

### 涉及 API

以下基础 API 受 Subnormal 影响：

| AscendC 基础 API | 兼容说明 |
|-----------------|---------|
| Exp、Ln、Reciprocal、Sqrt、Rsqrt、Div | 需通过 Config 结构体的 `algo` 参数配置 Subnormal 计算模式 |

### Config 配置方式

以 `Ln` 接口为例，通过 `LnConfig` 结构体的 `algo` 参数配置：

| algo 取值 | 行为 |
|-----------|------|
| `LnAlgo::INTRINSIC` | 使用单指令计算，所有 Subnormal 被近似为 0（**默认值**） |
| `LnAlgo::PRECISION_1ULP_FTZ_TRUE` | 使用单指令计算，所有 Subnormal 被近似为 0 |
| `LnAlgo::PRECISION_1ULP_FTZ_FALSE` | 支持 Subnormal 数据计算（软件模拟，精度扩展） |

默认配置：

```cpp
constexpr LnConfig DEFAULT_LN_CONFIG = { LnAlgo::INTRINSIC };
```

### 代码示例

**220x 版本**（默认支持 Subnormal）：

```cpp
AscendC::Ln(dstLocal, srcLocal, count);
```

**351x 版本 — 支持 Subnormal**：

```cpp
constexpr AscendC::LnConfig CONFIG = {
    AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
};
AscendC::Ln<T, CONFIG>(dstLocal, srcLocal, count);
```

**351x 版本 — 高性能模式**（Subnormal 视为 0）：

```cpp
constexpr AscendC::LnConfig CONFIG_FAST = {
    AscendC::LnAlgo::INTRINSIC
};
AscendC::Ln<T, CONFIG_FAST>(dstLocal, srcLocal, count);
```

**完整示例**：

```cpp
constexpr AscendC::LnConfig CONFIG = {
    AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE
};

template <typename T>
__aicore__ inline void Compute(GM_ADDR dst, GM_ADDR src, uint32_t count)
{
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    srcGlobal.SetGlobalBuffer((__gm__ T*)src);
    dstGlobal.SetGlobalBuffer((__gm__ T*)dst);
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueue, 1, count * sizeof(T));
    pipe.InitBuffer(outQueue, 1, count * sizeof(T));
    AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
    AscendC::DataCopy(srcLocal, srcGlobal, count);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Ln<T, CONFIG>(dstLocal, srcLocal, count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    AscendC::DataCopy(dstGlobal, dstLocal, count);
    inQueue.FreeTensor(srcLocal);
}
```

**其他 API 的 Config 类似**：`ExpConfig`、`SqrtConfig`、`RsqrtConfig`、`DivConfig`、`ReciprocalConfig` 等，均通过 `algo` 参数控制 Subnormal 处理模式。

---

## 相关参考文档路径

| 文档类别 | 路径 | 说明 |
|---------|------|------|
| 迁移相关官方文档 | `references/migration/` | 220x→351x 架构迁移指导、基础/高阶 API 迁移指导、算子编译迁移指导、兼容性说明 |
| Memory-based Vector 操作 | `references/memory-base-vector/` | 传统 AscendC Vector 编程模式参考（TPipe/TQue/DataCopy 等） |
| API 选型与概述 | `references/api-overview/` | API 兼容性分层、高阶/基础/MicroAPI/SIMT 接口概述 |
