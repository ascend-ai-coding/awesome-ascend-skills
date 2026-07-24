# A8W8 量化 MatMul 开发指南

> **适用场景**：量化矩阵乘 `out = (A * x1Scale) @ (B * x2Scale)`，A/B 为 int8 或 fp8（e4m3/e5m2，支持混合 dtype），x1Scale/x2Scale 为 float32，输出为 bfloat16。
>
> **路径**：blaze library（Blaze::Gemm 命名空间，非 blaze_custom）
>
> **命名说明**：ops-tensor 仓将此类量化 matmul 组件以 "A8W8" 命名（如 `block_mmad_a8w8_mix.h`），最初含义是 int8×int8（Activation 8-bit × Weight 8-bit）。文件名沿用此路径命名，但实际输入类型不限定为 `int8_t`，当前同时支持 int8 和 fp8（含混合 dtype 组合）。
>
> **路径关系**：与 MX 量化 MatMul 均为量化场景，但量化粒度不同。A8W8 的 scale 为 per-tensor / per-channel / per-token（整个矩阵或沿 M/N 轴共享一个 scale），MX 的 scale 为 per-group（每 32/64 个 K 元素共享一个 fp8_e8m0 scale）。

---

## §1 场景背景

**数学定义**：

```
out = (A * x1Scale) @ (B * x2Scale)
```

**NPU 计算流**（利用量化 dtype 的 Cube 矩阵乘加速能力）：

```
out = cast_to_fp32(A@B) * x2Scale * x1Scale   ->  bf16
```

AIC 执行量化 dtype 的 Cube Mmad 累加，AIV 在向量上施加 scale 反量化。

**输入输出**：

| 张量    | shape       | dtype                            | 说明                                      |
| ------- | ----------- | -------------------------------- | ----------------------------------------- |
| A       | (M, K)      | int8 / fp8_e4m3fn_t / fp8_e5m2_t | 量化左输入                                |
| B       | (K, N)      | int8 / fp8_e4m3fn_t / fp8_e5m2_t | 量化右输入                                |
| x1Scale | 标量 或 (M) | float32                          | A 矩阵 scale（per-tensor 或 per-token）   |
| x2Scale | 标量 或 (N) | float32                          | B 矩阵 scale（per-tensor 或 per-channel） |
| out     | (M, N)      | bfloat16_t                       | 输出矩阵                                  |

**支持的 dtype 组合**：

| A dtype      | B dtype      | L0C 累加类型 | 说明                       |
| ------------ | ------------ | ------------ | -------------------------- |
| int8_t       | int8_t       | int32_t      | int8×int8→int32          |
| fp8_e4m3fn_t | fp8_e4m3fn_t | float        | fp8×fp8→fp32             |
| fp8_e4m3fn_t | fp8_e5m2_t   | float        | 混合 fp8→fp32（硬件支持） |
| fp8_e5m2_t   | fp8_e4m3fn_t | float        | 混合 fp8→fp32（硬件支持） |
| fp8_e5m2_t   | fp8_e5m2_t   | float        | fp8×fp8→fp32             |

L0C 累加类型由 `block_mmad_a8w8_mix.h` 内部根据 AType 推导：`int8_t → int32_t`，其他（含 fp8）→ `float`。最终以 Tensor API Mmad 静态检查为准。

**量化模式**：

4 种量化模式由 x1QuantMode（控制 x1Scale）和 x2QuantMode（控制 x2Scale）组合决定，对所有 dtype 组合均适用：

| x1QuantMode          | x2QuantMode           | x1Scale shape | x2Scale shape |
| -------------------- | --------------------- | ------------- | ------------- |
| PERTENSOR_MODE (0x1) | PERTENSOR_MODE (0x1)  | 标量          | 标量          |
| PERTENSOR_MODE (0x1) | PERCHANNEL_MODE (0x2) | 标量          | [N]           |
| PERTOKEN_MODE (0x4)  | PERTENSOR_MODE (0x1)  | [M]           | 标量          |
| PERTOKEN_MODE (0x4)  | PERCHANNEL_MODE (0x2) | [M]           | [N]           |

x1QuantMode / x2QuantMode 取值来自 `QuantMode` 枚举（`include/blaze/gemm/utils/common_utils.h`）。

---

## §2 组件选择 — 两条路径

blaze 库提供两条 A8W8 量化路径，能力边界不同：

| 路径         | Kernel                  | BlockMmad                                  | Epilogue                 | 反量化位置  | 支持的量化模式                                                                    | 支持的 dtype         |
| ------------ | ----------------------- | ------------------------------------------ | ------------------------ | ----------- | --------------------------------------------------------------------------------- | -------------------- |
| FixpipeQuant | `GemmUniversal`       | `BlockMmad<MatmulWithScaleFixpipeQuant>` | 无                       | AIC Fixpipe | x1Scale 仅 per-tensor；x2Scale per-tensor / per-channel                           | int8 / fp8（含混合） |
| MIX          | `QbmmMixWithoutBatch` | `BlockMmad<MatmulWithScaleMix>`          | `BlockEpilogueDequant` | AIV Vector  | x1Scale per-tensor / per-token；x2Scale per-tensor / per-channel（全部 4 种组合） | int8 / fp8（含混合） |

**关键约束**：FixpipeQuant 路径在 Cube Fixpipe 阶段施加 scale，x1Scale 只能是 per-tensor 标量（Kernel 层将 scaleA 乘入 scaleB 后作为标量传入 Block），**无法支持 x1Scale 为 per-token 向量**。

MIX 路径将累加结果从 L0C 经 Fixpipe(NoQuant) 搬到 UB（int32 或 float，取决于 AType），由 AIV 的 `BlockEpilogueDequant` 在向量上施加 scale，独立支持 per-token / per-channel / per-tensor。

**选择规则**：

- 需要支持 per-token x1Scale 时，**必须用 MIX 路径**
- 只需 per-tensor x1Scale 时，两条路径均可（FixpipeQuant 更轻量，纯 AIC 不需 AIV）
- 两条路径均支持 int8 和 fp8 输入，dtype 不影响路径选择

---

## §3 组装代码 — MIX 路径完整示例

### L0CType 推导

L0C 累加类型由 AType 决定，必须与 `block_mmad_a8w8_mix.h` 内部推导一致。int8 输入累加为 int32_t（epilogue 做 int32→fp32 cast），fp8 输入累加为 float（epilogue 直接 raw copy，不做 cast）。

```cpp
template <typename AType>
using L0CTypeOf = AscendC::Std::conditional_t<
    AscendC::IsSameType<AType, int8_t>::value, int32_t, float>;
```

### Include 顺序

```cpp
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
```

### Kernel 入口函数

```cpp
template <typename AType, typename BType>
__global__ __aicore__ __mix__(1, 2) void quant_matmul_kernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleAGM, GM_ADDR scaleBGM, GM_ADDR outGM,
    const MyTilingData tilingData)
{
    using OutType = bfloat16_t;
    using X1ScaleType = float;  // scaleA
    using X2ScaleType = float;  // scaleB
    using BiasType = int32_t;   // 编译期占位，MIX 路径不施加 bias 时设 isBias=false
    using L0CType = L0CTypeOf<AType>;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // BTypeTuple 第 1 元素为权重类型，第 2 元素 MIX mmad 未使用，填 uint64_t
    using BTypeTuple = AscendC::Std::tuple<BType, uint64_t>;

    // DispatchPolicy 用默认 ScheduleType（KernelMmadWithScaleMix），不显式传 WithoutBatch
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMix<
        Blaze::Gemm::NONE_FULL_LOAD_MODE, false>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE, LayoutA, LayoutB, AType>;

    // BlockMmad 的 CType 参数和 Epilogue 的 L0CType 参数都用 L0CType
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB,
        L0CType, LayoutC, BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<
        OutType, int32_t, X2ScaleType, X1ScaleType, L0CType>;

    using KernelImpl = Blaze::Gemm::Kernel::QbmmMixWithoutBatch<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    // ... Params 填充见下方映射表
    KernelImpl kernel;
    kernel(params);
}
```

### Host 端 dtype 分发

```cpp
if (dtypeA == "e4m3" && dtypeB == "e4m3")
    quant_matmul_kernel<fp8_e4m3fn_t, fp8_e4m3fn_t><<<gridDim, nullptr, stream>>>(...);
else if (dtypeA == "e4m3" && dtypeB == "e5m2")
    quant_matmul_kernel<fp8_e4m3fn_t, fp8_e5m2_t><<<gridDim, nullptr, stream>>>(...);
else if (dtypeA == "e5m2" && dtypeB == "e4m3")
    quant_matmul_kernel<fp8_e5m2_t, fp8_e4m3fn_t><<<gridDim, nullptr, stream>>>(...);
else if (dtypeA == "e5m2" && dtypeB == "e5m2")
    quant_matmul_kernel<fp8_e5m2_t, fp8_e5m2_t><<<gridDim, nullptr, stream>>>(...);
else
    quant_matmul_kernel<int8_t, int8_t><<<gridDim, nullptr, stream>>>(...);
```

### TilingData → Params 映射

| Params 字段                                          | TilingData 来源            | 说明          |
| ---------------------------------------------------- | -------------------------- | ------------- |
| `mmParams.aGmAddr/bGmAddr`                         | dA/dB                      | A/B GM 地址   |
| `mmParams.l0TileShape`                             | `{baseM, baseN, baseK}`  | L0 tile       |
| `mmParams.kAL1/kBL1/l1BufferNum/enableL0CPingPong` | kAL1/kBL1/nBufferNum/dbL0C | L1/L0C 配置   |
| `schParams.baseM/baseN/...`                        | baseM/baseN/mTailTile/...  | 调度参数      |
| `epilogueParams.x1QuantMode/x2QuantMode`           | x1QuantMode/x2QuantMode    | 量化模式标志  |
| `epilogueParams.x1ScaleGmAddr/x2ScaleGmAddr`       | dScaleA/dScaleB            | scale GM 地址 |
| `epilogueParams.outGmAddr`                         | dOut                       | 输出 GM 地址  |

---

## §4 QuantMode 与运行时派发

QuantMode 枚举值表（`include/blaze/gemm/utils/common_utils.h` / `block_epilogue_dequant.h`）：

| 枚举            | 值  | 含义                 | 适用 scale       |
| --------------- | --- | -------------------- | ---------------- |
| PERTENSOR_MODE  | 0x1 | per-tensor 标量      | x1Scale, x2Scale |
| PERCHANNEL_MODE | 0x2 | per-channel 向量 [N] | x2Scale          |
| PERTOKEN_MODE   | 0x4 | per-token 向量 [M]   | x1Scale          |

- x1QuantMode 控制 x1Scale（A 矩阵 scale），x2QuantMode 控制 x2Scale（B 矩阵 scale）
- `BlockEpilogueDequant::Init` 根据 mode 设置 `isPerChannel_` / `isPerToken_` / `isX1PerTensor_`，运行时派发到对应 VF dequant 模板特化
- 4 种组合的 scale 元素数计算：

| x1QuantMode    | x1Scale 元素数 | x2QuantMode     | x2Scale 元素数 |
| -------------- | -------------- | --------------- | -------------- |
| PERTENSOR_MODE | 1              | PERTENSOR_MODE  | 1              |
| PERTENSOR_MODE | 1              | PERCHANNEL_MODE | N              |
| PERTOKEN_MODE  | M              | PERTENSOR_MODE  | 1              |
| PERTOKEN_MODE  | M              | PERCHANNEL_MODE | N              |

---

## §5 Tiling

- 复用 `MatmulTilingSwat`，`inputElemBytes=1`（int8 和 fp8 均为 1 字节，tiling 逻辑完全一致）
- 参考代码：

```cpp
MatmulTilingData base;
MatmulTilingSwat swat;
swat.GetTilingData(m, n, k, 1UL, base, false, false, false, false, false);
```

- 映射到 `QbmmMixWithoutBatch` 子 Params 时需补充 QuantMode 字段（base SWAT 不产出）
- C+V 复用原则：不新增 vector tiling，AIV dequant 消费 Cube tiling 后的剩余 UB

---

## §6 常见陷阱

| # | 陷阱                                                                  | 症状                                                    | 解决                                                         |
| - | --------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| 1 | 需 per-token x1Scale 但选了 FixpipeQuant 路径                         | 不支持                                                  | 切换到 MIX 路径                                              |
| 2 | BTypeTuple 第二个类型填错                                             | 编译错误                                                | MIX 路径填`uint64_t`（mmad 未使用该参数）                  |
| 3 | MIX 路径用`__cube__` 修饰符                                         | AIV 不执行，输出全零或 hang                             | MIX 路径必须`__mix__(1, 2)`                                |
| 4 | fp8 输入时 BlockMmad 的 CType 和 Epilogue 的 L0CType 仍硬编码 int32_t | 精度错误（epilogue 对 float L0C 误做 int32→fp32 cast） | 两者都用`L0CTypeOf<AType>` 推导：int8→int32_t，fp8→float |
