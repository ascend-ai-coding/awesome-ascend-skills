# MX 量化 MatMul 开发指南

> **适用场景**：MX 量化矩阵乘 C = dequant(A) × dequant(B)（mxfp8/mxfp4），默认使用 SWAT 流式路径。
>
> **路径**：blaze library（Blaze::Gemm 命名空间，非 blaze_custom）
>
> **路径关系**：MX 量化 MatMul 与普通 MatMul 单算子均走 blaze library，但 DispatchPolicy 不同：普通 MatMul 使用 `MatmulMultiBlockBasic`，MX 量化使用 `MatmulWithScaleMx`。
>
> **融合跳转**：如果需求是 MXFP8/MXFP4 MatMul + Mul/Add/GELU/SwiGLU 等 Vector Epilogue，不使用本文的纯 `GemmUniversal` 路径，转到 [C+V 融合 MatMul 开发指南](fusion-matmul-development.md) 的 MX C+V 小节。

---

## §1 场景背景

**数学定义**（分组反量化 + 矩阵乘）：

```
c[i,j] = Σ(g=0..ceil(K/G)-1):
    scaleA[i,g] × scaleB[g,j] × Σ(k'=0..G-1):
        a[i, g×G + k'] × b[g×G + k', j]
```

其中 `G = MX_GROUP_SIZE = 32`，每个 Scale 覆盖 K 方向 32 个元素。

**输入输出**：

| 张量 | shape | dtype | 说明 |
|------|-------|-------|------|
| A | (M, K) | fp8_e4m3 / fp4x2_e2m1 | 量化左输入 |
| B | (K, N) | fp8_e4m3 / fp4x2_e2m1 | 量化右输入 |
| ScaleA | (M, CeilDiv(K,64), 2) | fp8_e8m0_t | A 的分组 Scale |
| ScaleB | (CeilDiv(K,64), N, 2) | fp8_e8m0_t | B 的分组 Scale |
| C | (M, N) | bfloat16_t | 输出矩阵 |

**MXFP8 vs MXFP4 选择**：普通交付工程通常按用户需求固定为 MXFP8 或 MXFP4；除非用户明确要求通用 demo，不默认生成 MXFP8/MXFP4 runtime dispatch。

| 维度 | MXFP8 | MXFP4 |
|------|-------|-------|
| C++ 类型 | `fp8_e4m3fn_t` | `fp4x2_e2m1_t`（packed） |
| 硬件 C0 | 32 | 64 |
| 内存节省（vs FP16） | ~50% | ~75% |
| 精度 | 较好 | 较低 |
| 适用场景 | 训练/推理，精度敏感 | 推理，带宽敏感 |

---

## §2 组件选择

本节只适用于纯 MX MatMul 单算子。MX C+V 融合是受控组合态，使用 `Kernel::MxMatmulKernelFused` 桥接 blaze library MX Block/Scheduler 与自定义 Epilogue。

| 组件 | 选择 | 来源 |
|------|------|------|
| 路径 | blaze library | `third_party/blaze/gemm/` |
| Kernel | `Blaze::Gemm::Kernel::GemmUniversal` | `kernel/gemm_universal.h` |
| BlockMmad | `Blaze::Gemm::Block::BlockMmad` | `block/block_mmad.h` |
| Scheduler | `Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3` | `block/block_scheduler.h` |
| DispatchPolicy | `Blaze::Gemm::MatmulWithScaleMx` | `policy/dispatch_policy.h` |
| Tiling | `QuantMatmulTilingSwat` | `assets/op_tiling/mx/` |

---

## §3 组装代码

### Kernel 入口函数

```cpp
#include "blaze/gemm/kernel/gemm_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler.h"
#include "blaze/gemm/policy/dispatch_policy.h"

template <bool TransA, bool TransB, uint64_t FullLoadMode>
__global__ __aicore__ __cube__ void mx_matmul_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData tilingData)
{
    using TypeA = fp8_e4m3fn_t;
    using TypeB = fp8_e4m3fn_t;
    using TypeC = bfloat16_t;
    using BiasType = float;
    using TypeScaleA = fp8_e8m0_t;
    using TypeScaleB = fp8_e8m0_t;

    using LayoutA = AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FullLoadMode, LayoutA, LayoutB, TypeA>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, void, BlockScheduler>;

    using Params = typename KernelImpl::Params;
    ProblemShape problemShape{tilingData.m, tilingData.n, tilingData.k, 1L};

    Params params;
    params.problemShape = problemShape;
    params.mmaddParams = {dA, dB, dC, dScaleA, dScaleB};
    params.blockSchedulerParams = {
        tilingData.baseM, tilingData.baseN, tilingData.baseK,
        tilingData.scaleKL1,
        tilingData.mTailTile, tilingData.nTailTile,
        tilingData.mBaseTailSplitCnt, tilingData.nBaseTailSplitCnt,
        tilingData.mTailMain, tilingData.nTailMain};
    params.tilingParams = {
        tilingData.baseM, tilingData.baseN, tilingData.baseK,
        tilingData.stepK, tilingData.nBufferNum, tilingData.dbL0c};

    KernelImpl kernel;
    kernel(params);
}
```

### Scale 数据格式

- **类型**：`fp8_e8m0_t`（8-bit 纯指数，无尾数，表示 2 的幂次）
- **分组大小**：`MX_GROUP_SIZE = 32`（每 32 个 K 方向元素共享一个 Scale 值）
- **Scale shape**：`(rows, CeilDiv(K, 64), 2)`，尾部 `2 = TILING_MXFP_MULTI_BASE_SIZE`
- **Host 端 size**：`(rows × CeilDiv(K, 64) × 2) × sizeof(uint8_t)`

### TilingData → Params 映射

| Params 字段 | TilingData 来源 | 说明 |
|------------|----------------|------|
| `problemShape` | `{m, n, k, 1}` | batch 固定为 1 |
| `mmaddParams` | `{dA, dB, dC, dScaleA, dScaleB}` | 5 个 GM 地址 |
| `blockSchedulerParams` | `{baseM, baseN, baseK, scaleKL1, mTailTile, nTailTile, ...}` | 多核调度参数 |
| `tilingParams` | `{baseM, baseN, baseK, stepK, nBufferNum, dbL0c}` | L0/L1 切分参数 |

---

## §4 MX 特有约束

### K 轴对齐（MXFP_DIVISOR_SIZE=64）

- `baseK` 必须是 64 的倍数（MX MMAD 指令约束）
- K 不满足 64 对齐时需要零填充
- Tiling 引擎自动处理对齐（`Align(k, TILING_MXFP_DIVISOR_SIZE)`）

### FP4 打包格式

两个 `fp4_e2m1` 值打包到 1 字节（低 4 位 + 高 4 位），打包沿矩阵**内轴**进行。

**内轴偶数约束**：

| 矩阵 | trans=false | trans=true |
|------|------------|-----------|
| A | K 必须为偶数 | M 必须为偶数 |
| B（ND） | N 必须为偶数 | K 必须为偶数 |
| B（NZ） | 天然满足 | 天然满足 |

### ND2NZ 不支持 B4

- `ND2NZ` 指令不支持 `B4`（fp4）数据类型
- FP4 数据必须视为 `B8` 处理：stride/shape 除以 2

### MMAD gemv 必须关闭

MX 量化场景下 MMAD 指令的 gemv 功能必须禁用。Blaze 库的 MX BlockMmad 已内部处理此约束。

### L1 Bank 冲突

- 双缓冲的两个 buffer 必须分布在不同的 256KB bank 中
- 否则 MTE1 带宽下降，MMAD 流水中断
- Blaze 库的 MX BlockMmad 已内部处理 bank 分配

---

## §5 Tiling 参数

**Tiling 引擎**：`QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN>`（来自 `assets/op_tiling/mx/`）

```cpp
QuantMatmulTilingData tilingData;
QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
tilingEngine.GetTilingData(m, n, k, tilingData, transA, transB, isNzA, isNzB, hasBias);
```

**关键输出字段**：

| 字段 | 含义 | Kernel 端使用 |
|------|------|-------------|
| `usedCoreNum` | 启动核数 | `<<<usedCoreNum, ...>>>` |
| `baseM/baseN/baseK` | L0 切分颗粒 | `tilingParams` |
| `scaleKL1` | Scale K 方向 L1 复用窗口 | `blockSchedulerParams` |
| `stepK` | K 方向 L1 depth 步长 | `tilingParams` |
| `nBufferNum` | L1 buffer 数（2 或 4） | `tilingParams` |
| `mTailTile/nTailTile` | M/N 方向是否有尾块 | `blockSchedulerParams` |
| `dbL0c` | L0C ping-pong 级数 | `tilingParams` |

---

## §6 常见陷阱

| # | 陷阱 | 症状 | 解决 |
|---|------|------|------|
| 1 | Scale shape 计算错误 | 精度异常 | 使用 `(rows, CeilDiv(K, 64), 2)` 而非 `(rows, CeilDiv(K, 32))` |
| 2 | FP4 未打包 | size 错误 | FP4 数据必须经过 `pack_b4_to_b8()` 打包 |
| 3 | K 未对齐 64 | 编译/精度错误 | Tiling 引擎自动 Pad，勿手动跳过 |
| 4 | Scale Layout 手动指定 | 类型不匹配 | Scale Layout 由 Kernel 内部推导，BlockMmad 中不指定 |
| 5 | FP4 内轴为奇数 | 数据生成 ValueError | 检查转置组合对应的内轴偶数约束 |
| 6 | MXFP8 `ceil(k/32)` 为奇数 | ND 布局精度错误 | 手动 K 方向填充 |
| 7 | 对 FP4 数据调用 ND2NZ | 编译错误 | 视为 B8 处理，stride/shape 除以 2 |
| 8 | ND 和 NZ 数据混用 | 精度异常 | ND 格式和 Weight NZ 格式的数据文件不可混用 |
| 9 | MX C+V 仍套纯 `GemmUniversal` | 无法接入自定义 Vector Epilogue | 转到 `fusion-matmul-development.md` 使用 `MxMatmulKernelFused` |

---

## §7 Layout 与变体边界

- A/B 支持 ND/NZ 与 4 种 transpose 组合，tiling 通过 `isNzA/isNzB/transA/transB` 计算对齐。
- Full-load、4-buffer、WeightQuant 专用 tiling 不属于本 skill 默认路径。
- Grouped MX MatMul 使用 `{totalM,N,K}` 复用 `QuantMatmulTilingSwat`，`groupList/groupNum` 独立传入 grouped kernel。

---

## §8 Golden 数据生成

MX 量化 matmul 的数据生成比普通 matmul 复杂，需要额外处理 Scale 生成、分组反量化和 FP4 打包。

### MXFP8 Golden 计算核心流程

```python
import math
import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn

def build_scale_broadcast(scale, target_shape, chunk_axis):
    """将 Scale 按 group_size=32 广播到目标 shape"""
    scale_repeat = np.repeat(scale.astype(np.float32), 32, axis=-1)
    if chunk_axis == 1:
        scale_broadcast = scale_repeat.reshape(scale.shape[0], -1)[..., :target_shape[1]]
    elif chunk_axis == 0:
        scale_broadcast = np.transpose(scale_repeat, (0, 2, 1)).reshape(-1, scale.shape[1])[:target_shape[0], ...]
    return scale_broadcast

def dequant_mxfp8(fp8_input, scale, chunk_axis):
    """分组反量化：fp8 × scale → float32"""
    scale_broadcast = build_scale_broadcast(scale, fp8_input.shape, chunk_axis)
    return fp8_input.astype(np.float32) * scale_broadcast

def gen_golden(m, k, n, trans_a=False, trans_b=True):
    # 1. 生成量化数据
    a_shape = (k, m) if trans_a else (m, k)
    a_fp8 = np.random.uniform(1, 8, a_shape).astype(float8_e4m3fn)
    b_shape = (n, k) if trans_b else (k, n)
    b_fp8 = np.random.uniform(1, 8, b_shape).astype(float8_e4m3fn)

    # 2. 生成 Scale（e8m0 格式，每 32 个 K 元素共享一个 Scale）
    a_scale_shape = (math.ceil(k / 64), m, 2) if trans_a else (m, math.ceil(k / 64), 2)
    b_scale_shape = (n, math.ceil(k / 64), 2) if trans_b else (math.ceil(k / 64), n, 2)
    a_scale = np.random.uniform(1, 8, size=a_scale_shape).astype(float8_e8m0)
    b_scale = np.random.uniform(1, 8, size=b_scale_shape).astype(float8_e8m0)

    # 3. 分组反量化 → golden = matmul(dequant(A), dequant(B))
    a_chunk_axis = 0 if trans_a else 1
    b_chunk_axis = 1 if trans_b else 0
    a_dequant = dequant_mxfp8(a_fp8, a_scale, a_chunk_axis)
    b_dequant = dequant_mxfp8(b_fp8, b_scale, b_chunk_axis)
    a_matmul = np.swapaxes(a_dequant, -1, -2) if trans_a else a_dequant
    b_matmul = np.swapaxes(b_dequant, -1, -2) if trans_b else b_dequant
    golden = torch.matmul(torch.from_numpy(a_matmul), torch.from_numpy(b_matmul)).to(torch.bfloat16)

    # 4. 输出：量化数据（view uint8）+ Scale + golden
    return a_fp8.view(np.uint8), b_fp8.view(np.uint8), a_scale, b_scale, golden
```

### MXFP4 额外步骤

MXFP4 在 MXFP8 基础上增加两个步骤：

**1. FP4 打包**：两个 fp4 值打包到 1 字节

```python
def pack_b4_to_b8(b4_data):
    packed_shape = [b4_data.shape[0], b4_data.shape[1] // 2]
    shift = np.array([0, 4], dtype=np.int8)
    b4_data = b4_data.reshape(-1, 2).view(np.int8)
    return np.sum(np.bitwise_and(b4_data, 0b00001111) << shift, axis=1, dtype=np.int8).reshape(packed_shape)
```

**2. 内轴偶数校验**：FP4 打包要求内轴长度为偶数

| 矩阵 | trans 标志 | 内轴 | 约束 |
|------|-----------|------|------|
| A | `transA=false` | K | K 必须为偶数 |
| A | `transA=true` | M | M 必须为偶数 |
| B | `transB=true` | K | K 必须为偶数 |
| B | `transB=false` | N | N 必须为偶数 |

### Scale shape 速查

| 矩阵 | 非转置（ND） | 转置（DN） |
|------|-------------|-----------|
| ScaleA | `(M, ceil(K/64), 2)` | `(ceil(K/64), M, 2)` |
| ScaleB | `(ceil(K/64), N, 2)` | `(N, ceil(K/64), 2)` |
