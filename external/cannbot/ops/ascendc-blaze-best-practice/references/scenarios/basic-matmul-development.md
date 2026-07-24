# 基础 MatMul 开发指南

> **适用场景**：普通矩阵乘 C = A × B（fp16/bf16/fp32），可选 MMAD bias 输入。
>
> **路径**：blaze library（`Blaze::Gemm` 命名空间，非 blaze_custom）
>
> **验证状态**：已通过 `blaze_basic_matmul_probe` 工程穿刺，覆盖 fp16/bf16/fp32、ND/NZ、transpose、bias、shape 组合，共 480 cases。普通开发模板按用户需求固定 dtype，仅保留 ND/NZ 与 transpose 的组合分发。

---

## §1 场景背景

**数学定义**：

```
C[M,N] = A[M,K] × B[K,N]
```

带 bias 时：

```
C[i,j] = Σ(k=0..K-1) A[i,k] × B[k,j] + bias[j]
```

**输入输出**：

| 张量 | shape | dtype | 说明 |
|------|-------|-------|------|
| A | transA=false: `(M,K)`；transA=true: `(K,M)` | fp16/bf16/fp32 | 左输入矩阵 |
| B | transB=false: `(K,N)`；transB=true: `(N,K)` | fp16/bf16/fp32 | 右输入矩阵 |
| Bias | `(N)` | fp32 | 可选，按列加到输出 |
| C | `(M,N)` | 同 A/B | 输出矩阵，默认 ND |

**dtype 固定原则**：普通开发场景通常由用户需求指定 dtype，不需要在一个算子工程中同时支持 fp16/bf16/fp32 runtime dispatch。dtype 在 kernel 类型别名和 host 侧元素字节数中固定。

| 用户需求 dtype | `AType/BType/CType` | `IsFp32` | NZ C0 |
|-------|---------------------|----------|-------|
| fp16 | `half` | `false` | 16 |
| bf16 | `bfloat16_t` | `false` | 16 |
| fp32 | `float` | `true` | 8 |

---

## §2 组件选择

基础 MatMul 单算子使用 ops-tensor 的 blaze library Basic MatMul 组件链。

| 组件 | 选择 | 来源 |
|------|------|------|
| 路径 | blaze library | `op_kernel/include/blaze/` |
| Kernel | `Blaze::Gemm::Kernel::GemmUniversal` | `blaze/gemm/kernel/kernel_universal.h` + `kernel_matmul_basic.h` |
| DispatchPolicy | `Blaze::Gemm::MatmulMultiBlockBasic` | `blaze/gemm/policy/dispatch_policy.h` |
| BlockMmad | `Blaze::Gemm::Block::BlockMmad` | `blaze/gemm/block/block_mmad.h` + `block_mmad_matmul_basic.h` |
| Scheduler | `Blaze::Gemm::Block::BlockSchedulerMatmulBasic` | `blaze/gemm/block/block_scheduler_matmul_basic.h` |
| Epilogue | `Blaze::Gemm::Block::BlockEpilogueEmpty` | `blaze/epilogue/block/block_epilogue_empty.h` |

**能力边界**：

- 默认使用 SWAT 流式路径。
- Basic Kernel 只使用 `BlockEpilogueEmpty`，不承载 vector epilogue 融合。
- 如果需求包含 ReLU/Add/Cast/Scale 等后处理，路由到 `fusion-matmul-development.md`。
- Full-load、StreamK、4-buffer 不属于本 skill 的默认路径。

---

## §3 Kernel 组装代码

### Include 顺序

部分 ops-tensor Basic MatMul 头文件中使用了非限定名 `Shape`、`Coord`、`SetMMLayoutTransform`。在 include blaze 头文件前引入对应 using，避免编译期找不到符号。

```cpp
#include "kernel_operator.h"

using AscendC::Coord;
using AscendC::SetMMLayoutTransform;
using AscendC::Shape;

#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
```

### TilingData

```cpp
#include "op_tiling/matmul/blaze_matmul_tiling_data.h"

enum class CubeFormat : uint32_t {
    ND = 0,
    NZ = 1,
};
```

`MatmulTilingData` 由 `MatmulTilingSwat` 生成，可映射到 BlockMmad 和 Scheduler 参数。Grouped MatMul 也复用该结构，不增加 group 字段。

### Kernel 入口函数

```cpp
template <bool TransA, bool TransB, CubeFormat FormatA, CubeFormat FormatB>
__global__ __aicore__ __cube__ void matmul_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dBias, GM_ADDR dC,
    const MatmulTilingData tilingData)
{
    // [MODIFY] 根据用户需求固定 dtype。bf16 改为 bfloat16_t，fp32 改为 float。
    using AType = half;
    using BType = half;
    using CType = half;
    using BiasType = float;

    using LayoutA = AscendC::Std::conditional_t<
        (FormatA == CubeFormat::NZ),
        AscendC::Std::conditional_t<TransA, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    using LayoutB = AscendC::Std::conditional_t<
        (FormatB == CubeFormat::NZ),
        AscendC::Std::conditional_t<TransB, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>,
        AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    constexpr uint64_t FUSED_OP_TYPE = 0;
    constexpr bool IS_ND_FORMAT = (FormatA == CubeFormat::ND && FormatB == CubeFormat::ND);
    constexpr bool IS_FP32 = AscendC::Std::is_same_v<AType, float>;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<NO_FULL_LOAD_MODE, FUSED_OP_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
        ProblemShape, NO_FULL_LOAD_MODE, IS_FP32, IS_ND_FORMAT>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;

    ProblemShape problemShape{
        static_cast<int64_t>(tilingData.m),
        static_cast<int64_t>(tilingData.n),
        static_cast<int64_t>(tilingData.k),
        1L};

    BlockMmadParams mmParams;
    mmParams.aGmAddr = dA;
    mmParams.bGmAddr = dB;
    mmParams.cGmAddr = dC;
    mmParams.biasGmAddr = hasBias ? dBias : nullptr;
    mmParams.ml1 = tilingData.mL1;
    mmParams.nl1 = tilingData.nL1;
    mmParams.kl1 = tilingData.kL1;
    mmParams.ml0 = tilingData.baseM;
    mmParams.nl0 = tilingData.baseN;
    mmParams.kl0 = tilingData.baseK;
    mmParams.l1Stages = tilingData.l1BufferNum;
    mmParams.l0cStages = static_cast<uint16_t>(tilingData.l0cDB);

    BlockSchedulerParams schParams;
    schParams.mL1 = tilingData.mL1;
    schParams.nL1 = tilingData.nL1;
    schParams.kL1 = tilingData.kL1;
    schParams.baseM = tilingData.baseM;
    schParams.baseN = tilingData.baseN;
    schParams.baseK = tilingData.baseK;
    schParams.mTailCnt = tilingData.mTailCnt;
    schParams.nTailCnt = tilingData.nTailCnt;
    schParams.mBaseTailSplitCnt = tilingData.mBaseTailSplitCnt;
    schParams.nBaseTailSplitCnt = tilingData.nBaseTailSplitCnt;
    schParams.mTailMain = tilingData.mTailMain;
    schParams.nTailMain = tilingData.nTailMain;
    schParams.isHf32 = 0;
    schParams.l2CacheMode = Blaze::Gemm::L2_CACHE_DEFAULT;
    schParams.sliceM = 0;
    schParams.srcNdStride = 1;
    schParams.innerBatch = 1;

    BlockEpilogueParams epilogueParams;

    Params params;
    params.problemShape = problemShape;
    params.mmadParams = mmParams;
    params.epilogueParams = epilogueParams;
    params.schParams = schParams;

    KernelImpl kernel;
    kernel(params);
}
```

### TilingData → Params 映射

| Params 字段 | TilingData 来源 | 说明 |
|------------|----------------|------|
| `problemShape` | `{m, n, k, 1}` | batch 固定为 1 |
| `mmadParams.aGmAddr` | `dA` | A GM 地址 |
| `mmadParams.bGmAddr` | `dB` | B GM 地址 |
| `mmadParams.cGmAddr` | `dC` | C GM 地址 |
| `mmadParams.biasGmAddr` | `hasBias ? dBias : nullptr` | 可选 fp32 bias；`hasBias` 是 launcher/runtime 参数，不来自 `TilingData` |
| `mmadParams.ml1/nl1/kl1` | `mL1/nL1/kL1` | L1 tile 尺寸 |
| `mmadParams.ml0/nl0/kl0` | `baseM/baseN/baseK` | L0 tile 尺寸 |
| `mmadParams.l1Stages` | `l1BufferNum` | 推荐直接映射当前 SWAT 的 L1 buffer 数 |
| `mmadParams.l0cStages` | `l0cDB` | 推荐直接映射当前 SWAT 的 L0C buffer 数 |
| `schParams.mL1/nL1/kL1` | `mL1/nL1/kL1` | Scheduler tile 尺寸 |
| `schParams.baseM/baseN/baseK` | `baseM/baseN/baseK` | Scheduler L0 粒度 |
| `schParams.mTailCnt/nTailCnt` | `mTailCnt/nTailCnt` | 尾块 split 参数；在普通 SWAT 中语义等价于 tail split factor，默认值为 `1/1` |
| `epilogueParams` | `BlockEpilogueEmpty::Params{}` | 空后处理 |

---

## §4 Layout 与 Launcher 分发

普通开发模板固定 dtype，仅分发：

```
transA × transB × formatA × formatB = 16 种组合
```

推荐使用分层分发，避免手写 16 个扁平分支。

```cpp
template <bool TransA, bool TransB, CubeFormat FormatA>
void LaunchByBLayout(
    aclrtStream stream,
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dBias, GM_ADDR dC,
    const MatmulTilingData tilingData,
    CubeFormat formatB)
{
    if (formatB == CubeFormat::NZ) {
        matmul_kernel<TransA, TransB, FormatA, CubeFormat::NZ>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dBias, dC, tilingData);
    } else {
        matmul_kernel<TransA, TransB, FormatA, CubeFormat::ND>
            <<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dBias, dC, tilingData);
    }
}

template <bool TransA, bool TransB>
void LaunchByALayout(
    aclrtStream stream,
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dBias, GM_ADDR dC,
    const MatmulTilingData tilingData,
    CubeFormat formatA,
    CubeFormat formatB)
{
    if (formatA == CubeFormat::NZ) {
        LaunchByBLayout<TransA, TransB, CubeFormat::NZ>(stream, dA, dB, dBias, dC, tilingData, formatB);
    } else {
        LaunchByBLayout<TransA, TransB, CubeFormat::ND>(stream, dA, dB, dBias, dC, tilingData, formatB);
    }
}
```

完整链路：

```
LaunchByTransA
  → LaunchByTransB
    → LaunchByALayout
      → LaunchByBLayout
```

---

## §5 Tiling 参数

Basic MatMul 使用 `assets/op_tiling/matmul/MatmulTilingSwat`，不要手写启发式 tiling。

```cpp
MatmulTilingData tilingData;
MatmulTilingSwat tiling;
tiling.GetTilingData(m, n, k, inputElemBytes, tilingData, transA, transB, isANz, isBNz, hasBias);
```

**关键字段**：

| 字段 | 含义 | Kernel 端使用 |
|------|------|-------------|
| `usedCoreNum` | 启动核数 | `<<<usedCoreNum, ...>>>` |
| `m/n/k` | 问题规模 | `ProblemShape` |
| `mL1/nL1/kL1` | L1 tile 尺寸 | `BlockMmadParams` + `BlockSchedulerParams` |
| `baseM/baseN/baseK` | L0 tile 尺寸 | `BlockMmadParams` + `BlockSchedulerParams` |
| `mTailCnt/nTailCnt` | 尾块切分份数（禁用尾块二次切分时固定为 `1/1`） | `BlockSchedulerParams` |
| `mBaseTailSplitCnt/nBaseTailSplitCnt` | 尾块合并数量 | `BlockSchedulerParams` |

---

## §6 Bias 输入

blaze library Basic MatMul 的 bias 不需要手写 L1/L0 copy 流水，直接通过 `BlockMmad::Params::biasGmAddr` 传入。

| 项 | 约定 |
|----|------|
| Bias dtype | `float` |
| Bias shape | `(N)` |
| Bias layout | `AscendC::Te::NDExtLayoutPtn` |
| 无 bias | `biasGmAddr = nullptr` |

```cpp
using BiasType = float;
using LayoutBias = AscendC::Te::NDExtLayoutPtn;
mmParams.biasGmAddr = hasBias ? dBias : nullptr;
```

如果需求是 bias + activation / bias + add / bias + cast 等 vector 后处理，不属于 Basic Kernel 范围，应切换到 C+V 融合场景。

---

## §7 常见陷阱

| # | 陷阱 | 症状 | 解决 |
|---|------|------|------|
| 1 | 未在 blaze include 前引入 `Shape/Coord/SetMMLayoutTransform` | 编译报未声明标识符 | 在 include 前添加 `using AscendC::...` |
| 2 | `usedCoreNum=1` 但 M/N 多 tile | 输出部分列/行全 0 | `usedCoreNum` 按 `tileM × tileN` 与硬件核数取 min |
| 3 | 把验证 harness 的 dtype dispatch 带入交付模板 | 代码复杂、实例化膨胀 | 普通开发按用户需求固定 dtype |
| 4 | NZ size 按 ND 计算 | H2D/D2H 越界或精度异常 | 按 `ceil(cols/C0) * ceil(rows/16) * 16 * C0` 计算 |
| 5 | fp32 仍使用 C0=16 | NZ 数据错位 | fp32 C0=8，fp16/bf16 C0=16 |
| 6 | Basic Kernel 承载 vector epilogue | 设计不匹配 | Basic Kernel 只配 `BlockEpilogueEmpty`，融合需求走 fusion 文档 |
| 7 | 在本 skill 中切换 full-load/StreamK/4-buffer | 缺少 tiling 与 block 契约 | 默认只使用 SWAT |

---

## §8 验证建议

普通 MatMul 单算子建议至少验证：

- 典型 shape：小 shape、方阵、非方阵、多 tile、非 64/128 规则 shape。
- layout：ND/ND、ND/NZ、NZ/ND、NZ/NZ。
- transpose：`false/false`、`true/false`、`false/true`、`true/true`。
- bias：无 bias / 有 bias。

精度阈值建议：

| dtype | rtol | atol |
|-------|------|------|
| fp16 | 1e-3 | 1e-3 |
| bf16 | 2e-2 | 2e-2 |
| fp32 | 1e-4 | 1e-4 |

---

**下一步**：→ `references/development/step3-launcher.md`（编写 Launcher）
