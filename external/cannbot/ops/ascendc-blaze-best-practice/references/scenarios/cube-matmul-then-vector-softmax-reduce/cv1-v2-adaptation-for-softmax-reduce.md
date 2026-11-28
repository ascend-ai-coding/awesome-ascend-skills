# 从 group_matmul_kernel_cv1_v2.h 适配为 Softmax/Reduce 两阶段 Kernel

本指导说明如何将 `assets/blaze_custom/kernel/group_matmul_kernel_cv1_v2.h`（C+V1+V2 两阶段组合）改写为 matmul+softmax/reduce 的两阶段 kernel。不新增 kernel 资产，在项目 `blaze_custom/kernel/` 中生成适配后的文件。

## 1. 何时需要本指导

- 场景为 `cube-matmul-then-vector-softmax-reduce`，需要 `tuple<V1, V2>` 两阶段 kernel
- kernel 结构特征与 [cv-sync §0](cv-sync-and-two-phase-entry.md) 匹配（栈 epilogue + idle return 在 Init 之前）。其他结构特征的 kernel 适配步骤需参照 §0 的结构维度表调整
- V1（PerTileEpilogue）已适配标准 BlockEpilogue 接口（`Init` + `operator()` + 析构）
- V2（CrossCoreEpilogue）已实现 `Init(params) + ReduceAll()` 接口
- 需要一个 `GemmUniversal` 特化来编排 Phase 1（Cv1Kernel）+ Phase 2（V2），而不是在 kernel entry 中内联

## 2. 适配前提

- 已读取 [CV Sync 与两阶段 Kernel 编排专题](cv-sync-and-two-phase-entry.md)
- 已读取 [Online Softmax/Reduce Epilogue 设计专题](online-softmax-reduce-epilogue-design.md)
- V1 PerTileEpilogue 资产已复制到项目
- V2 CrossCoreEpilogue 资产已复制到项目
- `blaze_custom/utils/common_utils.h` 已就位

## 3. cv1_v2 与 softmax 的差异分析

`cv1_v2` 是 C+V1+V2 两阶段编排的通用 kernel 模式，当前 per-token-quant 场景使用了此模式。其现有实现有 4 层逻辑与 softmax 不兼容：

| 层 | cv1_v2 (现有实现) | softmax 需求 | 原因 |
|----|--------------------------|-------------|------|
| **V2 调用约定** | `operator()(ws.Slice, out.Slice, aux.Slice, rowCount)` | `Init(params) + ReduceAll()` | softmax V2 内部做行分配和 GM 访问，不接受外部 Tensor slice |
| **行分配** | `RunV2` 外部做行分配，传 slice 给 V2 | V2 `Init` 内部做（跨核归约需读所有核数据） | softmax V2 需跨核读取 onlineMax/onlineSum，不能只读自己分到的行 |
| **Tensor 构造** | 1 workspace + 1 output + 1 aux | 4 workspace + 1 output，布局不同 | softmax 有 onlineMax/onlineSum/mHistory/expWorkspace 四个 GM workspace |
| **Params 字段** | `realM`/`workspaceRowPitch`/`outputGmAddr`/`auxOutputGmAddr` | 不需要这些外部字段 | softmax V2 从自己的 Params 获取所有 GM 地址 |

## 4. 逐步改写指南

### 4.1 复制源文件

将 `assets/blaze_custom/kernel/group_matmul_kernel_cv1_v2.h` 复制到项目 `blaze_custom/kernel/matmul_softmax_kernel.h`。

### 4.2 重命名 trait 和头文件守卫

| 原始 | 改写为 |
|------|--------|
| `GROUP_MATMUL_KERNEL_CV1_V2_H` | `MATMUL_SOFTMAX_KERNEL_H` |
| `IsCv1V2EpiloguePipeline` | `IsTwoPhaseSoftmaxEpilogue` |
| `GROUP_MATMUL_CV1_V2_SYNC_ALL_CONFIG` | `MATMUL_SOFTMAX_SYNC_ALL_CONFIG`（或直接用 `AscendC::SyncAll()`） |

### 4.3 删除现有实现特有的 V2 外部接口类型别名

删除以下 3 行（:57-59）：

```cpp
// 删除：
using V2InputType = typename BlockEpilogueV2::InputType;
using V2OutputType = typename BlockEpilogueV2::OutputType;
using V2AuxOutputType = typename BlockEpilogueV2::AuxOutputType;
```

softmax 的 V2 不暴露 `InputType`/`OutputType`/`AuxOutputType`，不需要这些别名。

### 4.4 删除现有实现特有的外部 Params 字段

将 `Params` 结构体从：

```cpp
struct Params {
    typename Cv1Kernel::Params cv1Params{};
    typename BlockEpilogueV2::Params epilogueV2Params{};
    int64_t realM{0};
    int64_t workspaceRowPitch{0};
    GM_ADDR epilogueV2OutputGmAddr{nullptr};
    GM_ADDR epilogueV2AuxOutputGmAddr{nullptr};
};
```

简化为：

```cpp
struct Params {
    typename Cv1Kernel::Params cv1Params{};
    typename BlockEpilogueV2::Params epilogueV2Params{};
    Params() = default;
};
```

`realM`/`workspaceRowPitch`/`epilogueV2OutputGmAddr`/`epilogueV2AuxOutputGmAddr` 是现有实现的外部参数。softmax 的 V2 从自己的 `Params`（`onlineMaxAddr`/`onlineSumAddr`/`mHistoryAddr`/`expWorkspaceAddr`/`softmaxOutAddr`）获取所有 GM 地址。

### 4.5 修改 operator() 签名

从：

```cpp
__aicore__ inline void operator()(const Params& params, __gm__ V2InputType* workspaceGmAddr)
```

改为：

```cpp
__aicore__ inline void operator()(Params const& params)
```

去掉 `workspaceGmAddr` 参数，匹配标准 `GemmUniversal` 的 `operator()(Params const&)` 签名。

### 4.6 修改 Cv1Kernel 调用

从：

```cpp
Cv1Kernel cv1Kernel;
cv1Kernel(params.cv1Params, workspaceGmAddr);
```

改为：

```cpp
Cv1Kernel cv1Kernel;
cv1Kernel(params.cv1Params);
```

softmax 的 Cv1Kernel（`GemmUniversal<BlockMmad, V1>`）不需要 `workspaceGmAddr`。

### 4.7 重写 RunV2

将整个 `RunV2` 方法（:97-143）替换为：

```cpp
__aicore__ inline static void RunV2(Params const& params)
{
    BlockEpilogueV2 epilogueV2;
    epilogueV2.Init(params.epilogueV2Params);
    epilogueV2.ReduceAll();
}
```

删除原有的行分配逻辑（`rank`/`workers`/`rowStart`/`rowEnd`/`rowCount`）、Tensor 构造（`workspace`/`output`/`auxiliary`）和 V2 `operator()` 调用。softmax 的 V2 内部自己做行分配（`Init` 中 `myRows_ = M / vecCoreNum`）和 GM 访问。

### 4.8 删除 MakeRowMajorLayout 辅助函数

删除整个 `MakeRowMajorLayout` 方法（:84-95）。适配后的 `RunV2` 不再需要构造 Tensor。

## 5. 适配后的完整代码骨架

```cpp
#ifndef MATMUL_SOFTMAX_KERNEL_H
#define MATMUL_SOFTMAX_KERNEL_H

#include "kernel_basic_intf.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/kernel/kernel_universal.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class BlockEpilogue>
struct IsTwoPhaseSoftmaxEpilogue : public AscendC::Std::false_type {};

template <class BlockEpilogueV1, class BlockEpilogueV2>
struct IsTwoPhaseSoftmaxEpilogue<AscendC::Std::tuple<BlockEpilogueV1, BlockEpilogueV2>>
    : public AscendC::Std::true_type {};

template <
    class ProblemShape_, class BlockMmad_, class BlockEpilogueV1_, class BlockEpilogueV2_, class BlockScheduler_,
    typename Enable_>
class GemmUniversal<
    ProblemShape_, BlockMmad_, AscendC::Std::tuple<BlockEpilogueV1_, BlockEpilogueV2_>, BlockScheduler_, Enable_> {
public:
    using ProblemShape = ProblemShape_;
    using BlockMmad = BlockMmad_;
    using BlockEpilogueV1 = BlockEpilogueV1_;
    using BlockEpilogueV2 = BlockEpilogueV2_;
    using BlockEpilogue = AscendC::Std::tuple<BlockEpilogueV1, BlockEpilogueV2>;
    using BlockScheduler = BlockScheduler_;
    using Cv1Kernel = GemmUniversal<ProblemShape, BlockMmad, BlockEpilogueV1, BlockScheduler>;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockEpilogueV1Params = typename BlockEpilogueV1::Params;
    using BlockEpilogueV2Params = typename BlockEpilogueV2::Params;

    struct Params {
        typename Cv1Kernel::Params cv1Params{};
        BlockEpilogueV2Params epilogueV2Params{};
        Params() = default;
    };

    __aicore__ inline void operator()(Params const& params)
    {
        // Phase 0: workspace 初始化（所有核参与，含 idle）
        // 前提：kernel 为栈 epilogue + idle return 在 Init 之前（见 cv-sync §0），
        // Init 中的 SyncAll 对 idle 核死锁，需移到此处
        // 注意：下方 InitWorkspaceGlobal 为 softmax 变体实现（初始化 onlineMax=-inf + onlineSum=0）
        // reduce 变体需改为初始化 partialResult（见 reduce-adaptation-guide §3.3）
        InitWorkspaceGlobal(params);
        AscendC::SyncAll();

        // Phase 1: MatMul + PerTile online softmax (AIC BlockMmad + AIV PerTileEpilogue)
        {
            Cv1Kernel cv1Kernel;
            cv1Kernel(params.cv1Params);
        }

        // Phase 2: Cross-core reduction + final rescale (AIV only)
        AscendC::SyncAll();
        if ASCEND_IS_AIV {
            RunV2(params);
        }
    }

private:
    using V1Params = typename BlockEpilogueV1::Params;
    using ComputeType = typename BlockEpilogueV1::ComputeType;
    using MakeLayoutND = Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<ComputeType>>;

    __aicore__ inline static void InitWorkspaceGlobal(Params const& params)
    {
        if ASCEND_IS_AIV {
            constexpr int32_t FLOAT32_NEG_INF = 0xFF800000;
            constexpr int64_t INIT_BUF_ELEMS = 128;
            constexpr int64_t INIT_BUF_BYTES = INIT_BUF_ELEMS * sizeof(ComputeType);

            const auto& epiParams = params.cv1Params.epilogueParams;
            int64_t M = static_cast<int64_t>(epiParams.m);
            int64_t cubeCoreNum = static_cast<int64_t>(epiParams.cubeCoreNum);
            uint32_t VL = VECTOR_REG_WIDTH / sizeof(ComputeType);

            auto gmOnlineMax = reinterpret_cast<__gm__ ComputeType*>(epiParams.onlineMaxAddr);
            auto gmOnlineSum = reinterpret_cast<__gm__ ComputeType*>(epiParams.onlineSumAddr);

            int64_t ubOffNegInf = 0;
            int64_t ubOffZero = ubOffNegInf + INIT_BUF_BYTES;
            uint16_t initVfN = static_cast<uint16_t>((INIT_BUF_ELEMS + VL - 1) / VL);

            __VEC_SCOPE__
            {
                Reg::MaskReg allMask = Reg::CreateMask<ComputeType, Reg::MaskPattern::ALL>();

                __ubuf__ ComputeType* negInfAddr = reinterpret_cast<__ubuf__ ComputeType*>(
                    asc_get_phy_buf_addr(0) + ubOffNegInf);
                Reg::RegTensor<ComputeType> vregNegInf;
                Reg::Duplicate(vregNegInf, *reinterpret_cast<const ComputeType*>(&FLOAT32_NEG_INF), allMask);
                for (uint16_t i = 0; i < initVfN; ++i) {
                    Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_NORM_B32>(
                        negInfAddr + i * VL, vregNegInf, allMask);
                }

                __ubuf__ ComputeType* zeroAddr = reinterpret_cast<__ubuf__ ComputeType*>(
                    asc_get_phy_buf_addr(0) + ubOffZero);
                Reg::RegTensor<ComputeType> vregZero;
                Reg::Duplicate(vregZero, ComputeType(0), allMask);
                for (uint16_t i = 0; i < initVfN; ++i) {
                    Reg::StoreAlign<ComputeType, Reg::StoreDist::DIST_NORM_B32>(
                        zeroAddr + i * VL, vregZero, allMask);
                }
            }

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            int64_t blk = static_cast<int64_t>(GetBlockIdx());
            int64_t totalElems = cubeCoreNum * M;
            int64_t totalAiv = static_cast<int64_t>(GetBlockNum()) * static_cast<int64_t>(GetTaskRation());
            int64_t perCore = CeilDiv(totalElems, totalAiv);
            int64_t start = blk * perCore;
            int64_t end = (start + perCore > totalElems) ? totalElems : (start + perCore);

            auto gmMaxT = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(gmOnlineMax), MakeLayoutND{}(1UL, totalElems));
            auto gmSumT = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(gmOnlineSum), MakeLayoutND{}(1UL, totalElems));

            for (int64_t off = start; off < end; off += INIT_BUF_ELEMS) {
                int64_t cur = (off + INIT_BUF_ELEMS > end) ? (end - off) : INIT_BUF_ELEMS;

                auto ubMaxT = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOffNegInf), MakeLayoutND{}(1UL, cur));
                auto gmMaxSlice = gmMaxT.Slice(Te::MakeCoord(0UL, off), Te::MakeShape(1UL, cur));
                Te::Copy(Te::MakeCopy(Te::CopyUB2GM{}), gmMaxSlice, ubMaxT);

                auto ubSumT = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::UB, ComputeType>(ubOffZero), MakeLayoutND{}(1UL, cur));
                auto gmSumSlice = gmSumT.Slice(Te::MakeCoord(0UL, off), Te::MakeShape(1UL, cur));
                Te::Copy(Te::MakeCopy(Te::CopyUB2GM{}), gmSumSlice, ubSumT);
            }
        }
    }

    __aicore__ inline static void RunV2(Params const& params)
    {
        BlockEpilogueV2 epilogueV2;
        epilogueV2.Init(params.epilogueV2Params);
        epilogueV2.ReduceAll();
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze

#endif // MATMUL_SOFTMAX_KERNEL_H
```

## 6. 验证检查清单

| 检查项 | 预期 |
|--------|------|
| trait 重命名 | `IsTwoPhaseSoftmaxEpilogue`，不与原 `IsCv1V2EpiloguePipeline` 冲突 |
| operator() 签名 | `(Params const& params)` — 无 `workspaceGmAddr` 参数 |
| Cv1Kernel 调用 | `cv1Kernel(params.cv1Params)` — 无 `workspaceGmAddr` 参数 |
| RunV2 | 只调用 `Init(params.epilogueV2Params) + ReduceAll()`，无行分配/Tensor 构造 |
| Params | 只含 `cv1Params` + `epilogueV2Params`，无现有实现特有字段 |
| MakeRowMajorLayout | 已删除 |
| V2InputType/V2OutputType/V2AuxOutputType | 已删除 |
| 头文件守卫 | `MATMUL_SOFTMAX_KERNEL_H` |
| namespace | `Blaze::Gemm::Kernel`（与 cv1_v2 一致） |
| InitWorkspaceGlobal | 在 Cv1Kernel 之前调用 |
| SyncAll（InitWorkspaceGlobal 之后） | 所有核到达，含 idle |
| V1 epilogue 析构 | 有 `initialized_` guard，idle 核跳过 `CleanUpSyncFlag` |
| kernel 结构特征 | 与 cv-sync §0 匹配（栈 epilogue + idle return 在 Init 之前）；不匹配时按 §0 适配方向调整 |

## 7. 模板匹配优先级

适配后的特化与现有特化的 C++ partial ordering 关系：

| 特化 | ScheduleType 约束 | Epilogue 约束 | 匹配优先级 |
|------|------------------|-------------|-----------|
| 非 tuple 的 Blaze 库特化（如 fixpipe_opti、qbmm_mx 等） | `KernelMmadMultiBlockFixpipeOpti` 等 | 非 tuple | 非 tuple 时不竞争 |
| `cv1_v2`（Skill 资产） | 无 | `tuple<V1, V2>` | 与本特化同时匹配 tuple，但本特化无额外约束 |
| **本特化**（适配后） | 无 | `tuple<V1, V2>` | 与 cv1_v2 **同等具体** |

**注意**：本特化与 `cv1_v2` 的 `Enable_` 都是 `void`（默认），模板参数列表相同。如果两个特化同时存在于编译单元中，编译器会报歧义错误。

**解决方法**：项目只需包含适配后的 `matmul_softmax_kernel.h`，不同时包含原始 `group_matmul_kernel_cv1_v2.h`。如果项目同时需要两个场景（per-token-quant + softmax），则需要在适配后的特化中增加 ScheduleType 约束以区分。
