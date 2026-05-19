// ============================================================
// [PATTERN] 融合 Kernel 层 — MIX 模式（AIC + AIV 统一循环）
//
// 本文件为**融合算子 Kernel 模板**。Developer 通常无需改动本文件，
// 仅需把自定义 Epilogue 类作为模板参数传入。
//
// 架构（参考 ops-nn kernel_matmul_mix_without_que.h）:
//   - AIC 和 AIV 共享同一个 operator() 循环
//   - 通过 ASCEND_IS_AIC / ASCEND_IS_AIV 在循环体内分发
//   - CV 同步由 Kernel 层统一管理（不在 BlockMmad/Epilogue）
//   - BlockMmad 仅负责计算 + FixPipe (L0C→UB)
//   - Epilogue 仅负责融合计算 + 写出 GM (可选再发 AIV→AIC Flag)
//
// 使用方式（Developer 侧）:
//   #include "kernel/matmul_kernel_fused.h"
//   #include "epilogue/my_custom_epilogue.h"   // 自实现
//   using FusedKernel = Kernel::MatmulKernelFused<
//       ProblemShape, BlockMmad, BlockScheduler, MyCustomEpilogue>;
//
// Epilogue 合约（必须满足）：详见 references/matmul_fusion_guide.md 的“三接口合约”与“机制说明”章节
//   1) 成员类型 Params
//   2) Init(const Params&, baseM, baseN, problemShape4)
//   3) operator()(BlockShape, gmOffset, aivAicFlagId)
// ============================================================
#ifndef MATMUL_KERNEL_FUSED_H
#define MATMUL_KERNEL_FUSED_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"
#include "tensor.h"

#include "../block/block_mmad_swat.h"
#include "../block/block_scheduler.h"
#include "../utils/constants.h"
#include "../epilogue/cv_sync_constants.h"
// 本模板默认不 include 任何具体 Epilogue，避免产生不必要耦合。

namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue>
class MatmulKernelFused {
public:
    // ---- CV Sync Constants (与 ops-nn kernel_matmul_mix_without_que.h 一致) ----
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;
    static constexpr int16_t AIV_SYNC_AIC_FLAG = CvSync::AIV_TO_AIC_FLAG;
    static constexpr int16_t AIC_SYNC_AIV_FLAG = CvSync::AIC_TO_AIV_FLAG;
    static constexpr int16_t FLAG_ID_MAX = 16;
    static constexpr int16_t COUNT_ID_MAX = CvSync::COUNT_ID_MAX;
    static constexpr int16_t COUNT_FLAG = CvSync::COUNT_FLAG;

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using EpilogueParams = typename Epilogue::Params;
    using ProblemShapeType = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    using MakeLayoutA = AscendC::Te::NDLayoutFormat<AType>;
    using MakeLayoutB = AscendC::Te::DNLayoutFormat<BType>;
    using MakeLayoutScaleA = AscendC::Te::ScaleANDLayoutFormat<fp8_e8m0_t>;
    using MakeLayoutScaleB = AscendC::Te::ScaleBDNLayoutFormat<fp8_e8m0_t>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint8_t dbL0C;
    };

    // ---- Params 结构 ----
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
        EpilogueParams epilogueParams;
    };

    // ---- 统一入口 (AIC + AIV 共享) ----
    __aicore__ inline void operator()(const Params& params)
    {
        // ---- 1. 构造 Scheduler (AIC/AIV 共享同一个实例) ----
        //   BlockScheduler 内部通过 AscendC::GetBlockIdx() / GetBlockNum() 获取索引，
        //   无需外部传入 curBlockIdx / blockNum。
        BlockSchedulerOp bs(params.problemShape, params.schParams);

        // ---- 2. 初始化 Epilogue (AIV 侧 UB 布局) ----
        Epilogue epilogueOp;
        ProblemShapeType problemShape4 = {
            params.problemShape.m, params.problemShape.n, params.problemShape.k, 1};
        epilogueOp.Init(params.epilogueParams, params.qbmmParams.baseM, params.qbmmParams.baseN, problemShape4);

        // ---- 3. 初始化 BlockMmad (AIC 侧 L1/L0 流水线) ----
        BlockMmad blockMmadOp;
        if ASCEND_IS_AIC {
            TupleShape problemShape3 = {
                params.problemShape.m, params.problemShape.n, params.problemShape.k};
            BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
            bool enableL0cPingPong = (params.qbmmParams.dbL0C > 1);
            blockMmadOp.Init(problemShape3, l0TileShape, params.l1Params, enableL0cPingPong);
        }

        // ---- 4. 构建 GM Tensor views (AIC 侧需要) ----
        int64_t kScaleSize =
            ::CeilDiv(static_cast<int64_t>(params.problemShape.k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        auto layoutA = MakeLayoutA{}(params.problemShape.m, params.problemShape.k);
        auto layoutScaleA = MakeLayoutScaleA{}(params.problemShape.m, kScaleSize);
        auto layoutB = MakeLayoutB{}(params.problemShape.k, params.problemShape.n);
        auto layoutScaleB = MakeLayoutScaleB{}(kScaleSize, params.problemShape.n);

        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr)), layoutA);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr)), layoutScaleA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr)), layoutB);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr)), layoutScaleB);

        // gmC: 在融合模式下不直接写 GM，但 BlockMmad::operator() 签名需要
        auto layoutC = AscendC::Te::MakeNDLayout<CType>(params.problemShape.m, params.problemShape.n);
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)), layoutC);

        // ---- 5. 统一 tile 循环 (仿 kernel_matmul_mix_without_que.h:195-257) ----
        int64_t n = params.problemShape.n;
        int64_t count = 0;
        int64_t countId = 0;
        bool enableCVSync = false;
        constexpr int64_t kPos = 0L;

        BlockCoord blockIdx;
        while (bs.GetTileIdx(blockIdx)) {
            int64_t mPos = Get<MNK_M>(blockIdx);
            int64_t nPos = Get<MNK_N>(blockIdx);
            BlockShape singleShape = bs.GetBlockShape(blockIdx);
            int64_t curM = Get<MNK_M>(singleShape);
            int64_t curN = Get<MNK_N>(singleShape);
            if (curM <= 0 || curN <= 0) {
                return;
            }

            // offsetC: RowMajor GM 偏移
            int64_t offsetC = mPos * n + nPos;

            // ==== AIC 侧: matmul + FixPipe L0C→UB + CV SetFlag ====
            if ASCEND_IS_AIC {
                // 背压等待: Wait 上一轮 AIV 完成 (仿 ops-nn:223-229)
                if (enableCVSync) {
                    countId = count / COUNT_ID_MAX % COUNT_FLAG;
                    // 等待 AIV0
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + countId);
                    // 等待 AIV1
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                        AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
                }

                // 切片 GM tensors
                auto gmBlockA =
                    gmA(AscendC::Te::MakeCoord(mPos, kPos),
                        AscendC::Te::MakeShape(curM, params.problemShape.k));
                auto gmBlockScaleA =
                    gmScaleA(AscendC::Te::MakeCoord(mPos, kPos),
                             AscendC::Te::MakeShape(curM, kScaleSize));
                auto gmBlockB =
                    gmB(AscendC::Te::MakeCoord(kPos, nPos),
                        AscendC::Te::MakeShape(params.problemShape.k, curN));
                auto gmBlockScaleB =
                    gmScaleB(AscendC::Te::MakeCoord(kPos, nPos),
                             AscendC::Te::MakeShape(kScaleSize, curN));
                auto gmBlockC =
                    gmC(AscendC::Te::MakeCoord(mPos, nPos),
                        AscendC::Te::MakeShape(curM, curN));

                // 执行 BlockMmad: 计算 + Fixpipe (L0C→UB offset 0, SPLIT_M 双 AIV)
                blockMmadOp(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockC, singleShape);

                // CV SetFlag: 通知 AIV 数据就绪 (仿 ops-nn:236-243)
                enableCVSync = true;
                count++;
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                // 通知 AIV0
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIC_SYNC_AIV_FLAG + countId);
                // 通知 AIV1
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIC_SYNC_AIV_FLAG + countId + FLAG_ID_MAX);
            }

            // ==== AIV 侧: CV WaitFlag + Epilogue(Div + 写出) ====
            if ASCEND_IS_AIV {
                count++;
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                // 等待 AIC FixPipe 完成 (仿 ops-nn:250)
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(
                    AIC_SYNC_AIV_FLAG + countId);
                // 执行 Epilogue: 融合计算 + 写出 GM + SetFlag(AIV→AIC)
                // 约定：Epilogue::operator()(singleShape, gmOffset, aivAicFlagId)
                epilogueOp({curM, curN, 1, 1}, offsetC, (AIV_SYNC_AIC_FLAG + countId));
            }
        }

        // ---- 6. AIC drain: 等待最后一轮 AIV 完成 (仿 ops-nn:259-264) ----
        if ASCEND_IS_AIC {
            if (enableCVSync) {
                countId = count / COUNT_ID_MAX % COUNT_FLAG;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIV_SYNC_AIC_FLAG + countId);
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(
                    AIV_SYNC_AIC_FLAG + countId + FLAG_ID_MAX);
            }
        }
    }
};

} // namespace Kernel

#endif // MATMUL_KERNEL_FUSED_H
