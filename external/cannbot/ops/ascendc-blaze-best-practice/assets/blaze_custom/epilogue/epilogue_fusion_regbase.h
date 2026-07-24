/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_EPILOGUE_FUSION_REGBASE_H
#define EPILOGUE_EPILOGUE_FUSION_REGBASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "../utils/common_utils.h"

using namespace AscendC;

// ============================================================================
// RegBase Epilogue 参考样例 — matmul + vector 后融合
//
// 展示一个完整的 RegBase Epilogue 骨架，覆盖：
//   - SplitM 偏移计算（UB 读取无偏移 / GM 读写有偏移）
//   - tile 级与 stage 级 extra input 的双路同步（不同 eventID）
//   - Init 预发射反向依赖 / 析构排空
//   - DataCopyPad stride 参数（UB 侧传 0，GM 侧传 bytes）
//
// Vector 计算段只保留 Load → [USER COMPUTE] → Store 骨架，
// 开发者按需替换具体公式（Cast / Mul / Add / Exp 等）。
//
// 三接口合约（与 MatmulKernelFused 兼容）：
//   1) struct Params { ... };
//   2) void Init(Params, baseM, baseN, ProblemShape)
//   3) auto GetTensor()  — 返回 cLocal_
//   4) void operator()(BlockShape, gmOffset, flagId)
// ============================================================================

#ifdef __CCE_AICORE__

class EpilogueFusionRegBase {
public:
    using L0CDataType = int32_t;
    using ComputeType = float;
    using OutputType = bfloat16_t;

    static constexpr uint16_t ZERO_FLAG = 0;
    // tile 级 extra input 使用 eventID 0，stage 级 extra input 使用 eventID 1
    static constexpr uint16_t TILE_EVENT_ID = 0;
    static constexpr uint16_t STAGE_EVENT_ID = 1;

    constexpr static AscendC::Reg::CastTrait castTraitL0CToCompute = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    constexpr static AscendC::Reg::CastTrait castTraitComputeToOutput = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    struct Params {
        GM_ADDR extraInputAddr{nullptr};   // tile 级额外输入（如 per-channel scale）
        GM_ADDR extraInputBAddr{nullptr};  // stage 级额外输入（如 per-token scale）
        GM_ADDR outputGmAddr{nullptr};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    // cLocal_: matmul 结果（Fixpipe 写入 UB offset 0，不可释放）
    AscendC::LocalTensor<L0CDataType> cLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};

    // extraBufA_: tile 级额外输入（列依赖，加载一次跨 stage 只读复用）
    AscendC::LocalTensor<ComputeType> extraBufA_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};

    // extraBufB_: stage 级额外输入（行依赖，每 stage 覆盖）
    AscendC::LocalTensor<ComputeType> extraBufB_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};

    // outBuf_: 输出 staging
    AscendC::LocalTensor<OutputType> outBuf_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};

    AscendC::GlobalTensor<ComputeType> extraInputAGlobal_;
    AscendC::GlobalTensor<ComputeType> extraInputBGlobal_;
    AscendC::GlobalTensor<OutputType> outputGlobal_;

    int64_t stageRows_{0};
    int64_t nAlignL0C_{0};
    ProblemShape problemShape_;

    __aicore__ inline void Init(
        Params const& params, int64_t baseM, int64_t baseN, ProblemShape& problemShape)
    {
        // nAlign: UB 行对齐宽度（32B / sizeof(L0CDataType) 元素一组）
        nAlignL0C_ = ::CeilDiv(baseN, ALIGN_ELEM) * ALIGN_ELEM;

        // matmulAreaBytes: cLocal_ 占用空间，行步长是 nAlignL0C_（不是 L0C cube 边长 16）
        int64_t splitTaskRation = static_cast<int64_t>(AscendC::GetTaskRation());
        int64_t splitMRows = ::CeilDiv(baseM, splitTaskRation);
        int64_t matmulAreaBytes = splitMRows * nAlignL0C_ * sizeof(L0CDataType);

        // extraBufA_: 1 行 tile 级输入
        int64_t extraBufABytes = nAlignL0C_ * sizeof(ComputeType);

        // extraBufB_: stageRows 行 stage 级输入
        int64_t remainBytes = AscendC::TOTAL_UB_SIZE - matmulAreaBytes - extraBufABytes;
        // stagePerRowBytes = extraBufB_ 每行 + outBuf_ 每行
        int64_t stagePerRowBytes = nAlignL0C_ * sizeof(ComputeType) + nAlignL0C_ * sizeof(OutputType);
        stageRows_ = remainBytes / stagePerRowBytes;
        if (stageRows_ <= 0) {
            stageRows_ = 1;
        }

        int64_t extraBufAOffset = matmulAreaBytes / sizeof(L0CDataType);
        extraBufA_ = cLocal_[extraBufAOffset].template ReinterpretCast<ComputeType>();

        int64_t extraBufBOffset = (matmulAreaBytes + extraBufABytes) / sizeof(L0CDataType);
        extraBufB_ = cLocal_[extraBufBOffset].template ReinterpretCast<ComputeType>();

        int64_t extraBufBAreaBytes = stageRows_ * nAlignL0C_ * sizeof(ComputeType);
        int64_t outBufOffset = (matmulAreaBytes + extraBufABytes + extraBufBAreaBytes) / sizeof(L0CDataType);
        outBuf_ = cLocal_[outBufOffset].template ReinterpretCast<OutputType>();

        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutputType*>(params.outputGmAddr));
        extraInputAGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ComputeType*>(params.extraInputAddr));
        extraInputBGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ ComputeType*>(params.extraInputBAddr));

        // 预发射首轮反向依赖：
        // tile 级 extraBufA_ 的 V_MTE2 (eventID 0)
        // stage 级 extraBufB_ 的 V_MTE2 (eventID 1)
        // outBuf_ 的 MTE3_V (eventID 0)
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(TILE_EVENT_ID);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(STAGE_EVENT_ID);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        (void)flagId;
        int64_t curM = Get<0>(blockShape);
        int64_t curN = Get<1>(blockShape);

        // ---- SplitM 行数计算 ----
        int64_t halfM = ::CeilDiv(curM, AscendC::GetTaskRation());
        // 奇数 M 时 V1 比 V0 少一行；极端场景（curM=1）V1 localRows=0
        int64_t localRows = ((static_cast<uint64_t>(curM) & 1UL) > 0UL)
                              ? (halfM - AscendC::GetSubBlockIdx()) : halfM;
        if (localRows <= 0) {
            return;  // V1 无数据，CV 同步由 kernel 层处理
        }

        int64_t N = Get<MNK_N>(problemShape_);
        int64_t tileM0 = dstOffset / N;
        int64_t tileN0 = dstOffset % N;
        // GM offset 需要加 sub-block 偏移
        int64_t subM0 = tileM0 + AscendC::GetSubBlockIdx() * halfM;

        // per-call nAlign（从 blockShapeN，非 Init 时的 baseN，正确处理 tail tile）
        int64_t nAlign = ::CeilDiv(curN, ALIGN_ELEM) * ALIGN_ELEM;
        uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(ComputeType);
        uint16_t vfLoopNum = (static_cast<uint32_t>(nAlign) + VL - 1) / VL;

        // ---- tile 级 extra input：加载一次，跨 stage 只读复用 ----
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(TILE_EVENT_ID);
        {
            uint32_t rowBytes = static_cast<uint32_t>(curN * sizeof(ComputeType));
            uint32_t gmRowGap = static_cast<uint32_t>((N - curN) * sizeof(ComputeType));
            // GM→UB: srcStride=GM 侧 bytes, dstStride=UB 侧 32B 单位（nAlign 对齐时传 0）
            AscendC::DataCopyExtParams cp{1, rowBytes, gmRowGap, 0, 0};
            AscendC::DataCopyPadExtParams<ComputeType> pp{false, 0, 0, 0};
            AscendC::DataCopyPad(extraBufA_, extraInputAGlobal_[tileN0], cp, pp);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(TILE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(TILE_EVENT_ID);

        // UB 物理地址：SplitM 下每个 AIV 的 UB 已硬件分片，从 offset 0 读取，不需要 sub-block 偏移
        __ubuf__ L0CDataType* srcAddr = (__ubuf__ L0CDataType*)cLocal_.GetPhyAddr();
        __ubuf__ ComputeType* extraAAddr = (__ubuf__ ComputeType*)extraBufA_.GetPhyAddr();
        __ubuf__ ComputeType* extraBAddr = (__ubuf__ ComputeType*)extraBufB_.GetPhyAddr();
        __ubuf__ OutputType* dstAddr = (__ubuf__ OutputType*)outBuf_.GetPhyAddr();

        // ---- Stage 循环 ----
        for (int64_t stageOffset = 0; stageOffset < localRows; stageOffset += stageRows_) {
            int64_t rowsThisStage = AscendC::Std::min(stageRows_, localRows - stageOffset);
            int64_t stageM0 = subM0 + stageOffset;

            // stage 级 extra input：每 stage 加载 row-dependent 数据
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(STAGE_EVENT_ID);
            {
                uint16_t nRows = static_cast<uint16_t>(rowsThisStage);
                uint32_t rowBytes = static_cast<uint32_t>(sizeof(ComputeType));
                uint32_t gmRowGap = 0;
                AscendC::DataCopyExtParams cp{nRows, rowBytes, gmRowGap, 0, 0};
                AscendC::DataCopyPadExtParams<ComputeType> pp{false, 0, 0, 0};
                AscendC::DataCopyPad(extraBufB_, extraInputBGlobal_[stageM0], cp, pp);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(STAGE_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(STAGE_EVENT_ID);

            // 等待上一轮 MTE3 读完 outBuf_
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);

            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<L0CDataType> vregL0C;
                AscendC::Reg::RegTensor<ComputeType> vregCompute;
                AscendC::Reg::RegTensor<ComputeType> vregExtraA;
                AscendC::Reg::RegTensor<ComputeType> vregExtraB;
                AscendC::Reg::RegTensor<OutputType> vregOutput;
                AscendC::Reg::MaskReg mask;

                for (uint16_t row = 0; row < static_cast<uint16_t>(rowsThisStage); ++row) {
                    // UB 读取从 offset 0，SplitM 已硬件分片
                    __ubuf__ L0CDataType* rowSrc = srcAddr + (stageOffset + row) * nAlign;
                    __ubuf__ OutputType* rowDst = dstAddr + row * nAlign;

                    // [USER] 加载 stage 级行输入（广播模式示例）
                    AscendC::Reg::LoadAlign<ComputeType, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                        vregExtraB, extraBAddr + row);

                    for (uint16_t i = 0; i < vfLoopNum; ++i) {
                        uint32_t active = static_cast<uint32_t>(nAlign) - static_cast<uint32_t>(i) * VL;
                        if (active > VL) { active = VL; }
                        mask = AscendC::Reg::UpdateMask<ComputeType>(active);

                        AscendC::Reg::LoadAlign(vregL0C, rowSrc + i * VL);
                        AscendC::Reg::Cast<ComputeType, L0CDataType, castTraitL0CToCompute>(
                            vregCompute, vregL0C, mask);

                        // [USER] 加载 tile 级列输入
                        AscendC::Reg::LoadAlign(vregExtraA, extraAAddr + i * VL);

                        // [USER] 融合计算链 — 替换为实际公式
                        //   Reg::Mul(vregCompute, vregCompute, vregExtraA, mask);
                        //   Reg::Mul(vregCompute, vregCompute, vregExtraB, mask);
                        //   Reg::Add / Sub / Exp / Sqrt / ...

                        AscendC::Reg::Cast<OutputType, ComputeType, castTraitComputeToOutput>(
                            vregOutput, vregCompute, mask);
                        AscendC::Reg::StoreAlign<OutputType, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            rowDst + i * VL, vregOutput, mask);
                    }
                }
            }

            // 通知 MTE2 可覆盖 extraBufB_
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(STAGE_EVENT_ID);

            // 等待 V 完成，MTE3 才能读 outBuf_
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            // UB→GM 写回：GM offset 含 sub-block 偏移
            {
                int64_t gmRowOffset = subM0 * N + tileN0 + stageOffset * N;
                uint16_t nRows = static_cast<uint16_t>(rowsThisStage);
                uint32_t rowBytes = static_cast<uint32_t>(curN * sizeof(OutputType));
                uint32_t gmRowGap = static_cast<uint32_t>((N - curN) * sizeof(OutputType));
                // UB→GM: srcStride=UB 侧 32B 单位（nAlign 对齐时传 0），dstStride=GM 侧 bytes
                AscendC::DataCopyExtParams outParams{nRows, rowBytes, 0, gmRowGap, 0};
                AscendC::DataCopyPad<OutputType>(outputGlobal_[gmRowOffset], outBuf_, outParams);
            }

            // 通知 V 可覆盖 outBuf_
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
        }

        // 通知 MTE2 可覆盖 extraBufA_（tile 级，跨 stage 只读复用完毕）
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(TILE_EVENT_ID);
    }

    __host_aicore__ static Params InitParams(Params const& args) { return args; }

    __aicore__ ~EpilogueFusionRegBase()
    {
        // 排空所有反向依赖
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(TILE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(STAGE_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
    }

private:
    static constexpr int64_t ALIGN_ELEM = 32 / sizeof(L0CDataType);
};

#endif // __CCE_AICORE__

#endif // EPILOGUE_EPILOGUE_FUSION_REGBASE_H
