/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_EPILOGUE_FUSION_MEMBASE_H
#define EPILOGUE_EPILOGUE_FUSION_MEMBASE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "epilogue/cv_sync_constants.h"
#include "kernel_utils/common_utils.h"

using namespace AscendC;

// ============================================================================
// MemBase Epilogue 参考样例 — matmul + Mul 融合
//
// 本文件是 MemBase 后融合开发路径的参考样例，展示一个完整的 MemBase
// Epilogue 实现（Output = matmul(A, B) * D）。
//
// 路径选择：
//   - 推荐 RegBase 路径（epilogue_fusion_regbase.h），适用于复杂公式
//     （GELU / SwiGLU / LayerNorm 等多中间值链），详见
//     matmul_fixpopti_regbase_epilogue.md
//   - MemBase 路径仅适用于简单 vector 场景：仅存在一个 vector 操作且
//     具有明确可用的 AscendC API 接口（如 AscendC::Mul/Add/Div）
//
// 开发方法：复制本文件为新 epilogue，修改：
//   1. Params 结构体（额外输入的 GM 地址）
//   2. StageNum（UB 分区数）
//   3. operator() 中的融合计算语句（替换 AscendC::Mul）
//
// 计算：Output = matmul(A, B) * D
//   - matmul 结果由 AIC Fixpipe 写入 UB offset 0 (float)
//   - D 为第二路输入 [M, N] (float)
//   - Output 为最终输出 [M, N] (float)
//
// UB 布局 (stageNum=2):
//   [0, matmulArea)                    : cLocal_ (matmul 结果)
//   [matmulArea, +stageSize_)          : dLocal_ (第二路输入 D)
//   [matmulArea+stageSize_, +stageSize_): cLocalTmp_ (计算结果)
// ============================================================================

class MulEpilogue {
public:
    using DataType = float;

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;
    static constexpr int StageNum = 2;

    struct Params {
        GM_ADDR multiplierGmAddr{nullptr};
        GM_ADDR outputGmAddr{nullptr};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    AscendC::LocalTensor<DataType> cLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<DataType> cLocalTmp_{AscendC::TPosition::VECIN, 0, UB_SIZE};
    AscendC::LocalTensor<DataType> dLocal_{AscendC::TPosition::VECIN, 0, UB_SIZE};

    AscendC::GlobalTensor<DataType> outputGlobal_;
    AscendC::GlobalTensor<DataType> multiplierGlobal_;

    int64_t stageSize_{0};
    ProblemShape problemShape_;

    __aicore__ inline void Init(
        Params const& params, int64_t l1M, int64_t l1N, ProblemShape& problemShape)
    {
        constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
        int64_t l1NAlign = ::CeilDiv(l1N, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t splitTaskRation = static_cast<int64_t>(AscendC::GetTaskRation());
        int64_t l1MSplit = ::CeilDiv(l1M, splitTaskRation);
        int64_t matmulArea = l1MSplit * l1NAlign;

        int64_t lastUBBytes = UB_SIZE - matmulArea * sizeof(DataType);
        int64_t usableElems = (lastUBBytes > 0) ? (lastUBBytes / StageNum / sizeof(DataType)) : 0;
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(usableElems / l1NAlign * l1NAlign),
            matmulArea);

        int64_t ubOffset = matmulArea;
        dLocal_ = cLocal_[ubOffset];
        ubOffset += stageSize_;
        cLocalTmp_ = cLocal_[ubOffset];

        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(params.outputGmAddr));
        multiplierGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(params.multiplierGmAddr));
    }

    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t blockShapeN = Get<1>(blockShape);

        int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                          ? (halfM - AscendC::GetSubBlockIdx()) : halfM;

        constexpr int64_t ALIGN_ELEM = 32 / sizeof(DataType);
        int64_t nAlign = ::CeilDiv(blockShapeN, ALIGN_ELEM) * ALIGN_ELEM;
        int64_t inputSize = blockShapeM * nAlign;
        int64_t stageSize = AscendC::Std::min(stageSize_, inputSize) / nAlign * nAlign;
        int64_t N = Get<MNK_N>(problemShape_);

        if (stageSize <= 0) {
            return;
        }

        int64_t stageOffset = 0;

        while (stageOffset < inputSize) {
            int64_t curStageSize = AscendC::Std::min(stageSize, inputSize - stageOffset);
            int64_t offset = dstOffset + (stageOffset / nAlign) * N;
            offset += AscendC::GetSubBlockIdx() * halfM * N;

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);

            uint16_t nRows = static_cast<uint16_t>(curStageSize / nAlign);
            uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(DataType));
            uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(DataType));
            constexpr uint32_t UB_DATA_BLOCK_BYTES = 32U;
            uint32_t ubStrideBytes =
                static_cast<uint32_t>(nAlign) * static_cast<uint32_t>(sizeof(DataType));
            uint32_t blockBytes =
                ::CeilDiv(rowBytes, UB_DATA_BLOCK_BYTES) * UB_DATA_BLOCK_BYTES;
            uint32_t ubRowGap = (ubStrideBytes - blockBytes) / UB_DATA_BLOCK_BYTES;
            AscendC::DataCopyExtParams dCopyParams{nRows, rowBytes, gmRowGap, ubRowGap, 0};
            AscendC::DataCopyPadExtParams<DataType> dPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(dLocal_, multiplierGlobal_[offset], dCopyParams, dPadParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

            AscendC::Mul(cLocalTmp_, cLocal_[stageOffset], dLocal_, curStageSize);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

            AscendC::DataCopyExtParams outParams{nRows, rowBytes, ubRowGap, gmRowGap, 0};
            AscendC::DataCopyPad<DataType>(outputGlobal_[offset], cLocalTmp_, outParams);

            stageOffset += curStageSize;
        }
    }

    __host_aicore__ static Params InitParams(Params const& args) { return args; }
};

#endif // EPILOGUE_EPILOGUE_FUSION_MEMBASE_H
