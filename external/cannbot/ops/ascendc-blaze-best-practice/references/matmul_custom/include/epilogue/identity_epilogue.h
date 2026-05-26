/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_IDENTITY_EPILOGUE_H
#define EPILOGUE_IDENTITY_EPILOGUE_H

#include "kernel_operator.h"
#include "kernel_utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "epilogue/cv_sync_constants.h"

using namespace AscendC;

// [MODIFY] 自定义 Epilogue 需满足三接口合约：
//   1) using Params = ...      — 参数类型
//   2) void Init(Params, baseM, baseN, problemShape) — 初始化 UB 布局
//   3) void operator()(BlockShape, gmOffset, flagId)  — 逐 tile 处理 + 写出 + SetFlag
template <typename CType_>
class IdentityEpilogue {
public:
    using CType = CType_;
    using FloatType = float;

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = CvSync::MODE;

    static constexpr uint32_t UB_SIZE = 512 * 1024;

    struct Params {
        GM_ADDR cGmAddr{nullptr};
    };

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    LocalTensor<FloatType> cLocal_{TPosition::VECIN, 0, UB_SIZE};
    LocalTensor<CType> castLocal_{TPosition::VECIN, 0, UB_SIZE};
    GlobalTensor<CType> outputGlobal_;
    ProblemShape problemShape_;

    __aicore__ inline void Init(
        Params const& params, uint32_t baseM, uint32_t baseN, ProblemShape& problemShape)
    {
        problemShape_ = problemShape;
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(params.cGmAddr));
    }

    __aicore__ inline auto GetTensor() { return cLocal_; }

    __aicore__ inline void operator()(
        BlockShape const& blockShape, int64_t dstOffset, int64_t flagId = CvSync::AIV_TO_AIC_FLAG)
    {
        int64_t blockShapeM = Get<0>(blockShape);
        int64_t blockShapeN = Get<1>(blockShape);

        constexpr int64_t ALIGN_ELEM_FLOAT = 32 / sizeof(FloatType);
        int64_t nAlignFloat = ::CeilDiv(blockShapeN, ALIGN_ELEM_FLOAT) * ALIGN_ELEM_FLOAT;

        // SPLIT_M: AIV0/AIV1 各处理一半 M
        int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());
        blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                          ? (halfM - AscendC::GetSubBlockIdx()) : halfM;

        if (blockShapeM <= 0) {
            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
            return;
        }

        int64_t N = Get<MNK_N>(problemShape_);

        // bf16 DataCopyPad 行 stride 对齐：32B / 2B = 16 元素
        constexpr int64_t ALIGN_ELEM_BF16 = 32 / static_cast<int64_t>(sizeof(CType));
        int64_t nAlignBf16 = ::CeilDiv(blockShapeN, ALIGN_ELEM_BF16) * ALIGN_ELEM_BF16;

        // SPLIT_M: 每个 AIV 有独立 UB 空间，float 数据从 offset 0 开始
        int64_t bf16BufByteOffset = static_cast<int64_t>(blockShapeM) * nAlignFloat *
            static_cast<int64_t>(sizeof(FloatType));
        int64_t gmRowOffset = AscendC::GetSubBlockIdx() * halfM * N;

        // Step 1: Cast float → bf16，逐行处理 stride 差异
        for (int64_t row = 0; row < blockShapeM; row++) {
            AscendC::Cast(castLocal_[(bf16BufByteOffset / sizeof(CType)) + row * nAlignBf16],
                          cLocal_[row * nAlignFloat],
                          RoundMode::CAST_RINT, static_cast<uint32_t>(blockShapeN));
        }

        // Step 2: DataCopyPad bf16 UB → bf16 GM
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);

        int64_t gmOffset = dstOffset + gmRowOffset;
        uint16_t nRows = static_cast<uint16_t>(blockShapeM);
        uint32_t rowBytes = static_cast<uint32_t>(blockShapeN * sizeof(CType));
        uint32_t gmRowGap = static_cast<uint32_t>((N - blockShapeN) * sizeof(CType));
        uint32_t ubStrideBytes = static_cast<uint32_t>(nAlignBf16) * static_cast<uint32_t>(sizeof(CType));

        AscendC::DataCopyExtParams outParams{nRows, rowBytes, 0 /* ubRowGap */, gmRowGap, 0};
        AscendC::DataCopyPad<CType>(
            outputGlobal_[gmOffset],
            castLocal_[bf16BufByteOffset / sizeof(CType)],
            outParams);

        // Step 3: CV sync — 通知 AIC 本 tile UB 已释放
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(flagId);
    }

    __host_aicore__ static Params InitParams(Params const& args) { return args; }
};

#endif // EPILOGUE_IDENTITY_EPILOGUE_H
