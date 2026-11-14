/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements_v2_common.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_V2_COMMON_H
#define GATHER_ELEMENTS_V2_COMMON_H

#include "kernel_operator.h"

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    return b == 0 ? 0 : (a + b - 1) / b * b;
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

constexpr uint64_t TRANS_LEN = 16;
constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BYTE4 = 4;
constexpr uint64_t BYTE2 = 2;
constexpr uint8_t X_PROCESS = 0;
constexpr uint8_t IDX_PROCESS = 1;

struct TransParam {
    uint64_t xGatherDimSlice;
    uint64_t xGatherDimTail;
    uint64_t xSliceNum;
    uint64_t idxGatherDimSlice;
    uint64_t idxGatherDimTail;
    uint64_t idxSliceNum;
    uint64_t yGatherDimSlice;
    uint64_t yGatherDimTail;
    uint64_t ySliceNum;
};

struct GatherParam {
    uint64_t xGatherDimSlice;
    uint64_t xGatherDimTail;
    uint64_t xSliceNum;
    uint64_t idxGatherDimSlice;
    uint64_t idxGatherDimTail;
    uint64_t idxSliceNum;
};

namespace AscendC {
template <typename T_X, typename T_IDX>
class GatherElementsV2KernelBase {
public:
    __aicore__ inline GatherElementsV2KernelBase() {}

protected:
    __aicore__ inline void InitBasicParams(const GatherElementsV2TilingData& tiling)
    {
        blockIdx_ = GetBlockIdx();
        xAlign_ = static_cast<uint32_t>(BLOCK_SIZE / sizeof(T_X));
        idxAlign_ = static_cast<uint32_t>(BLOCK_SIZE / sizeof(T_IDX));
        xPreDim_ = tiling.params.xPreDim;
        xGatherDim_ = tiling.params.xGatherDim;
        xPostDim_ = tiling.params.xPostDim;
        idxPreDim_ = tiling.params.idxPreDim;
        idxGatherDim_ = tiling.params.idxGatherDim;
        idxPostDim_ = tiling.params.idxPostDim;
    }

    __aicore__ inline void InitGroupParams(const GatherElementsV2TilingData& tiling)
    {
        coreGroupNum_ = tiling.params.coreGroupNum;
        formerGroupNum_ = tiling.params.formerGroupNum;
        tailGroupNum_ = tiling.params.tailGroupNum;

        formerGroupPreDim_ = tiling.params.formerGroupPreDim;
        tailGroupPreDim_ = tiling.params.tailGroupPreDim;
        formerGroupCoreNum_ = tiling.params.formerGroupCoreNum;
        tailGroupCoreNum_ = tiling.params.tailGroupCoreNum;

        formerGroupFormerNum_ = tiling.params.formerGroupFormerNum;
        formerGroupTailNum_ = tiling.params.formerGroupTailNum;
        formerGroupFormerPostDim_ = tiling.params.formerGroupFormerPostDim;
        formerGroupTailPostDim_ = tiling.params.formerGroupTailPostDim;

        tailGroupFormerNum_ = tiling.params.tailGroupFormerNum;
        tailGroupTailNum_ = tiling.params.tailGroupTailNum;
        tailGroupFormerPostDim_ = tiling.params.tailGroupFormerPostDim;
        tailGroupTailPostDim_ = tiling.params.tailGroupTailPostDim;
    }

    __aicore__ inline void ComputeGroupInfo(uint64_t& groupId, uint64_t& groupCoreId)
    {
        if (blockIdx_ < formerGroupNum_ * formerGroupCoreNum_) {
            groupId = blockIdx_ / formerGroupCoreNum_;
            groupCoreId = blockIdx_ % formerGroupCoreNum_;
        } else {
            uint64_t remainCore = blockIdx_ - formerGroupNum_ * formerGroupCoreNum_;
            groupId = formerGroupNum_ + remainCore / tailGroupCoreNum_;
            groupCoreId = remainCore % tailGroupCoreNum_;
        }
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void VToMTE2Sync()
    {
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }

    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }

    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a > b ? b : a;
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a < b ? b : a;
    }

protected:
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_;
    GlobalTensor<T_IDX> idxGm_;

    // basic params
    uint64_t blockIdx_;
    uint64_t xAlign_;
    uint64_t idxAlign_;
    uint64_t xPreDim_;
    uint64_t xGatherDim_;
    uint64_t xPostDim_;
    uint64_t idxPreDim_;
    uint64_t idxGatherDim_;
    uint64_t idxPostDim_;

    // group params
    uint64_t coreGroupNum_;
    uint64_t formerGroupNum_;
    uint64_t tailGroupNum_;

    uint64_t formerGroupPreDim_;
    uint64_t tailGroupPreDim_;
    uint64_t formerGroupCoreNum_;
    uint64_t tailGroupCoreNum_;

    // former group params
    uint64_t formerGroupFormerNum_;
    uint64_t formerGroupTailNum_;
    uint64_t formerGroupFormerPostDim_;
    uint64_t formerGroupTailPostDim_;

    // tail group params
    uint64_t tailGroupFormerNum_;
    uint64_t tailGroupTailNum_;
    uint64_t tailGroupFormerPostDim_;
    uint64_t tailGroupTailPostDim_;
};
} // namespace AscendC

#endif // GATHER_ELEMENTS_V2_COMMON_H
