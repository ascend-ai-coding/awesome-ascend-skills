/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

// Stateless-across-groups scheduler for the grouped GLU reference kernel. MM
// SWAT remains the only tiling engine; this class only enumerates its M/N tiles.
template <class ProblemShape_>
class BlockSchedulerGroupMatmul {
public:
    using ProblemShape = ProblemShape_;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        int32_t baseM{0};
        int32_t baseN{0};
        uint32_t usedCoreNum{0};
        uint32_t blockIdxDivisor{1};
    };

    __aicore__ inline explicit BlockSchedulerGroupMatmul(const Params& params)
        : baseM_(params.baseM), baseN_(params.baseN), usedCoreNum_(params.usedCoreNum)
    {
        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t divisor = params.blockIdxDivisor == 0 ? 1 : params.blockIdxDivisor;
        logicalCoreIdx_ = blockIdx / divisor;
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& shape)
    {
        m_ = AscendC::Te::Get<0>(shape);
        n_ = AscendC::Te::Get<1>(shape);
        k_ = AscendC::Te::Get<2>(shape);
        mTiles_ = CeilDiv(m_, static_cast<int64_t>(baseM_));
        nTiles_ = CeilDiv(n_, static_cast<int64_t>(baseN_));
        totalTiles_ = mTiles_ * nTiles_;
        nextTile_ = logicalCoreIdx_;
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& coord)
    {
        if (baseM_ <= 0 || baseN_ <= 0 || usedCoreNum_ == 0 || nextTile_ < 0 || nextTile_ >= totalTiles_) {
            return false;
        }
        const int64_t mTile = nextTile_ / nTiles_;
        const int64_t nTile = nextTile_ % nTiles_;
        coord = {mTile, nTile, 0, 0};
        nextTile_ += static_cast<int64_t>(usedCoreNum_);
        return true;
    }

    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& coord) const
    {
        const int64_t mOffset = AscendC::Te::Get<0>(coord) * baseM_;
        const int64_t nOffset = AscendC::Te::Get<1>(coord) * baseN_;
        const int64_t curM = Min(baseM_, m_ - mOffset);
        const int64_t curN = Min(baseN_, n_ - nOffset);
        // Contract: {tileM, tileN, K, reserved}. M/N offsets are derived from
        // BlockCoord and baseM/baseN; fields 2/3 are not address offsets.
        return {curM, curN, k_, 0};
    }

private:
    __aicore__ inline static int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
    }

    __aicore__ inline static int64_t Min(int64_t lhs, int64_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    int64_t baseM_{0};
    int64_t baseN_{0};
    uint32_t usedCoreNum_{0};
    int64_t logicalCoreIdx_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t mTiles_{0};
    int64_t nTiles_{0};
    int64_t totalTiles_{0};
    int64_t nextTile_{0};
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
