/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUP_MATMUL_BLOCK_SCHEDULER_H
#define GROUP_MATMUL_BLOCK_SCHEDULER_H

#include "kernel_utils/common_utils.h"
#include "kernel_utils/tuple_utils.h"

#include "./block_scheduler_utils.h"

namespace Block {

class GroupMatmulBlockSchedulerSplitM {
public:
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        int32_t baseM{0};
        int32_t baseN{0};
    };

private:
    static constexpr int64_t WINDOW_LEN = 4LL;

    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t mTailAlign_{1};
    int64_t nTailAlign_{1};
    int64_t tailCnt_{1};
    int64_t tailBlockBase_{0};
    int64_t mainMWindow_{1};
    int64_t tailWindow_{1};
    int64_t mainRow_{0};
    int64_t round_{0};
    int64_t roundIdx_{0};
    int32_t baseM_{0};
    int32_t baseN_{0};
    int32_t mBaseTail_{0};
    int32_t nBaseTail_{0};
    uint32_t blockNum_{static_cast<uint32_t>(AscendC::GetBlockNum())};
    uint32_t blockIdx_{static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetTaskRation())};
    uint32_t startBlockIdx_{0};
    uint32_t endBlockIdx_{blockNum_ - 1};

public:
    __aicore__ inline explicit GroupMatmulBlockSchedulerSplitM(const Params& params)
        : baseM_(params.baseM), baseN_(params.baseN)
    {}

    __aicore__ inline GroupMatmulBlockSchedulerSplitM(int32_t baseM, int32_t baseN, int32_t /*baseK*/)
        : GroupMatmulBlockSchedulerSplitM(Params{baseM, baseN})
    {}

    __aicore__ inline void UpdateNextProblem(const TupleShape& problemShape)
    {
        k_ = Get<MNK_K>(problemShape);
        if (m_ != Get<MNK_M>(problemShape) || n_ != Get<MNK_N>(problemShape)) {
            m_ = Get<MNK_M>(problemShape);
            n_ = Get<MNK_N>(problemShape);
            mCnt_ = CeilDiv(m_, baseM_);
            nCnt_ = CeilDiv(n_, baseN_);
            mBaseTail_ = static_cast<int32_t>(m_ - (mCnt_ - 1) * baseM_);
            nBaseTail_ = static_cast<int32_t>(n_ - (nCnt_ - 1) * baseN_);
            totalCnt_ = mCnt_ * nCnt_;
            mainMWindow_ = Min(WINDOW_LEN, mCnt_);
            mainRow_ = mCnt_ / mainMWindow_ - 1;
            tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;
        }
        roundIdx_ = 0;
        round_ = CeilDiv(totalCnt_, static_cast<int64_t>(blockNum_));
        startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
        endBlockIdx_ = static_cast<uint32_t>((totalCnt_ + startBlockIdx_ - 1) % blockNum_);
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    __aicore__ inline void UpdateBaseM(uint32_t baseM)
    {
        baseM_ = static_cast<int32_t>(baseM);
    }

    __aicore__ inline void SetTailAlign(uint32_t mTailAlign, uint32_t nTailAlign)
    {
        mTailAlign_ = mTailAlign;
        nTailAlign_ = nTailAlign;
    }

    __aicore__ inline int64_t GetTailTileCnt()
    {
        return Min(static_cast<int64_t>(endBlockIdx_ + 1), totalCnt_);
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
    {
        mTailCnt_ = mTailCnt;
        nTailCnt_ = nTailCnt;
        tailCnt_ = mTailCnt_ * nTailCnt_;
        int64_t tailOriCnt = GetTailTileCnt();
        int64_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (tailCnt_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailCnt_ = 1;
            nTailCnt_ = 1;
            tailCnt_ = 1;
            tailBlockBase_ = 0;
        } else if (tailCnt_ > 1) {
            tailBlockBase_ = endBlockIdx_ + 1 - tailOriCnt;
        } else {
            tailBlockBase_ = 0;
        }
        endBlockIdx_ = static_cast<uint32_t>(newEndBlockIdx);
    }

    __aicore__ inline void UpdateTailTile()
    {
        int64_t tailTileCnt = GetTailTileCnt();
        int64_t remainTile = (static_cast<int64_t>(AscendC::GetBlockNum()) - endBlockIdx_ - 1) / tailTileCnt + 1;
        if (remainTile <= 1) {
            return;
        }

        int64_t mMin = Min(static_cast<int64_t>(AscendC::BLOCK_CUBE), mTailAlign_);
        int64_t nMin = Min(static_cast<int64_t>(AscendC::BLOCK_CUBE), nTailAlign_);
        int64_t mTile = Min(CeilDiv(static_cast<int64_t>(mBaseTail_), mMin), remainTile);
        int64_t nTile = Min(CeilDiv(static_cast<int64_t>(nBaseTail_), nMin), remainTile);
        while (mTile * nTile > remainTile) {
            if (mTile >= nTile) {
                mTile -= 1;
            } else {
                nTile -= 1;
            }
        }
        UpdateTailTile(static_cast<uint32_t>(mTile), static_cast<uint32_t>(nTile));
    }

    __aicore__ inline int64_t GetTileNum() const
    {
        return round_;
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (round_ == 0 || roundIdx_ > round_ - 1) {
            return false;
        }
        int64_t newBlockIdx = static_cast<int64_t>(blockIdx_);
        if (roundIdx_ == round_ - 1 && tailCnt_ > 1) {
            newBlockIdx = (tailBlockBase_ + ((newBlockIdx - tailBlockBase_) / tailCnt_) * tailCnt_) / tailCnt_;
        }
        int64_t index = newBlockIdx + roundIdx_ * blockNum_;
        if (blockIdx_ < startBlockIdx_) {
            index += blockNum_ - startBlockIdx_;
        } else if (tailCnt_ > 1 && endBlockIdx_ + 1 >= tailCnt_ * totalCnt_) {
            index -= (tailBlockBase_ + ((startBlockIdx_ - tailBlockBase_) / tailCnt_) * tailCnt_) / tailCnt_;
        } else {
            index -= startBlockIdx_;
        }

        int64_t rowIdx = index / nCnt_ / mainMWindow_;
        if (rowIdx < mainRow_) {
            Get<MNK_M>(blockCoord) = rowIdx * mainMWindow_ + index % mainMWindow_;
            Get<MNK_N>(blockCoord) = (index / mainMWindow_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = index - mainRow_ * mainMWindow_ * nCnt_;
            Get<MNK_M>(blockCoord) = mainRow_ * mainMWindow_ + tailIndex % tailWindow_;
            Get<MNK_N>(blockCoord) = (tailIndex / tailWindow_) % nCnt_;
        }

        if (rowIdx & 1) {
            Get<MNK_N>(blockCoord) = nCnt_ - 1 - Get<MNK_N>(blockCoord);
        }
        roundIdx_++;
        return true;
    }

    __aicore__ inline TupleShape GetBlockShape(const BlockCoord& blockCoord)
    {
        return GetBlockShape(blockCoord, roundIdx_ - 1);
    }

    __aicore__ inline TupleShape GetBlockShape(const BlockCoord& blockCoord, int64_t roundIdx) const
    {
        int64_t singleCoreM = Get<MNK_M>(blockCoord) != (mCnt_ - 1) ? baseM_ : mBaseTail_;
        int64_t singleCoreN = Get<MNK_N>(blockCoord) != (nCnt_ - 1) ? baseN_ : nBaseTail_;
        if (tailCnt_ == 1 || roundIdx < round_ - 1) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = CeilAlign(CeilDiv(singleCoreM, mTailCnt_), mTailAlign_);
        int64_t singleCoreNSplit = CeilAlign(CeilDiv(singleCoreN, nTailCnt_), nTailAlign_);
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = Min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
        singleCoreN = Min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline int64_t GetEndBlockIdx() const
    {
        return endBlockIdx_;
    }

    __aicore__ inline int64_t GetStartBlockIdx() const
    {
        return startBlockIdx_;
    }
};

} // namespace Block

#endif // GROUP_MATMUL_BLOCK_SCHEDULER_H
