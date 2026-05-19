/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================
// Tiling SWAT — Streaming 模式 Tiling 引擎
// [CONFIG] baseM/baseN/baseK 的计算参数可按目标 Shape 调优
// [EXTEND] 融合模式需增加 UB 容量检查逻辑
// ============================================================

/*!
 * \file tiling_swat.h
 * \brief SWAT tiling specialization for the MX non-full-load path.
 */

#ifndef TILING_SWAT_H
#define TILING_SWAT_H

#include <algorithm>

#include "tiling_base.h"

template <DataType aDataType, DataType bDataType>
class QuantMatmulTilingSwat : public QuantMatmulTilingBase<aDataType, bDataType> {
public:
    QuantMatmulTilingSwat() = default;
    ~QuantMatmulTilingSwat() override = default;

private:
    using Base = QuantMatmulTilingBase<aDataType, bDataType>;
    using Base::args_;
    using Base::platformInfo_;
    using Base::runInfo_;

protected:
    const char* TilingName() const override
    {
        return "swat";
    }

    void DoOpTiling(QuantMatmulTilingData& tilingData) override
    {
        // The streaming path can reuse the common base block search directly,
        // then specializes only the tail split and L1-depth decisions.
        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalcPathSpecificL1();

        uint32_t scaleKL1 = CalcScaleKL1();
        uint8_t nBufferNum = CalculateDefaultNBufferNum(scaleKL1);
        BuildTilingData(tilingData, scaleKL1, nBufferNum);
    }

private:
    uint32_t CalcScaleKL1() const
    {
        // Scale reuse is bounded by the smaller reusable K window on the A and
        // B sides, because both scale tensors must stay valid for the same K range.
        return static_cast<uint32_t>(std::min(
            runInfo_.scaleFactorA * runInfo_.stepKa * runInfo_.baseK,
            runInfo_.scaleFactorB * runInfo_.stepKb * runInfo_.baseK));
    }

    uint8_t CalculateDefaultNBufferNum(uint32_t scaleKL1) const
    {
        // The default streaming layout tries four L1 buffers first and falls
        // back to double buffering if the combined A/B/scale footprint is too large.
        uint64_t stepK = std::min(runInfo_.stepKa, runInfo_.stepKb);
        uint64_t kL1 = stepK * runInfo_.baseK;
        uint64_t usedL1Size = GetSizeWithDataType<bDataType>(runInfo_.baseN * kL1) * L1_FOUR_BUFFER;
        usedL1Size += runInfo_.baseN * CeilDiv(static_cast<uint64_t>(scaleKL1), MX_GROUP_SIZE) * DB_SIZE;
        usedL1Size += GetSizeWithDataType<aDataType>(runInfo_.baseM * kL1) * L1_FOUR_BUFFER;
        usedL1Size += runInfo_.baseM * CeilDiv(static_cast<uint64_t>(scaleKL1), MX_GROUP_SIZE) * DB_SIZE;
        return static_cast<uint8_t>(usedL1Size < platformInfo_.l1Size ? L1_FOUR_BUFFER : DB_SIZE);
    }

    void BuildTilingData(QuantMatmulTilingData& tilingData, uint32_t scaleKL1, uint8_t nBufferNum) const
    {
        // Flatten the host-side search result into the POD payload consumed by
        // the launcher and device kernel.
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mTailTile = static_cast<uint32_t>(runInfo_.mTailTile);
        tilingData.nTailTile = static_cast<uint32_t>(runInfo_.nTailTile);
        tilingData.mBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.mBaseTailSplitCnt);
        tilingData.nBaseTailSplitCnt = static_cast<uint32_t>(runInfo_.nBaseTailSplitCnt);
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.nTailMain);
        tilingData.usedCoreNum =
            static_cast<uint32_t>((runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL)
                                      ? platformInfo_.aicNum
                                      : runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile);
        tilingData.dbL0c = static_cast<uint8_t>(runInfo_.dbL0c);
        tilingData.scaleKL1 = scaleKL1;
        tilingData.stepK = static_cast<uint8_t>(std::min(runInfo_.stepKa, runInfo_.stepKb));
        tilingData.nBufferNum = nBufferNum;
    }

    void CalcTailBasicBlock()
    {
        if (runInfo_.tailBlockCnt == 0UL) {
            return;
        }

        // Non-full-load can split both M and N tail tiles. Grow the heavier
        // edge first, but keep the total tail work within the available cores.
        uint64_t mTile = 1UL;
        uint64_t nTile = 1UL;
        uint64_t preSplit = 1UL;
        uint64_t secSplit = 1UL;
        uint64_t& preSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? mTile : nTile;
        uint64_t& secSplitValid = runInfo_.mTailSize >= runInfo_.nTailSize ? nTile : mTile;
        uint64_t tileMax = platformInfo_.aicNum / runInfo_.tailBlockCnt;
        uint64_t mTileMax = std::min(tileMax, CeilDiv(runInfo_.baseM, CUBE_BLOCK));
        uint64_t nTileMax = std::min(tileMax, CeilDiv(runInfo_.baseN, CUBE_BLOCK));
        uint64_t preSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? mTileMax : nTileMax;
        uint64_t secSplitMax = runInfo_.mTailSize >= runInfo_.nTailSize ? nTileMax : mTileMax;
        while ((CalUsedCoreNum(runInfo_, preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) ||
               (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax)) {
            if (CalUsedCoreNum(runInfo_, preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) {
                preSplitValid = ++preSplit;
            }
            if (CalUsedCoreNum(runInfo_, preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax) {
                secSplitValid = ++secSplit;
            }
        }

        runInfo_.mTailTile = mTile;
        runInfo_.nTailTile = nTile;
    }

    void CalcPathSpecificL1()
    {
        // This path allocates A, B, scaleA, and scaleB symmetrically, so it
        // first searches a shared depth and then refines A/B independently.
        uint64_t baseASize = GetSizeWithDataType<aDataType>(runInfo_.baseM * runInfo_.baseK);
        uint64_t baseBSize = GetSizeWithDataType<bDataType>(runInfo_.baseN * runInfo_.baseK);

        uint64_t baseScaleASize =
            Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseM;
        uint64_t baseScaleBSize =
            Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseN;
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        uint64_t depthInit = GetDepthA1B1(runInfo_, platformInfo_.l1Size, baseL1Size, 1UL);
        uint64_t leftL1SizeByDepthInit = platformInfo_.l1Size - depthInit * baseL1Size;
        uint64_t depthASec =
            GetDepthA1B1(runInfo_, leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        uint64_t depthBSec =
            GetDepthA1B1(runInfo_, leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        // Start from a symmetric A/B depth and only break symmetry if the
        // budget cannot sustain the larger common depth on both sides.
        runInfo_.depthA1 = std::max(depthASec, depthBSec);
        runInfo_.depthB1 = runInfo_.depthA1;
        if (runInfo_.depthA1 * baseL1Size > platformInfo_.l1Size) {
            runInfo_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
            runInfo_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
        }
        CalStepKs(args_, runInfo_);
        CalScaleFactors(args_, platformInfo_, runInfo_, baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    }

    void AdjustBasicBlock()
    {
        // Re-balance the initial M/N split when the first guess underutilizes
        // cores or produces a highly skewed tile.
        uint64_t mMaxtile = CeilDiv(args_.m, CUBE_BLOCK);
        uint64_t nMaxtile = CeilDiv(args_.n, CUBE_BLOCK);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;

        uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);

        if (mMaxtile > nMaxtile) {
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
            nCnt = CeilDiv(args_.n, tempBaseN);
            mCnt = platformInfo_.aicNum / nCnt;
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
        } else {
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
            mCnt = CeilDiv(args_.m, tempBaseM);
            nCnt = platformInfo_.aicNum / mCnt;
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
        }

        while (tempBaseN > tempBaseM * BASEM_BASEN_RATIO && nCnt < platformInfo_.aicNum / NUM_TWO &&
               tempBaseN != CUBE_BLOCK) {
            nCnt = nCnt * NUM_TWO;
            mCnt = platformInfo_.aicNum / nCnt;
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
            mCnt = CeilDiv(args_.m, tempBaseM);
            nCnt = CeilDiv(args_.n, tempBaseN);
        }
        while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCnt < platformInfo_.aicNum / NUM_TWO &&
               tempBaseM != CUBE_BLOCK) {
            mCnt = mCnt * NUM_TWO;
            nCnt = platformInfo_.aicNum / mCnt;
            tempBaseM = Align(CeilDiv(args_.m, mCnt), CUBE_BLOCK);
            tempBaseN = Align(CeilDiv(args_.n, nCnt), CUBE_BLOCK);
            mCnt = CeilDiv(args_.m, tempBaseM);
            nCnt = CeilDiv(args_.n, tempBaseN);
        }

        uint64_t kAlignValue = Align(args_.k, BASIC_BLOCK_SIZE_256);
        uint64_t kMaxValue = GetShapeWithDataType<aDataType>(platformInfo_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
        kMaxValue = FloorAlign(kMaxValue, BASIC_BLOCK_SIZE_256);
        if (kMaxValue >= BASIC_BLOCK_SIZE_256) {
            runInfo_.baseM = tempBaseM;
            runInfo_.baseN = tempBaseN;
            runInfo_.baseK = std::min(kAlignValue, kMaxValue);
            runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256)
                                                          : runInfo_.baseK;
        }
    }

    void CalcBasicBlock()
    {
        // Start from a 256-sized candidate tile, then refine it and capture
        // the tail statistics used by later scheduling decisions.
        runInfo_.baseM = Align(std::min(args_.m, BASIC_BLOCK_SIZE_256), CUBE_BLOCK);
        runInfo_.baseN = Align(std::min(args_.n, BASIC_BLOCK_SIZE_256), CUBE_BLOCK);
        runInfo_.baseK = Align(std::min(args_.k, aDataType == DataType::FP4 ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128), TILING_MXFP_DIVISOR_SIZE);

        uint64_t blockNum = CeilDiv(args_.m, runInfo_.baseM) * CeilDiv(args_.n, runInfo_.baseN);
        if (blockNum < platformInfo_.aicNum) {
            AdjustBasicBlock();
        }
        CHECK_COND(
            runInfo_.baseM != 0UL && runInfo_.baseN != 0UL && runInfo_.baseK != 0UL,
            "Failed to derive a valid tiling base shape: baseM, baseN, and baseK must all be non-zero.");

        runInfo_.mBlockCnt = CeilDiv(args_.m, runInfo_.baseM);
        runInfo_.nBlockCnt = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.totalBlockCnt = runInfo_.mBlockCnt * runInfo_.nBlockCnt;
        runInfo_.tailBlockCnt = runInfo_.totalBlockCnt % platformInfo_.aicNum;
        runInfo_.mTailSize = args_.m - (runInfo_.mBlockCnt - 1UL) * runInfo_.baseM;
        runInfo_.nTailSize = args_.n - (runInfo_.nBlockCnt - 1UL) * runInfo_.baseN;
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1U;
    }

    void OptimizeEdgeBasicBlock()
    {
        // Merge tiny M-edge tiles when K is cache-line aligned so the tail
        // block behaves more like the steady-state region.
        if (runInfo_.mBlockCnt == 1UL) {
            return;
        }
        uint64_t mTailSize = args_.m % runInfo_.baseM;
        bool isInnerAxisAlign = GetSizeWithDataType<aDataType>(args_.k) % MTE2_CACHELINE_SIZE == 0UL;
        if (mTailSize > 0UL && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
            uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
            uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;

            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax; mergeLen += windowSize,
                          ++mergeWindowNum) {
                uint64_t newTailMain =
                    Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                uint64_t curPerf =
                    (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.mTailMain = newTailMain;
                    runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    uint64_t CalUsedCoreNum(const QuantMatmulRunInfo& runInfo, uint64_t mTile, uint64_t nTile)
    {
        return mTile * nTile * runInfo.tailBlockCnt;
    }

    uint64_t GetDepthA1B1(const QuantMatmulRunInfo& runInfo, uint64_t leftSize, uint64_t perDepthSize,
                                 uint64_t depthInit)
    {
        // The first pass grows by powers of two to find a feasible region; the
        // second pass snaps the result to a DMA-friendly K granularity.
        if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
            return depthInit;
        }
        uint64_t depthScale = leftSize / perDepthSize;
        if (depthInit > 1UL) {
            uint64_t baseKSize = GetSizeWithDataType<aDataType>(runInfo.baseK);
            while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
                   (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
                depthScale -= 1UL;
            }
            if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
                (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
                depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
            }
            depthScale = std::max(depthScale, 1UL);
        } else {
            constexpr uint64_t SCALE_INDEX = 2UL;
            depthScale = 1UL;
            while (depthScale * perDepthSize < leftSize) {
                depthScale *= SCALE_INDEX;
            }
            depthScale = depthScale == 1UL ? depthScale : depthScale / SCALE_INDEX;
        }
        return depthInit * depthScale;
    }

    void CalStepKs(const QuantMatmulArgs& args, QuantMatmulRunInfo& runInfo)
    {
        // Convert L1 depth to step-K counts and keep A/B synchronized so both
        // sides advance through K with the same outer scheduling cadence.
        runInfo.stepKa = runInfo.depthA1 / DB_SIZE;
        runInfo.stepKb = runInfo.depthB1 / DB_SIZE;

        if (runInfo.stepKa * runInfo.baseK > args.k) {
            runInfo.stepKa = CeilDiv(args.k, runInfo.baseK);
        }

        if (runInfo.stepKb * runInfo.baseK > args.k) {
            runInfo.stepKb = CeilDiv(args.k, runInfo.baseK);
        }

        if (runInfo.stepKa > runInfo.stepKb) {
            runInfo.stepKa = runInfo.stepKa / runInfo.stepKb * runInfo.stepKb;
        }
        if (runInfo.stepKb > runInfo.stepKa) {
            runInfo.stepKb = runInfo.stepKb / runInfo.stepKa * runInfo.stepKa;
        }

        runInfo.stepKa = std::min(runInfo.stepKa, 4UL);
        runInfo.stepKb = std::min(runInfo.stepKb, 4UL);

        runInfo.depthA1 = runInfo.stepKa * DB_SIZE;
        runInfo.depthB1 = runInfo.stepKb * DB_SIZE;
    }

    void CalScaleFactors(const QuantMatmulArgs& args, const QuantMatmulPlatformInfo& platformInfo,
                                QuantMatmulRunInfo& runInfo, uint64_t baseASize, uint64_t baseBSize,
                                uint64_t baseScaleASize, uint64_t baseScaleBSize)
    {
        // Scale reuse is solved after A/B depth is fixed. The search keeps the
        // two scale paths balanced while staying inside the leftover L1 budget.
        uint64_t scaleFactorAMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleASize, SCALER_FACTOR_MAX);
        uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleBSize, SCALER_FACTOR_MAX);
        uint64_t scaleFactorA = args.k / (runInfo.stepKa * runInfo.baseK);
        uint64_t scaleFactorB = args.k / (runInfo.stepKb * runInfo.baseK);
        runInfo.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
        runInfo.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
        runInfo.scaleFactorA = std::min(scaleFactorAMax, runInfo.scaleFactorA);
        runInfo.scaleFactorB = std::min(scaleFactorBMax, runInfo.scaleFactorB);

        // `scaleInit` is the balanced reuse factor both sides can afford
        // without favoring either A or B. Any leftover space is then assigned
        // to the side that can still benefit from deeper scale reuse.
        uint64_t leftL1Size = platformInfo.l1Size - (runInfo.depthA1 * baseASize + runInfo.depthB1 * baseBSize);
        uint64_t scaleInit = leftL1Size / (runInfo.depthA1 * baseScaleASize + runInfo.depthB1 * baseScaleBSize);
        if (runInfo.scaleFactorA <= scaleInit && runInfo.scaleFactorB > scaleInit) {
            leftL1Size -= runInfo.scaleFactorA * runInfo.depthA1 * baseScaleASize;
            runInfo.scaleFactorB = std::min(leftL1Size / (runInfo.depthB1 * baseScaleBSize), runInfo.scaleFactorB);
        } else if (runInfo.scaleFactorB <= scaleInit && runInfo.scaleFactorA > scaleInit) {
            leftL1Size -= runInfo.scaleFactorB * runInfo.depthB1 * baseScaleBSize;
            runInfo.scaleFactorA = std::min(leftL1Size / (runInfo.depthA1 * baseScaleASize), runInfo.scaleFactorA);
        } else if (runInfo.scaleFactorA > scaleInit && runInfo.scaleFactorB > scaleInit) {
            leftL1Size -= scaleInit * runInfo.depthB1 * baseScaleBSize + scaleInit * runInfo.depthA1 * baseScaleASize;
            uint64_t scaleASec =
                std::min(leftL1Size / (runInfo.depthA1 * baseScaleASize), runInfo.scaleFactorA - scaleInit);
            uint64_t scaleBSec =
                std::min(leftL1Size / (runInfo.depthB1 * baseScaleBSize), runInfo.scaleFactorB - scaleInit);
            runInfo.scaleFactorA = scaleASec >= scaleBSec ? scaleASec + scaleInit : scaleInit;
            runInfo.scaleFactorB = scaleASec < scaleBSec ? scaleBSec + scaleInit : scaleInit;
        }
    }
};

#endif // TILING_SWAT_H
