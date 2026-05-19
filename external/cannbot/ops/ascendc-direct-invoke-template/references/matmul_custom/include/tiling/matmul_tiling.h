/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_tiling.h
 * \brief Host-side SWAT tiling engine for the matmul non-full-load path.
 */

#ifndef MATMUL_TILING_H
#define MATMUL_TILING_H

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
#include "tiling/matmul_tiling_base.h"
#include "host_utils/common_utils.h"

// ============================================================================
// Matmul SWAT Tiling —— 根据 (m, n, k) 与平台缓存大小挑选：
//   baseM/baseN/baseK、L1 depth、tail-split 策略、usedCoreNum、dbL0c
//
// 算法纲要：
//   1. FormulateBasicBlock: 256x256 起步，结合 mCore*nCore >= aicNum 做 N 轴/M 轴分裂
//   2. OptimizeEdgeBasicBlock: 合并尾块以改善最后一行/列的 load-balance
//   3. CalcTailBasicBlock: 当余量 < aicNum 时把尾块 split 多份
//   4. CalL1Tiling: 依据 L1 容量决定 depthA1/depthB1 与 stepKa/stepKb
//
// [MODIFY] 该 tiling 按 fp16/bf16 byte size（DATA_SIZE_FP16=2）做 L1 预算。
// 切换数据类型（fp8=1 / fp32=4 / fp4x2=0.5）时，修改 DATA_SIZE_FP16 或新增类型分支。
// ============================================================================

class MatmulTilingSwat : public MatmulTilingBase {
public:
    MatmulTilingSwat() = default;
    ~MatmulTilingSwat() override = default;

protected:
    const char* TilingName() const override
    {
        return "swat";
    }

    void DoOpTiling(MatmulTilingData& tilingData) override
    {
        ResetBase();
        FormulateBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalL1Tiling();

        FormulateLoadBalanceBlock();
        if (runInfo_.baseM == BASIC_BLOCK_SIZE_256 && runInfo_.baseN == BASIC_BLOCK_SIZE_256) {
            OptimizeEdgeBasicBlock();
        }
        uint64_t remainSizeForAL1BL1 =
            args_.hasBias ? (platformInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) : platformInfo_.l1Size;
        runInfo_.stepKa =
            remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) / DATA_SIZE_FP16;
        runInfo_.stepKb = runInfo_.stepKa; // has bias, adjust stepK to suitable value
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;

        BuildTilingData(tilingData);
    }

private:
    void ResetBase()
    {
        runInfo_.usedCoreNum = platformInfo_.aicNum;
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_256; // 256 is better base
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / DATA_SIZE_FP16;
        runInfo_.stepM = 1;
        runInfo_.stepN = 1;
        runInfo_.iterateOrder = 0;
        runInfo_.dbL0c = 1;
        runInfo_.singleCoreK = args_.k;
        runInfo_.singleCoreM = runInfo_.baseM;
        runInfo_.singleCoreN = runInfo_.baseN;
        runInfo_.mBaseTailSplitCnt = 1;
        runInfo_.nBaseTailSplitCnt = 1;
        runInfo_.tailInfo.mTailMain = 0;
        runInfo_.tailInfo.nTailMain = 0;
    }

    void FormulateBasicBlock()
    {
        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore * nCore >= platformInfo_.aicNum) {
            runInfo_.baseM = std::min(Align(args_.m, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
            runInfo_.baseN = std::min(Align(args_.n, BASIC_BLOCK_SIZE_16), runInfo_.baseN);
            return;
        }
        CalcBasicBlock();
        mCore = CeilDiv(args_.m, runInfo_.baseM);
        nCore = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.usedCoreNum = mCore * nCore;
        uint64_t kValueAlign = Align(args_.k, BASIC_BLOCK_SIZE_16);
        uint64_t kValueMax = FloorAlign(
            platformInfo_.l0aSize / DB_SIZE / DATA_SIZE_FP16 / std::max(runInfo_.baseM, runInfo_.baseN),
            BASIC_BLOCK_SIZE_16);
        runInfo_.baseK = std::min(kValueAlign, kValueMax);
    }

    void FormulateLoadBalanceBlock()
    {
        runInfo_.baseM = std::min(Align(args_.m, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
        runInfo_.baseN = std::min(Align(args_.n, BASIC_BLOCK_SIZE_16), runInfo_.baseN);

        runInfo_.defaultBalance =
            CalcMultiCoreBalance(args_.m, args_.n, platformInfo_.aicNum, runInfo_.baseM, runInfo_.baseN);
        runInfo_.redundantData = CalcRedundantDataMovement(runInfo_.baseM, runInfo_.baseN, args_.m, args_.n);

        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);

        double singleBlockNum = static_cast<double>(mCore * nCore / platformInfo_.aicNum);

        constexpr double LOAD_BALANCE_RATE_LIMIT = 0.88;
        constexpr double MAX_SINGLE_CORE_ROUND = 10.0;
        bool needReselect = singleBlockNum >= 1.0 && singleBlockNum <= MAX_SINGLE_CORE_ROUND &&
                            runInfo_.defaultBalance < LOAD_BALANCE_RATE_LIMIT;

        if (needReselect) {
            uint64_t higherSingleX;
            uint64_t lowerSingleX;
            CalcSingleX(higherSingleX, lowerSingleX);

            uint64_t minMN = Align(std::min(args_.m, args_.n), BASIC_BLOCK_SIZE_16);
            uint64_t maxMN = Align(std::max(args_.m, args_.n), BASIC_BLOCK_SIZE_16);
            bool isMLarger = (args_.m > args_.n);
            if (lowerSingleX >= minMN) {
                HandleLargeSingleSide(minMN, maxMN, isMLarger);
            } else {
                HandleLargeBothSides(higherSingleX, lowerSingleX, minMN, isMLarger);
            }
        }

        if (singleBlockNum < 1.0) {
            CalcBasicBlock();
        }
        runInfo_.baseM = Align(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = Align(runInfo_.baseN, BASIC_BLOCK_SIZE_16);
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1UL;

        mCore = CeilDiv(args_.m, runInfo_.baseM);
        nCore = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.usedCoreNum = std::min(mCore * nCore, static_cast<uint64_t>(platformInfo_.aicNum));
        uint64_t kValueAlign = Align(args_.k, BASIC_BLOCK_SIZE_16);
        uint64_t kValueMax = FloorAlign(
            platformInfo_.l0aSize / DB_SIZE / DATA_SIZE_FP16 / std::max(runInfo_.baseM, runInfo_.baseN),
            BASIC_BLOCK_SIZE_16);
        runInfo_.baseK = std::min(kValueAlign, kValueMax);
    }

    void CalcBasicBlock()
    {
        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore == 0UL || nCore == 0UL) {
            return;
        }
        if (mCore <= nCore) {
            runInfo_.baseM = Align(CeilDiv(args_.m, mCore), BASIC_BLOCK_SIZE_16);
            mCore = CeilDiv(args_.m, runInfo_.baseM);
            nCore = runInfo_.usedCoreNum / mCore;
            runInfo_.baseN = Align(CeilDiv(args_.n, nCore), BASIC_BLOCK_SIZE_16);
        } else {
            runInfo_.baseN = Align(CeilDiv(args_.n, nCore), BASIC_BLOCK_SIZE_16);
            nCore = CeilDiv(args_.n, runInfo_.baseN);
            mCore = runInfo_.usedCoreNum / nCore;
            runInfo_.baseM = Align(CeilDiv(args_.m, mCore), BASIC_BLOCK_SIZE_16);
        }

        while (runInfo_.baseN >= runInfo_.baseM * NUM_TWO && nCore < runInfo_.usedCoreNum / NUM_TWO) {
            nCore = nCore * NUM_TWO;
            mCore = runInfo_.usedCoreNum / nCore;
            runInfo_.baseM = Align(CeilDiv(args_.m, mCore), BASIC_BLOCK_SIZE_16);
            runInfo_.baseN = Align(CeilDiv(args_.n, nCore), BASIC_BLOCK_SIZE_16);
            mCore = CeilDiv(args_.m, runInfo_.baseM);
            nCore = CeilDiv(args_.n, runInfo_.baseN);
        }

        while (runInfo_.baseM >= runInfo_.baseN * NUM_TWO && mCore < runInfo_.usedCoreNum / NUM_TWO) {
            mCore = mCore * NUM_TWO;
            nCore = runInfo_.usedCoreNum / mCore;
            runInfo_.baseM = Align(CeilDiv(args_.m, mCore), BASIC_BLOCK_SIZE_16);
            runInfo_.baseN = Align(CeilDiv(args_.n, nCore), BASIC_BLOCK_SIZE_16);
            mCore = CeilDiv(args_.m, runInfo_.baseM);
            nCore = CeilDiv(args_.n, runInfo_.baseN);
        }
    }

    void OptimizeEdgeBasicBlock()
    {
        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore * nCore < platformInfo_.aicNum || mCore == 1UL || nCore == 1UL) {
            return;
        }
        uint64_t mBaseTail = args_.m % runInfo_.baseM;
        uint64_t nBaseTail = args_.n % runInfo_.baseN;

        if (mBaseTail > 0UL && !args_.isATrans && (nBaseTail == 0UL || mBaseTail <= nBaseTail)) {
            GetOuterAxisTailCnt(false, runInfo_.mBaseTailSplitCnt, runInfo_.tailInfo.mTailMain);
        } else if (nBaseTail > 0UL && args_.isBTrans) {
            GetOuterAxisTailCnt(true, runInfo_.nBaseTailSplitCnt, runInfo_.tailInfo.nTailMain);
        }
    }

    void GetOuterAxisTailCnt(bool nLoadBalance, uint32_t& baseTailSplitCnt, uint64_t& tailMain)
    {
        uint64_t aicNum = platformInfo_.aicNum;
        uint64_t x = args_.m;
        uint64_t y = args_.n;
        uint64_t baseX = runInfo_.baseM;
        uint64_t baseY = runInfo_.baseN;
        if (nLoadBalance) {
            x = args_.n;
            y = args_.m;
            baseX = runInfo_.baseN;
            baseY = runInfo_.baseM;
        }

        uint64_t xCnt = CeilDiv(x, baseX);
        uint64_t yCnt = CeilDiv(y, baseY);
        uint64_t xTail = x % baseX;

        uint64_t totalWindows = CeilDiv(xCnt * yCnt, aicNum);
        uint64_t mainWindows = CeilDiv((xCnt - 1UL) * yCnt + yCnt % aicNum, aicNum);
        if (yCnt % aicNum == 0UL && (xCnt % WINDOW_LEN == 0UL || WINDOW_LEN % xCnt == 0UL)) {
            mainWindows = totalWindows;
        }
        uint64_t tailWindows = totalWindows - mainWindows;
        uint64_t perfRes = mainWindows * baseX + tailWindows * xTail;

        uint64_t baseTailCntMax = 1UL;
        baseTailCntMax = std::min((baseX - xTail) / BASIC_BLOCK_SIZE_16, xCnt);

        for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
            uint64_t newTailMain = Align(CeilDiv((mergeLen * baseX + xTail), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
            uint64_t newTailLast = mergeLen * (baseX - newTailMain) + xTail;
            uint64_t newMainRound = 0UL;
            uint64_t newTailRound = 0UL;
            if (mergeLen < xCnt - 1UL) {
                newMainRound = CeilDiv(((xCnt - 1UL - mergeLen) * yCnt + (mergeLen + 1UL) * yCnt) % aicNum, aicNum);
            }
            if (mergeLen > 0UL) {
                newTailRound = std::min(CeilDiv(mergeLen * yCnt + yCnt % aicNum, aicNum), totalWindows - newMainRound);
            }
            uint64_t curPerf = newMainRound * baseX + newTailRound * newTailMain +
                               (totalWindows - newMainRound - newTailRound) * newTailLast;
            if (curPerf < perfRes || (!nLoadBalance && curPerf == perfRes)) {
                perfRes = curPerf;
                tailMain = static_cast<uint32_t>(newTailMain);
                baseTailSplitCnt = static_cast<uint32_t>(mergeLen + 1UL);
            }
        }
    }

    void CalcTailBasicBlock()
    {
        uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);
        uint64_t mnCnt = mCnt * nCnt;
        uint64_t tailCnt = mnCnt <= platformInfo_.aicNum ? 0UL : mnCnt % platformInfo_.aicNum;
        runInfo_.tailInfo.mCnt = 1UL;
        runInfo_.tailInfo.nCnt = 1UL;
        if (tailCnt != 0UL) {
            while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= platformInfo_.aicNum) {
                runInfo_.tailInfo.mCnt += 1UL;
                if (runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= platformInfo_.aicNum) {
                    runInfo_.tailInfo.nCnt += 1UL;
                }
            }
        }
    }

    void CalL1Tiling()
    {
        uint64_t totalL1Size = platformInfo_.l1Size;
        uint64_t reserveBTSize = 0UL;
        runInfo_.depthA1 = totalL1Size / NUM_TWO / runInfo_.baseM / runInfo_.baseK / DATA_SIZE_FP16; // 2: half of l1
        runInfo_.depthB1 = totalL1Size / NUM_TWO / runInfo_.baseN / runInfo_.baseK / DATA_SIZE_FP16; // 2: half of l1

        uint64_t depthASize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * DATA_SIZE_FP16;
        uint64_t depthBSize = runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * DATA_SIZE_FP16;
        if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
            if (runInfo_.baseM <= runInfo_.baseN) {
                runInfo_.depthA1 = std::max(runInfo_.depthA1 / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
            } else {
                runInfo_.depthB1 = std::max(runInfo_.depthB1 / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
            }
        }
        runInfo_.stepKa = std::max(runInfo_.depthA1 / DB_SIZE, 1UL);
        runInfo_.stepKb = std::max(runInfo_.depthB1 / DB_SIZE, 1UL);

        // When aligned and base block is [256, 256], adjust stepK
        if (runInfo_.baseM == BASIC_BLOCK_SIZE_256 && runInfo_.baseN == BASIC_BLOCK_SIZE_256 &&
            args_.m % BASIC_BLOCK_SIZE_16 == 0 && args_.n % BASIC_BLOCK_SIZE_16 == 0 &&
            args_.k % BASIC_BLOCK_SIZE_16 == 0 && runInfo_.singleCoreK <= BASIC_BLOCK_SIZE_256) {
            runInfo_.stepKa = std::min(runInfo_.stepKa, 2UL);
            runInfo_.stepKb = std::min(runInfo_.stepKb, 2UL);
        }

        // Adjust stepKa and stepKb to be integer multiples of each other
        if (runInfo_.stepKa >= runInfo_.stepKb) {
            runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
        } else {
            runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
        }

        // Enable double buffer by default
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE; // depth % (stepKa * stepM) == 0
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE; // depth % (stepKb * stepN) == 0
        runInfo_.singleCoreM = runInfo_.baseM;
        runInfo_.singleCoreN = runInfo_.baseN;
        return;
    }

    double CalcMultiCoreBalance(uint64_t M, uint64_t N, uint64_t coreNum, uint64_t baseM, uint64_t baseN)
    {
        if (baseM == 0UL || baseN == 0UL) {
            return 0.0;
        }
        uint64_t mCnt = CeilDiv(M, baseM);
        uint64_t nCnt = CeilDiv(N, baseN);
        uint64_t mTail = M % baseM;
        uint64_t nTail = N % baseN;
        uint64_t mMainCnt = M / baseM;
        uint64_t nMainCnt = N / baseN;
        uint64_t totalTiles = mCnt * nCnt;
        uint64_t totalMainTiles = mMainCnt * nMainCnt;
        if (totalTiles == 0UL || coreNum == 0UL) {
            return 0.0;
        }
        double avgLoad = static_cast<double>(M * N) / coreNum;

        uint64_t singleMaxTail = std::max(baseM * nTail, baseN * mTail);

        uint64_t coreTailNum = totalTiles % coreNum;
        uint64_t tailRatio = coreTailNum != 0UL ? coreNum / coreTailNum : 1UL;

        auto getDiv = [](uint64_t base) -> uint64_t {
            if (base % BASIC_BLOCK_SIZE_256 == 0UL) {
                return BASIC_BLOCK_SIZE_16;
            } else if (base % BASIC_BLOCK_SIZE_128 == 0UL) {
                return BASIC_BLOCK_SIZE_16 / NUM_TWO;
            } else if (base % BASIC_BLOCK_SIZE_64 == 0UL) {
                return NUM_TWO * NUM_TWO;
            } else if (base % BLOCK_BYTE_SIZE == 0UL) {
                return NUM_TWO;
            } else {
                return 1UL;
            }
        };
        uint64_t mDiv = getDiv(baseM);
        uint64_t nDiv = getDiv(baseN);

        tailRatio = std::min(tailRatio, mDiv * nDiv);
        uint64_t tailNum = CeilDiv(mCnt * nCnt, coreNum) - CeilDiv(mMainCnt * nMainCnt, coreNum);
        uint64_t noDivTailNum = tailNum >= NUM_TWO ? tailNum - 1UL : 0UL;

        double maxLoad;
        if (tailNum == 0UL) {
            maxLoad = static_cast<double>(totalMainTiles / coreNum) * baseM * baseN +
                      static_cast<double>(CeilDiv(totalMainTiles, coreNum) - totalMainTiles / coreNum) * baseM * baseN /
                          tailRatio;
        } else {
            maxLoad = static_cast<double>(
                CeilDiv(totalMainTiles, coreNum) * baseM * baseN + noDivTailNum * singleMaxTail +
                (tailNum - noDivTailNum) * singleMaxTail / tailRatio);
        }
        return avgLoad / maxLoad;
    }

    uint64_t CalcRedundantDataMovement(uint64_t baseM, uint64_t baseN, uint64_t mValue, uint64_t nValue)
    {
        uint64_t mBlocks = CeilDiv(mValue, baseM);
        uint64_t nBlocks = CeilDiv(nValue, baseN);
        uint64_t totalMovement = mBlocks * nValue + nBlocks * mValue;
        return totalMovement - (mValue + nValue);
    }

    void CalcSingleX(uint64_t& higherSingleX, uint64_t& lowerSingleX)
    {
        double data = static_cast<double>(args_.m * args_.n) / platformInfo_.aicNum;

        while (data > static_cast<double>(BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256)) {
            data /= NUM_TWO;
        }

        double bestValue = std::sqrt(data);
        higherSingleX = Align(static_cast<uint64_t>(std::ceil(bestValue)), BASIC_BLOCK_SIZE_16);
        lowerSingleX = FloorAlign(static_cast<uint64_t>(std::floor(bestValue)), BASIC_BLOCK_SIZE_16);
    }

    uint64_t Floor(uint64_t value, uint64_t align)
    {
        return (value / align) * align;
    }

    void HandleLargeSingleSide(uint64_t minMN, uint64_t maxMN, bool isMLarger)
    {
        if (isMLarger) {
            runInfo_.baseN = minMN;
            runInfo_.baseM = platformInfo_.l0cSize / runInfo_.dbL0c / runInfo_.baseN;
            runInfo_.baseM = Floor(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
            CalcLargeSingleSide(maxMN, runInfo_.baseM, isMLarger);
        } else {
            runInfo_.baseM = minMN;
            runInfo_.baseN = platformInfo_.l0cSize / runInfo_.dbL0c / runInfo_.baseM;
            runInfo_.baseN = Floor(runInfo_.baseN, BLOCK_BYTE_SIZE);
            CalcLargeSingleSide(maxMN, runInfo_.baseN, isMLarger);
        }
    }

    void CalcLargeSingleSide(uint64_t maxMN, uint64_t& targetBase, bool isMLarger)
    {
        constexpr uint64_t MIN_BASE_BLOCK = 112UL;
        constexpr uint64_t MAX_BASE_BLOCK = 336UL;
        constexpr uint64_t NUM_NINE = 9UL;
        constexpr uint64_t NUM_TEN = 10UL;
        constexpr int MAX_LOOP_NUM = 20;

        uint64_t minCoreNum = (platformInfo_.aicNum + 1UL) * NUM_NINE / NUM_TEN;
        for (uint64_t tmpCoreNum = platformInfo_.aicNum; tmpCoreNum >= minCoreNum; tmpCoreNum--) {
            int loop = 1;
            while (loop <= MAX_LOOP_NUM) {
                uint64_t baseBlock = CeilDiv(maxMN, tmpCoreNum * loop);
                baseBlock = UpdateBaseBlock(baseBlock, isMLarger);
                if (baseBlock >= MIN_BASE_BLOCK && baseBlock <= MAX_BASE_BLOCK) {
                    targetBase = baseBlock;
                    return;
                }
                loop++;
            }
        }
        return;
    }

    uint64_t UpdateBaseBlock(uint64_t baseBlock, bool isMLarger)
    {
        if (!isMLarger) {
            return Align(baseBlock, BASIC_BLOCK_SIZE_128 / DATA_SIZE_FP16);
        } else {
            return Align(baseBlock, BASIC_BLOCK_SIZE_16);
        }
    }

    void HandleLargeBothSides(uint64_t higherSingleX, uint64_t lowerSingleX, uint64_t minMN, bool isMLarger)
    {
        constexpr uint64_t MIN_BASE_BLOCK = 112UL;
        constexpr uint64_t MAX_BASE_BLOCK = 336UL;

        CalcParams params1 = {
            lowerSingleX,
            MIN_BASE_BLOCK,
            true,
            runInfo_.defaultBalance,
            BASIC_BLOCK_SIZE_256,
            BASIC_BLOCK_SIZE_256,
            BASIC_BLOCK_SIZE_128 / DATA_SIZE_FP16};
        if (CalcBestBalance(params1, isMLarger)) {
            return;
        }

        CalcParams params2 = {higherSingleX,
                              std::min(MAX_BASE_BLOCK, minMN),
                              false,
                              runInfo_.defaultBalance,
                              BASIC_BLOCK_SIZE_256,
                              BASIC_BLOCK_SIZE_256,
                              BASIC_BLOCK_SIZE_128 / DATA_SIZE_FP16};
        if (CalcBestBalance(params2, isMLarger)) {
            return;
        }

        runInfo_.baseM = params1.baseM;
        runInfo_.baseN = params1.baseN;
        runInfo_.baseK = params1.baseK;
        if (params1.bestBalance < params2.bestBalance) {
            runInfo_.baseM = params2.baseM;
            runInfo_.baseN = params2.baseN;
            runInfo_.baseK = params2.baseK;
        }
    }

    struct CalcParams {
        uint64_t baseStart;
        uint64_t baseEnd;
        bool isNegativeSign;
        double bestBalance;
        uint64_t baseM;
        uint64_t baseN;
        uint64_t baseK;
    };

    bool CalcBestBalance(CalcParams& params, bool isMLarger)
    {
        constexpr double LOAD_BALANCING_THRESHOLD = 0.98;
        constexpr double MIN_EQUALIZATION_COEFFICIENT = 1.02;
        constexpr double BALANCE_REDUNDANT_THRESHOLD = 0.97;

        uint64_t startIndex;
        uint64_t count;
        uint64_t baseX = params.baseStart;
        bool condition = params.isNegativeSign ? baseX >= params.baseEnd : baseX <= params.baseEnd;
        while (condition) {
            bool isFindStartIndex = FindLoadBalanceInfo(baseX, startIndex, count);
            if (!isFindStartIndex) {
                baseX = params.isNegativeSign ? baseX - BASIC_BLOCK_SIZE_16 : baseX + BASIC_BLOCK_SIZE_16;
                condition = params.isNegativeSign ? baseX >= params.baseEnd : baseX <= params.baseEnd;
                continue;
            }

            for (uint64_t i = 0; i < count; i++) {
                if (startIndex + i >= BLOCK_TABLE.size()) {
                    break;
                }
                auto [x1, x2, x3, x4, x5] = BLOCK_TABLE[startIndex + i];
                (void)x5;
                if (x3 > args_.k) {
                    continue;
                }
                uint64_t currentBaseM = isMLarger ? x2 : x1;
                uint64_t currentBaseN = isMLarger ? x1 : x2;

                double balance =
                    CalcMultiCoreBalance(args_.m, args_.n, platformInfo_.aicNum, currentBaseM, currentBaseN) / x4;
                double removeRatio = static_cast<double>(
                    CalcRedundantDataMovement(currentBaseM, currentBaseN, args_.m, args_.n) / runInfo_.redundantData);
                bool isUpdateBaseBlock = false;
                if (balance - runInfo_.defaultBalance > removeRatio - BALANCE_REDUNDANT_THRESHOLD) {
                    isUpdateBaseBlock = UpdateBothBaseBlock(balance, params, currentBaseM, currentBaseN, x3);
                }
                if (isUpdateBaseBlock) {
                    return true;
                }
            }
            baseX = params.isNegativeSign ? baseX - BASIC_BLOCK_SIZE_16 : baseX + BASIC_BLOCK_SIZE_16;
            condition = params.isNegativeSign ? baseX >= params.baseEnd : baseX <= params.baseEnd;
        }
        return false;
    }

    bool UpdateBothBaseBlock(
        double balance, CalcParams& params, uint64_t currentBaseM, uint64_t currentBaseN, uint64_t baseK)
    {
        constexpr double LOAD_BALANCING_THRESHOLD = 0.98;
        constexpr double MIN_EQUALIZATION_COEFFICIENT = 1.02;

        if (balance > LOAD_BALANCING_THRESHOLD) {
            runInfo_.baseM = currentBaseM;
            runInfo_.baseN = currentBaseN;
            runInfo_.baseK = baseK;
            return true;
        } else if (
            (balance > params.bestBalance) && (balance > MIN_EQUALIZATION_COEFFICIENT * runInfo_.defaultBalance)) {
            params.bestBalance = balance;
            params.baseM = currentBaseM;
            params.baseN = currentBaseN;
            params.baseK = baseK;
        }
        return false;
    }

    bool FindLoadBalanceInfo(uint64_t block, uint64_t& startIndex, uint64_t& count)
    {
        for (const auto& entry : BLOCK_LOOKUP_TABLE) {
            if (std::get<0>(entry) == block) {
                startIndex = std::get<NUM_TWO>(entry);
                count = std::get<1>(entry);
                return true;
            }
        }
        return false;
    }

    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> BLOCK_LOOKUP_TABLE = {
        {336, 6, 0},   {320, 5, 6},   {304, 6, 11},  {288, 7, 17},  {272, 8, 24},
        {256, 9, 32},  {240, 10, 41}, {224, 10, 51}, {208, 11, 61}, {192, 12, 72},
        {176, 12, 84}, {160, 11, 96}, {144, 9, 107}, {128, 7, 116}, {112, 1, 123}};

    std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>> BLOCK_TABLE = {
        {336, 192, 48, 1, 1.126},     {336, 176, 48, 1, 1.126},     {336, 160, 48, 1, 1.126},
        {336, 144, 48, 1.073, 1.126}, {336, 128, 48, 1.167, 1.126}, {336, 112, 48, 1.287, 1.126},
        {320, 192, 48, 1, 1.126},     {320, 176, 48, 1, 1.126},     {320, 160, 48, 1.014, 1.126},
        {320, 144, 48, 1.089, 1.126}, {320, 128, 48, 1.183, 1.126}, {304, 208, 48, 1, 1.126},
        {304, 192, 48, 1, 1.126},     {304, 176, 48, 1, 1.126},     {304, 160, 48, 1.032, 1.126},
        {304, 144, 48, 1.107, 1.126}, {304, 128, 48, 1.201, 1.126}, {288, 224, 48, 1, 1.126},
        {288, 208, 48, 1, 1.126},     {288, 192, 48, 1, 1.126},     {288, 176, 48, 1, 1.126},
        {288, 160, 48, 1.051, 1.126}, {288, 144, 48, 1.126, 1.126}, {288, 128, 48, 1.220, 1.126},
        {272, 240, 48, 1, 1.126},     {272, 224, 48, 1, 1.126},     {272, 208, 48, 1, 1.126},
        {272, 192, 48, 1, 1.126},     {272, 176, 48, 1.012, 1.126}, {272, 160, 48, 1.073, 1.126},
        {272, 144, 48, 1.148, 1.126}, {272, 128, 48, 1.242, 1.126}, {256, 256, 64, 1, 1},
        {256, 240, 64, 1, 1},         {256, 224, 64, 1, 1},         {256, 208, 64, 1, 1},
        {256, 192, 64, 1, 1},         {256, 176, 64, 1.037, 1},     {256, 160, 64, 1.098, 1},
        {256, 144, 64, 1.173, 1},     {256, 128, 64, 1.267, 1},     {240, 272, 48, 1, 1.126},
        {240, 256, 64, 1, 1},         {240, 240, 64, 1, 1},         {240, 224, 64, 1, 1},
        {240, 208, 64, 1, 1},         {240, 192, 64, 1.014, 1},     {240, 176, 64, 1.065, 1},
        {240, 160, 64, 1.126, 1},     {240, 144, 64, 1.201, 1},     {240, 128, 64, 1.295, 1},
        {224, 288, 48, 1, 1.126},     {224, 272, 48, 1, 1.126},     {224, 256, 64, 1, 1},
        {224, 240, 64, 1, 1},         {224, 224, 64, 1, 1},         {224, 208, 64, 1.003, 1},
        {224, 192, 64, 1.046, 1},     {224, 176, 64, 1.097, 1},     {224, 160, 64, 1.159, 1},
        {224, 144, 64, 1.234, 1},     {208, 304, 48, 1, 1.126},     {208, 288, 48, 1, 1.126},
        {208, 272, 48, 1, 1.126},     {208, 256, 64, 1, 1},         {208, 240, 64, 1, 1},
        {208, 224, 64, 1.003, 1},     {208, 208, 64, 1.04, 1},      {208, 192, 64, 1.083, 1},
        {208, 176, 64, 1.134, 1},     {208, 160, 64, 1.196, 1},     {208, 144, 64, 1.271, 1},
        {192, 336, 48, 1, 1.126},     {192, 320, 48, 1, 1.126},     {192, 304, 48, 1, 1.126},
        {192, 288, 48, 1, 1.126},     {192, 272, 48, 1, 1.126},     {192, 256, 64, 1, 1},
        {192, 240, 64, 1.014, 1},     {192, 224, 64, 1.046, 1},     {192, 208, 64, 1.083, 1},
        {192, 192, 80, 1.126, 1},     {192, 176, 80, 1.178, 1},     {192, 160, 80, 1.239, 1},
        {176, 336, 48, 1, 1.126},     {176, 320, 48, 1, 1.126},     {176, 304, 48, 1, 1.126},
        {176, 288, 48, 1, 1.126},     {176, 272, 48, 1.012, 1.126}, {176, 256, 64, 1.037, 1},
        {176, 240, 64, 1.065, 1},     {176, 224, 64, 1.097, 1},     {176, 208, 64, 1.134, 1},
        {176, 192, 80, 1.178, 1},     {176, 176, 80, 1.229, 1},     {176, 160, 80, 1.290, 1},
        {160, 336, 48, 1, 1.126},     {160, 320, 48, 1.014, 1.126}, {160, 304, 48, 1.032, 1.126},
        {160, 288, 48, 1.051, 1.126}, {160, 272, 48, 1.073, 1.126}, {160, 256, 64, 1.098, 1},
        {160, 240, 64, 1.126, 1},     {160, 224, 64, 1.159, 1},     {160, 208, 64, 1.196, 1},
        {160, 192, 80, 1.239, 1},     {160, 176, 80, 1.290, 1},     {144, 336, 48, 1.073, 1.126},
        {144, 320, 48, 1.089, 1.126}, {144, 304, 48, 1.107, 1.126}, {144, 288, 48, 1.126, 1.126},
        {144, 272, 48, 1.148, 1.126}, {144, 256, 64, 1.173, 1},     {144, 240, 64, 1.201, 1},
        {144, 224, 64, 1.234, 1},     {144, 208, 64, 1.271, 1},     {128, 336, 48, 1.167, 1.126},
        {128, 320, 48, 1.183, 1.126}, {128, 304, 48, 1.201, 1.126}, {128, 288, 48, 1.220, 1.126},
        {128, 272, 48, 1.242, 1.126}, {128, 256, 64, 1.267, 1},     {128, 240, 64, 1.295, 1},
        {112, 336, 48, 1.287, 1.126}};

    void BuildTilingData(MatmulTilingData& tilingData) const
    {
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mL1 = std::min(Align(args_.m, BASIC_BLOCK_SIZE_16), runInfo_.baseM * runInfo_.stepM);
        tilingData.nL1 = std::min(Align(args_.n, BASIC_BLOCK_SIZE_16), runInfo_.baseN * runInfo_.stepN);
        int32_t stepKa = std::min(runInfo_.stepKb, runInfo_.stepKa);
        int32_t STEPKA_THERSHOLD = 4;
        stepKa = std::min(STEPKA_THERSHOLD, stepKa);
        tilingData.kL1 = runInfo_.baseK * static_cast<uint32_t>(stepKa);
        tilingData.mTailCnt = static_cast<uint32_t>(runInfo_.tailInfo.mCnt);
        tilingData.nTailCnt = static_cast<uint32_t>(runInfo_.tailInfo.nCnt);
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = runInfo_.tailInfo.mTailMain;
        tilingData.nTailMain = runInfo_.tailInfo.nTailMain;
        tilingData.usedCoreNum = static_cast<uint32_t>(runInfo_.usedCoreNum);
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.l0cDB = static_cast<uint8_t>(runInfo_.dbL0c);
    }
};

#endif // MATMUL_TILING_H
