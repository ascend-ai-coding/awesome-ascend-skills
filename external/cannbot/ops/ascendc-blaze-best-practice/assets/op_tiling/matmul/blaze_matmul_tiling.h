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
 * \file blaze_matmul_tiling.h
 * \brief Host-side SWAT tiling engine for ordinary matmul kernels.
 *
 * Ordinary matmul means A/B input matrices, optional bias, and one output C.
 * Grouped matmul reuses this engine with totalM; group metadata is not part of
 * the tiling data. Full-load, StreamK, and 4-buffer variants are intentionally
 * not provided by this skill.
 */

#ifndef BLAZE_MATMUL_TILING_H
#define BLAZE_MATMUL_TILING_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "platform/platform_ascendc.h"
#include "blaze_matmul_tiling_data.h"

namespace blaze_matmul_tiling {

constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t NUM_TWO = 2UL;

#define BLAZE_TILING_CHECK_COND(cond, msg)                                                                        \
    do {                                                                                                          \
        if (!(cond)) {                                                                                            \
            throw std::runtime_error(                                                                             \
                std::string("Error: ") + msg + "\nFile: " + __FILE__ + "\nLine: " + std::to_string(__LINE__)); \
        }                                                                                                         \
    } while (0)

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
inline T Align(T a, T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
inline T FloorAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

struct PlatformInfo {
    uint32_t aicNum{0};
    uint32_t aivNum{0};
    uint64_t ubSize{0};
    uint64_t l1Size{0};
    uint64_t l0aSize{0};
    uint64_t l0bSize{0};
    uint64_t l0cSize{0};
    uint64_t l2Size{0};
    uint64_t btSize{0};
    platform_ascendc::SocVersion socVersion{0};
};

struct Args {
    uint64_t m{0};
    uint64_t n{0};
    uint64_t k{0};
    uint64_t inputElemBytes{2};
    uint64_t biasElemBytes{4};
    bool hasBias{false};
    bool isATrans{false};
    bool isBTrans{false};
    bool isANz{false};
    bool isBNz{false};
};

struct TailInfo {
    uint64_t mCnt{1UL};
    uint64_t nCnt{1UL};
    uint64_t mTailMain{0UL};
    uint64_t nTailMain{0UL};
};

struct RunInfo {
    uint64_t baseM{1UL};
    uint64_t baseN{1UL};
    uint64_t baseK{1UL};
    uint64_t singleCoreM{1UL};
    uint64_t singleCoreN{1UL};
    uint64_t singleCoreK{1UL};
    uint32_t mBaseTailSplitCnt{1U};
    uint32_t nBaseTailSplitCnt{1U};
    uint64_t usedCoreNum{1UL};
    uint64_t depthA1{1UL};
    uint64_t depthB1{1UL};
    uint64_t stepKa{1UL};
    uint64_t stepKb{1UL};
    uint64_t stepM{1UL};
    uint64_t stepN{1UL};
    uint64_t dbL0c{1UL};
    uint64_t l1BufferNum{2UL};
    TailInfo tailInfo;
};

inline PlatformInfo LoadPlatformInfo()
{
    PlatformInfo info{};
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    info.aicNum = ascendcPlatform->GetCoreNumAic();
    info.aivNum = ascendcPlatform->GetCoreNumAiv();
    info.socVersion = ascendcPlatform->GetSocVersion();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0aSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0bSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0cSize);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, info.l2Size);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, info.btSize);
    return info;
}

} // namespace blaze_matmul_tiling

class MatmulTilingSwat {
public:
    void GetTilingData(uint64_t m, uint64_t n, uint64_t k, uint64_t inputElemBytes, MatmulTilingData& tilingData,
        bool isATrans = false, bool isBTrans = false, bool isANz = false, bool isBNz = false,
        bool hasBias = false, uint64_t biasElemBytes = blaze_matmul_tiling::DATA_SIZE_FP32)
    {
        using namespace blaze_matmul_tiling;

        BLAZE_TILING_CHECK_COND(inputElemBytes > 0UL, "inputElemBytes must be greater than zero.");
        platformInfo_ = LoadPlatformInfo();
        args_ = {m, n, k, inputElemBytes, biasElemBytes, hasBias, isATrans, isBTrans, isANz, isBNz};
        runInfo_ = {};

        ResetBase();
        FormulateLoadBalanceBlock();
        if (runInfo_.baseM == BASIC_BLOCK_SIZE_256 && runInfo_.baseN == BASIC_BLOCK_SIZE_256) {
            OptimizeEdgeBasicBlock();
        }
        CalcTailBasicBlock();
        CalL1Tiling();
        BuildTilingData(tilingData);
        PrintTilingData(tilingData);
    }

    void GetTilingData(uint64_t m, uint64_t n, uint64_t k, MatmulTilingData& tilingData,
        bool isATrans = false, bool isBTrans = false, bool isANz = false, bool isBNz = false)
    {
        GetTilingData(m, n, k, 2UL, tilingData, isATrans, isBTrans, isANz, isBNz, false,
            blaze_matmul_tiling::DATA_SIZE_FP32);
    }

private:
    using PlatformInfo = blaze_matmul_tiling::PlatformInfo;
    using Args = blaze_matmul_tiling::Args;
    using RunInfo = blaze_matmul_tiling::RunInfo;

    PlatformInfo platformInfo_{};
    Args args_{};
    RunInfo runInfo_{};

    uint64_t FractalC0() const
    {
        return blaze_matmul_tiling::BLOCK_BYTE_SIZE / args_.inputElemBytes;
    }

    uint64_t MAlignment() const
    {
        return args_.isANz && args_.isATrans ? FractalC0() : blaze_matmul_tiling::BASIC_BLOCK_SIZE_16;
    }

    uint64_t NAlignment() const
    {
        return args_.isBNz && !args_.isBTrans ? FractalC0() : blaze_matmul_tiling::BASIC_BLOCK_SIZE_16;
    }

    uint64_t KAlignment() const
    {
        bool kIsAInnerAxis = args_.isANz && !args_.isATrans;
        bool kIsBInnerAxis = args_.isBNz && args_.isBTrans;
        return (kIsAInnerAxis || kIsBInnerAxis) ? FractalC0() : blaze_matmul_tiling::BASIC_BLOCK_SIZE_16;
    }

    void ResetBase()
    {
        using namespace blaze_matmul_tiling;
        runInfo_.usedCoreNum = platformInfo_.aicNum;
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / args_.inputElemBytes;
        runInfo_.stepM = 1UL;
        runInfo_.stepN = 1UL;
        runInfo_.dbL0c = 1UL;
        runInfo_.singleCoreK = args_.k;
        runInfo_.singleCoreM = runInfo_.baseM;
        runInfo_.singleCoreN = runInfo_.baseN;
    }

    void FormulateLoadBalanceBlock()
    {
        using namespace blaze_matmul_tiling;
        runInfo_.baseM = std::min(Align(args_.m, MAlignment()), runInfo_.baseM);
        runInfo_.baseN = std::min(Align(args_.n, NAlignment()), runInfo_.baseN);

        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore * nCore < platformInfo_.aicNum) {
            CalcBasicBlock();
        }

        runInfo_.baseM = Align(runInfo_.baseM, MAlignment());
        runInfo_.baseN = Align(runInfo_.baseN, NAlignment());
        runInfo_.dbL0c =
            runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 * DB_SIZE <= platformInfo_.l0cSize ? DB_SIZE : 1UL;

        mCore = CeilDiv(args_.m, runInfo_.baseM);
        nCore = CeilDiv(args_.n, runInfo_.baseN);
        runInfo_.usedCoreNum = std::max<uint64_t>(1UL,
            std::min(mCore * nCore, static_cast<uint64_t>(platformInfo_.aicNum)));

        uint64_t kAlign = KAlignment();
        uint64_t kValueAlign = Align(args_.k, kAlign);
        uint64_t kValueMax = FloorAlign(
            platformInfo_.l0aSize / DB_SIZE / args_.inputElemBytes / std::max(runInfo_.baseM, runInfo_.baseN), kAlign);
        BLAZE_TILING_CHECK_COND(kValueMax >= kAlign, "Failed to derive valid baseK from L0A capacity.");
        runInfo_.baseK = std::min(kValueAlign, kValueMax);
    }

    void CalcBasicBlock()
    {
        using namespace blaze_matmul_tiling;
        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore == 0UL || nCore == 0UL) {
            return;
        }
        if (mCore <= nCore) {
            runInfo_.baseM = Align(CeilDiv(args_.m, mCore), MAlignment());
            mCore = CeilDiv(args_.m, runInfo_.baseM);
            nCore = std::max<uint64_t>(1UL, runInfo_.usedCoreNum / mCore);
            runInfo_.baseN = Align(CeilDiv(args_.n, nCore), NAlignment());
        } else {
            runInfo_.baseN = Align(CeilDiv(args_.n, nCore), NAlignment());
            nCore = CeilDiv(args_.n, runInfo_.baseN);
            mCore = std::max<uint64_t>(1UL, runInfo_.usedCoreNum / nCore);
            runInfo_.baseM = Align(CeilDiv(args_.m, mCore), MAlignment());
        }
    }

    void OptimizeEdgeBasicBlock()
    {
        uint64_t mCore = blaze_matmul_tiling::CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = blaze_matmul_tiling::CeilDiv(args_.n, runInfo_.baseN);
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
        using namespace blaze_matmul_tiling;
        uint64_t aicNum = platformInfo_.aicNum;
        uint64_t x = nLoadBalance ? args_.n : args_.m;
        uint64_t y = nLoadBalance ? args_.m : args_.n;
        uint64_t baseX = nLoadBalance ? runInfo_.baseN : runInfo_.baseM;
        uint64_t baseY = nLoadBalance ? runInfo_.baseM : runInfo_.baseN;
        uint64_t xCnt = CeilDiv(x, baseX);
        uint64_t yCnt = CeilDiv(y, baseY);
        uint64_t xTail = x % baseX;
        uint64_t totalWindows = CeilDiv(xCnt * yCnt, aicNum);
        uint64_t mainWindows = CeilDiv((xCnt - 1UL) * yCnt + yCnt % aicNum, aicNum);
        uint64_t tailWindows = totalWindows - mainWindows;
        uint64_t perfRes = mainWindows * baseX + tailWindows * xTail;
        uint64_t baseTailCntMax = std::min((baseX - xTail) / BASIC_BLOCK_SIZE_16, xCnt);
        for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
            uint64_t newTailMain = Align(CeilDiv((mergeLen * baseX + xTail), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
            uint64_t newTailLast = mergeLen * (baseX - newTailMain) + xTail;
            uint64_t newMainRound = mergeLen < xCnt - 1UL ?
                CeilDiv(((xCnt - 1UL - mergeLen) * yCnt + (mergeLen + 1UL) * yCnt) % aicNum, aicNum) : 0UL;
            uint64_t newTailRound = std::min(CeilDiv(mergeLen * yCnt + yCnt % aicNum, aicNum), totalWindows - newMainRound);
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
        using namespace blaze_matmul_tiling;
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
        using namespace blaze_matmul_tiling;
        uint64_t biasBytes = args_.hasBias ? Align(args_.n, NAlignment()) * args_.biasElemBytes : 0UL;
        BLAZE_TILING_CHECK_COND(platformInfo_.l1Size > biasBytes, "L1 space is insufficient after reserving bias.");
        uint64_t totalL1Size = platformInfo_.l1Size - biasBytes;
        runInfo_.depthA1 = std::max(totalL1Size / NUM_TWO / runInfo_.baseM / runInfo_.baseK / args_.inputElemBytes, 1UL);
        runInfo_.depthB1 = std::max(totalL1Size / NUM_TWO / runInfo_.baseN / runInfo_.baseK / args_.inputElemBytes, 1UL);
        uint64_t depthASize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * args_.inputElemBytes;
        uint64_t depthBSize = runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * args_.inputElemBytes;
        if (depthASize + depthBSize > totalL1Size) {
            if (runInfo_.baseM <= runInfo_.baseN) {
                runInfo_.depthA1 = std::max(runInfo_.depthA1 / NUM_TWO, 1UL);
            } else {
                runInfo_.depthB1 = std::max(runInfo_.depthB1 / NUM_TWO, 1UL);
            }
        }
        runInfo_.stepKa = std::max(runInfo_.depthA1 / DB_SIZE, 1UL);
        runInfo_.stepKb = std::max(runInfo_.depthB1 / DB_SIZE, 1UL);
        if (runInfo_.stepKa >= runInfo_.stepKb) {
            runInfo_.stepKa = std::max(runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb, 1UL);
        } else {
            runInfo_.stepKb = std::max(runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa, 1UL);
        }
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
        runInfo_.singleCoreM = runInfo_.baseM;
        runInfo_.singleCoreN = runInfo_.baseN;
    }

    void BuildTilingData(MatmulTilingData& tilingData) const
    {
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mL1 = std::min(blaze_matmul_tiling::Align(args_.m, MAlignment()), runInfo_.baseM * runInfo_.stepM);
        tilingData.nL1 = std::min(blaze_matmul_tiling::Align(args_.n, NAlignment()), runInfo_.baseN * runInfo_.stepN);
        uint64_t stepK = std::min<uint64_t>(4UL, std::min(runInfo_.stepKa, runInfo_.stepKb));
        tilingData.kL1 = static_cast<uint32_t>(runInfo_.baseK * stepK);
        tilingData.mTailCnt = static_cast<uint32_t>(runInfo_.tailInfo.mCnt);
        tilingData.nTailCnt = static_cast<uint32_t>(runInfo_.tailInfo.nCnt);
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.tailInfo.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.tailInfo.nTailMain);
        tilingData.usedCoreNum = static_cast<uint32_t>(runInfo_.usedCoreNum);
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.l0cDB = static_cast<uint8_t>(runInfo_.dbL0c);
    }

    void PrintTilingData(const MatmulTilingData& tilingData) const
    {
        std::printf("[Matmul Strategy]\n");
        std::printf("  strategy           : swat\n");
        std::printf("[Matmul Tiling Data]\n");
        std::printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        std::printf("  m                  : %u\n", tilingData.m);
        std::printf("  n                  : %u\n", tilingData.n);
        std::printf("  k                  : %u\n", tilingData.k);
        std::printf("  mL1                : %u\n", tilingData.mL1);
        std::printf("  nL1                : %u\n", tilingData.nL1);
        std::printf("  kL1                : %u\n", tilingData.kL1);
        std::printf("  baseM              : %u\n", tilingData.baseM);
        std::printf("  baseN              : %u\n", tilingData.baseN);
        std::printf("  baseK              : %u\n", tilingData.baseK);
        std::printf("  mTailCnt           : %u\n", tilingData.mTailCnt);
        std::printf("  nTailCnt           : %u\n", tilingData.nTailCnt);
        std::printf("  mBaseTailSplitCnt  : %u\n", tilingData.mBaseTailSplitCnt);
        std::printf("  nBaseTailSplitCnt  : %u\n", tilingData.nBaseTailSplitCnt);
        std::printf("  mTailMain          : %u\n", tilingData.mTailMain);
        std::printf("  nTailMain          : %u\n", tilingData.nTailMain);
        std::printf("  l1BufferNum        : %u\n", tilingData.l1BufferNum);
        std::printf("  l0cDB              : %u\n", tilingData.l0cDB);
    }
};

#endif // BLAZE_MATMUL_TILING_H
