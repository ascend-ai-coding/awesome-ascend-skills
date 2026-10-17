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

/*!
 * \file weight_quant_tiling.h
 * \brief Host-side SWAT tiling engine for weight-quant matmul kernels.
 *
 * Based on MatmulTilingSwat with an additional AIV UB constraint.
 * Weight-quant matmul is V+C fusion: A is bf16, B is int8 (dequantized to
 * bf16 in AIV UB before MMAD). The UB constraint accounts for the AIV
 * dequant buffers (weightIn int8 + weightOut bf16, ping-pong + scale/offset).
 *
 * SWAT three-layer load balancing is inherited:
 *   1. CalcBasicBlock — shrink baseM/baseN to fill all AIC cores
 *   2. OptimizeEdgeBasicBlock — merge small tail blocks
 *   3. CalcTailBasicBlock — split tail blocks across idle cores
 */

#ifndef WEIGHT_QUANT_TILING_H
#define WEIGHT_QUANT_TILING_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "platform/platform_ascendc.h"
#include "weight_quant_tiling_data.h"

namespace weight_quant_tiling {

constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t DATA_SIZE_BF16 = 2UL;
constexpr uint64_t DATA_SIZE_INT8 = 1UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t UB_ALIGN_64 = 64UL;
constexpr uint64_t UB_ALIGN_32 = 32UL;

#define WQ_TILING_CHECK_COND(cond, msg)                                                                             \
    do {                                                                                                            \
        if (!(cond)) {                                                                                              \
            throw std::runtime_error(                                                                               \
                std::string("Error: ") + msg + "\nFile: " + __FILE__ + "\nLine: " + std::to_string(__LINE__));     \
        }                                                                                                           \
    } while (0)

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) { return a; }
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
    if (b == 0) { return a; }
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
    bool transB{false};
    bool hasOffset{false};
    bool hasBias{false};
    uint64_t biasElemBytes{DATA_SIZE_FP32};
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
    uint64_t kL1{1UL};
    uint64_t nUbSize{1UL};
    uint64_t kUbSize{1UL};
    uint32_t mBaseTailSplitCnt{1U};
    uint32_t nBaseTailSplitCnt{1U};
    uint64_t usedCoreNum{1UL};
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

} // namespace weight_quant_tiling

class WeightQuantTilingSwat {
public:
    void GetTilingData(uint64_t m, uint64_t n, uint64_t k,
                       WeightQuantMatmulTilingData& tilingData,
                       bool transB = false, bool hasOffset = false, bool hasBias = false,
                       uint64_t biasElemBytes = weight_quant_tiling::DATA_SIZE_FP32)
    {
        using namespace weight_quant_tiling;

        WQ_TILING_CHECK_COND(m > 0 && n > 0 && k > 0, "m, n, k must be greater than zero.");
        if (transB) {
            WQ_TILING_CHECK_COND(n % BASIC_BLOCK_SIZE_16 == 0,
                "n must be 16-aligned when transB=true (N is the fractal axis).");
        } else {
            WQ_TILING_CHECK_COND(k % BASIC_BLOCK_SIZE_16 == 0,
                "k must be 16-aligned when transB=false (K is the fractal axis).");
        }
        platformInfo_ = LoadPlatformInfo();
        args_ = {m, n, k, transB, hasOffset, hasBias, biasElemBytes};
        runInfo_ = {};

        ResetBase();
        FormulateLoadBalanceBlock();
        if (runInfo_.baseM == BASIC_BLOCK_SIZE_256 && runInfo_.baseN == BASIC_BLOCK_SIZE_256) {
            OptimizeEdgeBasicBlock();
        }
        CalcTailBasicBlock();
        CalL1AndUbTiling();
        BuildTilingData(tilingData);
        PrintTilingData(tilingData);
    }

private:
    using PlatformInfo = weight_quant_tiling::PlatformInfo;
    using Args = weight_quant_tiling::Args;
    using RunInfo = weight_quant_tiling::RunInfo;

    PlatformInfo platformInfo_{};
    Args args_{};
    RunInfo runInfo_{};

    static constexpr uint64_t elemBytesA() { return weight_quant_tiling::DATA_SIZE_BF16; }
    static constexpr uint64_t elemBytesB() { return weight_quant_tiling::DATA_SIZE_INT8; }
    static constexpr uint64_t elemBytesDequantB() { return weight_quant_tiling::DATA_SIZE_BF16; }

    uint64_t MAlignment() const { return weight_quant_tiling::BASIC_BLOCK_SIZE_16; }
    uint64_t NAlignment() const { return weight_quant_tiling::BASIC_BLOCK_SIZE_16; }
    uint64_t KAlignment() const { return weight_quant_tiling::BASIC_BLOCK_SIZE_16; }

    void ResetBase()
    {
        using namespace weight_quant_tiling;
        runInfo_.usedCoreNum = platformInfo_.aicNum;
        runInfo_.baseM = BASIC_BLOCK_SIZE_256;
        runInfo_.baseN = BASIC_BLOCK_SIZE_256;
        runInfo_.baseK = BASIC_BLOCK_SIZE_128 / elemBytesA();
        runInfo_.dbL0c = 1UL;
        runInfo_.l1BufferNum = 2UL;
    }

    void FormulateLoadBalanceBlock()
    {
        using namespace weight_quant_tiling;
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
            platformInfo_.l0aSize / DB_SIZE / elemBytesA() / std::max(runInfo_.baseM, runInfo_.baseN), kAlign);
        WQ_TILING_CHECK_COND(kValueMax >= kAlign, "Failed to derive valid baseK from L0A capacity.");
        runInfo_.baseK = std::min(kValueAlign, kValueMax);
    }

    void CalcBasicBlock()
    {
        using namespace weight_quant_tiling;
        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore == 0UL || nCore == 0UL) { return; }
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
        using namespace weight_quant_tiling;
        uint64_t mCore = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCore = CeilDiv(args_.n, runInfo_.baseN);
        if (mCore * nCore < platformInfo_.aicNum || mCore == 1UL || nCore == 1UL) { return; }
        uint64_t mBaseTail = args_.m % runInfo_.baseM;
        uint64_t nBaseTail = args_.n % runInfo_.baseN;
        if (mBaseTail > 0UL && (nBaseTail == 0UL || mBaseTail <= nBaseTail)) {
            GetOuterAxisTailCnt(false, runInfo_.mBaseTailSplitCnt, runInfo_.tailInfo.mTailMain);
        } else if (nBaseTail > 0UL) {
            GetOuterAxisTailCnt(true, runInfo_.nBaseTailSplitCnt, runInfo_.tailInfo.nTailMain);
        }
    }

    void GetOuterAxisTailCnt(bool nLoadBalance, uint32_t& baseTailSplitCnt, uint64_t& tailMain)
    {
        using namespace weight_quant_tiling;
        uint64_t aicNum = platformInfo_.aicNum;
        uint64_t x = nLoadBalance ? args_.n : args_.m;
        uint64_t y = nLoadBalance ? args_.m : args_.n;
        uint64_t baseX = nLoadBalance ? runInfo_.baseN : runInfo_.baseM;
        uint64_t xCnt = CeilDiv(x, baseX);
        uint64_t yCnt = CeilDiv(y, nLoadBalance ? runInfo_.baseM : runInfo_.baseN);
        uint64_t xTail = x % baseX;
        if (xTail == 0UL) { return; }
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
        using namespace weight_quant_tiling;
        uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
        uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);
        uint64_t mnCnt = mCnt * nCnt;
        uint64_t tailCnt = mnCnt <= platformInfo_.aicNum ? 0UL : mnCnt % platformInfo_.aicNum;
        runInfo_.tailInfo.mCnt = 1UL;
        runInfo_.tailInfo.nCnt = 1UL;
        if (tailCnt != 0UL) {
            while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= platformInfo_.aicNum) {
                runInfo_.tailInfo.mCnt += 1UL;
                // transB=true 时 N 是分形轴，尾轮切分会破坏 16 对齐，禁止 N 方向切分
                if (!args_.transB &&
                    runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= platformInfo_.aicNum) {
                    runInfo_.tailInfo.nCnt += 1UL;
                }
            }
        }
    }

    void CalL1AndUbTiling()
    {
        using namespace weight_quant_tiling;

        uint64_t biasBytes = args_.hasBias ?
            Align(runInfo_.baseN * args_.biasElemBytes, UB_ALIGN_64) : 0UL;
        WQ_TILING_CHECK_COND(platformInfo_.l1Size > biasBytes, "L1 space is insufficient after reserving bias.");
        uint64_t availL1 = platformInfo_.l1Size - biasBytes;

        uint64_t nUbSize;
        if (args_.transB) {
            // transB=true: N 被 2 AIV 切分。nHalf = (baseN/32)*16，
            // AIV1 拿到 baseN - nHalf，是最大份额。baseN 是所有 tile 中最大的 curN，
            // maxNUbLen 单调递增，故 nUbSize = baseN - (baseN/32)*16。
            nUbSize = runInfo_.baseN - (runInfo_.baseN / 32UL) * BASIC_BLOCK_SIZE_16;
        } else {
            // transB=false: N 不切分，AIV 拿到完整 curN。baseN 是最大 curN，128 对齐预留 padding。
            nUbSize = Align(runInfo_.baseN, BASIC_BLOCK_SIZE_128);
        }

        uint64_t kL1MaxL1 = availL1 / (NUM_TWO * (runInfo_.baseM + runInfo_.baseN) * elemBytesDequantB());
        kL1MaxL1 = FloorAlign(kL1MaxL1, BASIC_BLOCK_SIZE_16);

        uint64_t scaleOffsetUbBytes = nUbSize * elemBytesA() * 2UL * (1UL + (args_.hasOffset ? 1UL : 0UL));
        WQ_TILING_CHECK_COND(platformInfo_.ubSize > scaleOffsetUbBytes, "UB space is insufficient for scale/offset.");
        uint64_t availUb = platformInfo_.ubSize - scaleOffsetUbBytes;
        uint64_t ubBytesPerK = DB_SIZE * nUbSize * (elemBytesB() + elemBytesDequantB());
        uint64_t kUbSizeMax = availUb / ubBytesPerK;

        uint64_t kL1MaxUb;
        if (args_.transB) {
            kL1MaxUb = FloorAlign(kUbSizeMax, BASIC_BLOCK_SIZE_128);
        } else {
            kL1MaxUb = FloorAlign(kUbSizeMax * NUM_TWO, BASIC_BLOCK_SIZE_16);
        }

        uint64_t kL1 = std::min({kL1MaxL1, kL1MaxUb, args_.k});
        kL1 = FloorAlign(kL1, BASIC_BLOCK_SIZE_16);
        if (kL1 == 0UL) { kL1 = std::min(args_.k, kL1MaxL1); }
        if (kL1 == 0UL) { kL1 = args_.k; }

        uint64_t kUbSize;
        if (args_.transB) {
            kUbSize = Align(kL1, BASIC_BLOCK_SIZE_128);
        } else {
            kUbSize = CeilDiv(kL1, 32UL) * BASIC_BLOCK_SIZE_16;
        }

        runInfo_.kL1 = kL1;
        runInfo_.nUbSize = nUbSize;
        runInfo_.kUbSize = kUbSize;
    }

    void BuildTilingData(WeightQuantMatmulTilingData& tilingData) const
    {
        using namespace weight_quant_tiling;
        tilingData = {};
        tilingData.m = static_cast<uint32_t>(args_.m);
        tilingData.n = static_cast<uint32_t>(args_.n);
        tilingData.k = static_cast<uint32_t>(args_.k);
        tilingData.baseM = static_cast<uint32_t>(runInfo_.baseM);
        tilingData.baseN = static_cast<uint32_t>(runInfo_.baseN);
        tilingData.baseK = static_cast<uint32_t>(runInfo_.baseK);
        tilingData.mL1 = static_cast<uint32_t>(std::min(Align(args_.m, MAlignment()), runInfo_.baseM));
        tilingData.nL1 = static_cast<uint32_t>(std::min(Align(args_.n, NAlignment()), runInfo_.baseN));
        tilingData.kL1 = static_cast<uint32_t>(runInfo_.kL1);
        tilingData.mTailCnt = static_cast<uint32_t>(runInfo_.tailInfo.mCnt);
        tilingData.nTailCnt = static_cast<uint32_t>(runInfo_.tailInfo.nCnt);
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = static_cast<uint32_t>(runInfo_.tailInfo.mTailMain);
        tilingData.nTailMain = static_cast<uint32_t>(runInfo_.tailInfo.nTailMain);
        tilingData.usedCoreNum = static_cast<uint32_t>(runInfo_.usedCoreNum);
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.l0cDB = static_cast<uint8_t>(runInfo_.dbL0c);
        tilingData.nUbSize = static_cast<uint32_t>(runInfo_.nUbSize);
        tilingData.kUbSize = static_cast<uint32_t>(runInfo_.kUbSize);
        tilingData.transB = args_.transB ? 1 : 0;
        tilingData.hasOffset = args_.hasOffset ? 1 : 0;
        tilingData.hasBias = args_.hasBias ? 1 : 0;
    }

    void PrintTilingData(const WeightQuantMatmulTilingData& tilingData) const
    {
        std::printf("[WeightQuant Strategy]\n");
        std::printf("  strategy           : swat\n");
        std::printf("[WeightQuant Tiling Data]\n");
        std::printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        std::printf("  m                  : %u\n", tilingData.m);
        std::printf("  n                  : %u\n", tilingData.n);
        std::printf("  k                  : %u\n", tilingData.k);
        std::printf("  baseM              : %u\n", tilingData.baseM);
        std::printf("  baseN              : %u\n", tilingData.baseN);
        std::printf("  baseK              : %u\n", tilingData.baseK);
        std::printf("  kL1                : %u\n", tilingData.kL1);
        std::printf("  nUbSize            : %u\n", tilingData.nUbSize);
        std::printf("  kUbSize            : %u\n", tilingData.kUbSize);
        std::printf("  transB             : %u\n", tilingData.transB);
        std::printf("  hasOffset          : %u\n", tilingData.hasOffset);
        std::printf("  hasBias            : %u\n", tilingData.hasBias);
        std::printf("  l1BufferNum        : %u\n", tilingData.l1BufferNum);
        std::printf("  l0cDB              : %u\n", tilingData.l0cDB);
    }
};

#endif // WEIGHT_QUANT_TILING_H
