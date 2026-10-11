/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_MATMUL_MX_TILING_H
#define QUANT_MATMUL_MX_TILING_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "platform/platform_ascendc.h"
#include "quant_matmul_mx_tiling_data.h"

namespace blaze_mx_tiling {

constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t FP8_C0_SIZE = 32UL;
constexpr uint64_t DATA_SIZE_L0C = 4UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t TILING_MXFP_DIVISOR_SIZE = 64UL;
constexpr uint64_t TILING_MXFP_MULTI_BASE_SIZE = 2UL;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t BASEK_LIMIT = 4095UL;
constexpr uint64_t BASEM_BASEN_RATIO = 2UL;
constexpr uint64_t SCALER_FACTOR_MIN = 1UL;
constexpr uint64_t SCALER_FACTOR_MAX = 127UL;
constexpr uint64_t MTE2_MIN_LOAD_SIZE = 32768UL;
constexpr uint64_t MTE2_CACHELINE_SIZE = 128UL;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t L2_ALIGN_SIZE = 128UL;
constexpr uint64_t WINDOW_LEN = 4UL;

#define BLAZE_MX_TILING_CHECK_COND(cond, msg)                                                                     \
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

namespace mm {
enum class DataType {
    DT_FLOAT8_E4M3FN,
    DT_FLOAT4_E2M1,
};
} // namespace mm

template <mm::DataType dataType>
inline uint64_t GetSizeWithDataType(uint64_t shape)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        return CeilDiv(shape, 2UL);
    } else {
        return shape;
    }
}

template <mm::DataType dataType>
inline uint64_t GetShapeWithDataType(uint64_t bytes)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        return bytes * 2UL;
    } else {
        return bytes;
    }
}

struct PlatformInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0cSize{0UL};
    uint64_t l0aSize{0UL};
    uint64_t l0bSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion;
};

struct Args {
    uint64_t m{0UL};
    uint64_t k{0UL};
    uint64_t n{0UL};
    bool transA{false};
    bool transB{true};
    bool isANz{false};
    bool isBNz{false};
    bool hasBias{false};
    uint64_t biasElemBytes{DATA_SIZE_FP32};
};

struct RunInfo {
    uint64_t baseM{0UL};
    uint64_t baseN{0UL};
    uint64_t baseK{0UL};
    uint64_t stepKa{0UL};
    uint64_t stepKb{0UL};
    uint64_t depthA1{0UL};
    uint64_t depthB1{0UL};
    uint64_t dbL0c{0UL};
    uint64_t mBlockCnt{0UL};
    uint64_t nBlockCnt{0UL};
    uint64_t totalBlockCnt{0UL};
    uint64_t mTailTile{1UL};
    uint64_t nTailTile{1UL};
    uint64_t mTailSize{0UL};
    uint64_t nTailSize{0UL};
    uint64_t tailBlockCnt{0UL};
    uint64_t mBaseTailSplitCnt{1UL};
    uint64_t mTailMain{0UL};
    uint64_t nBaseTailSplitCnt{1UL};
    uint64_t nTailMain{0UL};
    uint64_t scaleFactorA{0UL};
    uint64_t scaleFactorB{0UL};
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

} // namespace blaze_mx_tiling

namespace mm = blaze_mx_tiling::mm;

template <mm::DataType aDataType, mm::DataType bDataType>
class QuantMatmulTilingSwat {
public:
    void GetTilingData(uint64_t m, uint64_t n, uint64_t k, QuantMatmulTilingData& tilingData,
        bool transA = false, bool transB = true, bool isANz = false, bool isBNz = false,
        bool hasBias = false, uint64_t biasElemBytes = blaze_mx_tiling::DATA_SIZE_FP32)
    {
        platformInfo_ = blaze_mx_tiling::LoadPlatformInfo();
        args_ = {m, k, n, transA, transB, isANz, isBNz, hasBias, biasElemBytes};
        runInfo_ = {};

        CalcBasicBlock();
        OptimizeEdgeBasicBlock();
        CalcTailBasicBlock();
        CalcPathSpecificL1();

        uint32_t scaleKL1 = CalcScaleKL1();
        BuildTilingData(tilingData, scaleKL1, static_cast<uint8_t>(blaze_mx_tiling::DB_SIZE));
        PrintTilingData(tilingData);
    }

private:
    using PlatformInfo = blaze_mx_tiling::PlatformInfo;
    using Args = blaze_mx_tiling::Args;
    using RunInfo = blaze_mx_tiling::RunInfo;
    using DataType = blaze_mx_tiling::mm::DataType;

    PlatformInfo platformInfo_{};
    Args args_{};
    RunInfo runInfo_{};

    uint64_t AFractalC0() const
    {
        return blaze_mx_tiling::GetShapeWithDataType<aDataType>(blaze_mx_tiling::BLOCK_BYTE_SIZE);
    }

    uint64_t BFractalC0() const
    {
        return blaze_mx_tiling::GetShapeWithDataType<bDataType>(blaze_mx_tiling::BLOCK_BYTE_SIZE);
    }

    uint64_t L1Budget() const
    {
        uint64_t biasBytes = args_.hasBias ? blaze_mx_tiling::Align(args_.n, BFractalC0()) * args_.biasElemBytes : 0UL;
        BLAZE_MX_TILING_CHECK_COND(platformInfo_.l1Size > biasBytes, "L1 space is insufficient after reserving bias.");
        return platformInfo_.l1Size - biasBytes;
    }

    uint32_t CalcScaleKL1() const
    {
        return static_cast<uint32_t>(std::min(
            runInfo_.scaleFactorA * runInfo_.stepKa * runInfo_.baseK,
            runInfo_.scaleFactorB * runInfo_.stepKb * runInfo_.baseK));
    }

    void BuildTilingData(QuantMatmulTilingData& tilingData, uint32_t scaleKL1, uint8_t nBufferNum) const
    {
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
        tilingData.usedCoreNum = static_cast<uint32_t>(
            (runInfo_.totalBlockCnt > 1UL || runInfo_.tailBlockCnt == 0UL) ?
                platformInfo_.aicNum :
                runInfo_.tailBlockCnt * runInfo_.mTailTile * runInfo_.nTailTile);
        tilingData.dbL0c = static_cast<uint8_t>(runInfo_.dbL0c);
        tilingData.scaleKL1 = scaleKL1;
        tilingData.stepK = static_cast<uint8_t>(std::min(runInfo_.stepKa, runInfo_.stepKb));
        tilingData.nBufferNum = nBufferNum;
    }

    void CalcTailBasicBlock()
    {
        using namespace blaze_mx_tiling;
        if (runInfo_.tailBlockCnt == 0UL) {
            return;
        }
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
        while ((CalUsedCoreNum(preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) ||
               (CalUsedCoreNum(preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax)) {
            if (CalUsedCoreNum(preSplit + 1UL, secSplit) <= platformInfo_.aicNum && preSplit < preSplitMax) {
                preSplitValid = ++preSplit;
            }
            if (CalUsedCoreNum(preSplit, secSplit + 1UL) <= platformInfo_.aicNum && secSplit < secSplitMax) {
                secSplitValid = ++secSplit;
            }
        }
        runInfo_.mTailTile = mTile;
        runInfo_.nTailTile = nTile;
    }

    uint64_t CalUsedCoreNum(uint64_t mTile, uint64_t nTile) const
    {
        return mTile * nTile * runInfo_.tailBlockCnt;
    }

    uint64_t GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit) const
    {
        using namespace blaze_mx_tiling;
        if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE) {
            return depthInit;
        }
        uint64_t depthScale = leftSize / perDepthSize;
        if (depthInit > 1UL) {
            uint64_t baseKSize = GetSizeWithDataType<aDataType>(runInfo_.baseK);
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
            depthScale = 1UL;
            while (depthScale * perDepthSize < leftSize) {
                depthScale *= 2UL;
            }
            depthScale = depthScale == 1UL ? depthScale : depthScale / 2UL;
        }
        return depthInit * depthScale;
    }

    void CalStepKs()
    {
        using namespace blaze_mx_tiling;
        runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
        runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
        if (runInfo_.stepKa * runInfo_.baseK > args_.k) {
            runInfo_.stepKa = CeilDiv(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKb * runInfo_.baseK > args_.k) {
            runInfo_.stepKb = CeilDiv(args_.k, runInfo_.baseK);
        }
        if (runInfo_.stepKa > runInfo_.stepKb) {
            runInfo_.stepKa = runInfo_.stepKa / runInfo_.stepKb * runInfo_.stepKb;
        }
        if (runInfo_.stepKb > runInfo_.stepKa) {
            runInfo_.stepKb = runInfo_.stepKb / runInfo_.stepKa * runInfo_.stepKa;
        }
        runInfo_.stepKa = std::min(runInfo_.stepKa, 4UL);
        runInfo_.stepKb = std::min(runInfo_.stepKb, 4UL);
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
    }

    void CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize, uint64_t baseScaleBSize)
    {
        using namespace blaze_mx_tiling;
        uint64_t scaleFactorAMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleASize, SCALER_FACTOR_MAX);
        uint64_t scaleFactorBMax = std::min(MTE2_MIN_LOAD_SIZE / baseScaleBSize, SCALER_FACTOR_MAX);
        uint64_t scaleFactorA = args_.k / (runInfo_.stepKa * runInfo_.baseK);
        uint64_t scaleFactorB = args_.k / (runInfo_.stepKb * runInfo_.baseK);
        runInfo_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
        runInfo_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
        runInfo_.scaleFactorA = std::min(scaleFactorAMax, runInfo_.scaleFactorA);
        runInfo_.scaleFactorB = std::min(scaleFactorBMax, runInfo_.scaleFactorB);

        uint64_t biasBytes = args_.hasBias ? Align(args_.n, BFractalC0()) * args_.biasElemBytes : 0UL;
        BLAZE_MX_TILING_CHECK_COND(platformInfo_.l1Size > biasBytes, "L1 space is insufficient after reserving bias.");
        uint64_t leftL1Size = platformInfo_.l1Size - biasBytes -
            (runInfo_.depthA1 * baseASize + runInfo_.depthB1 * baseBSize);
        uint64_t scaleInit = leftL1Size / (runInfo_.depthA1 * baseScaleASize + runInfo_.depthB1 * baseScaleBSize);
        if (runInfo_.scaleFactorA <= scaleInit && runInfo_.scaleFactorB > scaleInit) {
            leftL1Size -= runInfo_.scaleFactorA * runInfo_.depthA1 * baseScaleASize;
            runInfo_.scaleFactorB = std::min(leftL1Size / (runInfo_.depthB1 * baseScaleBSize), runInfo_.scaleFactorB);
        } else if (runInfo_.scaleFactorB <= scaleInit && runInfo_.scaleFactorA > scaleInit) {
            leftL1Size -= runInfo_.scaleFactorB * runInfo_.depthB1 * baseScaleBSize;
            runInfo_.scaleFactorA = std::min(leftL1Size / (runInfo_.depthA1 * baseScaleASize), runInfo_.scaleFactorA);
        } else if (runInfo_.scaleFactorA > scaleInit && runInfo_.scaleFactorB > scaleInit) {
            leftL1Size -= scaleInit * runInfo_.depthB1 * baseScaleBSize + scaleInit * runInfo_.depthA1 * baseScaleASize;
            uint64_t scaleASec = std::min(leftL1Size / (runInfo_.depthA1 * baseScaleASize), runInfo_.scaleFactorA - scaleInit);
            uint64_t scaleBSec = std::min(leftL1Size / (runInfo_.depthB1 * baseScaleBSize), runInfo_.scaleFactorB - scaleInit);
            runInfo_.scaleFactorA = scaleASec >= scaleBSec ? scaleASec + scaleInit : scaleInit;
            runInfo_.scaleFactorB = scaleASec < scaleBSec ? scaleBSec + scaleInit : scaleInit;
        }
    }

    void CalcPathSpecificL1()
    {
        using namespace blaze_mx_tiling;
        uint64_t baseASize = GetSizeWithDataType<aDataType>(runInfo_.baseM * runInfo_.baseK);
        uint64_t baseBSize = GetSizeWithDataType<bDataType>(runInfo_.baseN * runInfo_.baseK);
        uint64_t baseScaleASize = Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseM;
        uint64_t baseScaleBSize = Align(CeilDiv(runInfo_.baseK, MX_GROUP_SIZE), TILING_MXFP_MULTI_BASE_SIZE) * runInfo_.baseN;
        uint64_t l1Budget = L1Budget();
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        uint64_t depthInit = GetDepthA1B1(l1Budget, baseL1Size, 1UL);
        uint64_t leftL1SizeByDepthInit = l1Budget - depthInit * baseL1Size;
        uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        runInfo_.depthA1 = std::max(depthASec, depthBSec);
        runInfo_.depthB1 = runInfo_.depthA1;
        if (runInfo_.depthA1 * baseL1Size > l1Budget) {
            runInfo_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
            runInfo_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
        }
        CalStepKs();
        CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    }

    void AdjustBasicBlock()
    {
        using namespace blaze_mx_tiling;
        uint64_t baseMAlignNum = args_.isANz && args_.transA ? AFractalC0() :
            (args_.transA ? GetShapeWithDataType<aDataType>(L2_ALIGN_SIZE) : CUBE_BLOCK);
        uint64_t baseNAlignNum = args_.isBNz && !args_.transB ? BFractalC0() :
            (args_.transB ? CUBE_BLOCK : GetShapeWithDataType<bDataType>(L2_ALIGN_SIZE));
        uint64_t baseKAlignNum = (args_.transA && !args_.transB) ? GetShapeWithDataType<aDataType>(FP8_C0_SIZE) :
            GetShapeWithDataType<aDataType>(L2_ALIGN_SIZE);
        if (args_.isANz && !args_.transA) {
            baseKAlignNum = AFractalC0();
        }
        if (args_.isBNz && args_.transB) {
            baseKAlignNum = std::max(baseKAlignNum, BFractalC0());
        }
        if (args_.transA || !args_.transB) {
            baseKAlignNum = std::max(baseKAlignNum, GetShapeWithDataType<aDataType>(MXFP_DIVISOR_SIZE));
        }
        uint64_t mMaxtile = CeilDiv(args_.m, baseMAlignNum);
        uint64_t nMaxtile = CeilDiv(args_.n, baseNAlignNum);
        uint64_t tempBaseM = runInfo_.baseM;
        uint64_t tempBaseN = runInfo_.baseN;

        if (mMaxtile * nMaxtile >= platformInfo_.aicNum || (!args_.transA && args_.transB)) {
            uint64_t mCnt = CeilDiv(args_.m, runInfo_.baseM);
            uint64_t nCnt = CeilDiv(args_.n, runInfo_.baseN);
            if (mMaxtile > nMaxtile) {
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
                nCnt = CeilDiv(args_.n, tempBaseN);
                mCnt = platformInfo_.aicNum / nCnt;
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
            } else {
                tempBaseM = Align(CeilDiv(args_.m, mCnt), baseMAlignNum);
                mCnt = CeilDiv(args_.m, tempBaseM);
                nCnt = platformInfo_.aicNum / mCnt;
                tempBaseN = Align(CeilDiv(args_.n, nCnt), baseNAlignNum);
            }
            uint64_t kAlignValue = Align(args_.k, baseKAlignNum);
            uint64_t kMaxValue = GetShapeWithDataType<aDataType>(platformInfo_.l0aSize / DB_SIZE) / std::max(tempBaseM, tempBaseN);
            kMaxValue = FloorAlign(kMaxValue, baseKAlignNum);
            if (kMaxValue >= baseKAlignNum) {
                runInfo_.baseM = tempBaseM;
                runInfo_.baseN = tempBaseN;
                runInfo_.baseK = std::min(kAlignValue, kMaxValue);
                runInfo_.baseK = runInfo_.baseK > BASEK_LIMIT ? Align(runInfo_.baseK / NUM_TWO, BASIC_BLOCK_SIZE_256) : runInfo_.baseK;
            }
        }
    }

    void CalcBasicBlock()
    {
        using namespace blaze_mx_tiling;
        runInfo_.baseM = std::min(args_.m, BASIC_BLOCK_SIZE_256);
        runInfo_.baseM = args_.isANz && args_.transA ? Align(runInfo_.baseM, AFractalC0()) :
            (!args_.transA ? Align(runInfo_.baseM, CUBE_BLOCK) : Align(runInfo_.baseM, GetShapeWithDataType<aDataType>(L1_ALIGN_SIZE)));
        runInfo_.baseN = std::min(args_.n, BASIC_BLOCK_SIZE_256);
        runInfo_.baseN = args_.isBNz && !args_.transB ? Align(runInfo_.baseN, BFractalC0()) :
            (args_.transB ? Align(runInfo_.baseN, CUBE_BLOCK) : Align(runInfo_.baseN, GetShapeWithDataType<bDataType>(L1_ALIGN_SIZE)));
        runInfo_.baseK = Align(
            std::min(args_.k, aDataType == DataType::DT_FLOAT4_E2M1 ? BASIC_BLOCK_SIZE_256 : BASIC_BLOCK_SIZE_128),
            TILING_MXFP_DIVISOR_SIZE);
        uint64_t blockNum = CeilDiv(args_.m, runInfo_.baseM) * CeilDiv(args_.n, runInfo_.baseN);
        if (blockNum < platformInfo_.aicNum) {
            AdjustBasicBlock();
        }
        BLAZE_MX_TILING_CHECK_COND(
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
        using namespace blaze_mx_tiling;
        if (runInfo_.mBlockCnt == 1UL && runInfo_.nBlockCnt == 1UL) {
            return;
        }
        bool isInnerAxisAlign = GetSizeWithDataType<aDataType>(args_.k) % MTE2_CACHELINE_SIZE == 0UL;
        uint64_t mTailSize = args_.m % runInfo_.baseM;
        if (runInfo_.mBlockCnt > 1UL && mTailSize > 0UL && !args_.transA && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.mBlockCnt);
            uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.mBlockCnt);
            uint64_t mainWindowNum = runInfo_.mBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.mBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseM;
            uint64_t mergeWindowNum = 1UL;
            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax; mergeLen += windowSize, ++mergeWindowNum) {
                uint64_t newTailMain = Align(CeilDiv((mergeLen * runInfo_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseM + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.mTailMain = newTailMain;
                    runInfo_.mBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
        uint64_t nTailSize = args_.n % runInfo_.baseN;
        if (runInfo_.nBlockCnt > 1UL && nTailSize > 0UL && args_.transB && isInnerAxisAlign) {
            uint64_t baseTailCntMax = std::min((runInfo_.baseN - nTailSize) / BASIC_BLOCK_SIZE_16, runInfo_.nBlockCnt);
            uint64_t windowSize = std::min(WINDOW_LEN, runInfo_.nBlockCnt);
            uint64_t mainWindowNum = runInfo_.nBlockCnt / windowSize - 1UL;
            uint64_t tailWindowSize = runInfo_.nBlockCnt - mainWindowNum * windowSize;
            uint64_t perfRes = (mainWindowNum + 1UL) * runInfo_.baseN;
            uint64_t mergeWindowNum = 1UL;
            for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax; mergeLen += windowSize, ++mergeWindowNum) {
                uint64_t newTailMain = Align(CeilDiv((mergeLen * runInfo_.baseN + nTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
                uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * runInfo_.baseN + mergeWindowNum * newTailMain;
                if (curPerf <= perfRes) {
                    perfRes = curPerf;
                    runInfo_.nTailMain = newTailMain;
                    runInfo_.nBaseTailSplitCnt = mergeLen + 1UL;
                }
            }
        }
    }

    void PrintTilingData(const QuantMatmulTilingData& tilingData) const
    {
        std::printf("[QuantMatmul Strategy]\n");
        std::printf("  strategy           : no_full\n");
        std::printf("[QuantMatmul Tiling Data]\n");
        std::printf("  usedCoreNum        : %u\n", tilingData.usedCoreNum);
        std::printf("  m                  : %u\n", tilingData.m);
        std::printf("  n                  : %u\n", tilingData.n);
        std::printf("  k                  : %u\n", tilingData.k);
        std::printf("  baseM              : %u\n", tilingData.baseM);
        std::printf("  baseN              : %u\n", tilingData.baseN);
        std::printf("  baseK              : %u\n", tilingData.baseK);
        std::printf("  scaleKL1           : %u\n", tilingData.scaleKL1);
        std::printf("  mTailTile          : %u\n", tilingData.mTailTile);
        std::printf("  nTailTile          : %u\n", tilingData.nTailTile);
        std::printf("  mBaseTailSplitCnt  : %u\n", tilingData.mBaseTailSplitCnt);
        std::printf("  nBaseTailSplitCnt  : %u\n", tilingData.nBaseTailSplitCnt);
        std::printf("  mTailMain          : %u\n", tilingData.mTailMain);
        std::printf("  nTailMain          : %u\n", tilingData.nTailMain);
        std::printf("  stepK              : %u\n", tilingData.stepK);
        std::printf("  nBufferNum         : %u\n", tilingData.nBufferNum);
        std::printf("  dbL0c              : %u\n", tilingData.dbL0c);
    }
};

#endif
