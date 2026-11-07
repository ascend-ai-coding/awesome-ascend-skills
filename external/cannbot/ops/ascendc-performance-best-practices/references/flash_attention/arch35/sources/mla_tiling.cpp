/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mla_tiling.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "op_kernel/deps/attention/common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "op_kernel/deps/attention/flash_mla_with_kvcache_metadata/op_kernel_aicpu/flash_mla_with_kvcache_metadata.h"

void ValidateParams(const MlaParams& p)
{
    if (!p.batch || !p.heads || !p.sq || !p.skv || !p.cores || p.cores > 36 || p.blockSize < 16 || p.blockSize > 1024 ||
        p.blockSize % 16 || !p.maxBlocks || p.bf16 > 1 || p.lse > 1 ||
        (p.layout != 0 && p.layout != 1 && p.layout != 2 && p.layout != 4) || p.kvLayout < 1 || p.kvLayout > 3 ||
        (p.mask != 0 && p.mask != 3) || !std::isfinite(p.scale) || p.skv > uint64_t(p.blockSize) * p.maxBlocks ||
        p.usedSize != (p.usedSize ? p.batch : 0) || p.cuSize != ((p.layout == 2 || p.layout == 4) ? p.batch + 1 : 0))
        throw std::invalid_argument("Invalid MLA geometry/layout/sequence contract");
}
size_t MlaOutputElements(const MlaParams& p)
{
    return size_t(p.layout == 2 || p.layout == 4 ? size_t(p.totalQ) : size_t(p.batch) * p.sq) * p.heads * 512;
}
size_t MlaWorkspaceBytes(const MlaParams& p)
{
    return size_t(p.cores) * 2 * 96 * (512 + 2 * 8) * sizeof(float);
}
optiling::FlashMlaWithKvcacheTilingData ComputeMlaTiling(const MlaParams& p)
{
    ValidateParams(p);
    optiling::FlashMlaWithKvcacheTilingData t;
    std::memset(&t, 0, sizeof(t));
    auto& b = t.baseTiling.flashMlaWithKvcacheBaseParams;
    b.bSize = p.batch;
    b.t1Size = p.cuSize ? p.totalQ : 0;
    b.n2Size = 1;
    b.gSize = p.heads;
    b.s1Size = p.cuSize ? p.totalQ : p.sq;
    b.s2Size = p.maxBlocks * p.blockSize;
    b.dSize = 512;
    b.dSizeV = 512;
    b.dSizeRope = 64;
    b.actualSeqLengthsQSize = p.cuSize;
    b.actualSeqLengthsKVSize = p.batch;
    b.scaleValue = p.scale;
    b.isSoftMaxLseEnable = p.lse;
    b.coreNum = p.cores;
    b.outputLayout = p.layout == 4 ? 5 : p.layout == 2 ? 3 : p.layout;
    b.l2CacheOffFlag = uint64_t(p.heads) * b.s1Size <= 128;
    b.keyStrides.bnStride = uint64_t(p.blockSize) * 576;
    b.keyStrides.n2Stride = p.kvLayout == 1 ? 576 : uint64_t(p.blockSize) * 576;
    b.valueStrides = b.keyStrides;
    b.kRopeStrides = b.keyStrides;
    auto& a = t.baseTiling.flashMlaWithKvcacheAttenMaskParams;
    a.sparseMode = p.mask;
    a.preTokens = INT32_MAX;
    a.nextTokens = INT32_MAX;
    if (p.mask) {
        a.attenMaskBatch = 1;
        a.attenMaskS1Size = 2048;
        a.attenMaskS2Size = 2048;
    }
    auto& pa = t.baseTiling.flashMlaWithKvcachePageAttentionParams;
    pa.paLayoutType = p.kvLayout == 1 ? 1 : p.kvLayout == 2 ? 0 : 2;
    pa.blockSize = p.blockSize;
    pa.maxBlockNumPerBatch = p.maxBlocks;
    t.baseTiling.flashMlaWithKvcacheSystemPrefixParams.isActualSharedPrefixLenNull = 1;
    auto& w = t.baseTiling.flashMlaWithKvcacheWorkspaceParams;
    w.accumOutSize = p.cores * 2 * 96 * 512;
    w.logSumExpSize = p.cores * 2 * 96 * 8;
    auto& e = t.baseTiling.flashMlaWithKvcacheEmptyTensorParams;
    e.totalOutputSize = MlaOutputElements(p);
    e.totalSoftMaxLseOutputSize = p.lse ? e.totalOutputSize / 512 : 0;
    e.singleCoreSize = (e.totalOutputSize + p.cores * 2 - 1) / (p.cores * 2);
    e.needInit = 1;
    return t;
}
std::vector<uint32_t> ComputeMlaMetadata(const MlaParams& p, const int32_t* cache, const int32_t* cu,
                                         const int32_t* used)
{
    ValidateParams(p);
    if (!cache || (p.cuSize && !cu) || (p.usedSize && !used))
        throw std::invalid_argument("Missing sequence tensor");
    load_balance::DeviceInfo deviceInfo;
    deviceInfo.aicCoreMaxNum = deviceInfo.aicCoreMinNum = p.cores;
    deviceInfo.aivCoreMaxNum = deviceInfo.aivCoreMinNum = p.cores * 2;
    load_balance::BaseInfo baseInfo;
    baseInfo.batchSize = p.batch;
    baseInfo.querySeqSize = p.sq;
    baseInfo.queryHeadNum = p.heads;
    baseInfo.kvSeqSize = p.skv;
    baseInfo.kvHeadNum = 1;
    baseInfo.headDimQk = 576;
    baseInfo.headDimV = 512;
    baseInfo.attenMaskFlag = p.mask != 0;
    baseInfo.sparseMode = p.mask ? p.mask : static_cast<uint32_t>(load_balance::SparseMode::BUTT);
    baseInfo.preToken = baseInfo.nextToken = UINT32_MAX;
    baseInfo.layoutQuery = load_balance::ConvertToLayout(p.layout == 0 ? "BSND" : p.layout == 1 ? "BNSD" : "TND");
    baseInfo.queryType = baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeKvSeq = false;
    baseInfo.actualKvSeqSize.assign(cache, cache + p.batch);
    if (p.cuSize && (cu[0] != 0 || cu[p.batch] != int64_t(p.totalQ)))
        throw std::invalid_argument("Invalid cumulative Q endpoints");
    for (uint32_t i = 0; i < p.batch; ++i) {
        int64_t qlen = p.cuSize ? int64_t(cu[i + 1]) - cu[i] : p.sq;
        if (cache[i] < 0 || cache[i] > int64_t(p.skv) || qlen < 0 || qlen > p.sq ||
            (p.usedSize && (used[i] < 0 || used[i] > qlen)))
            throw std::invalid_argument("Sequence length exceeds storage/max length");
    }
    if (p.usedSize) {
        baseInfo.isCumulativeQuerySeq = false;
        baseInfo.actualQuerySeqSize.assign(used, used + p.batch);
    } else if (p.cuSize) {
        baseInfo.isCumulativeQuerySeq = true;
        baseInfo.actualQuerySeqSize.assign(cu + 1, cu + p.batch + 1);
    }
    load_balance::SectionStreamKParam param;
    param.mBaseSize = 96;
    param.s2BaseSize = 112;
    param.l2Byte = uint64_t(p.heads) * p.sq <= 192 ? 0 : 96U * 1024U * 1024U;
    param.fdTolerance = 10;
    param.fdLeastBlock = 3;
    param.fdOn = true;
    load_balance::SectionStreamKResult splitRes{};
    if (load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) != SECTION_STREAM_K_SUCCESS)
        throw std::runtime_error("SectionStreamK failed");
    std::vector<uint32_t> data((1 + size_t(splitRes.sectionNum) * p.cores * 3) * 16);
    using namespace optiling;
    detail::FaMetadata faMetadata(data.data(), splitRes.sectionNum, static_cast<uint32_t>(p.cores),
                                  static_cast<uint32_t>((p.cores * 2)));
    faMetadata.Clear(); // set to all 0

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    // Keep metadata tile sizes consistent with the fixed M96/S2=112 kernel.
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, HEAD_M_BASE_SIZE_MLA);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, HEAD_S2_BASE_SIZE_MLA);
    faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, static_cast<FA_METADATA_T>(p.cores));
    faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, static_cast<FA_METADATA_T>((p.cores * 2)));
    faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX,
                               static_cast<FA_METADATA_T>(load_balance::OutputLayout::BN2_S1G));
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }

    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(p.cores)}; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto& prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX, faRes.bNEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }

        auto& fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX, fdRes.bNIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
    return data;
}
static thread_local std::string lastError;
extern "C" const char* mla_last_error()
{
    return lastError.c_str();
}
extern "C" size_t mla_tiling_size()
{
    return sizeof(optiling::FlashMlaWithKvcacheTilingData);
}
extern "C" int mla_prepare(const MlaParams* p, const int32_t* cache, const int32_t* cu, const int32_t* used,
                           void* tiling, uint32_t* metadata, size_t capacity, size_t* words, size_t* workspaceBytes)
{
    try {
        if (!p || !words || !workspaceBytes)
            throw std::invalid_argument("Null prepare argument");
        auto t = ComputeMlaTiling(*p);
        auto m = ComputeMlaMetadata(*p, cache, cu, used);
        *words = m.size();
        *workspaceBytes = MlaWorkspaceBytes(*p);
        if (tiling)
            std::memcpy(tiling, &t, sizeof(t));
        if (metadata) {
            if (capacity < m.size())
                throw std::invalid_argument("Metadata capacity too small");
            std::memcpy(metadata, m.data(), m.size() * 4);
        }
        lastError.clear();
        return 0;
    } catch (const std::exception& e) {
        lastError = e.what();
        return -1;
    }
}
