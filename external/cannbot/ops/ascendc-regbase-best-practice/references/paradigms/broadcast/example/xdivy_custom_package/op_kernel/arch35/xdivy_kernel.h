/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Xdivy Kernel — XdivyKernel<T, RANK>
// y = (x1 == 0) ? 0 : (x1 / x2)，广播二元算子，FP16/BF16 走 Cast 流水

#ifndef XDIVY_KERNEL_H_
#define XDIVY_KERNEL_H_

#include "kernel_operator.h"
#include "xdivy_tiling_struct.h"
#include "xdivy_struct.h"

// GetCoreRange: 算本核 tile 区间 [start, end) — 前 mainCoreNum 个核各处理
// mainTiles 块，其余核各处理 mainTiles-1 块（母模板核间均衡切分公式）
__aicore__ inline void GetCoreRange(int64_t coreId, int64_t mainTiles, int64_t mainCoreNum, int64_t& start,
                                    int64_t& end)
{
    if (coreId < mainCoreNum) {
        start = coreId * mainTiles;
        end = start + mainTiles;
    } else {
        start = mainCoreNum * mainTiles + (coreId - mainCoreNum) * (mainTiles - 1);
        end = start + mainTiles - 1;
    }
}

// GetUBSplitRange: 返回本 tile 切分轴实际段长（尾块取 ubTail，其余取 ubFactor）
__aicore__ inline int64_t GetUBSplitRange(int64_t ubBlockIdx, int64_t ubOuter, int64_t ubFactor, int64_t ubTail)
{
    return (ubBlockIdx == ubOuter - 1) ? ubTail : ubFactor;
}

// FlatToEffectiveCoord: 平铺 tile 序号 → 广播坐标系起点坐标
// （flat%ubOuter → 切分轴块起点，flat/ubOuter → 外层维混合进制分解，内层维保持 0）
__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t ubSplitIdx,
                                            int64_t ubFactor, int64_t ubOuter, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++) {
        effCoord[d] = 0;
    }
    int64_t ubBlockIdx = flat % ubOuter;
    int64_t outer = flat / ubOuter;
    for (int64_t d = ubSplitIdx - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[ubSplitIdx] = ubBlockIdx * ubFactor;
    return true;
}

// CalcOffset: 坐标×stride → GM 元素偏移（广播维 stride=0，地址不前进）
__aicore__ inline int64_t CalcOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++) {
        offset += effCoord[d] * strides[d];
    }
    return offset;
}

// CalcTransferCount: 算输出侧实际搬运元素数（Host 侧已保证输出稠密，dim=1 分支仅作防御性保留）
__aicore__ inline int64_t CalcTransferCount(const int64_t* normalShape, int64_t rank, int64_t ubSplitIdx,
                                            int64_t ubBlockLength)
{
    int64_t splitElems = (normalShape[ubSplitIdx] == 1) ? 1 : ubBlockLength;
    int64_t innerElems = 1;
    for (int64_t d = ubSplitIdx + 1; d < rank; d++) {
        innerElems *= normalShape[d];
    }
    return splitElems * innerElems;
}

template <typename T>
__simd_vf__ inline void XdivyVF(__ubuf__ T* dstAddr, __ubuf__ T* srcXAddr, __ubuf__ T* srcYAddr, uint32_t count,
                                uint32_t oneRepeatSize, uint16_t repeatTimes);

template <typename T, int64_t RANK>
class XdivyKernel {
    static constexpr int64_t MAX_RANK = 8;
    static constexpr int64_t MAX_NDDMA_DIMS = 5;
    static constexpr int64_t NDDMA_DIMS = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;

    static constexpr bool NEED_CAST = !std::is_same_v<T, float>;

    static constexpr int64_t BUF_COUNT_NO_CAST = 3;
    static constexpr int64_t BUF_COUNT_CAST = 4;
    static constexpr int64_t NUM_BUF = NEED_CAST ? BUF_COUNT_CAST : BUF_COUNT_NO_CAST;

    AscendC::TPipe pipe_;
    const XdivyTilingData<RANK>* td_ = nullptr;
    AscendC::GlobalTensor<T> gmIn_[MAX_INPUT_SLOTS];
    AscendC::GlobalTensor<T> gmOut_[MAX_OUTPUT_SLOTS];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[NUM_BUF];
    AscendC::MultiCopyParams<T, NDDMA_DIMS> nddmaParams_[MAX_INPUT_SLOTS];
    int64_t nddmaOuterIters_[MAX_INPUT_SLOTS];
    int64_t nddmaDims_ = 0;

public:
    // Init: 绑定 GM、初始化 TBuf，并预构每输入的 NDDMA 描述（半成品，运行期只 patch 切分轴长度）
    __aicore__ inline void Init(GM_ADDR* inputs, GM_ADDR* outputs, const XdivyTilingData<RANK>* td)
    {
        td_ = td;
        for (int32_t i = 0; i < MAX_INPUT_SLOTS; i++) {
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        }
        for (int32_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        }
        for (int32_t i = 0; i < NUM_BUF; i++) {
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);
        }

        const int64_t* dstShape = td_->maxBroShape;
        int64_t k = td_->split.ubSplitIdx;
        nddmaDims_ = (RANK - k <= NDDMA_DIMS) ? (RANK - k) : NDDMA_DIMS;
        for (int32_t inp = 0; inp < MAX_INPUT_SLOTS; inp++) {
            int64_t inner = 1;
            int64_t nd = 0;
            for (int64_t d = RANK - 1; d >= k && nd < NDDMA_DIMS; d--) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[inp][d];
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
                inner *= (d == k) ? td_->split.ubFactor : dstShape[d];
                nd++;
            }
            for (; nd < NDDMA_DIMS; nd++) {
                nddmaParams_[inp].loopInfo.loopSize[nd] = 1;
                nddmaParams_[inp].loopInfo.loopSrcStride[nd] = 0;
                nddmaParams_[inp].loopInfo.loopDstStride[nd] = inner;
                nddmaParams_[inp].loopInfo.loopLpSize[nd] = 0;
                nddmaParams_[inp].loopInfo.loopRpSize[nd] = 0;
            }
            nddmaOuterIters_[inp] = 1;
            for (int64_t d = k; d < RANK - nddmaDims_; d++) {
                nddmaOuterIters_[inp] *= (d == k) ? td_->split.ubFactor : dstShape[d];
            }
        }
    }

    // Process: 主循环 — 逐 tile { 解码 tile → 搬入 → 计算 → 搬出 }，
    // Mutex 串行化 MTE2/V/MTE3 三级流水，FP32/FP16 按编译期分发
    __aicore__ inline void Process()
    {
        uint8_t mutexId = AscendC::AllocMutexID();

        int64_t start = 0;
        int64_t end = 0;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.mainTiles, td_->multicore.mainCoreNum, start, end);

        int64_t innerCount = 1;
        for (int64_t d = td_->split.ubSplitIdx + 1; d < RANK; d++) {
            innerCount *= td_->maxBroShape[d];
        }

        int64_t coord[MAX_RANK] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t ubBlockLength =
                GetUBSplitRange(flat % td_->split.ubOuter, td_->split.ubOuter, td_->split.ubFactor, td_->split.ubTail);
            int64_t count = ubBlockLength * innerCount;
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.ubSplitIdx, td_->split.ubFactor,
                                 td_->split.ubOuter, coord);

            constexpr int32_t inX1 = 0;
            constexpr int32_t inX2 = 1;
            constexpr int32_t outY = 0;

            if constexpr (NEED_CAST) {
                ProcessFP16(coord, inX1, inX2, outY, ubBlockLength, count, mutexId);
            } else {
                ProcessFP32(coord, inX1, inX2, outY, ubBlockLength, count, mutexId);
            }
        }

        AscendC::ReleaseMutexID(mutexId);
    }

private:
    // ProcessFP32: FP32 路径 — 搬入两输入 → VF 计算 → 搬出
    __aicore__ inline void ProcessFP32(const int64_t* coord, int32_t inX1, int32_t inX2, int32_t outY,
                                       int64_t ubBlockLength, int64_t count, uint8_t mutexId)
    {
        constexpr int32_t bX1 = 0;
        constexpr int32_t bX2 = 1;
        constexpr int32_t bY = 2;

        AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
        CopyInOne(coord, inX1, bX1, ubBlockLength);
        CopyInOne(coord, inX2, bX2, ubBlockLength);
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

        AscendC::Mutex::Lock<PIPE_V>(mutexId);
        CallXdivyVF(bY, bX1, bX2, count);
        AscendC::Mutex::Unlock<PIPE_V>(mutexId);

        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId);
        CopyOutOne(coord, outY, bY, ubBlockLength);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId);
    }

    // ProcessFP16: FP16/BF16 路径 — Cast 到 FP32 计算后再 Cast 回原 dtype
    __aicore__ inline void ProcessFP16(const int64_t* coord, int32_t inX1, int32_t inX2, int32_t outY,
                                       int64_t ubBlockLength, int64_t count, uint8_t mutexId)
    {
        constexpr int32_t bTempX1 = 0;
        constexpr int32_t bX1 = 1;
        constexpr int32_t bX2 = 2;
        constexpr int32_t bTempX2 = 3;

        AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
        CopyInOne(coord, inX1, bTempX1, ubBlockLength);
        CopyInOne(coord, inX2, bTempX2, ubBlockLength);
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

        AscendC::Mutex::Lock<PIPE_V>(mutexId);
        AscendC::Cast(buf_[bX1].template Get<float>(), buf_[bTempX1].template Get<T>(), AscendC::RoundMode::CAST_NONE,
                      count);
        AscendC::Cast(buf_[bX2].template Get<float>(), buf_[bTempX2].template Get<T>(), AscendC::RoundMode::CAST_NONE,
                      count);

        CallXdivyVF(bTempX2, bX1, bX2, count);

        AscendC::Cast(buf_[bTempX1].template Get<T>(), buf_[bTempX2].template Get<float>(),
                      AscendC::RoundMode::CAST_RINT, count);
        AscendC::Mutex::Unlock<PIPE_V>(mutexId);

        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId);
        CopyOutOne(coord, outY, bTempX1, ubBlockLength);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId);
    }

    // CallXdivyVF: 调用寄存器级 VF 完成 (x1==0)?0:(x1/x2)，按 repeat 分批
    __aicore__ inline void CallXdivyVF(int32_t dstSlot, int32_t srcXSlot, int32_t srcYSlot, int64_t count)
    {
        constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(float);
        uint16_t repeatTimes = AscendC::CeilDivision(count, oneRepeatSize);
        __ubuf__ float* dstAddr = (__ubuf__ float*)buf_[dstSlot].template Get<float>().GetPhyAddr();
        __ubuf__ float* srcXAddr = (__ubuf__ float*)buf_[srcXSlot].template Get<float>().GetPhyAddr();
        __ubuf__ float* srcYAddr = (__ubuf__ float*)buf_[srcYSlot].template Get<float>().GetPhyAddr();
        asc_vf_call<XdivyVF<float>>(dstAddr, srcXAddr, srcYAddr, static_cast<uint32_t>(count), oneRepeatSize,
                                    repeatTimes);
    }

    // CopyInOne: 带广播的 GM→UB 搬入 — RANK≤5 单条 NDDMA；RANK>5 时
    // NDDMA 管最内 5 维，外层 gap 维用拍平的 oi 循环
    __aicore__ inline void CopyInOne(const int64_t* coord, int32_t inputIdx, int32_t slot, int64_t ubBlockLength)
    {
        int64_t k = td_->split.ubSplitIdx;
        int64_t off = CalcOffset(coord, td_->inputStrides[inputIdx], RANK);
        const int64_t* dstShape = td_->maxBroShape;
        auto params = nddmaParams_[inputIdx];
        int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd) {
                params.loopInfo.loopSize[nd] = ubBlockLength;
            }
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }
        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};
        if constexpr (RANK <= MAX_NDDMA_DIMS) {
            AscendC::DataCopy<T, NDDMA_DIMS, cfg>(buf_[slot].template Get<T>(), gmIn_[inputIdx][off], params);
        } else {
            AscendC::LocalTensor<T> buf = buf_[slot].template Get<T>();
            int64_t elemBase = off;
            for (int64_t oi = 0; oi < nddmaOuterIters_[inputIdx]; oi++) {
                int64_t elemAdj = 0;
                int64_t tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? ubBlockLength : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[inputIdx][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<T, NDDMA_DIMS, cfg>(buf[oi * inner], gmIn_[inputIdx][elemBase + elemAdj], params);
            }
        }
    }

    // CopyOutOne: UB→GM 搬出 — DataCopyPad 单 block 线性搬运
    __aicore__ inline void CopyOutOne(const int64_t* coord, int32_t outputIdx, int32_t slot, int64_t ubBlockLength)
    {
        int64_t off = CalcOffset(coord, td_->outputStrides[outputIdx], RANK);
        int64_t cnt = CalcTransferCount(td_->outputShapes[outputIdx], RANK, td_->split.ubSplitIdx, ubBlockLength);
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[slot].template Get<T>(), extParams);
    }
};

// XdivyVF: 寄存器级 VF — Div + EQ 比较 + Select 实现 (x1==0)?0:(x1/x2)
template <typename T>
__simd_vf__ inline void XdivyVF(__ubuf__ T* dstAddr, __ubuf__ T* srcXAddr, __ubuf__ T* srcYAddr, uint32_t count,
                                uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> regX, regY, regDiv, regZero, regFinal;
    AscendC::Reg::MaskReg mask, maskEq;
    AscendC::Reg::AddrReg aReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);

        AscendC::Reg::LoadAlign(regX, srcXAddr, aReg);
        AscendC::Reg::LoadAlign(regY, srcYAddr, aReg);

        AscendC::Reg::Div(regDiv, regX, regY, mask);

        AscendC::Reg::Sub(regZero, regX, regX, mask);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(maskEq, regX, regZero, mask);

        AscendC::Reg::Select<float>(regFinal, regZero, regDiv, maskEq);

        AscendC::Reg::StoreAlign(dstAddr, regFinal, aReg, mask);
    }
}

#endif // XDIVY_KERNEL_H_
