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
 * \file euclidean_norm_empty.h
 * \brief EuclideanNorm 空 tensor 模板 kernel 类实现（新范式）。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_
#define OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "euclidean_norm_tiling_data.h"
#include "euclidean_norm_tiling_key.h"

namespace NsEuclideanNorm {

using namespace AscendC;

constexpr uint32_t VL_BYTES_EMPTY = Ops::Base::GetVRegSize();
constexpr float EMPTY_R_OUTPUT_VALUE = 0.0f; // empty_r_output_value：EMPTY_R 输出固化值

// EMPTY_R 输出填充 VF：Duplicate 固化值 sqrt(0)=0 到 outBuf，供后续 CopyOut 循环复用。
template <typename DType>
__simd_vf__ inline void DuplicateEmptyROutputVfImpl(__ubuf__ DType* outPtr, DType value, uint32_t totalElems,
                                                    uint16_t repeatTime)
{
    constexpr uint32_t repPerVf = VL_BYTES_EMPTY / sizeof(DType);
    AscendC::Reg::RegTensor<DType> dReg;
    AscendC::Reg::Duplicate(dReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(repPerVf);
        mask = AscendC::Reg::UpdateMask<DType>(remaining);
        AscendC::Reg::StoreAlign(outPtr + off, dReg, mask);
    }
}

template <typename DType>
class EuclideanNormEmptyKernel {
public:
    using DT = DType;

    __aicore__ inline EuclideanNormEmptyKernel() {}

    // Empty 初始化：缓存 EmptyTilingData 指针（切分字段经 td_ 直读，不逐字段拷贝到成员），
    // 绑定输出 GM，仅分配 outBuf 一个 UB buffer。
    __aicore__ inline void Init(GM_ADDR y, const EuclideanNormEmptyTilingData* td, TPipe* pipe)
    {
        td_ = td;

        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));
        pipe_ = pipe;
        pipe_->InitBuffer(outBuf_, td_->postBufSize);
    }

    // Empty 主流程：EMPTY_A 时 usedCoreNum=0 全核直接返回；EMPTY_R 时按大小核均分输出区间，
    // outBuf 填 0 后循环 CopyOut
    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        // 超出启用核数的核直接退出（EMPTY_A 时 usedCoreNum=0 全核早退）
        if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
            return;
        }

        // 解码本核输出区间 [aStart, aEnd)（按 a 元素数均分）
        int64_t aStart = 0;
        int64_t aEnd = 0;
        if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
            aStart = blockIdx * td_->aBigCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aBigCoreLoopCnt * td_->aUbFactor;
        } else {
            aStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt * td_->aUbFactor +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aSmallCoreLoopCnt * td_->aUbFactor;
        }
        // 末核受输出总数约束（防越界）
        if (aEnd > td_->aTotal) {
            aEnd = td_->aTotal;
        }
        if (aStart >= aEnd) {
            return;
        }

        // 申请核内同步 MutexID（V 填充 → MTE3 搬出两段同 id 链式串行）
        mutexId_ = AscendC::AllocMutexID();

        // 计算段（V）：outBuf 一次填固化值，供后续 CopyOut 循环复用
        AscendC::Mutex::Lock<PIPE_V>(mutexId_);
        DuplicateEmptyROutputVf();
        AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

        // 搬出段（MTE3）：按 aUbFactor 分块循环直写 GM y
        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId_);
        for (int64_t aOff = aStart; aOff < aEnd; aOff += td_->aUbFactor) {
            const int64_t aLen = (aOff + td_->aUbFactor > aEnd) ? (aEnd - aOff) : td_->aUbFactor;
            CopyOut(aOff, aLen);
        }
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId_);

        AscendC::ReleaseMutexID(mutexId_);
    }

private:
    // 调 DuplicateEmptyROutputVfImpl 按 td_->aUbFactor 填 outBuf 并做 V→MTE3 事件交接（V 写完才允许 MTE3 搬出）。
    __aicore__ inline void DuplicateEmptyROutputVf()
    {
        __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.Get<DT>().GetPhyAddr());
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor);
        const uint32_t repDType = VL_BYTES_EMPTY / sizeof(DT);
        // repeat 按 dtype 的寄存器元素数分段
        const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, repDType));

        asc_vf_call<DuplicateEmptyROutputVfImpl<DT>>(outPtr, static_cast<DT>(EMPTY_R_OUTPUT_VALUE), totalElems,
                                                     repeatTime);
    }

    __aicore__ inline void CopyOut(int64_t outOff, int64_t aLen)
    {
        auto outLocal = outBuf_.Get<DT>();

        // 单 burst 直写 GM y（输出 dense、无 pad）
        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm_[outOff], outLocal, outParams);
    }

    const EuclideanNormEmptyTilingData* td_ = nullptr;

    GlobalTensor<DT> yGm_; // 输出 GM
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> outBuf_; // 唯一 UB buffer（固化值输出）
    uint8_t mutexId_ = 0;               // 核内同步 MutexID（Process 内申请/释放）
};

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_EMPTY_H_
