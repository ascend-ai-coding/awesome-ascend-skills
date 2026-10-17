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
 * \file paired_segment_rotate_vf.h
 * \brief 成对分段旋转的 Reg VF 集成骨架
 *
 * 适用架构：A5 / DAV_3510（__CCE_AICORE__ != 310 时 VL = 256B / sizeof(float) = 64）。
 * 命名空间：源码统一写 AscendC::Reg::。计算指令 Mul/Add/Sub/Muls 在 `using namespace AscendC;` 后裸调用。
 *
 * 核心结构：
 *   1. rotate-half 的前后半错位 = 用不同 offset 把同一 UB 的两半 load 进两个寄存器，不再需要 UB→UB 搬运；
 *   2. cos/sin/in 的乘加全在 RegTensor<float> 上一拍算完（half 6 条指令），无中间 LocalTensor；
 *   3. fp16/bf16 的 Cast 与 load/store 融合（DIST_UNPACK_B16 / DIST_PACK_B32），无独立 Cast pass、
 *      无额外 fp32 中转 buffer —— 一份 VF 计算对 fp16/bf16/fp32 完全同构；
 *   4. 尾块用 Reg::UpdateMask<float>(cnt) 覆盖。
 *
 * 拷贝使用：放进项目 op_kernel/，最小适配 dLen/dAlign/currSNum/currDNum 的来源（通常从 host tiling 读），
 * 接入后须在目标工程完成编译、精度和性能验收。
 */
#ifndef PAIRED_SEGMENT_ROTATE_VF_H
#define PAIRED_SEGMENT_ROTATE_VF_H

#include "kernel_operator.h"

using namespace AscendC;

// ============================================================================
// A. Load/Store + Cast 封装（摘自 load_store_utils.h，namespace ops）
//    通用的 fp16/bf16 <-> fp32 VF load/store cast，任何 "寄存器内 fp32 计算" 的
//    elementwise/vector VF 中可复用，不依赖本骨架的旋转公式。
// ============================================================================
namespace ops {
using namespace AscendC;

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// load 对齐的 half/bf16/fp32 input(UB) 到 fp32 dst(寄存器)
template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(__ubuf__ T* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ half*)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, ((__ubuf__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign(dst, ((__ubuf__ float*)(input) + (offset)));
    }
}

// load 2 个对齐 Tensor 到 2 个寄存器（half 的前/后半、interleave 的偶/奇一次装两块）
template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(__ubuf__ T* src1, __ubuf__ T* src2, Reg::RegTensor<float>& dst1,
                                              Reg::RegTensor<float>& dst2, Reg::MaskReg& dst1Preg,
                                              Reg::MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16Q;
        Reg::RegTensor<half> xFp16R;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ half*)(src1) + (src1Offset)));
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xFp16Q;
        Reg::RegTensor<bfloat16_t> xFp16R;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xFp16Q,
                                                                   ((__ubuf__ bfloat16_t*)(src1) + (src1Offset)));
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xFp16R,
                                                                   ((__ubuf__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        Reg::LoadAlign(dst1, ((__ubuf__ float*)(src1) + (src1Offset)));
        Reg::LoadAlign(dst2, ((__ubuf__ float*)(src2) + (src2Offset)));
    }
}

// store 对齐的 fp32 src(寄存器) 到 output(UB)，回写时按 T Cast（fp16/bf16 走 DIST_PACK_B32）
template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(__ubuf__ T* output, Reg::RegTensor<float>& src, Reg::MaskReg& preg,
                                               uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> yFp16;
        Cast<half, float, castTraitB322B16>(yFp16, src, preg);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>(((__ubuf__ half*)output + offset), yFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(((__ubuf__ bfloat16_t*)output + offset), xBf16,
                                                                   preg);
    } else {
        Reg::StoreAlign(((__ubuf__ float*)output + offset), src, preg);
    }
}
} // namespace ops

// 本骨架不依赖来源工程的 op_kernel/math_util.h；把所需的两个整数工具留在本文件内。
template <typename T>
__aicore__ inline constexpr T VfExampleCeilDiv(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline constexpr T VfExampleCeilAlign(T value, T alignment)
{
    return VfExampleCeilDiv(value, alignment) * alignment;
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

constexpr uint32_t VL_FLOAT32_SIZE = GetVRegSize() / sizeof(float); // A5: 64
constexpr uint32_t BLOCK_TYPE_SIZE = 32U;
constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t QUARTER_MODE_COEF = 4;

// ============================================================================
// A2. 对齐搬运（摘自 apply_rotary_pos_emb_common.h）
// 来源实现让前半 [0, D/2) 与后半 [D/2, D) 在 UB 上都按 32B 对齐。做法：
// CopyIn 分两次 DataCopyPad，
// 前半放 UB 偏移 0，后半放 UB 偏移 dstStride = CeilAlign(halfD, 32/sizeof(T))
// （= VF 里的 halfDAlign）。该 UB 布局是计算地址契约，不能只改搬运或只改 VF offset。
// ============================================================================
template <typename T>
__aicore__ inline void CopyInHalfMode(LocalTensor<T>& inTensor, GlobalTensor<T>& gmTensor, const int64_t offset,
                                      const uint32_t blockCount, const uint32_t blockLen,
                                      const DataCopyPadExtParams<T>& padParams, const int64_t D, const int64_t realDim,
                                      const uint32_t dSplitCoef)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = blockLen;
    int64_t srcStride = D - realDim / dSplitCoef; // GM 侧跳过另一半的间隙（多行批量）
    copyParams.srcStride = srcStride * sizeof(T);
    int64_t dstStride = VfExampleCeilAlign<int64_t>(realDim / dSplitCoef, BLOCK_TYPE_SIZE / sizeof(T)); // = halfDAlign
    copyParams.dstStride = dstStride * sizeof(T) / BLOCK_TYPE_SIZE;
    DataCopyPad(inTensor, gmTensor[offset], copyParams, padParams); // 前半 -> UB 偏移 0
    DataCopyPad(inTensor[dstStride], gmTensor[offset + realDim / dSplitCoef], copyParams,
                padParams); // 后半 -> 对齐偏移
}

template <typename T>
__aicore__ inline void CopyOutHalfMode(LocalTensor<T>& outTensor, GlobalTensor<T>& gmTensor, const int64_t offset,
                                       const uint32_t blockCount, const uint32_t blockLen, const int64_t D,
                                       const int64_t realDim, const uint32_t dSplitCoef)
{
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = blockCount;
    copyOutParams.blockLen = blockLen;
    int64_t srcStride = VfExampleCeilAlign<int64_t>(realDim / dSplitCoef, BLOCK_TYPE_SIZE / sizeof(T)); // = halfDAlign
    copyOutParams.srcStride = srcStride * sizeof(T) / BLOCK_TYPE_SIZE;
    int64_t dstStride = D - realDim / dSplitCoef;
    copyOutParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(gmTensor[offset], outTensor, copyOutParams);                                   // 前半：对齐 UB -> GM
    DataCopyPad(gmTensor[offset + realDim / dSplitCoef], outTensor[srcStride], copyOutParams); // 后半
}

// ============================================================================
// B. VF 计算核（摘自 apply_rotary_pos_emb_common.h）
// 注意 HalfAlignVF/QuarterAlignVF 里 halfOffset = offset + halfDAlign 与上面
// CopyInHalfMode 的 dstStride 完全一致 —— UB 布局对齐前后半，VF 的两个半区
// load/store 才都是对齐微指令。搬运与计算的对齐口径必须一致。
// ============================================================================

// ---------------------------------------------------------------------------
// B1. Half 模式
//   qOut[0] = q[0] * cos[0] - q[1] * sin[0]
//   qOut[1] = q[1] * cos[1] + q[0] * sin[1]
// 前半 [0..D/2) 与后半 [D/2..D) 分别装进 vregIn/vregHalfIn，一拍 6 条指令算完。
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void HalfAlignVF(const AscendC::LocalTensor<T>& sinTensor, const AscendC::LocalTensor<T>& cosTensor,
                                   const AscendC::LocalTensor<T>& inTensor, const AscendC::LocalTensor<T>& outTensor,
                                   uint32_t dLen, uint32_t dAlign, uint16_t currSNum, uint16_t currDNum)
{
    __ubuf__ T* sinUb = (__ubuf__ T*)sinTensor.GetPhyAddr();
    __ubuf__ T* cosUb = (__ubuf__ T*)cosTensor.GetPhyAddr();
    __ubuf__ T* inUb = (__ubuf__ T*)inTensor.GetPhyAddr();
    __ubuf__ T* outUb = (__ubuf__ T*)outTensor.GetPhyAddr();
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = VfExampleCeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = VfExampleCeilDiv(halfD, VL_FLOAT32_SIZE);
    __ubuf__ T* currInUb;
    __ubuf__ T* currOutUb;
    __ubuf__ T* currSinUb;
    __ubuf__ T* currCosUb;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vregIn;
        Reg::RegTensor<float> vregHalfIn;
        Reg::RegTensor<float> vregSin;
        Reg::RegTensor<float> vregHalfSin;
        Reg::RegTensor<float> vregCos;
        Reg::RegTensor<float> vregHalfCos;
        Reg::RegTensor<float> vregOut;
        Reg::RegTensor<float> vregHalfOut;
        Reg::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * dAlign;
            currCosUb = cosUb + sIdx * dAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlign;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = Reg::UpdateMask<float>(updateCnt); // 尾块自动窄化 mask
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset,
                                                   halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset,
                                                   halfOffset);

                    Mul(vregSin, vregSin, vregHalfIn, preg);          // sin*q_back
                    Mul(vregHalfOut, vregHalfSin, vregIn, preg);      // sin*q_front
                    Mul(vregCos, vregCos, vregIn, preg);              // cos*q_front
                    Sub(vregOut, vregCos, vregSin, preg);             // out_front = cos*q_front - sin*q_back
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);  // cos*q_back
                    Add(vregHalfOut, vregHalfOut, vregHalfCos, preg); // out_back = sin*q_front + cos*q_back

                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregOut, preg, offset);
                    ops::StoreOneTensorForDtypeT<T>(currOutUb, vregHalfOut, preg, halfOffset);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 交错、Quarter、Batch 等分支不在本骨架中。遇到这些布局时记录范例缺口，
// 不按本文件注释补写。

#endif // PAIRED_SEGMENT_ROTATE_VF_H
