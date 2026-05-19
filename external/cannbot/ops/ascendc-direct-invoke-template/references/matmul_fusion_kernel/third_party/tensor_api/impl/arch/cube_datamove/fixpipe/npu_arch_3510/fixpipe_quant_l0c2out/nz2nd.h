/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "tensor_api/impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2out/nz2nd.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file nz2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2OUT_NZ2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2OUT_NZ2ND_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class Fixpipe2OutNZ2NDSimpleQuant3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void Run(const T& dst, const U& src, const V& quant, const Params&... params)
    {
        SetRegisterImpl<trait, T, U, V>(dst, src, quant);
        DataCopyImpl<trait, quantPre, T, U>(dst, src, params...);
    }

private:

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename V>
    __aicore__ inline static auto SetRegisterImpl(const T& dst, const U& src, const V& quant)
    {
        uint32_t ndNum = 1;
        uint32_t srcNDStride = 0;
        uint32_t dstNDStride = 0;
        SetRegister3510::SetRegister(quant, ndNum, dstNDStride, srcNDStride);
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src, const FixpipeParams& params)
    {
        CheckTemplate<trait, T, U>();
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        uint32_t mSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout));
        uint32_t nSize = Std::min(
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout));
        
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        if constexpr (GetHardPos<T>() == Hardware::GM) {
            uint8_t cacheMode = GetCacheModeFromTensor(dst);
            bool isChannelSplit = trait.enableChannelSplit;
            CopyMatrixCcToGm3510::DataCopy<quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride,
                                                                  cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn,
                                                                  nz2dnEn);
        } else {
            uint8_t dualDstCtl = trait.dualDstCtl;
            bool subBlockId = false;
            CopyMatrixCcToUb3510::DataCopy<quantPre, T, U>(dst, src, nSize, mSize, srcStride, dstStride,
                                                                      dualDstCtl, reluEn, unitFlag, subBlockId, nz2ndEn,
                                                                      nz2dnEn);
        }
    }
};

class Fixpipe2OutNZ2NDVector3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void FixpipeNZ2NDVectorEntrance(const T& dst, const U& src, const V& quant, const Params& ...params)
    {
        FixpipeNZ2NDVectorCompute<trait, quantPre, T, U, V>(dst, src, quant, params...);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, bool isTail>
    __aicore__ inline static auto GenParams(const T& dst, const U& src, const FixpipeParams& params)
    {
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        if constexpr (isTail) {
            nSize = nSize % MAIN_LOOP_N_SIZE_3510;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE_3510) {
                nSize = MAIN_LOOP_N_SIZE_3510;
            }
        }
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = params.unitFlag;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        if constexpr (GetHardPos<T>() == Hardware::GM) {
            uint8_t cacheMode = GetCacheModeFromTensor(dst);
            bool isChannelSplit = trait.enableChannelSplit;
            auto fixpipeParams = Std::make_tuple(
                nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
            return fixpipeParams;
        } else {
            if (trait.dualDstCtl == DUAL_DST_SPLIT_N) {
                dstStride = dstStride >> 1;
            }
            uint8_t dualDstCtl = trait.dualDstCtl;
            bool subBlockId = false;
            auto fixpipeParams = Std::make_tuple(
                nSize, mSize, srcStride, dstStride, dualDstCtl, reluEn, unitFlag, subBlockId, nz2ndEn, nz2dnEn);
            return fixpipeParams;
        }
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void FixpipeNZ2NDVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
        uint32_t calNSize, uint32_t tailNSize, const Params&... params)
    {
        auto mainLoopParam = GenParams<trait, T, U, false>(dst, src, params...);
        for (uint16_t i = 0; i < nIterNum; ++i) {
            CopyDeqTensorToFbuf3510::CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertSync();
            auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * CBURST_NUM_3510));
            auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * MAIN_LOOP_N_SIZE_3510));
            DataCopyWrapper<trait, quantPre>(dst(dstCoord), src(srcCoord),
                mainLoopParam, tuple_sequence<decltype(mainLoopParam)>{});
        }
        if (tailNSize) {
            auto tailParam = GenParams<trait, T, U, true>(dst, src, params...);
            CopyDeqTensorToFbuf3510::CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertSync();
            auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * CBURST_NUM_3510));
            auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * MAIN_LOOP_N_SIZE_3510));
            DataCopyWrapper<trait, quantPre>(dst(dstCoord), src(srcCoord),
                tailParam, tuple_sequence<decltype(tailParam)>{});
        }
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline static void DataCopyWrapper(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        if constexpr (GetHardPos<T>() == Hardware::GM) {
            CopyMatrixCcToGm3510::DataCopy<quantPre>(dst, src, Std::get<Is>(tupleParams)...);
        } else {
            CopyMatrixCcToUb3510::DataCopy<quantPre>(dst, src, Std::get<Is>(tupleParams)...);
        }
    }

};

class Fixpipe2OutNZ2NDVectorQuant3510 : public Fixpipe2OutNZ2NDVector3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void Run(const T& dst, const U& src, const V& quant, const Params&... params)
    {
        SetRegisterImpl<trait, T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U, V>(dst, src, quant, params...);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline static constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src, const V& quant, const Params&... params)
    {
        CheckTemplate<trait, T, U>();
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        uint32_t nSize = Std::min(
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout));
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        uint16_t nIterNum = 1;
        uint32_t calNSize = nSize;
        uint32_t tailNSize = 0;
        if (calNSize > MAIN_LOOP_N_SIZE_3510) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE_3510;
            tailNSize = nSize % MAIN_LOOP_N_SIZE_3510;
            calNSize = MAIN_LOOP_N_SIZE_3510;
        }
        FixpipeNZ2NDVectorEntrance<trait, quantPre, T, U, V>(dst, src, quant, nIterNum, calNSize, tailNSize, params...);
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline static void SetRegisterImpl(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNDStride = 0;
        uint32_t dstNDStride = 0;
        SetRegister3510::SetRegister(ndNum, dstNDStride, srcNDStride);
    }
};

}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2OUT_NZ2ND_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
