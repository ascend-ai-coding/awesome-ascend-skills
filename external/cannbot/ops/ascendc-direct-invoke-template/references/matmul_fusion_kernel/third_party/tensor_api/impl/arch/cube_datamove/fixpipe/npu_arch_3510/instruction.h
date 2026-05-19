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
    "tensor_api/impl/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_INSTRUCTION_H

#include "impl/arch/utils/arch_utils.h"
#include "impl/arch/cube_datamove/fixpipe/fixpipe_utils.h"

namespace AscendC {
namespace Te {

class CopyMatrixCcToGm3510 {
public:
    template <QuantMode_t quantPre, typename T, typename U, typename... Params>
    __aicore__ inline static void DataCopy(const T& dst, const U& src, const Params& ...params)
    {
        CopyMatrixCcToGm<quantPre>(dst.Data().Get(), src.Data().Get(), params...);
    }

private:
    template <QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static void CopyMatrixCcToGm(__gm__ T *dst, __cc__ U *src, uint32_t nSize, uint32_t mSize,
        uint32_t srcStride, uint32_t dstStride, uint8_t cacheMode, bool reluEn, uint8_t unitFlag, bool isChannelSplit,
        bool nz2ndEn, bool nz2dnEn)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, cacheMode, 0, unitFlag, static_cast<uint64_t>(quantPre),
                reluEn, isChannelSplit, nz2ndEn, static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, false, 0, false, false, false, false, false, 
                nz2dnEn); 
        }
    }
};

class CopyMatrixCcToUb3510 {
public:
template <QuantMode_t quantPre, typename T, typename U, typename... Params>
    __aicore__ inline static void DataCopy(const T& dst, const U& src, const Params& ...params)
    {
        CopyMatrixCcToUb<quantPre>(dst.Data().Get(), src.Data().Get(), params...);
    }

private:

    template <QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline static void CopyMatrixCcToUb(__ubuf__ T *dst, __cc__ U *src, uint32_t nSize, uint32_t mSize,
        uint32_t srcStride, uint32_t dstStride, uint8_t dualDstCtl, bool reluEn, uint8_t unitFlag, bool subBlockId,
        bool nz2ndEn, bool nz2dnEn)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_matrix_cc_to_ub(dst, src, 0, nSize, mSize, dstStride, srcStride, dualDstCtl, subBlockId, 0, unitFlag, static_cast<uint64_t>(quantPre),
                reluEn, false, nz2ndEn, static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, false, 0, false, false, false, false, false, 
                nz2dnEn); 
        }
    }
};


class SetRegister3510 {
public:
    template <typename... Params>
    __aicore__ inline static void SetRegister(const uint64_t& quant, const Params& ...params)
    {
        SetQuantPre(quant);
        SetParamsToRegister<uint64_t>(params...);
    }

    template <typename... Params>
    __aicore__ inline static void SetRegister(const Params& ...params)
    {
        SetParamsToRegister<uint64_t>(params...);
    }

private:
    template <typename T>
    __aicore__ inline static void SetQuantPre(const T& quant)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            set_quant_pre(quant);
        }
    }

    template <typename T>
    __aicore__ inline static void SetParamsToRegister(uint32_t ndNum, uint32_t dstNDStride, uint32_t srcNDStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            T loop3Para = static_cast<T>(dstNDStride) << 32;
            loop3Para |= static_cast<T>(srcNDStride) << 16;
            loop3Para |= static_cast<T>(ndNum);
            set_loop3_para(loop3Para);
        }
    }

    template <typename T>
    __aicore__ inline static void SetParamsToRegister(uint32_t dnNum, uint32_t dstDNStride, uint32_t srcNZMatrixStride, uint32_t srcNZC0Stride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            T loop3Para = static_cast<T>(dstDNStride) << 32;
            loop3Para |= static_cast<T>(srcNZMatrixStride) << 16;
            loop3Para |= static_cast<T>(dnNum);
            set_loop3_para(loop3Para);
            T channelPara = static_cast<T>(srcNZC0Stride) << 48;
            set_channel_para(channelPara);
        }
    }
};

}
}

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_INSTRUCTION_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
