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
    "tensor_api/impl/arch/cube_datamove/load_data/npu_arch_3510/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_INSTRUCTION_H

#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"
#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {
class LoadCbufToCa3510 {
public:
    template <const LoadDataTrait& trait, typename T, typename U, typename... Params>
    __aicore__ inline static void LoadData(const T& dst, const U& src, const Params& ...params)
    {
        LoadCbufToCa<trait.transposed>(dst.Data().Get(), src.Data().Get(), params...);
    }

private:
    template <bool transpose, typename T>
    __aicore__ inline static void LoadCbufToCa(__ca__ T* dst, __cbuf__ T* src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            load_cbuf_to_ca(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
        }
    }
};

class LoadCbufToCaS43510 {
public:
    template <const LoadDataTrait& trait, typename T, typename U, typename... Params>
    __aicore__ inline static void LoadData(const T& dst, const U& src, const Params& ...params)
    {
        LoadCbufToCa<trait.transposed>(dst.Data().Get(), src.Data().Get(), params...);
    }

private:
    template <bool transpose, typename T>
    __aicore__ inline static void LoadCbufToCa(__ca__ T* dst, __cbuf__ T* src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            load_cbuf_to_ca_s4(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
        }
    }
};

class LoadCbufToCb3510 {
public:
    template <const LoadDataTrait& trait, typename T, typename U, typename... Params>
    __aicore__ inline static void LoadData(const T& dst, const U& src, const Params& ...params)
    {
        LoadCbufToCb<trait.transposed>(dst.Data().Get(), src.Data().Get(), params...);
    }

private:
    template <bool transpose, typename T>
    __aicore__ inline static void LoadCbufToCb(__cb__ T* dst, __cbuf__ T* src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
        }
    }
};

class LoadCbufToCbS43510 {
public:
    template <const LoadDataTrait& trait, typename T, typename U, typename... Params>
    __aicore__ inline static void LoadData(const T& dst, const U& src, const Params& ...params)
    {
        LoadCbufToCb<trait.transposed>(dst.Data().Get(), src.Data().Get(), params...);
    }

private:
    template <bool transpose, typename T>
    __aicore__ inline static void LoadCbufToCb(__cb__ T* dst, __cbuf__ T* src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            load_cbuf_to_cb_s4(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_INSTRUCTION_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
