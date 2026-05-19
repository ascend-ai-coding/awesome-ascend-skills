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
    "tensor_api/impl/arch/cube_datamove/fixpipe/fixpipe_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file fixpipe_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_UTILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_UTILS_H

#include "impl/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te{

constexpr uint32_t MAIN_LOOP_N_SIZE_3510 = 512;
constexpr uint32_t CBURST_NUM_3510 = MAIN_LOOP_N_SIZE_3510 / BLOCK_CUBE;

template <typename T>
__aicore__ inline auto AllocTempBuf(const T& calNSize)
{
    uint64_t deqTensorTempBuf = 0;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        deqTensorTempBuf = reinterpret_cast<uint64_t>(get_imm(0));
    }
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void SetFpc(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        uint64_t deqTensorAddr = (reinterpret_cast<uint64_t>(deqTensorTempBuf) >> static_cast<uint64_t>(7)) << 8;
        set_fpc(deqTensorAddr);
    }
}

__aicore__ inline void InsertSync()
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 || 
                    CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        pipe_barrier(PIPE_FIX);
    }
}

class CopyDeqTensorToFbuf3510 {
public:
    template <typename T>
    __aicore__ inline static void CopyDeqTensorToFbufImpl(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        auto dstAddr = reinterpret_cast<__fbuf__ uint64_t*>(AllocTempBuf(calNSize));
        auto dst = MakeTensor(MakeFixbufmemPtr(dstAddr), src.Layout());
        auto tileSrc = TileSrcTensor(src, calNSize, nIterIndex);
        DataCopyL12FB3510::Run<DEFAULT_DATA_COPY_TRAIT>(dst, tileSrc);
        SetFpc(dstAddr);
    }
private:
    template <typename T>
    __aicore__ inline static decltype(auto) TileSrcTensor(const T& src, uint16_t calNSize, uint16_t nIterIndex) {
        auto coord = MakeCoord(MakeCoord(Std::Int<0>{}, Std::Int<0>{}), MakeCoord(Std::Int<0>{}, nIterIndex * MAIN_LOOP_N_SIZE_3510));
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, Std::Int<1>{}), MakeShape(Std::Int<1>{}, calNSize));
        return src(coord, shape);
    }
};

template <const FixpipeTrait& trait, typename T, typename U, typename S = void>
__aicore__ inline constexpr QuantMode_t GetFixpipeQuantPre()
{
    using srcType = typename U::elementType;
    using dstType = typename T::elementType;
    constexpr bool isTensor = IsTileTensorV<S>;
    constexpr bool isScalar = Std::is_same_v<S, uint64_t>;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    if constexpr (trait.roundMode == RoundMode::HYBRID) {
        static_assert(
            (Std::is_same_v<srcType, __cc__ float> && Std::is_one_of_v<dstType, __gm__ hifloat8_t, __ubuf__ hifloat8_t>),
            "Only when L0CType is float and output Type is hifloat8_t support RoundMode::HYBRID in Fixpipe");
    }
    if constexpr (isTensor) {
        if constexpr (Std::is_same_v<srcType, __cc__ int32_t> && Std::is_one_of_v<dstType, __gm__ half, __ubuf__ half>) {
            return QuantMode_t::VDEQF16;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t, __ubuf__ uint8_t, __ubuf__ int8_t>) {
            return QuantMode_t::VQF322B8_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ int32_t> &&
            Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t, __ubuf__ uint8_t, __ubuf__ int8_t>) {
            return QuantMode_t::VREQ8;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ fp8_e4m3fn_t, __ubuf__ fp8_e4m3fn_t>) {
            return QuantMode_t::VQF322FP8_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ hifloat8_t, __ubuf__ hifloat8_t>) {
            if constexpr (trait.roundMode == RoundMode::HYBRID) {
                return QuantMode_t::VQF322HIF8_PRE_HYBRID;
            } else {
                return QuantMode_t::VQF322HIF8_PRE;
            }
        } else if constexpr (Std::is_same_v<srcType, __cc__ int32_t> &&
            Std::is_one_of_v<dstType, __gm__ bfloat16_t, __ubuf__ bfloat16_t>) {
            return QuantMode_t::VQS322BF16_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ half, __ubuf__ half>) {
            return QuantMode_t::VQF322F16_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ bfloat16_t, __ubuf__ bfloat16_t>) {
            return QuantMode_t::VQF322BF16_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ float, __ubuf__ float>) {
            return QuantMode_t::VQF322F32_PRE;
        }
    } else if constexpr (isScalar) {
        if constexpr (Std::is_same_v<srcType, __cc__ int32_t> &&
            Std::is_one_of_v<dstType, __gm__ half, __ubuf__ half>) {
            return QuantMode_t::DEQF16;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t, __ubuf__ uint8_t, __ubuf__ int8_t>) {
            return QuantMode_t::QF322B8_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ int32_t> &&
            Std::is_one_of_v<dstType, __gm__ uint8_t, __gm__ int8_t, __ubuf__ uint8_t, __ubuf__ int8_t>) {
            return QuantMode_t::REQ8;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ fp8_e4m3fn_t, __ubuf__ fp8_e4m3fn_t>) {
            return QuantMode_t::QF322FP8_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ hifloat8_t, __ubuf__ hifloat8_t>) {
            if constexpr (trait.roundMode == RoundMode::HYBRID) {
                return QuantMode_t::QF322HIF8_PRE_HYBRID;
            } else {
                return QuantMode_t::QF322HIF8_PRE;
            }
        } else if constexpr (Std::is_same_v<srcType, __cc__ int32_t> &&
            Std::is_one_of_v<dstType, __gm__ bfloat16_t, __ubuf__ bfloat16_t>) {
            return QuantMode_t::QS322BF16_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ half, __ubuf__ half>) {
            return QuantMode_t::QF322F16_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float>
            && Std::is_one_of_v<dstType, __gm__ bfloat16_t, __ubuf__ bfloat16_t>) {
            return QuantMode_t::QF322BF16_PRE;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ float, __ubuf__ float>) {
            return QuantMode_t::QF322F32_PRE;
        }
    } else {
        if constexpr (Std::is_same_v<srcType, __cc__ float> && Std::is_one_of_v<dstType, __gm__ half, __ubuf__ half>) {
            return QuantMode_t::F322F16;
        } else if constexpr (Std::is_same_v<srcType, __cc__ float> &&
            Std::is_one_of_v<dstType, __gm__ bfloat16_t, __ubuf__ bfloat16_t>) {
            return QuantMode_t::F322BF16;
        } else {
            return QuantMode_t::NoQuant;
        }
    }
    return QuantMode_t::NoQuant;
#else
    return QuantMode_t::NoQuant;
#endif
}

enum class Format3510 : uint8_t { None, NZ, ND, DN };
enum class QuantMode3510 : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format3510 GetDataFormat()
{
    if constexpr (IsL0cNZFormat<T>::value || IsNZFormat<T>::value) {
        return Format3510::NZ;
    } else if constexpr (IsNDFormat<T>::value) {
        return Format3510::ND;
    } else if constexpr (IsDNFormat<T>::value) {
        return Format3510::DN;
    }
    return Format3510::None;
}

template <const QuantMode_t quantPre>
__aicore__ inline constexpr QuantMode3510 GetQuantMode()
{
    if constexpr (IsVectorQuantMode<quantPre>()) {
        return QuantMode3510::Vector;
    } else if constexpr (IsScalarQuantMode<quantPre>()) {
        return QuantMode3510::Scalar;
    } else if constexpr (IsDirectQuantMode<quantPre>()) {
        return QuantMode3510::Direct;
    }
    return QuantMode3510::None;
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
