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
    "tensor_api/impl/arch/utils/arch_utils.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file arch_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_ARCH_UTILS_H
#define IMPL_TENSOR_API_ARCH_UTILS_ARCH_UTILS_H

#include "impl/arch/utils/check_data_type_3510.h"
#include "impl/arch/utils/check_format.h"
#include "impl/arch/utils/is_format.h"

namespace AscendC {
namespace Te {

template<const LoadDataTrait& trait, bool transpose> 
constexpr LoadDataTrait TransTrait = LoadDataTrait(trait, transpose); 

template <typename T>
__aicore__ inline uint8_t GetCacheModeFromTensor(const T& tensor) {
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        return static_cast<uint8_t>((reinterpret_cast<uint64_t>(tensor.Data().Get())) >> L2_CACHE_OFFSET);
    } else {
        return 0;
    }
}

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define SCALAR_QUANT_MODE QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8,\
    QuantMode_t::QS322BF16_PRE, QuantMode_t::QF322F16_PRE, QuantMode_t::QF322BF16_PRE, QuantMode_t::QF322FP8_PRE,\
    QuantMode_t::QF322HIF8_PRE, QuantMode_t::QF322HIF8_PRE_HYBRID, QuantMode_t::QF322F32_PRE
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define SCALAR_QUANT_MODE QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8
#else
#define SCALAR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsScalarQuantMode = Std::is_one_of_value<QuantMode_t, quantPre, SCALAR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8,\
    QuantMode_t::VQS322BF16_PRE, QuantMode_t::VQF322F16_PRE, QuantMode_t::VQF322BF16_PRE, QuantMode_t::VQF322FP8_PRE,\
    QuantMode_t::VQF322HIF8_PRE, QuantMode_t::VQF322HIF8_PRE_HYBRID, QuantMode_t::VQF322F32_PRE
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8
#else
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsVectorQuantMode = Std::is_one_of_value<QuantMode_t, quantPre, TILE_OP_INTERNAL_TENSOR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#else
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsDirectQuantMode = Std::is_one_of_value<QuantMode_t, quantPre, TILE_OP_INTERNAL_DIRECT_QUANT_MODE>;

template <typename T, AttrInfo info1, AttrInfo info2, size_t dim>
__aicore__ inline constexpr decltype(auto) GetEleFromLayout(const T& layout) {
    if constexpr (info1 == AttrInfo::SHAPE && info2 == AttrInfo::ROW) {
        return Std::get<dim>(Std::get<0>(layout.Shape()));
    } else if constexpr (info1 == AttrInfo::SHAPE && info2 == AttrInfo::COLUMN) {
        return Std::get<dim>(Std::get<1>(layout.Shape()));
    } else if constexpr (info1 == AttrInfo::STRIDE && info2 == AttrInfo::ROW) {
        return Std::get<dim>(Std::get<0>(layout.Stride()));
    } else if constexpr (info1 == AttrInfo::STRIDE && info2 == AttrInfo::COLUMN) {
        return Std::get<dim>(Std::get<1>(layout.Stride()));
    }        
}


} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_UTILS_ARCH_UTILS_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
