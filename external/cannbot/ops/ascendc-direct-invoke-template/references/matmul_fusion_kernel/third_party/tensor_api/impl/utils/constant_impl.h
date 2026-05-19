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
    "tensor_api/impl/utils/constant_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file constant_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H
#define IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#include <cstdint>
#include <utility>
#include <type_traits>
#include "impl/utils/macro_impl.h"
#include "utils/std/tuple.h"
#include "utils/std/type_traits.h"
#include "utils/std/utility.h"
#include "utils/std/algorithm.h"

namespace AscendC {
namespace Te {
constexpr size_t TWO_DIM_DATA = 2;
constexpr size_t FOUR_DIM_DATA = 4;
constexpr size_t FRACTAL_FIXED = 16;
constexpr size_t DISABLE_COORD = 0;
constexpr size_t ENABLE_COORD = 1;
constexpr size_t SHIFT_LEFT_16 = 0x00010000;
constexpr size_t L2_CACHE_OFFSET = 60;
constexpr size_t MX_SCALE_K0 = 2;
constexpr uint32_t BLOCK_CUBE = 16;

struct ArchVersion {
    static constexpr uint32_t V3510 = 3510;
    static constexpr uint32_t V2201 = 2201;
};

struct GetArchVersion {
    __aicore__ inline constexpr uint32_t operator()() const {
#ifdef __NPU_ARCH__
        return __NPU_ARCH__;
#else
        return 0;
#endif
    }
};

constexpr uint32_t CURRENT_ARCH_VERSION = GetArchVersion{}();

enum class LayoutFormat : uint8_t { NZ, ZN, ZZ, DN, ND, NN};

enum class TupleFormat : uint8_t { Shape, Stride, Coord};

template <typename TupleType>
using tuple_sequence = Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<TupleType>>>;

template<typename T>
__aicore__ inline constexpr auto GetHardPos()
{
   return T::iterator::hardPos;
}

template <typename ElementType, typename DataType>
inline constexpr bool is_one_of_attr_v = Std::is_one_of_v<ElementType, __gm__ DataType, __cbuf__ DataType, __ca__ DataType, 
                                                        __cb__ DataType, __cc__ DataType, __ubuf__ DataType, DataType>;

template <typename DataType>
inline constexpr bool is_b4_type = is_one_of_attr_v<DataType, fp4x2_e1m2_t> || is_one_of_attr_v<DataType, fp4x2_e2m1_t>;

template<typename T = Std::ignore_t>
__aicore__ inline constexpr size_t GetC0Size() {
    constexpr size_t c0Size = 32;
    if constexpr (is_b4_type<T>) {
        return c0Size * 2;
    } else {
        return c0Size;
    }
}

template<typename T = Std::ignore_t>
constexpr size_t C0_SIZE = GetC0Size<T>();

template<typename T>
constexpr size_t C0_ELEMENT = C0_SIZE<T> / sizeof(T);

template <size_t N, typename = Std::make_index_sequence<N>>
struct EmptyGenerator;

template <size_t N, size_t... Idx>
struct EmptyGenerator<N, Std::index_sequence<Idx...>>
{   
    using type = Std::tuple<Std::Int<Idx * 0>...>;
};

template <size_t N>
struct TupleEmptyGenerator
{   
    static_assert((N > 0 && (N & 1) == 0), "N must be greater than 0, and must be even.");
    using type = Std::tuple<typename EmptyGenerator<N / 2>::type, 
        typename EmptyGenerator<N / 2>::type>;
};

template <size_t N>
using EmptyShape = typename TupleEmptyGenerator<N>::type;

template <size_t N>
using EmptyStride = typename TupleEmptyGenerator<N>::type;

template <size_t N>
using EmptyCoord = typename TupleEmptyGenerator<N>::type;

// IsIntegralConstant
template <typename T>
struct IsIntegralConstant : Std::false_type {};

template <size_t Value>
struct IsIntegralConstant<Std::Int<Value>> : Std::true_type {};

template <typename T>
constexpr bool IsIntegralConstantV = IsIntegralConstant<T>::value;

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif