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
    "tensor_api/impl/arch/utils/is_format.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file is_format.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H
#define IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#include "impl/utils/utils_impl.h"
#include "impl/tensor/pointer_impl.h"
#include "impl/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {

template <typename T>
struct GetTypeFromFourDimTrait;

template <Hardware hPos, typename Pointer, typename Shape1, typename Shape2, typename Stride1, typename Stride2>
struct GetTypeFromFourDimTrait<LocalTensor<TensorAttribute<ViewEngine<HardwareMemPtr<hPos, Pointer>>, Layout<Shape<Shape1, Shape2>, Stride<Stride1, Stride2>>>>> {
    using ShapeRowsZeroDim = typename Std::tuple_element<0, Shape1>::type;
    using ShapeRowsOneDim = typename Std::tuple_element<1, Shape1>::type;
    using ShapeColumnsZeroDim = typename Std::tuple_element<0, Shape2>::type;
    using ShapeColumnsOneDim = typename Std::tuple_element<1, Shape2>::type;

    using StrideRowsZeroDim = typename Std::tuple_element<0, Stride1>::type;
    using StrideRowsOneDim = typename Std::tuple_element<1, Stride1>::type;
    using StrideColumnsZeroDim = typename Std::tuple_element<0, Stride2>::type;
    using StrideColumnsOneDim = typename Std::tuple_element<1, Stride2>::type;
};

enum class AttrInfo : uint8_t {SHAPE, STRIDE, ROW, COLUMN};

template <typename T, AttrInfo info1, AttrInfo info2, size_t dim> 
struct GetFourDimType;

template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeRowsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeRowsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeColumnsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::ShapeColumnsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideRowsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideRowsOneDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideColumnsZeroDim>;
};
template <typename T>
struct GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1> {
    using type = Std::remove_cvref_t<typename GetTypeFromFourDimTrait<T>::StrideColumnsOneDim>;
};

template <typename T>
struct CheckArrangement {
    using type = typename T::elementType;
    using ShapeRow0Type = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
    using ShapeRow1Type = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 1>::type;
    using ShapeColumn0Type = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
    using StrideRow0Type = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
    using StrideRow1Type = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
    using StrideColumn0Type = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
    using StrideColumn1Type = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;

    __aicore__ inline static constexpr bool IsScaleType() {
 	    return is_one_of_attr_v<type, fp8_e8m0_t>;
 	}

    static constexpr ShapeRow0Type ShapeRow0{};
    static constexpr ShapeRow1Type ShapeRow1{};
    static constexpr ShapeColumn0Type ShapeColumn0{};
    static constexpr StrideRow0Type StrideRow0{};
    static constexpr StrideRow1Type StrideRow1{};
    static constexpr StrideColumn0Type StrideColumn0{};
    static constexpr StrideColumn1Type StrideColumn1{};
    static constexpr auto c0Ele = C0_ELEMENT<type>;
};

__aicore__ inline constexpr bool CheckPairs() {
    return true;
}

template <typename T, typename U, typename... Args>
__aicore__ inline constexpr bool CheckPairs(const T& left, const U& right, Args ...args) {
    return (left == right) && CheckPairs(args...);
}

template <typename... Args>
__aicore__ inline constexpr bool CheckEvenPairs(Args... args) {
    static_assert((sizeof...(args) % 2) == 0, "parameters number must be an even number.");
    return CheckPairs(args...);
}

template <typename T>
struct IsZZFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalZZFormatNormal() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<FRACTAL_FIXED>{}, arg.ShapeRow0, 
                                      Std::Int<arg.c0Ele>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<arg.c0Ele>{}, arg.StrideRow0,
                                      Std::Int<1>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalScaleZZFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<FRACTAL_FIXED>{}, arg.ShapeRow0,
                                      Std::Int<MX_SCALE_K0>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<MX_SCALE_K0>{}, arg.StrideRow0,
                                      Std::Int<1>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalZZFormat() {
        using ResultType = Std::conditional_t<arg.IsScaleType(),
                           Std::bool_constant<IsFractalScaleZZFormat()>,
                           Std::bool_constant<IsFractalZZFormatNormal()>>;
        return ResultType::value;
    }
public:
    static constexpr bool value = IsFractalZZFormat();
};

template <typename T>
struct IsNNFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalNNFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<MX_SCALE_K0>{}, arg.ShapeRow0,
               Std::Int<FRACTAL_FIXED>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<1>{}, arg.StrideRow0,
               Std::Int<MX_SCALE_K0>{}, arg.StrideColumn0);

        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNNFormat();
};

template <typename T>
struct IsZNFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalZNFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<arg.c0Ele>{}, arg.ShapeRow0,
               Std::Int<FRACTAL_FIXED>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<1>{}, arg.StrideRow0,
               Std::Int<arg.c0Ele>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalZNFormat();
};

template <typename T>
struct IsNZFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalNZFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<FRACTAL_FIXED>{}, arg.ShapeRow0,
               Std::Int<arg.c0Ele>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<arg.c0Ele>{}, arg.StrideRow0,
               Std::Int<1>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalNZFormat();
};

template <typename T>
struct IsL0cNZFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalL0cNZFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<FRACTAL_FIXED>{}, arg.ShapeRow0,
               Std::Int<FRACTAL_FIXED>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<FRACTAL_FIXED>{}, arg.StrideRow0,
               Std::Int<1>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }
public:
    static constexpr bool value = IsFractalL0cNZFormat();
};

template <typename T>
struct IsNDFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalNDFormatNormal() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<1>{}, arg.ShapeRow0,
               Std::Int<1>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<0>{}, arg.StrideRow0,
               Std::Int<0>{}, arg.StrideColumn0, Std::Int<1>{}, arg.StrideColumn1);
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalScaleNDFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<2>{}, arg.ShapeRow0,
               Std::Int<1>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<1>{}, arg.StrideRow0,
               Std::Int<0>{}, arg.StrideColumn0, Std::Int<2>{}, arg.StrideColumn1);
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalNDFormatOneDim() {
        return CheckPairs(Std::Int<1>{}, arg.ShapeRow1);
    }

    __aicore__ inline static constexpr bool IsFractalNDFormat() {
        using ResultType = Std::conditional_t<arg.IsScaleType(),
                           Std::bool_constant<IsFractalScaleNDFormat()>,
                           Std::bool_constant<IsFractalNDFormatNormal()>>;
        return ResultType::value;
    }
public:
    static constexpr bool value = IsFractalNDFormat();
    static constexpr bool normalValue = IsFractalNDFormatNormal();
    static constexpr bool oneDimValue = IsFractalNDFormatOneDim();
};

template <typename T>
struct IsDNFormat {
private:
    static constexpr CheckArrangement<T> arg{};

    __aicore__ inline static constexpr bool IsFractalDNFormatNormal() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<1>{}, arg.ShapeRow0,
               Std::Int<1>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<0>{}, arg.StrideRow0,
               Std::Int<1>{}, arg.StrideRow1, Std::Int<0>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalScaleDNFormat() {
        constexpr bool isShapeRight = CheckEvenPairs(Std::Int<1>{}, arg.ShapeRow0,
               Std::Int<2>{}, arg.ShapeColumn0);
        constexpr bool isStrideRight = CheckEvenPairs(Std::Int<0>{}, arg.StrideRow0,
               Std::Int<2>{}, arg.StrideRow1, Std::Int<1>{}, arg.StrideColumn0);
        return (isShapeRight && isStrideRight);
    }

    __aicore__ inline static constexpr bool IsFractalDNFormat() {
        using ResultType = Std::conditional_t<arg.IsScaleType(),
                           Std::bool_constant<IsFractalScaleDNFormat()>,
                           Std::bool_constant<IsFractalDNFormatNormal()>>;
        return ResultType::value;
    }
public:
    static constexpr bool value = IsFractalDNFormat();
    static constexpr bool normalValue = IsFractalDNFormatNormal();
};

template <typename T>
struct IsScaleANDFormat { // shape = ((1, row),(1,col)) stride = ((0, col),(0, 1))
    static constexpr bool value = IsNDFormat<T>::normalValue;
};

template <typename T>
struct IsScaleADNFormat { // shape = ((1, row),(2,col/2)) stride = ((0, 2),(1, row*2))
    static constexpr bool value = IsDNFormat<T>::value;
};

template <typename T>
struct IsScaleBNDFormat { // shape = ((2, row/2),(1,col)) stride = ((1, 2*col),(0, 2))
    static constexpr bool value = IsNDFormat<T>::value;
};

template <typename T>
struct IsScaleBDNFormat { // shape = ((1, row),(1,col)) stride = ((0, 1),(0, row))
    static constexpr bool value = IsDNFormat<T>::normalValue;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_UTILS_IS_FORMAT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
