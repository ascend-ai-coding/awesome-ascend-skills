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
    "tensor_api/impl/tensor/pointer_mem_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file pointer_mem_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_MEM_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_MEM_IMPL_H

#include "impl/tensor/pointer_adaptor_impl.h"

namespace AscendC {
namespace Te {

template <Hardware hPos, typename Pointer>
struct HardwareMemPtr : IterAdaptor<Pointer, HardwareMemPtr<hPos, Pointer>> {
    using IterAdaptor<Pointer, HardwareMemPtr<hPos, Pointer>>::IterAdaptor;
    static constexpr const Hardware hardPos = hPos;
};

// is hardware mem
template <Hardware hardPos, typename Pointer, typename = void>
struct IsHardwareMem : Std::false_type {};

template <Hardware hardPos, typename Pointer>
struct IsHardwareMem<hardPos, HardwareMemPtr<hardPos, Pointer>> : Std::true_type {};

template <Hardware hardPos, typename Pointer>
struct IsHardwareMem<hardPos, Pointer, void_t<typename Pointer::iterator>> : IsHardwareMem<hardPos, typename Pointer::iterator> {};

template <Hardware hardPos, typename Pointer>
constexpr bool IsHardwareMemV = IsHardwareMem<hardPos, Pointer>::value;

template <Hardware hardPos, typename Iterator>
__aicore__ inline constexpr auto MakeMemPtr(Iterator iter) 
{
    if constexpr (IsHardwareMem<hardPos, Iterator>::value) {
        return iter;
    } else {
        return HardwareMemPtr<hardPos, Iterator>{iter};
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_MEM_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
