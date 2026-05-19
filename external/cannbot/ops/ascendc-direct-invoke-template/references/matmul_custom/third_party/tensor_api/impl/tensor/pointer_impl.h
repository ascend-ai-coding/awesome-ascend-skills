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
    "tensor_api/impl/tensor/pointer_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file pointer_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_IMPL_H
#define IMPL_TENSOR_API_TENSOR_POINTER_IMPL_H

#include "impl/tensor/pointer_adaptor_impl.h"
#include "impl/tensor/pointer_mem_impl.h"
#include "impl/tensor/engine_impl.h"

namespace AscendC {
namespace Te {

template <typename Iterator>
__aicore__ inline constexpr auto MakeGMmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::GM, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeUBmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::UB, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL1memPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L1, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0AmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L0A, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0BmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L0B, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0CmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::L0C, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeBiasmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::BIAS, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeFixbufmemPtr(Iterator iter) {
    return MakeMemPtr<Hardware::FIXBUF, Iterator>(iter);
}

template <typename T, typename U>
__aicore__ inline auto MakeUBmemPtr(const U& byteOffset) {
    return MakeUBmemPtr(reinterpret_cast<__ubuf__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline auto MakeL1memPtr(const U& byteOffset) {
    return MakeL1memPtr(reinterpret_cast<__cbuf__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline auto MakeL0AmemPtr(const U& byteOffset) {
    return MakeL0AmemPtr(reinterpret_cast<__ca__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline auto MakeL0BmemPtr(const U& byteOffset) {
    return MakeL0BmemPtr(reinterpret_cast<__cb__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline auto MakeL0CmemPtr(const U& byteOffset) {
    return MakeL0CmemPtr(reinterpret_cast<__cc__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline auto MakeBiasmemPtr(const U& byteOffset) {
    return MakeBiasmemPtr(reinterpret_cast<__biasbuf__ T*>(get_imm(0) + byteOffset));
}

template <typename T, typename U>
__aicore__ inline auto MakeFixbufmemPtr(const U& byteOffset) {
    return MakeFixbufmemPtr(reinterpret_cast<__fbuf__ T*>(get_imm(0) + byteOffset));
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
