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
    "tensor_api/impl/atom/mad_traits_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file mad_traits_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ATOM_MAD_TRAITS_IMPL_H
#define IMPL_TENSOR_API_ATOM_MAD_TRAITS_IMPL_H

namespace AscendC {
namespace Te {

template <typename MadOperation, typename... MadOpArgs>
struct MmadTraits{};

template <typename MadOp, typename MadTraits, typename MadOpWith, typename MadTraitsWith>
struct MmadTraits<MadOp, MadTraits, MadOpWith, MadTraitsWith>
{
    using TraitType = typename MadTraits::TraitType;
    static constexpr const TraitType defaultTrait = MadTraits::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args& ...args) const {
        MadOp::template Mad<TraitType, trait, Args...>(args...);
    }

    template <typename... Args>
    __aicore__ inline constexpr MmadTraits<MadOpWith, MadTraitsWith>
    with(const Args& ...args) const
    {
        return {args...};
    }
};

}
}

#endif // IMPL_TENSOR_API_ATOM_COPY_TRAITS_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
