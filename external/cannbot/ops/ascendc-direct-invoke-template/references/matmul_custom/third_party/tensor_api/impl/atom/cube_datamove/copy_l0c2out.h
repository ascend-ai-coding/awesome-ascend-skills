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
    "tensor_api/impl/atom/cube_datamove/copy_l0c2out.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file copy_l0c2out.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_L0C2OUT_H
#define IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_L0C2OUT_H


#include "impl/utils/utils_impl.h"
#include "impl/arch/cube_datamove/fixpipe/fixpipe_impl.h"
#include "impl/atom/copy_traits_impl.h"

namespace AscendC {
namespace Te {

struct FixpipeTraitDefault {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = DEFAULT_FIXPIPE_TRAIT;
};

struct CopyL0C2Out {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args& ...args)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Fixpipe<traits>(args...);
    }
};


 struct CopyL0C2OutWith { 
     template <typename Tp, const Tp& traits, typename... Args> 
     __aicore__ inline static void Copy(const Args& ...args) 
     { 
         // custom function Fixpipe<traits, Args...>(args...); 
     } 
 }; 
 
 
 template <typename TraitStruct> 
 struct CopyTraits<CopyL0C2OutWith, TraitStruct> 
 { 
     using TraitType = typename TraitStruct::TraitType; 
     static constexpr const TraitType defaultTrait = TraitStruct::value; 
 
 
     template <const TraitType& trait = defaultTrait, typename... Args> 
     __aicore__ inline void CopyUnpack(const Args& ...args) const { 
       CopyL0C2OutWith::Copy<TraitType, trait, Args...>(args..., params); 
     } 
 
 
     FixpipeParams params; 
 };

template <typename Traits>
struct CopyTraits<CopyL0C2Out, Traits> : public CopyTraits<CopyL0C2Out, Traits, CopyL0C2OutWith, FixpipeTraitDefault> {};

template <>
struct CopyTraits<CopyL0C2Out> : public CopyTraits<CopyL0C2Out, FixpipeTraitDefault> {};

using CopyL0C2GM = CopyL0C2Out;
using CopyL0C2UB = CopyL0C2Out;
using CopyL0C2GMWith = CopyL0C2OutWith;
using CopyL0C2UBWith = CopyL0C2OutWith;

}
}

#endif // IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_L0C2OUT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
