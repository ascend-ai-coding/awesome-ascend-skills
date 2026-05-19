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
    "tensor_api/impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2out.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file fixpipe_l0c2out.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2OUT_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2OUT_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2out/nz2nz.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2out/nz2nd.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2out/nz2dn.h"

namespace AscendC {
namespace Te {

class FixpipeL0C2Out3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename... Params>
    __aicore__ inline static void Run(const T& dst, const U& src, const Params&... params) {
        Execute<trait>(dst, src, params...);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename... Params>
    __aicore__ inline static void Execute(const T& dst, const U& src, const Params&... params) {
        constexpr auto quantPre = GetFixpipeQuantPre<trait, T, U>();
        if constexpr (IsL0cNZFormat<U>::value && (IsNZFormat<T>::value || IsL0cNZFormat<T>::value)) {
            Fixpipe2OutNz2Nz3510::Run<trait, quantPre, T, U>(dst, src, params...);
        } else if constexpr (IsL0cNZFormat<U>::value && IsNDFormat<T>::value) {
            Fixpipe2OutNz2Nd3510::Run<trait, quantPre, T, U>(dst, src, params...);
        } else if constexpr (IsL0cNZFormat<U>::value && IsDNFormat<T>::value) {
            Fixpipe2OutNz2Dn3510::Run<trait, quantPre, T, U>(dst, src, params...);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2OUT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
