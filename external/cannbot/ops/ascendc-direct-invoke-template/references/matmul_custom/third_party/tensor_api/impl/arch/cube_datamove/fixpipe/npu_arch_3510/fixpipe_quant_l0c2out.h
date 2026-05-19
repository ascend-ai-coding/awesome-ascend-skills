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
    "tensor_api/impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2out.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file fixpipe_quant_l0c2out.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2OUT_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2OUT_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2out/nz2dn.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2out/nz2nd.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2out/nz2nz.h"

namespace AscendC {
namespace Te {

class FormatRegistorIgnore3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename ...Args>
    __aicore__ inline static void Run(const Args&... args) {}
};

template <Format3510 dstFormat, Format3510 srcFormat, QuantMode3510 QuantMode3510>
struct FormatRegistorFixpipe2Out3510 {
    using type = FormatRegistorIgnore3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Direct> {
    using type = Fixpipe2OutNZ2NZSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::ND, Format3510::NZ, QuantMode3510::Direct> {
    using type = Fixpipe2OutNZ2NDSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::DN, Format3510::NZ, QuantMode3510::Direct> {
    using type = Fixpipe2OutNZ2DNSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Scalar> {
    using type = Fixpipe2OutNZ2NZSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::ND, Format3510::NZ, QuantMode3510::Scalar> {
    using type = Fixpipe2OutNZ2NDSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::DN, Format3510::NZ, QuantMode3510::Scalar> {
    using type = Fixpipe2OutNZ2DNSimpleQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::NZ, Format3510::NZ, QuantMode3510::Vector> {
    using type = Fixpipe2OutNZ2NZVectorQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::ND, Format3510::NZ, QuantMode3510::Vector> {
    using type = Fixpipe2OutNZ2NDVectorQuant3510;
};

template <>
struct FormatRegistorFixpipe2Out3510<Format3510::DN, Format3510::NZ, QuantMode3510::Vector> {
    using type = Fixpipe2OutNZ2DNVectorQuant3510;
};

class FixpipeQuantL0C2Out3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void Run(const T& dst, const U& src, const V& quant, const Params&... params) {
        Execute<trait>(dst, src, quant, params...);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename... Params>
    __aicore__ inline static void Execute(const T& dst, const U& src, const V& quant, const Params&... params)
    {
        constexpr auto quantPre = GetFixpipeQuantPre<trait, T, U, V>();
        using FixpipeQuantL0C2Out = typename FormatRegistorFixpipe2Out3510<
            GetDataFormat<T>(), GetDataFormat<U>(), GetQuantMode<quantPre>()>::type;
        FixpipeQuantL0C2Out::template Run<trait, quantPre, T, U, V>(dst, src, quant, params...);
    }
};

}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2OUT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
