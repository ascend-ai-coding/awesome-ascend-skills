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
    "tensor_api/impl/arch/cube_datamove/fixpipe/fixpipe_routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file fixpipe_routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_ROUTING_H

#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_l0c2out.h"
#include "impl/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_quant_l0c2out.h"

namespace AscendC {
namespace Te {

class FixpipeIgnore {
public:
    template <const FixpipeTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) {}
};

template <Hardware dstPos, Hardware srcpos, Hardware quantpos, uint32_t Version>
struct FixpipeTensor2Tensor {
    using type = FixpipeIgnore;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::MAX, ArchVersion::V3510> {
    using type = FixpipeL0C2Out3510;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::L1, ArchVersion::V3510> {
    using type = FixpipeQuantL0C2Out3510;
};

// L0C to UB routing for V3510
template <>
struct FixpipeTensor2Tensor<Hardware::UB, Hardware::L0C, Hardware::MAX, ArchVersion::V3510> {
    using type = FixpipeL0C2Out3510;
};

template <>
struct FixpipeTensor2Tensor<Hardware::UB, Hardware::L0C, Hardware::L1, ArchVersion::V3510> {
    using type = FixpipeQuantL0C2Out3510;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
