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
    "tensor_api/impl/arch/cube_datamove/data_copy/data_copy_routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy_routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_DATA_COPY_ROUTING_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_DATA_COPY_ROUTING_H

#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_gm2l1.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_l12bt.h"
#include "impl/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_l12fb.h"

namespace AscendC {
namespace Te {

class DataCopyIgnore {
public:
    template <const DataCopyTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) {}
};

template <Hardware dstTPos, Hardware srcTpos, uint32_t Version>
struct DataCopyTensor2Tensor {
    using type = DataCopyIgnore;
};

template <>
struct DataCopyTensor2Tensor<Hardware::L1, Hardware::GM, ArchVersion::V3510> {
    using type = DataCopyGM2L13510;
};

template <>
struct DataCopyTensor2Tensor<Hardware::BIAS, Hardware::L1, ArchVersion::V3510> {
    using type = DataCopyL12BT3510;
};

template <>
struct DataCopyTensor2Tensor<Hardware::FIXBUF, Hardware::L1, ArchVersion::V3510> {
    using type = DataCopyL12FB3510;
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_DATA_COPY_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
