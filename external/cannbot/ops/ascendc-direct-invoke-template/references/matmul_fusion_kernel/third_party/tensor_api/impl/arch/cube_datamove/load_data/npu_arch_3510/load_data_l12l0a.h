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
    "tensor_api/impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file load_data_l12l0a.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_H

#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a/nz2nz.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a/zn2nz.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a/zn2nzb8b4.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a/nz2nz_with_coord.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a/zn2nz_with_coord.h"
#include "impl/arch/cube_datamove/load_data/npu_arch_3510/load_data_l12l0a/zn2nzb8b4_with_coord.h"

namespace AscendC {
namespace Te {
class LoadDataL12L0A3510 {
public:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        Execute<trait>(dst, src);
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Execute(const T& dst, const U& src) {
        if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            LoadDataL12L0ANZ2NZ3510::Run<trait, T, U>(dst, src);
        } else if constexpr (IsZNFormat<U>::value && IsNZFormat<T>::value && (sizeof(typename U::elementType) == 1)) {
            LoadDataL12L0AZN2NZB8B43510::Run<trait, T, U>(dst, src);
        } else if constexpr (IsZNFormat<U>::value && IsNZFormat<T>::value) {
            LoadDataL12L0AZN2NZ3510::Run<trait, T, U>(dst, src);
        }
    }
};

class LoadDataL12L0AWithCoord3510 {
public:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        Execute<trait>(dst, src, coord);
    }

private:
    template <const LoadDataTrait& trait, typename T, typename U, class Coord>
    __aicore__ inline void Execute(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            LoadDataL12L0ANZ2NZWithCoord3510::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsNZFormat<T>::value && (sizeof(typename U::elementType) == 1)) {
            LoadDataL12L0AZN2NZB8B4WithCoord3510::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsNZFormat<T>::value) {
            LoadDataL12L0AZN2NZWithCoord3510::Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0A_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
