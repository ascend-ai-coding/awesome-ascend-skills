/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_mmad_mx.h
 * \brief Tile-level MMAD traits used by the SWAT MX kernels.
 */

#pragma once

#include "include/tensor_api/tensor.h"

namespace AscendC {
namespace Te {

constexpr MmadTrait MX_MMAD_TRAIT = MmadTrait{0, false, false, true, MmadType::MX};
struct MmadTraitMX {
    using TraitType = MmadTrait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};

template <>
struct MmadTraits<MmadOperation, MmadTraitMX>
    : public MmadTraits<MmadOperation, MmadTraitDefault, MmadOpWith, MmadTraitMX> {};

} // namespace Te
} // namespace AscendC