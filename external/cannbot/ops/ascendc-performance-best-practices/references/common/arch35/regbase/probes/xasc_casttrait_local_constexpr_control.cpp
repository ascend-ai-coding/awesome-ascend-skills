/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Diagnostic control for the CastTrait probe. This intentionally omits
// `static`; older C++/CANN contracts reject it as a reference NTTP argument.
#include "kernel_operator.h"

__aicore__ inline void XascCastTraitLocalConstexprControl()
{
    constexpr AscendC::Reg::CastTrait kWiden = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};

    AscendC::Reg::RegTensor<float> outputReg;
    AscendC::Reg::RegTensor<half> inputReg;
    AscendC::Reg::MaskReg mask;
    __VEC_SCOPE__
    {
        AscendC::Reg::Cast<float, half, kWiden>(outputReg, inputReg, mask);
    }
}
