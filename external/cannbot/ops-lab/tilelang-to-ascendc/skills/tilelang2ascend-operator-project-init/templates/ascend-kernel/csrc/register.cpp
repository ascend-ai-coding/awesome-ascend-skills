/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("helloworld(Tensor x, Tensor y) -> Tensor");
    m.def("avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor");

#ifdef BUILD_CATLASS_MODULE
    m.def("catlass_matmul_basic(Tensor tensor_a, Tensor tensor_b, Tensor(a!) tensor_c, str? format_mode=None) -> ()");
#endif

}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("helloworld", TORCH_FN(ascend_kernel::helloworld));
    m.impl("avg_pool3d", TORCH_FN(ascend_kernel::avg_pool3d));

#ifdef BUILD_CATLASS_MODULE
    m.impl("catlass_matmul_basic", TORCH_FN(ascend_kernel::catlass_matmul_basic));
#endif

}
}  // namespace
