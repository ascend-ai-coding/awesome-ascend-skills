/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */



#ifndef OPS_H
#define OPS_H


namespace ascend_kernel {

at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y);

at::Tensor avg_pool3d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override);

#ifdef BUILD_CATLASS_MODULE
void catlass_matmul_basic(const at::Tensor &tensor_a,
                          const at::Tensor &tensor_b, at::Tensor &tensor_c,
                          c10::optional<c10::string_view> format_mode);
#endif

} // namespace ascend_kernel

#endif // OPS_H
