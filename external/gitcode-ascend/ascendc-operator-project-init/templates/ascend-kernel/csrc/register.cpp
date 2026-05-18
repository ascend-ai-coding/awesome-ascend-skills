// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
