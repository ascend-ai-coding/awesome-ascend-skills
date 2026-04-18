// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/library.h>

#include "torch_aclnn_helper.h"

using namespace at;

namespace ascend_kernel {

int64_t CeilDiv(int64_t value, int64_t factor)
{
    int64_t value_num = 0;
    if (factor == 0) {
        return value_num;
    }
    if (value % factor == 0) {
        value_num = value / factor;
    } else {
        value_num = value / factor + 1;
    }

    return value_num;
}

c10::SmallVector<int64_t, 3> avg_pool3d_npu_output_size(const at::Tensor &self, c10::IntArrayRef kernel_size,
                                                           c10::IntArrayRef stride, c10::IntArrayRef padding,
                                                           bool ceil_mode)
{
    TORCH_CHECK(self.dim() == 4 || self.dim() == 5, "tensor self's dimension must be 4 or 5");
    TORCH_CHECK(kernel_size.size() == 3, "kernel_size length should be 3");
    TORCH_CHECK(stride.size() == 3, "stride length should be 3");
    TORCH_CHECK(stride[0] * stride[1] * stride[2] != 0, "stride should not contain zero");
    TORCH_CHECK(padding.size() == 3, "padding length should be 3");

    int self_d = self.size(-3);
    int self_h = self.size(-2);
    int self_w = self.size(-1);

    int64_t kernel_d = ceil_mode ? (CeilDiv(self_d + 2 * padding[0] - kernel_size[0], stride[0]) + 1) :
                                   ((self_d + 2 * padding[0] - kernel_size[0]) / stride[0] + 1);
    int64_t kernel_h = ceil_mode ? (CeilDiv(self_h + 2 * padding[1] - kernel_size[1], stride[1]) + 1) :
                                   ((self_h + 2 * padding[1] - kernel_size[1]) / stride[1] + 1);
    int64_t kernel_w = ceil_mode ? (CeilDiv(self_w + 2 * padding[2] - kernel_size[2], stride[2]) + 1) :
                                   ((self_w + 2 * padding[2] - kernel_size[2]) / stride[2] + 1);
    TORCH_CHECK(kernel_d > 0, "kernel_d has to be positive, but got ", kernel_d);
    TORCH_CHECK(kernel_h > 0, "kernel_h has to be positive, but got ", kernel_h);
    TORCH_CHECK(kernel_w > 0, "kernel_w has to be positive, but got ", kernel_w);

    if (ceil_mode) {
        if ((kernel_d - 1) * stride[0] >= self_d + padding[0]) {
            --kernel_d;
        }

        if ((kernel_h - 1) * stride[1] >= self_h + padding[1]) {
            --kernel_h;
        }

        if ((kernel_w - 1) * stride[2] >= self_w + padding[2]) {
            --kernel_w;
        }
    }

    c10::SmallVector<int64_t, 3> output_size;
    if (self.dim() == 4) {
        output_size = {self.size(0), kernel_d, kernel_h, kernel_w};
    } else {
        output_size = {self.size(0), self.size(1), kernel_d, kernel_h, kernel_w};
    }

    return output_size;
}

c10::SmallVector<int64_t, 3> calc_avg_pool3d_output_size(const at::Tensor &self, at::IntArrayRef kernel_size,
                                         at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
                                         bool count_include_pad, c10::optional<int64_t> divisor_override)
{
    // generalize kernels, strides and paddings to 3D-shape
    TORCH_CHECK(
        !kernel_size.empty(),
        "kernel_size must either be a single int, or a tuple of three ints");
    const int64_t k_d = kernel_size[0];
    const int64_t k_h = kernel_size.size() == 1 ? k_d : kernel_size[1];
    const int64_t k_w = kernel_size.size() == 1 ? k_d : kernel_size[2];
    c10::SmallVector<int64_t, 3> kernel_sizes = {k_d, k_h, k_w};
    at::IntArrayRef kernels = at::IntArrayRef(kernel_sizes);

    const int64_t s_d = stride.empty() ? k_d : stride[0];
    const int64_t s_h = stride.empty() ? k_h : stride.size() == 1 ? s_d : stride[1];
    const int64_t s_w = stride.empty() ? k_w : stride.size() == 1 ? s_d : stride[2];
    c10::SmallVector<int64_t, 3> stride_sizes = {s_d, s_h, s_w};
    TORCH_CHECK(s_d != 0 && s_h != 0 && s_w != 0, "stride should not be zero");
    at::IntArrayRef strides = at::IntArrayRef(stride_sizes);

    const int64_t pad_d = padding[0];
    const int64_t pad_h = padding.size() == 1 ? pad_d : padding[1];
    const int64_t pad_w = padding.size() == 1 ? pad_d : padding[2];
    c10::SmallVector<int64_t, 3> padding_sizes = {pad_d, pad_h, pad_w};
    TORCH_CHECK(pad_d >= 0 &&  pad_h >= 0 && pad_w >= 0, "pad should not be less than 0");
    TORCH_CHECK(
        pad_d <= k_d / 2 && pad_h <= k_h / 2 && pad_w <= k_w / 2,
        "pad should be smaller than or equal to half of kernel size");
    at::IntArrayRef paddings = at::IntArrayRef(padding_sizes);

    auto output_size = avg_pool3d_npu_output_size(self, kernels, strides, paddings, ceil_mode);

    return output_size;
}

at::Tensor avg_pool3d(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override)
{
    c10::SmallVector<int64_t, 3> output_size = calc_avg_pool3d_output_size(
        self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    at::Tensor result = at_npu::native::empty_with_format(output_size, self.options(), at_npu::native::get_npu_format(self));

    EXEC_NPU_CMD(aclnnAvgPool3d, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, result);

    return result;
}

}  // namespace ascend_kernel