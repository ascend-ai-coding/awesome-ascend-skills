// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
