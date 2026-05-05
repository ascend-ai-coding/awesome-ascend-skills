// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================
// 索引类算子 op_host 参考代码（单行更新 / index_select 风格）
// 适用: index_select, index_add 等 1D 共享索引算子
// 示例: 以 torch.index_select(input, dim, index) 为例
//       index 为 1D，所有行共享，无需 transpose
//       3D 布局: [outerSize, srcDimSize, innerSize]
//       动态核分配: coresPerOuter 按索引批次自适应
// ============================================================

#include "torch_kernel_helper.h"
#include "aclrtlaunch_<op_name>.h"
#include <torch/extension.h>

namespace ascend_kernel {

constexpr int64_t MAX_CORES = 40;
constexpr int64_t MAX_INDEX_BATCH = 512;  // indexBuf 每核最大容量

at::Tensor <op_name>(const at::Tensor& input, int64_t dim, const at::Tensor& index) {
    // ---- 输入校验 ----
    TORCH_CHECK(input.device().type() == at::DeviceType::PrivateUse1,
                "<op_name>: input must be on NPU device");
    TORCH_CHECK(index.device().type() == at::DeviceType::PrivateUse1,
                "<op_name>: index must be on NPU device");

    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim;

    TORCH_CHECK(dim >= 0 && dim < ndim,
                "<op_name>: dim out of range, got dim=", dim);
    TORCH_CHECK(index.dim() == 1,
                "<op_name>: index must be 1D");

    TORCH_CHECK(input.scalar_type() == at::kFloat ||
                input.scalar_type() == at::kHalf ||
                input.scalar_type() == at::kBFloat16,
                "<op_name>: only float32, float16 and bfloat16 are supported");

    // ---- 连续化 ----
    at::Tensor inputContiguous = input.contiguous();
    at::Tensor indexContiguous = index.contiguous();

    // ---- 3D 维度参数 ----
    // 无需 transpose，直接按 3D 布局计算偏移:
    //   input:  [outerSize, srcDimSize, innerSize]
    //   output: [outerSize, indexSize,  innerSize]
    int64_t outerSize = 1;
    for (int64_t i = 0; i < dim; i++) outerSize *= inputContiguous.size(i);

    int64_t srcDimSize = inputContiguous.size(dim);

    int64_t innerSize = 1;
    for (int64_t i = dim + 1; i < ndim; i++) innerSize *= inputContiguous.size(i);

    int64_t indexSize = indexContiguous.size(0);
    int64_t totalLength = outerSize * indexSize * innerSize;

    // 搬运类算子不做数值计算，bf16 可直接按 fp16 路径搬运
    int64_t dtypeSize = inputContiguous.element_size();

    // ---- Index 转 int32 ----
    auto indexInt32 = indexContiguous.to(at::kInt);
    // Host 端发送元素索引，Kernel 端直接按 3D 偏移访问

    // ---- 输出张量 ----
    // 输出形状: input.shape[:dim] + [indexSize] + input.shape[dim+1:]
    auto outputShape = inputContiguous.sizes().vec();
    outputShape[dim] = indexSize;
    auto output = at::empty(outputShape, inputContiguous.options());

    if (totalLength == 0) return output;

    // ---- Block 级 Tiling: 动态核分配 ----
    // 每个 outer 按 index 批次分配核数，最大化 indexBuf 利用率
    int64_t indexBatchSize = std::min(indexSize, MAX_INDEX_BATCH);
    int64_t idealCoresPerOuter = (indexSize + indexBatchSize - 1) / indexBatchSize;
    if (idealCoresPerOuter < 1) idealCoresPerOuter = 1;

    int64_t totalTasks = outerSize * idealCoresPerOuter;
    int64_t coresPerOuter;
    uint32_t blockDim;

    if (totalTasks <= MAX_CORES) {
        coresPerOuter = idealCoresPerOuter;
        blockDim = static_cast<uint32_t>(totalTasks);
    } else {
        coresPerOuter = MAX_CORES / outerSize;
        if (coresPerOuter < 1) coresPerOuter = 1;
        blockDim = MAX_CORES;
    }
    if (blockDim < 1) blockDim = 1;

    EXEC_KERNEL_CMD(<op_name>, blockDim, inputContiguous, indexInt32, output,
                    outerSize, innerSize, indexSize, srcDimSize,
                    totalLength, coresPerOuter);

    return output;
}

}  // namespace ascend_kernel
