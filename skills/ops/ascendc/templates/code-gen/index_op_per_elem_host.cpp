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
// 索引类算子 op_host 参考代码（逐元素索引模式）
// 适用: gather, scatter, scatter_add 等逐元素索引算子
// 示例: 以 torch.gather(input, dim, index) 为例
//       index 与 output 同形，每个输出位置有独立索引
// 使用: 本文件演示了 3D 维度统一化、逐行索引加载、分批 launch 等
// ============================================================

#include "torch_kernel_helper.h"
#include "aclrtlaunch_<op_name>.h"
#include <torch/extension.h>

namespace ascend_kernel {

constexpr int64_t MAX_CORES = 40;

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
    TORCH_CHECK(index.dim() == ndim,
                "<op_name>: index must have same ndim as input");

    TORCH_CHECK(input.scalar_type() == at::kFloat ||
                input.scalar_type() == at::kHalf ||
                input.scalar_type() == at::kBFloat16,
                "<op_name>: only float32, float16 and bfloat16 are supported");

    // ---- 连续化 ----
    // 搬运类算子不做数值计算，bf16 可直接按 fp16 路径搬运（dtypeSize 均为 2）
    at::Tensor inputContiguous = input.contiguous();
    at::Tensor indexContiguous = index.contiguous();

    // ---- 维度参数 ----
    int64_t N = inputContiguous.size(dim);   // 源维度大小
    int64_t K = indexContiguous.size(dim);   // 索引维度大小（每行索引数量）
    // bf16 与 fp16 同为 2 字节，kernel 统一用 half 路径搬运
    int64_t dtypeSize = inputContiguous.element_size();

    // ---- Index 转 int32 ----
    auto indexInt32 = indexContiguous.to(at::kInt);
    // 注意: Host 端发送元素索引，不做 index * dtypeSize
    // Kernel 端通过 Muls 转换为字节偏移

    // ---- 3D 维度统一化 ----
    // transpose(dim, -1) 将 dim 移到最后一维，再 reshape 为 [batch, rows, N/K]
    //   batch = dims[0] * ... * dims[dim-1]
    //   rows  = dims[dim+1] * ... * dims[ndim-1]
    //   N     = dims[dim] (input), K = index.size(dim) (index)
    int64_t batch = 1;
    for (int64_t i = 0; i < dim; i++) batch *= inputContiguous.size(i);
    int64_t rows = inputContiguous.numel() / (batch * N);

    auto inputFlat  = inputContiguous.transpose(dim, -1).contiguous()
                          .reshape({batch, rows, N});
    // gather: index 与 input 同 ndim，也需 transpose + reshape
    auto indexFlat  = indexInt32.transpose(dim, -1).contiguous()
                          .reshape({batch, rows, K});

    // bf16 用对应 dtype 分配输出，保持 dtypeSize=2
    auto outputFlat = at::empty({batch, rows, K},
        inputContiguous.options().dtype(
            inputContiguous.scalar_type() == at::kBFloat16
                ? at::kBFloat16 : inputContiguous.scalar_type()));

    // ---- 分批 launch ----
    int64_t totalRows = batch * rows;
    int64_t rowOffset = 0;
    uint32_t blockDim = static_cast<uint32_t>(std::min(totalRows, MAX_CORES));
    if (blockDim < 1) blockDim = 1;

    EXEC_KERNEL_CMD(<op_name>, blockDim, inputFlat, indexFlat, outputFlat,
                    totalRows, N, K, dtypeSize, totalRows, rowOffset);

    // ---- 输出 reshape ----
    // [batch, rows, K] → transpose 还原 → index.sizes()
    std::vector<int64_t> transposedOutShape;
    for (int64_t i = 0; i < ndim; i++) {
        if (i == dim) continue;
        transposedOutShape.push_back(indexContiguous.size(i));
    }
    transposedOutShape.push_back(K);

    auto outputContiguous = outputFlat.reshape(transposedOutShape)
                                .transpose(ndim - 1, dim).contiguous();

    return outputContiguous.reshape(index.sizes());
}

}  // namespace ascend_kernel
