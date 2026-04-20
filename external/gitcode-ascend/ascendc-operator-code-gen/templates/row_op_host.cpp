// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================
// 行处理算子 op_host 模板
// 适用: LayerNorm, Softmax, BatchNorm 等按行/维度归约算子
// 使用: 复制到 csrc/ops/<op_name>/op_host/<op_name>.cpp，
//       替换 <op_name>/<OpName> 占位符，修改参数和 tiling 逻辑
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_<op_name>.h"

namespace ascend_kernel {

at::Tensor <op_name>(const at::Tensor &self /* , 其他输入参数 */)
{
    // ---- 输入校验 ----
    TORCH_CHECK(self.dim() > 0, "<op_name>: input must have at least 1 dimension");
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat,
                "<op_name>: only float16 and float32 are supported, got ", self.scalar_type());

    // ---- 计算行维度 ----
    // 将 tensor 视为 [totalRows, dimLength]
    int64_t dimLength = self.size(-1);    // <-- 根据算子语义调整
    int64_t totalRows = self.numel() / dimLength;

    if (totalRows == 0 || dimLength == 0) {
        return at::empty_like(self);
    }

    at::Tensor x = self.contiguous();
    int64_t dtypeSize = x.element_size();

    // ---- 获取硬件参数 ----
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);

    // ---- 对齐 ----
    int64_t alignElements = 32 / dtypeSize;
    int64_t dimLengthAlign = ((dimLength + alignElements - 1) / alignElements) * alignElements;

    // ---- UB 容量检查 ----
    // bufferCoefficient = 每元素在 UB 中总占用字节, 从设计文档 UB 分配表推导
    // 示例: inQ*4 + outQ*4 + tmpBuf*4 = 12 (fp16 with fp32 compute)
    int64_t bufferCoefficient = (dtypeSize == 2) ? 12 : 16; // <-- 根据实际修改
    int64_t maxDimLength = ubSizeLimit / bufferCoefficient;
    maxDimLength = (maxDimLength / alignElements) * alignElements;
    TORCH_CHECK(dimLengthAlign <= maxDimLength,
                "<op_name>: dimLength ", dimLength, " exceeds UB capacity (max ", maxDimLength, ")");

    // ---- Padding ----
    at::Tensor kernelInput;
    if (dimLength != dimLengthAlign) {
        kernelInput = x.reshape({totalRows, dimLength});
        kernelInput = at::constant_pad_nd(kernelInput, {0, dimLengthAlign - dimLength}, 0.0);
        kernelInput = kernelInput.contiguous();
    } else {
        kernelInput = x.reshape({totalRows, dimLengthAlign}).contiguous();
    }
    at::Tensor kernelOutput = at::empty_like(kernelInput);

    // ---- Block 级 tiling (按行分配) ----
    int64_t usedCoreNum = std::min(totalRows, coreNum);
    int64_t formerLength = (totalRows + usedCoreNum - 1) / usedCoreNum;
    int64_t tailLength = formerLength - 1;
    int64_t formerNum = totalRows - tailLength * usedCoreNum;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(<op_name>, blockDim,
                    kernelInput, kernelOutput,
                    totalRows, dimLength, dimLengthAlign,
                    formerNum, formerLength, tailLength
                    /* , 其他算子特有参数 */);

    // ---- 去 padding + reshape ----
    at::Tensor output = kernelOutput;
    if (dimLength != dimLengthAlign) {
        output = output.narrow(-1, 0, dimLength).contiguous();
    }
    output = output.reshape(self.sizes());

    return output;
}

}  // namespace ascend_kernel
