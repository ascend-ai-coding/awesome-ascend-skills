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
// Elementwise 算子 op_host 模板
// 适用: ReLU, GELU, Add, Mul 等逐元素算子
// 使用: 复制到 csrc/ops/<op_name>/op_host/<op_name>.cpp，
//       替换 <op_name>/<OpName>/<dtype> 等占位符
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_<op_name>.h"

namespace ascend_kernel {

constexpr int64_t CACHE_LINE_BYTE_LENGTH = 512;

at::Tensor <op_name>(const at::Tensor &self /* , 其他输入参数 */)
{
    // ---- 输入校验 ----
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat,
                "<op_name>: only float16 and float32 are supported, got ", self.scalar_type());

    at::Tensor output = at::empty_like(self);

    int64_t totalLength = self.numel();
    if (totalLength == 0) {
        return output;
    }

    int64_t dtypeSize = self.element_size();

    // ---- 获取硬件参数 ----
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);

    // ---- Block 级 tiling (核间切分, Cache Line 对齐) ----
    int64_t totalLengthCore = (totalLength + coreNum - 1) / coreNum;
    int64_t totalLengthCoreAlign = (totalLengthCore + CACHE_LINE_BYTE_LENGTH - 1)
                                    / CACHE_LINE_BYTE_LENGTH * CACHE_LINE_BYTE_LENGTH;

    int64_t usedCoreNum = (totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign;
    int64_t formerNum = usedCoreNum - 1;
    int64_t formerLength = totalLengthCoreAlign;
    int64_t tailLength = totalLength - formerNum * formerLength;

    // ---- UB 级 tiling (核内切分) ----
    // bufferCoefficient = 每元素在 UB 中总占用字节, 从设计文档 UB 分配表推导
    // 示例: 单输入单输出 + double buffer
    //   fp16: 2 queues * 2(double buf) * 2B = 8
    //   fp32: 2 queues * 2(double buf) * 4B = 16
    int64_t bufferCoefficient = dtypeSize * 4; // <-- 根据实际 UB 分配表修改
    int64_t maxTileElements = ubSizeLimit / bufferCoefficient;
    int64_t alignElements = 32 / dtypeSize;
    int64_t tileLength = (maxTileElements / alignElements) * alignElements;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    EXEC_KERNEL_CMD(<op_name>, blockDim,
                    self, output,
                    formerNum, formerLength, tailLength, tileLength, dtypeSize);

    return output;
}

}  // namespace ascend_kernel
