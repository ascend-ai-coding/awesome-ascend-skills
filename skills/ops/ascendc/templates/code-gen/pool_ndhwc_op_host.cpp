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
// Pooling 算子 op_host 模板
// 适用: Pooling 等算子
// 使用: 复制到 csrc/ops/<op_name>/op_host/<op_name>.cpp，
//       替换 <op_name>/<OpName>/<dtype> 等占位符
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_<op_name>_fp32.h"
#include "aclrtlaunch_<op_name>_fp16.h"
#include "aclrtlaunch_<op_name>_bf16.h"


static inline int64_t AvgPool3DOutputShape(
    const int64_t inputSize, const int64_t kernelSize, const int64_t padL, const int64_t stride, const bool ceilMode) {
    int64_t outputSize = (stride == 0) ? -1 :
                         (inputSize + padL * 2 - kernelSize + (ceilMode ? stride - 1 : 0)) /stride + 1;

    if (ceilMode) {
        if ((outputSize - 1) * stride >= inputSize + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

static inline int64_t divRtn(const int64_t x, const int64_t y) {
  int64_t q = x / y;
  int64_t r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  };
  return q;
}

static inline int64_t MaxPool3DOutputShape(const int64_t inputSize, const int64_t kernelSize, const int64_t padL,
                                            const int64_t padR, const int64_t stride, const int64_t dilation,
                                            const bool ceilMode) {
  int64_t outputSize =
      divRtn(inputSize + padL + padR - dilation * (kernelSize - 1) - 1 + (ceilMode ? stride - 1 : 0), stride) + 1;

  if (ceilMode) {
    if ((outputSize - 1) * stride >= inputSize + padL) {
      --outputSize;
    }
  }
  return outputSize;
}


namespace ascend_kernel {
// 注意: return_indices=false时返回 std::vector<at::Tensor> 包含1个Tensor(仅output)
//       return_indices=true 时返回 std::vector<at::Tensor> 包含2个Tensor(output + indices)
std::vector<at::Tensor> pooling_op(const at::Tensor& self,
                        at::IntArrayRef kernel_size,
                        at::IntArrayRef stride,
                        at::IntArrayRef padding,
                        at::IntArrayRef dilation,    // MaxPool3d 需要 dilation，当前方案只考虑dilation=1场景
                        bool ceil_mode,
                        bool return_indices) {       // 根据具体算子增减参数
    
    // ---- 获取硬件参数 ----核数量，ub空间大小参数
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = static_cast<int64_t>(ascendc_platform->GetCoreNumAiv());
    uint64_t ubSize;
    ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubSizeLimit = static_cast<int64_t>(ubSize);
    
    // 参数解析
    int64_t N = self.size(0);
    int64_t C = self.size(1);
    int64_t inputD = self.size(2);
    int64_t inputH = self.size(3);
    int64_t inputW = self.size(4);
    int64_t kernelD = kernel_size[0];
    int64_t kernelH = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[1];
    int64_t kernelW = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[2];
    // ... stride, padding, dilation, ceil_mode, count_include_pad, divisor_override 解析
    // 参考2.3节按实际算子签名补全

    // 参数校验
    TORCH_CHECK(self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat || self.scalar_type() == at::kBFloat16,
                "<op_name>: only float16, float32 and bfloat16 are supported, got ", self.scalar_type()); 
    //数据类型、长度、取值范围校验...

    // 计算输出形状, ceilmode=1需要特殊处理
    // MaxPool3d: 调用 MaxPool3DOutputShape(inputSize, kernelSize, padL, padR, stride, dilation, ceilMode)
    // AvgPool3d: 调用 AvgPool3DOutputShape(inputSize, kernelSize, padL, stride, ceilMode)

    // 准备输入数据，NCDHWC → NDHWC格式
    at::Tensor xNDHWC = self.permute({0, 2, 3, 4, 1}).contiguous();
    at::Tensor outputNDHWC = at::empty({N, outputD, outputH, outputW, C}, 
                                 self.options().dtype(self.scalar_type()));

    // 计算tiling切分参数-核间处理，每个核处理数据量
    int64_t formerNum, tailNum, formerLength, tailLength, usedCoreNum; //...
    int64_t outputNum = N * outputD * outputH * outputW;
    if (outputNum < coreNum) {
        formerNum = outputNum;
        tailNum = 0UL;
        formerLength = 1UL;
        tailLength = 0UL;
        usedCoreNum = outputNum;
    } else if (outputNum % coreNum == 0UL) {
        formerNum = coreNum;
        tailNum = 0UL;
        formerLength = outputNum / coreNum;
        tailLength = 0UL;
        usedCoreNum = coreNum;
    } else {
        formerNum = outputNum % coreNum;
        tailNum = coreNum - formerNum;
        formerLength = outputNum / coreNum + 1UL;
        tailLength = outputNum / coreNum;
        usedCoreNum = coreNum;
    }

    // 计算tiling切分参数-核内处理，UB空间单次处理的数据量，UB空间使用最好预留1k
    int64_t elemSize = static_cast<int64_t>(self.element_size());
    int64_t alignNum = 32 / elemSize;
    int64_t alignC = ((C + alignNum - 1) / alignNum) * alignNum;
    //...其他参数

    // windowWNum: UB空间单次能处理的窗口数量
    // 各UB缓冲区: alignC 以32B block对齐, 相邻block地址步长按 (元素数/8) 个32B block计算
    // +4 为预留的32B block步长空间（一个block步长占4字节描述符）
    int64_t rowExtra = (kW - 1) * alignC * (elemSize + 4);
    int64_t perWindowBase = alignC * (sW * (elemSize + 4) + 4);

    // 无索引版本: 仅dataLocal + castLocal + maxBufLocal/sumBufLocal
    int64_t windowWNum = (ubSizeLimit - 1024 - rowExtra) / perWindowBase;
    if (windowWNum < 1) windowWNum = 1;
    if (windowWNum > oW) windowWNum = oW;

    // 有索引版本: 额外 indicesLocal + indicesUpdLocal + maskBufLocal (每窗口8*alignC字节 + ceil(C/64)*32字节掩码)
    int64_t perWindowIndices = perWindowBase + 8 * alignC + ((C + 63) / 64 * 32);
    int64_t windowWNumIndices = (ubSizeLimit - 1024 - rowExtra) / perWindowIndices;
    if (windowWNumIndices < 1) windowWNumIndices = 1;
    if (windowWNumIndices > oW) windowWNumIndices = oW;

    // 启动kernel (blockDim = usedCoreNum)
    // 多数据类型: 需要分别include aclrtlaunch_<op_name>.h / _fp16.h / _bf16.h
    // 有索引版本还需要 aclrtlaunch_<op_name>_indices.h / _fp16_indices.h / _bf16_indices.h
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    void* xNDHWC_ptr = const_cast<void*>(xNDHWC.data_ptr());
    void* output_ptr = const_cast<void*>(outputNDHWC.data_ptr());
    void* indices_ptr = nullptr;
    at::Tensor indicesNDHWC;

    // 无索引/有索引使用不同LAUNCH宏(windowWNum vs windowWNumIndices)，参数需要传递左值
    #define LAUNCH_NO_IDX(KERNEL) \
        EXEC_KERNEL_CMD(KERNEL, blockDim, xNDHWC_ptr, output_ptr, indices_ptr, \
            N, C, ..., formerNum, formerLength, tailLength, windowWNum)
    #define LAUNCH_IDX(KERNEL) \
        EXEC_KERNEL_CMD(KERNEL, blockDim, xNDHWC_ptr, output_ptr, indices_ptr, \
            N, C, ..., formerNum, formerLength, tailLength, windowWNumIndices)

    if (return_indices) {
        indicesNDHWC = at::empty({N, oD, oH, oW, C}, self.options().dtype(at::kInt));
        indices_ptr = const_cast<void*>(indicesNDHWC.data_ptr());
    } else {
        indices_ptr = output_ptr; // dummy, 避免空指针
    }

    if (return_indices) {
        if (self.scalar_type() == at::kFloat) { LAUNCH_IDX(<op_name>_indices); }
        else if (self.scalar_type() == at::kHalf) { LAUNCH_IDX(<op_name>_fp16_indices); }
        else { LAUNCH_IDX(<op_name>_bf16_indices); }
    } else {
        if (self.scalar_type() == at::kFloat) { LAUNCH_NO_IDX(<op_name>); }
        else if (self.scalar_type() == at::kHalf) { LAUNCH_NO_IDX(<op_name>_fp16); }
        else { LAUNCH_NO_IDX(<op_name>_bf16); }
    }
    #undef LAUNCH_NO_IDX
    #undef LAUNCH_IDX

    // NDHWC → NCDHW格式
    at::Tensor output = outputNDHWC.permute({0, 4, 1, 2, 3}).contiguous();

    if (return_indices) {
        at::Tensor indices = indicesNDHWC.permute({0, 4, 1, 2, 3}).contiguous();
        return {output, indices};
    }
    return {output};
}

}  // namespace ascend_kernel