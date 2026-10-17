// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// ----------------------------------------------------------------------------------------------------------

#include <ATen/Operators.h>
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/compiled_warmup_launch.h"

namespace ops_profiling_warmup {
namespace {
constexpr int64_t kExpectedElements = 128LL * 1024 * 1024;

void CheckInput(const torch::Tensor &x)
{
    TORCH_CHECK(x.dim() == 3, "compiled warmup expects a 3D tensor");
    TORCH_CHECK(
        x.size(0) == 128 && x.size(1) == 1024 && x.size(2) == 1024,
        "compiled warmup expects shape (128, 1024, 1024)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "compiled warmup expects float16");
    TORCH_CHECK(x.is_contiguous(), "compiled warmup expects a contiguous tensor");
    TORCH_CHECK(x.numel() == kExpectedElements, "compiled warmup tensor size mismatch");
}
}

torch::Tensor compiled_warmup_meta(const torch::Tensor &x)
{
    CheckInput(x);
    return torch::empty({}, x.options());
}

torch::Tensor compiled_warmup_npu(const torch::Tensor &x, const torch::Tensor &out)
{
    CheckInput(x);
    c10_npu::NPUGuard guard(x.device().index());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    int64_t blockDim = 0;
    int64_t blockLength = 0;
    uint32_t tileBytes = 0;
    std::tie(blockDim, blockLength, tileBytes) = CalcCompiledWarmupTiling();
    auto xPtr = reinterpret_cast<GM_ADDR>(x.data_ptr());
    auto outPtr = reinterpret_cast<GM_ADDR>(out.data_ptr());

    auto launch = [=]() -> int {
        LaunchCompiledWarmupHalf(
            xPtr,
            outPtr,
            x.numel(),
            blockLength,
            tileBytes,
            static_cast<uint32_t>(blockDim),
            stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("OpsProfilingWarmup", launch);
    return out;
}
}
