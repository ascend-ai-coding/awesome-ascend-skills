// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// ----------------------------------------------------------------------------------------------------------

#include <torch/all.h>
#include <torch/python.h>

namespace ops_profiling_warmup {
torch::Tensor compiled_warmup_npu(const torch::Tensor &x, const torch::Tensor &out);
torch::Tensor compiled_warmup_meta(const torch::Tensor &x);
}

PYBIND11_MODULE(_C, module)
{
    module.def("compiled_warmup_npu", &ops_profiling_warmup::compiled_warmup_npu);
    module.def("compiled_warmup_meta", &ops_profiling_warmup::compiled_warmup_meta);
}
