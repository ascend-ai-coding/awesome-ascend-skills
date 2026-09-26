/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file torch_adapter.cpp
 * @brief PyTorch 适配层 - ACLNN 两段式调用 + PyTorch 算子注册
 *
 * 两段式调用架构：
 *   - 主线程：分配所有内存（输出 tensor + workspace tensor）+ GetWorkspace
 *   - lambda：执行算子，通过 OpCommand 入 queue
 *
 * 内存管理：workspace 内存由 PyTorch tensor 管理（torch::empty），不使用 aclrtMalloc
 *
 * 编译：通过 CMake 构建（见 CMakeLists.txt）
 * 使用：
 *   import torch
 *   torch.ops.load_library("libtorch_adapter.so")
 *   x = torch.randn(2, 3, device="npu")
 *   y = torch.randn(2, 3, device="npu")
 *   z = torch.ops.add_example.forward(x, y)
 */

#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "aclnn_add_example.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#ifndef ACLNN_STATUS_DEFINED
typedef int aclnnStatus;
#define ACLNN_STATUS_DEFINED
#endif

namespace {

// ============================================================================
// PyTorch c10::ScalarType -> CANN aclDataType 映射
// Ref: c10/core/ScalarType.h, acl/acl_base_rt.h
// ============================================================================

aclDataType ScalarTypeToAclDType(c10::ScalarType st) {
    switch (st) {
        case c10::ScalarType::Byte:    return ACL_UINT8;
        case c10::ScalarType::Char:    return ACL_INT8;
        case c10::ScalarType::Int:     return ACL_INT32;
        case c10::ScalarType::Long:    return ACL_INT64;
        case c10::ScalarType::Half:    return ACL_FLOAT16;
        case c10::ScalarType::Float:   return ACL_FLOAT;
        case c10::ScalarType::Double:  return ACL_DOUBLE;
        case c10::ScalarType::Bool:    return ACL_BOOL;
        case c10::ScalarType::BFloat16:return ACL_BF16;
        default: return ACL_DT_UNDEFINED;
    }
}

// ============================================================================
// 辅助函数：从裸指针创建 aclTensor
// ============================================================================

static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

aclTensor* CreateAclTensor(const void* data_ptr,
                           const std::vector<int64_t>& shape,
                           aclDataType dtype) {
    auto strides = ComputeStrides(shape);
    aclTensor* tensor = aclCreateTensor(
        shape.data(),
        shape.size(),
        dtype,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        const_cast<void*>(data_ptr)
    );
    return tensor;
}

// ============================================================================
// Workspace 封装（不包含内存指针，内存由 PyTorch tensor 管理）
// ============================================================================

struct OpWorkspace {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclTensor* x1_acl = nullptr;
    aclTensor* x2_acl = nullptr;
    aclTensor* output_acl = nullptr;

    ~OpWorkspace() {
        if (x1_acl) { aclDestroyTensor(x1_acl); x1_acl = nullptr; }
        if (x2_acl) { aclDestroyTensor(x2_acl); x2_acl = nullptr; }
        if (output_acl) { aclDestroyTensor(output_acl); output_acl = nullptr; }
        executor = nullptr;
    }
};

OpWorkspace* OpGetWorkspace(const void* x1_ptr,
                            const void* x2_ptr,
                            void* output_ptr,
                            const std::vector<int64_t>& shape,
                            aclDataType dtype) {
    auto ws = new OpWorkspace();
    ws->x1_acl = CreateAclTensor(x1_ptr, shape, dtype);
    ws->x2_acl = CreateAclTensor(x2_ptr, shape, dtype);
    ws->output_acl = CreateAclTensor(output_ptr, shape, dtype);

    if (!ws->x1_acl || !ws->x2_acl || !ws->output_acl) {
        TORCH_CHECK(false, "OpGetWorkspace: CreateAclTensor failed, "
                    "x1=", (ws->x1_acl ? "ok" : "null"),
                    ", x2=", (ws->x2_acl ? "ok" : "null"),
                    ", out=", (ws->output_acl ? "ok" : "null"));
        delete ws;
        return nullptr;
    }

    aclnnStatus status = aclnnAddExampleGetWorkspaceSize(
        ws->x1_acl, ws->x2_acl, ws->output_acl, &ws->workspace_size, &ws->executor);

    if (status != ACL_SUCCESS) {
        TORCH_CHECK(false, "OpGetWorkspace: GetWorkspaceSize failed, aclnnStatus=", status);
        delete ws;
        return nullptr;
    }

    return ws;
}

aclnnStatus OpExecute(OpWorkspace* ws, void* workspace_ptr, aclrtStream stream) {
    if (!ws || !ws->executor) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclnnAddExample(workspace_ptr, ws->workspace_size, ws->executor, stream);
}

} // anonymous namespace

// ============================================================================
// Meta 函数：形状推导
// ============================================================================

static torch::Tensor forward_meta(const torch::Tensor& x1, const torch::Tensor& x2) {
    TORCH_CHECK(x1.sizes() == x2.sizes(),
                "forward: shapes must match, got ", x1.sizes(), " vs ", x2.sizes());
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
                "forward: dtypes must match, got ", x1.scalar_type(), " vs ", x2.scalar_type());
    return torch::empty_like(x1);
}

// ============================================================================
// NPU 实现：方式三 - stream(false) + OpCommand + lambda 入 queue
// ============================================================================

static torch::Tensor forward_npu(const torch::Tensor& x1, const torch::Tensor& x2) {
    auto z = torch::empty_like(x1).contiguous();
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    auto dtype = ScalarTypeToAclDType(x1.scalar_type());
    auto shape = x1.sizes().vec();

    OpWorkspace* ws = OpGetWorkspace(
        x1_contig.data_ptr(), x2_contig.data_ptr(), z.data_ptr(), shape, dtype);

    TORCH_CHECK(ws != nullptr, "OpGetWorkspace returned null (see above for details)");

    torch::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (ws->workspace_size > 0) {
        workspace_tensor = torch::empty({static_cast<int64_t>(ws->workspace_size)},
                                        torch::dtype(torch::kByte).device(x1.device()));
        workspace_ptr = workspace_tensor.data_ptr();
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

    auto acl_call = [ws, workspace_ptr, acl_stream]() -> int {
        aclnnStatus status = OpExecute(ws, workspace_ptr, acl_stream);
        delete ws;
        return status == ACL_SUCCESS ? 0 : 1;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_add_example", acl_call);

    return z;
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY_FRAGMENT(add_example, m) {
    m.def("forward(Tensor x1, Tensor x2) -> Tensor");
}

TORCH_LIBRARY_IMPL(add_example, Meta, m) {
    m.impl("forward", forward_meta);
}

TORCH_LIBRARY_IMPL(add_example, PrivateUse1, m) {
    m.impl("forward", forward_npu);
}
