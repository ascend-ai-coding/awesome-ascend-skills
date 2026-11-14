// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See the License for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// Worked example: CANN built-in Add, so the template builds and runs on any
// CANN install with no custom vendor package.  Adapting to a real project
// usually means a CUSTOM aclnn op from a vendor package; see the notes at the
// bottom of this file for the two pitfalls hit in real custom-op validation.

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>

#include <cstring>
#include <string>
#include <vector>

#include "target_adapter.h"

namespace {

struct Buffer {
    void* device = nullptr;
    size_t bytes = 0;
    void* host = nullptr;
};

struct Inputs {
    std::vector<int64_t> shape;
    Buffer self_buffer;
    Buffer other_buffer;
    Buffer out_buffer;
};

Inputs g_inputs;

Buffer AllocBuffer(size_t bytes)
{
    Buffer buffer;
    buffer.bytes = bytes;
    if (aclrtMalloc(&buffer.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
        return {};
    buffer.host = std::malloc(bytes);
    if (buffer.host == nullptr)
        return {};
    return buffer;
}

void FreeBuffer(Buffer& buffer)
{
    if (buffer.device != nullptr)
        aclrtFree(buffer.device);
    if (buffer.host != nullptr)
        std::free(buffer.host);
    buffer.device = nullptr;
    buffer.host = nullptr;
}

aclTensor* MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* device_ptr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), device_ptr);
}

} // namespace

const char* TARGET_OPERATOR_NAME = "Add";

// Exact profiler Op Names of one target invocation, in launch order.
const std::vector<std::string> TARGET_KERNEL_SEQUENCE = {"aclnnAdd_AddAiCore_Add"};

bool target_make_inputs(const CaseParams& params)
{
    g_inputs = Inputs{};
    const int64_t m = params.get("m");
    const int64_t n = params.get("n");
    if (m <= 0 || n <= 0)
        return false;
    g_inputs.shape = {m, n};

    const size_t bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
    g_inputs.self_buffer = AllocBuffer(bytes);
    g_inputs.other_buffer = AllocBuffer(bytes);
    g_inputs.out_buffer = AllocBuffer(bytes);
    if (g_inputs.self_buffer.device == nullptr || g_inputs.other_buffer.device == nullptr ||
        g_inputs.out_buffer.device == nullptr) {
        return false;
    }

    auto fill = [&](Buffer& buffer, float base) {
        auto* values = static_cast<float*>(buffer.host);
        const size_t count = bytes / sizeof(float);
        for (size_t i = 0; i < count; ++i)
            values[i] = base + static_cast<float>(i % 997) / 997.0f;
        return aclrtMemcpy(buffer.device, bytes, buffer.host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    };
    if (!fill(g_inputs.self_buffer, -2.0f))
        return false;
    if (!fill(g_inputs.other_buffer, 0.0f))
        return false;

    // Outputs of custom ops may rely on zero-initialized memory (the torch
    // path allocates with at::zeros); zero via host-side memset so no extra
    // kernel lands in the target sequence.
    if (aclrtMemset(g_inputs.out_buffer.device, bytes, 0, bytes) != ACL_SUCCESS)
        return false;
    return true;
}

bool target_invoke(void* stream)
{
    aclTensor* self_tensor = MakeTensor(g_inputs.shape, ACL_FLOAT, g_inputs.self_buffer.device);
    aclTensor* other_tensor = MakeTensor(g_inputs.shape, ACL_FLOAT, g_inputs.other_buffer.device);
    aclTensor* out_tensor = MakeTensor(g_inputs.shape, ACL_FLOAT, g_inputs.out_buffer.device);
    float alpha_value = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    if (self_tensor == nullptr || other_tensor == nullptr || out_tensor == nullptr || alpha == nullptr) {
        return false;
    }

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const int32_t status =
        aclnnAddGetWorkspaceSize(self_tensor, other_tensor, alpha, out_tensor, &workspace_size, &executor);
    bool ok = status == 0;
    void* workspace = nullptr;
    if (ok && workspace_size > 0) {
        ok = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }
    if (ok) {
        ok = aclnnAdd(workspace, workspace_size, executor, static_cast<aclrtStream>(stream)) == 0;
    }
    if (workspace != nullptr)
        aclrtFree(workspace);

    aclDestroyTensor(self_tensor);
    aclDestroyTensor(other_tensor);
    aclDestroyTensor(out_tensor);
    aclDestroyScalar(alpha);
    return ok;
}

void target_release()
{
    FreeBuffer(g_inputs.self_buffer);
    FreeBuffer(g_inputs.other_buffer);
    FreeBuffer(g_inputs.out_buffer);
}

// Adapting to a custom vendor op (real-world notes from a beam_search_group
// validation):
//   1. #include the vendor aclnn header (layout varies):
//        #if __has_include(<aclnnop/aclnn_<op>.h>)
//        #include <aclnnop/aclnn_<op>.h>
//        #else
//        #include <aclnn_<op>.h"
//        #endif
//   2. Build with --vendor-dir=<unpacked vendors/<name>> and source the
//      vendor's bin/set_env.bash before running: without it GetWorkspaceSize
//      fails with 161001 PARAM_NULLPTR.
//   3. Zero-init every output with aclrtMemset; uninitialized outputs can
//      hang kernels that assume at::zeros semantics.
//   4. Keep the tensor/dtype/attr mapping identical to the project's real
//      invocation (mirror its reference binding).
