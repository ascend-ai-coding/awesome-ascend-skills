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
 * =============================================================================
 * euclidean_norm_package/tests/aclnn/test_aclnn.cpp
 * =============================================================================
 * Role: ACLNN integration test for the EuclideanNorm operator.
 *        End-to-end test that calls aclnnEuclideanNormGetWorkspaceSize and
 *        aclnnEuclideanNorm via the public ACLNN API, verifies output
 *        correctness against a CPU reference, and reports pass/fail.
 *
 * Test topology:
 *   x = [rows, cols] (2-D),  axes = {1},  keepDims = false
 *   out[i] = sqrt( sum_j( x[i][j]^2 ) )
 *
 * Coverage (mirrors add_custom_package/tests/aclnn):
 *   - fp16 + fp32 via template <typename T>.
 *   - 14 shapes covering tiny / boundary / medium / large.
 *   - Index-dependent fill values for edge-case coverage.
 * =============================================================================
 */

#include <cstdio>
#include <cmath>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "aclnn_euclidean_norm.h"

#define CHECK_ACL(x) \
    do { \
        aclError __ret = (x); \
        if (__ret != ACL_SUCCESS) { \
            fprintf(stderr, "ACL error %d at %s:%d: %s\n", __ret, __FILE__, __LINE__, #x); \
            return -1; \
        } \
    } while (0)

#define CHECK_ACLNN(x) \
    do { \
        aclnnStatus __ret = (x); \
        if (__ret != OK) { \
            fprintf(stderr, "ACLNN error %d at %s:%d: %s\n", __ret, __FILE__, __LINE__, #x); \
            return -1; \
        } \
    } while (0)

template <typename T>
static T FromFloat(float v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloatToFloat16(v);
    } else {
        return v;
    }
}

template <typename T>
static float ToFloat(T v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloat16ToFloat(v);
    } else {
        return v;
    }
}

template <typename T>
static aclDataType AclDtypeOf()
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return ACL_FLOAT16;
    } else {
        return ACL_FLOAT;
    }
}

template <typename T>
static const char* DtypeName()
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return "fp16";
    } else {
        return "fp32";
    }
}

/**
 * TestEuclideanNorm<T>: run ACLNN test for shape [rows, cols], axes={1}.
 *
 * x[i][j] = (linearIdx % 32) * 0.1f   — bounded for fp16 (x^2 < 10, sum < 10K)
 * expected out[i] = sqrt( sum_j x[i][j]^2 )
 */
template <typename T>
static int TestEuclideanNorm(int64_t rows, int64_t cols)
{
    constexpr bool isFp16 = std::is_same_v<T, aclFloat16>;
    const aclDataType aclType = AclDtypeOf<T>();
    const char* dn = DtypeName<T>();

    size_t xNum = static_cast<size_t>(rows * cols);
    size_t xBytes = xNum * sizeof(T);
    size_t outNum = static_cast<size_t>(rows);
    size_t outBytes = outNum * sizeof(T);

    std::vector<T> xHost(xNum);
    std::vector<T> outHost(outNum);

    for (size_t i = 0; i < xNum; ++i) {
        xHost[i] = FromFloat<T>(static_cast<float>(i % 32) * 0.1f);
    }

    void* devX = nullptr;
    void* devOut = nullptr;
    CHECK_ACL(aclrtMalloc(&devX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    CHECK_ACL(aclrtMemcpy(devX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    const int64_t xDims[] = {rows, cols};
    const int64_t xStrides[] = {cols, 1};
    const int64_t xStorage[] = {rows, cols};
    const int64_t outDims[] = {rows};
    const int64_t outStrides[] = {1};
    const int64_t outStorage[] = {rows};

    aclTensor* x = aclCreateTensor(xDims, 2, aclType, xStrides, 0, ACL_FORMAT_ND, xStorage, 2, devX);
    aclTensor* out = aclCreateTensor(outDims, 1, aclType, outStrides, 0, ACL_FORMAT_ND, outStorage, 1, devOut);

    int64_t axesValues[] = {1};
    aclIntArray* axes = aclCreateIntArray(axesValues, 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnEuclideanNormGetWorkspaceSize(x, axes, false, out, &workspaceSize, &executor));

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    CHECK_ACLNN(aclnnEuclideanNorm(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(outHost.data(), outBytes, devOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // CPU reference: out[i] = sqrt( sum_j x[i][j]^2 )
    // Kernel accumulates internally in fp32 even for fp16 inputs, so the
    // reference is computed in double and compared with fp16 output tolerance.
    bool ok = true;
    for (int64_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            float v = ToFloat<T>(xHost[static_cast<size_t>(i * cols + j)]);
            acc += static_cast<double>(v) * v;
        }
        float expected = std::sqrt(static_cast<float>(acc));
        float got = ToFloat<T>(outHost[static_cast<size_t>(i)]);
        float tol = isFp16 ? 1.0f : 1e-3f;
        if (std::fabs(got - expected) > tol) {
            fprintf(stderr, "[%s] Mismatch at row %ld: got %f, expected %f\n", dn, i, got, expected);
            ok = false;
            break;
        }
    }

    fprintf(ok ? stdout : stderr, "Test %s rows=%ld cols=%ld %s\n", dn, rows, cols, ok ? "PASSED" : "FAILED");

    if (workspace) {
        aclrtFree(workspace);
    }
    aclDestroyIntArray(axes);
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    aclrtFree(devX);
    aclrtFree(devOut);
    aclrtDestroyStream(stream);
    return ok ? 0 : -1;
}

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    // Shape list: [rows, cols] — cols is the reduced dimension.
    // Covers tiny / boundary / medium / large, matching add_custom coverage.
    const std::vector<std::pair<int64_t, int64_t>> shapes = {
        {1, 1},    {1, 7},    {1, 16},   {4, 100},   {4, 1023},   {4, 1024},   {4, 1025},
        {4, 4096}, {4, 4097}, {4, 9973}, {8, 16384}, {8, 100000}, {4, 409600}, {4, 1048577},
    };

    int ret = 0;
    int passed = 0;
    for (auto [r, c] : shapes) {
        if (TestEuclideanNorm<aclFloat16>(r, c) == 0) {
            ++passed;
        } else {
            ret = -1;
        }
        if (TestEuclideanNorm<float>(r, c) == 0) {
            ++passed;
        } else {
            ret = -1;
        }
    }

    size_t total = shapes.size() * 2;
    fprintf(stdout, "\n==== ACLNN summary: %d/%zu cases PASSED (%zu shapes x {fp16, fp32}) ====\n", passed, total,
            shapes.size());

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return ret;
}
