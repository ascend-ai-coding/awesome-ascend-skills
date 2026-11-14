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
 * xdivy_custom_package/tests/aclnn/test_aclnn.cpp
 * =============================================================================
 * Role: ACLNN integration test for the Xdivy operator.
 *        End-to-end test that calls aclnnXdivyGetWorkspaceSize and aclnnXdivy
 *        via the public ACLNN API, verifies output correctness against a
 *        CPU reference computed with broadcast semantics, and reports pass/fail.
 *
 * Contents:
 *   - CHECK_ACL / CHECK_ACLNN macros: error-checking wrappers.
 *   - FromFloat<T> / ToFloat<T>: conversion helpers between float and fp16.
 *   - BroadcastInfo: describes a broadcast scenario (x1/x2/y shapes).
 *   - MakeBroadcastTensor: allocates an aclTensor with given shape + data.
 *   - GoldenXdivy: CPU reference for (x1==0) ? 0 : x1/x2 with broadcast.
 *   - TestXdivy<T>: templated end-to-end test for one scenario.
 *   - main: iterates over scenarios in both fp16 and fp32.
 *
 * Operator name variants:
 *   - PascalCase:   Xdivy
 *   - snake_case:   xdivy
 *   - UPPER_SNAKE:  XDIVY
 *   - camelCase:    xdivy
 *
 * To create a new operator (e.g. FooBar), replace:
 *   aclnnXdivyGetWorkspaceSize → aclnnFooBarGetWorkspaceSize
 *   aclnnXdivy → aclnnFooBar
 *   aclnn_xdivy.h → aclnn_foo_bar.h
 *   Xdivy → FooBar
 * =============================================================================
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

// acl.h: aclInit, aclrtMalloc, aclrtMemcpy, aclCreateTensor, aclrtStream, etc.
#include "acl/acl.h"
// aclnn_xdivy.h: our vendor operator's public ACLNN API header.
#include "aclnn_xdivy.h"

/**
 * CHECK_ACL(x): wrap an ACL API call and check its return code.
 */
#define CHECK_ACL(x) \
    do { \
        aclError __ret = (x); \
        if (__ret != ACL_SUCCESS) { \
            fprintf(stderr, "ACL error %d at %s:%d: %s\n", __ret, __FILE__, __LINE__, #x); \
            return -1; \
        } \
    } while (0)

/** CHECK_ACLNN(x): same idea, for aclnnStatus returns. */
#define CHECK_ACLNN(x) \
    do { \
        aclnnStatus __ret = (x); \
        if (__ret != OK) { \
            fprintf(stderr, "ACLNN error %d at %s:%d: %s\n", __ret, __FILE__, __LINE__, #x); \
            return -1; \
        } \
    } while (0)

/** FromFloat<T>: convert float to tensor element type T (fp16 or fp32). */
template <typename T>
static T FromFloat(float v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloatToFloat16(v);
    } else {
        return v;
    }
}

/** ToFloat<T>: convert tensor element T to float. */
template <typename T>
static float ToFloat(T v)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        return aclFloat16ToFloat(v);
    } else {
        return v;
    }
}

/** AclTypeOf<T>: maps C++ type T to the corresponding aclDataType. */
template <typename T>
static aclDataType AclTypeOf()
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        return ACL_FLOAT16;
    else
        return ACL_FLOAT;
}

/** DtypeName<T>: human-readable dtype name for logs. */
template <typename T>
static const char* DtypeName()
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        return "fp16";
    else
        return "fp32";
}

/**
 * BroadcastInfo: describes one broadcast test scenario.
 *   x1Shape, x2Shape — input shapes (broadcast-compatible).
 *   yShape            — expected output shape.
 */
struct BroadcastInfo {
    std::vector<int64_t> x1Shape;
    std::vector<int64_t> x2Shape;
    std::vector<int64_t> yShape;
};

/** ShapeSize: total element count = product of dims. */
static int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s)
        n *= d;
    return n;
}

/**
 * MakeAclTensor: allocate device memory, fill with deterministic data, and
 *   create an aclTensor descriptor.
 *
 * Data patterns (must match the golden computation):
 *   x1: ((flat * 7) % 13) * 0.5f  — includes zeros to test the x1==0 path.
 *   x2: ((flat + 1) % 11 + 1) * 1.5f — never zero.
 */
template <typename T>
static int32_t MakeAclTensor(const std::vector<int64_t>& shape, bool isX1, void** outDev, aclTensor** outTensor)
{
    int64_t n = (shape.empty()) ? 1 : ShapeSize(shape);
    size_t bytes = static_cast<size_t>(n) * sizeof(T);

    std::vector<T> host(n);
    for (int64_t i = 0; i < n; ++i) {
        float v = isX1 ? static_cast<float>((i * 7) % 13) * 0.5f : static_cast<float>((i + 1) % 11 + 1) * 1.5f;
        host[i] = FromFloat<T>(v);
    }

    CHECK_ACL(aclrtMalloc(outDev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(*outDev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // Build aclTensor with stride = 1 per dim (contiguous row-major).
    // For scalar input (empty shape), pass 0 dims and a 1-element storage.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int32_t i = (int32_t)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    *outTensor = aclCreateTensor(shape.data(), (int64_t)shape.size(), AclTypeOf<T>(), strides.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), (int64_t)shape.size(), *outDev);
    return 0;
}

/**
 * GoldenXdivy: compute the expected output on the CPU with broadcast.
 *   For each output position, look up x1 and x2 using broadcast rules and
 *   compute (x1==0) ? 0 : (x1/x2).
 */
template <typename T>
static std::vector<float> GoldenXdivy(const BroadcastInfo& bc)
{
    int64_t outN = ShapeSize(bc.yShape);
    int64_t n1 = (bc.x1Shape.empty()) ? 1 : ShapeSize(bc.x1Shape);
    int64_t n2 = (bc.x2Shape.empty()) ? 1 : ShapeSize(bc.x2Shape);

    // Build host input buffers (must match MakeAclTensor's data patterns).
    std::vector<T> x1(n1), x2(n2);
    for (int64_t i = 0; i < n1; ++i)
        x1[i] = FromFloat<T>(static_cast<float>((i * 7) % 13) * 0.5f);
    for (int64_t i = 0; i < n2; ++i)
        x2[i] = FromFloat<T>(static_cast<float>((i + 1) % 11 + 1) * 1.5f);

    // For each output position, compute multi-dim coord and look up x1/x2
    // via broadcast (right-aligned, dim==1 means read index 0).
    std::vector<float> out(outN);
    int64_t outRank = (int64_t)bc.yShape.size();
    int64_t rank1 = (int64_t)bc.x1Shape.size();
    int64_t rank2 = (int64_t)bc.x2Shape.size();

    for (int64_t idx = 0; idx < outN; ++idx) {
        // Decode flat idx to multi-dim coord using yShape.
        std::vector<int64_t> coord(outRank);
        int64_t tmp = idx;
        for (int64_t d = outRank - 1; d >= 0; --d) {
            coord[d] = tmp % bc.yShape[d];
            tmp /= bc.yShape[d];
        }

        // Look up x1 with broadcast.
        auto lookup = [&](const std::vector<T>& data, const std::vector<int64_t>& inShape, int64_t inRank) -> float {
            int64_t offset = outRank - inRank;
            int64_t inFlat = 0;
            int64_t stride = 1;
            for (int64_t i = inRank - 1; i >= 0; --i) {
                int64_t outD = i + offset;
                int64_t c = (outD >= 0) ? coord[outD] : 0;
                if (inShape[i] == 1)
                    c = 0;
                inFlat += c * stride;
                stride *= inShape[i];
            }
            return ToFloat<T>(data[inFlat]);
        };

        float x1v = lookup(x1, bc.x1Shape, rank1);
        float x2v = lookup(x2, bc.x2Shape, rank2);
        out[idx] = (x1v == 0.0f) ? 0.0f : x1v / x2v;
    }
    return out;
}

/**
 * TestXdivy<T>: run the full ACLNN test for a given broadcast scenario.
 */
template <typename T>
static int32_t TestXdivy(const BroadcastInfo& bc)
{
    int64_t outN = ShapeSize(bc.yShape);
    size_t outBytes = static_cast<size_t>(outN) * sizeof(T);

    // Create an ACL stream.
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    // Create input tensors (allocates device memory + fills with patterns).
    void* devX1 = nullptr;
    void* devX2 = nullptr;
    void* devY = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    if (MakeAclTensor<T>(bc.x1Shape, true, &devX1, &x1) != 0)
        return -1;
    if (MakeAclTensor<T>(bc.x2Shape, false, &devX2, &x2) != 0)
        return -1;

    // Allocate output tensor (uninitialised).
    CHECK_ACL(aclrtMalloc(&devY, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<int64_t> yStrides(bc.yShape.size(), 1);
    for (int32_t i = (int32_t)bc.yShape.size() - 2; i >= 0; --i) {
        yStrides[i] = yStrides[i + 1] * bc.yShape[i + 1];
    }
    aclTensor* y = aclCreateTensor(bc.yShape.data(), (int64_t)bc.yShape.size(), AclTypeOf<T>(), yStrides.data(), 0,
                                   ACL_FORMAT_ND, bc.yShape.data(), (int64_t)bc.yShape.size(), devY);

    // Phase 1: query workspace size and obtain executor.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnXdivyGetWorkspaceSize(x1, x2, y, &workspaceSize, &executor));

    // Allocate workspace on device if needed.
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Phase 2: launch the operator.
    CHECK_ACLNN(aclnnXdivy(workspace, workspaceSize, executor, stream));

    // Synchronise and copy output back.
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::vector<T> outHost(outN);
    CHECK_ACL(aclrtMemcpy(outHost.data(), outBytes, devY, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // Verify against CPU golden.
    auto expected = GoldenXdivy<T>(bc);
    bool ok = true;
    for (size_t i = 0; i < outHost.size(); ++i) {
        float got = ToFloat<T>(outHost[i]);
        if (std::fabs(got - expected[i]) > 1e-3f) {
            fprintf(stderr, "[%s] Mismatch at [%zu]: got %f, expected %f\n", DtypeName<T>(), i, got, expected[i]);
            ok = false;
            break;
        }
    }

    fprintf(ok ? stdout : stderr, "Test %s for %s yShapeSize=%ld %s\n", ok ? "PASSED" : "FAILED", DtypeName<T>(), outN,
            ok ? "" : "(see errors above)");

    // Cleanup.
    if (workspace)
        aclrtFree(workspace);
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(y);
    aclrtFree(devX1);
    aclrtFree(devX2);
    aclrtFree(devY);
    aclrtDestroyStream(stream);

    return ok ? 0 : -1;
}

/**
 * main: entry point for the ACLNN integration test.
 */
int32_t main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    // Broadcast scenarios covering:
    //   - same-shape 1D/2D/3D (no broadcast).
    //   - row/col broadcast.
    //   - scalar dividend (x1 rank 0).
    //   - small / medium / large element counts.
    const std::vector<BroadcastInfo> scenarios = {
        {{16}, {16}, {16}},             // 1d same shape
        {{}, {16}, {16}},               // scalar x1
        {{4, 8}, {4, 8}, {4, 8}},       // 2d same shape
        {{4, 8}, {1, 8}, {4, 8}},       // row broadcast
        {{4, 1}, {1, 8}, {4, 8}},       // col broadcast (full)
        {{2, 3, 4}, {3, 4}, {2, 3, 4}}, // 3d + 2d
        {{1000}, {1000}, {1000}},       // medium 1d
        {{1024}, {1}, {1024}},          // scalar divisor (broadcast over 1024)
    };

    int32_t ret = 0;
    int32_t passed = 0;
    for (const auto& bc : scenarios) {
        if (TestXdivy<aclFloat16>(bc) == 0)
            ++passed;
        else
            ret = -1;
        if (TestXdivy<float>(bc) == 0)
            ++passed;
        else
            ret = -1;
    }

    size_t total = scenarios.size() * 2;
    fprintf(stdout, "\n==== ACLNN broadcast summary: %d/%zu cases PASSED ====\n", passed, total);

    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return ret;
}
