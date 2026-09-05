/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn_matmul_blaze_example.h"

using bf16_t = uint16_t;

static constexpr double RTOL = std::pow(2.0, -7);
static constexpr double ATOL = 10.0 * RTOL;

static int64_t Size(const std::vector<int64_t>& s) {
    int64_t z = 1;
    for (auto v : s) z *= v;
    return z;
}

static std::vector<int64_t> Strides(const std::vector<int64_t>& s) {
    std::vector<int64_t> st(s.size(), 1);
    for (int i = (int)s.size() - 2; i >= 0; --i) st[i] = st[i + 1] * s[i + 1];
    return st;
}

static uint16_t FloatToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint16_t bf16 = static_cast<uint16_t>(u >> 16);
    if ((u & 0xFFFF) != 0) bf16 |= 1;
    return bf16;
}

static float Bf16ToFloat(uint16_t bf16) {
    uint32_t u = static_cast<uint32_t>(bf16) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static void Golden(const bf16_t* a, const bf16_t* b, bf16_t* c, int64_t M, int64_t K, int64_t N) {
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += Bf16ToFloat(a[m * K + k]) * Bf16ToFloat(b[k * N + n]);
            c[m * N + n] = FloatToBf16(acc);
        }
}

template <typename T>
static int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape,
                        aclDataType dt, void** dev, aclTensor** tensor) {
    size_t bytes = host.size() * sizeof(T);
    if (aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 1;
    if (aclrtMemcpy(*dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return 2;
    auto strides = Strides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *dev);
    return *tensor ? 0 : 3;
}

static int CreateOut(const std::vector<int64_t>& shape, aclDataType dt, size_t elemSize,
                     void** dev, aclTensor** tensor) {
    size_t bytes = Size(shape) * elemSize;
    if (aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return 1;
    auto strides = Strides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *dev);
    return *tensor ? 0 : 3;
}

struct TestResult {
    std::string name;
    bool pass;
    double maxRel;
    double maxAbs;
};

static std::vector<TestResult> g_results;

static bool RunPrecisionCase(const std::string& name, int64_t M, int64_t K, int64_t N, aclrtStream stream) {
    std::mt19937 g(42 + M * 1000 + K + N);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<bf16_t> a(M * K), b(K * N), golden(M * N);
    for (auto& v : a) v = FloatToBf16(d(g));
    for (auto& v : b) v = FloatToBf16(d(g));
    Golden(a.data(), b.data(), golden.data(), M, K, N);

    void *da = nullptr, *db = nullptr, *dc = nullptr;
    aclTensor *ta = nullptr, *tb = nullptr, *tc = nullptr;

    if (CreateTensor(a, {M, K}, ACL_BF16, &da, &ta) ||
        CreateTensor(b, {K, N}, ACL_BF16, &db, &tb) ||
        CreateOut({M, N}, ACL_BF16, sizeof(bf16_t), &dc, &tc)) {
        g_results.push_back({name, false, 0, 0});
        goto cleanup;
    }

    {
        uint64_t wsSize = 0;
        aclOpExecutor* exec = nullptr;
        auto ret = aclnnMatmulBlazeExampleGetWorkspaceSize(ta, tb, false, false, tc, &wsSize, &exec);
        if (ret != ACL_SUCCESS) {
            g_results.push_back({name, false, 0, 0});
            goto cleanup;
        }
        void* ws = nullptr;
        if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclnnMatmulBlazeExample(ws, wsSize, exec, stream);
        aclrtSynchronizeStream(stream);
        if (ws) aclrtFree(ws);
    }

    {
        std::vector<bf16_t> out(M * N);
        aclrtMemcpy(out.data(), out.size() * sizeof(bf16_t), dc,
                    out.size() * sizeof(bf16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        double maxRel = 0, maxAbs = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            double actual = Bf16ToFloat(out[i]);
            double gold = Bf16ToFloat(golden[i]);
            double absd = std::abs(actual - gold);
            double reld = absd / std::max(std::abs(gold), 1e-7);
            maxRel = std::max(maxRel, reld);
            maxAbs = std::max(maxAbs, absd);
        }
        bool ok = (maxRel < ATOL);
        g_results.push_back({name, ok, maxRel, maxAbs});
    }

cleanup:
    if (ta) aclDestroyTensor(ta);
    if (tb) aclDestroyTensor(tb);
    if (tc) aclDestroyTensor(tc);
    if (da) aclrtFree(da);
    if (db) aclrtFree(db);
    if (dc) aclrtFree(dc);
    return g_results.back().pass;
}

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS) { std::cerr << "aclInit failed" << std::endl; return 1; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { std::cerr << "aclrtSetDevice failed" << std::endl; return 1; }
    aclrtStream stream;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { std::cerr << "aclrtCreateStream failed" << std::endl; return 1; }

    RunPrecisionCase("L0_001 (16,16,16)", 16, 16, 16, stream);
    RunPrecisionCase("L0_002 (32,32,32)", 32, 32, 32, stream);
    RunPrecisionCase("L0_003 (64,64,64)", 64, 64, 64, stream);
    RunPrecisionCase("L0_004 (128,128,128)", 128, 128, 128, stream);
    RunPrecisionCase("L0_005 (256,256,256)", 256, 256, 256, stream);
    RunPrecisionCase("L1_001 (1,16,16) M=1", 1, 16, 16, stream);
    RunPrecisionCase("L1_002 (16,16,1) N=1", 16, 16, 1, stream);
    RunPrecisionCase("L1_003 (16,1,16) K=1", 16, 1, 16, stream);
    RunPrecisionCase("L2_001 (32,64,33) N-unaligned", 32, 64, 33, stream);
    RunPrecisionCase("L2_002 (33,64,32) M-odd", 33, 64, 32, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    int passed = 0, total = (int)g_results.size();
    std::cout << "\n=== Test Summary ===" << std::endl;
    for (const auto& r : g_results) {
        std::cout << (r.pass ? "PASS" : "FAIL") << "  " << r.name
                  << "  maxRel=" << r.maxRel << " maxAbs=" << r.maxAbs << std::endl;
        if (r.pass) ++passed;
    }
    std::cout << "===\nTotal: " << passed << "/" << total
              << (passed == total ? " PASSED" : " FAILED") << std::endl;
    return (passed == total) ? 0 : 1;
}
