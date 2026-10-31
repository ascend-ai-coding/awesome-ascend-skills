/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <platform/platform_ascendc.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#ifndef PROBE_CASE_HEADER
#define PROBE_CASE_HEADER "topk_cases.h"
#endif
#include PROBE_CASE_HEADER
#include "host_types.h"
#define ACL(x) \
    do { \
        auto e = (x); \
        if (e) { \
            std::cerr << #x << ":" << e << "\n"; \
            return 1; \
        } \
    } while (0)
size_t elem(int t)
{
    return t == 0 || t == 1 ? 1 : t == 3 ? 8 : t == 2 || t == 5 ? 4 : 2;
}
long double read(const std::vector<uint8_t>& a, size_t i, int t)
{
    auto p = a.data() + i * elem(t);
    switch (t) {
        case 0:
            return *(int8_t*)p;
        case 1:
            return *p;
        case 2: {
            int32_t v;
            memcpy(&v, p, 4);
            return v;
        }
        case 3: {
            int64_t v;
            memcpy(&v, p, 8);
            return v;
        }
        case 5: {
            float v;
            memcpy(&v, p, 4);
            return v;
        }
        default: {
            uint16_t v;
            memcpy(&v, p, 2);
            return decode(v, t == 6);
        }
    }
}
void write(std::vector<uint8_t>& a, size_t i, int t, long double v)
{
    auto p = a.data() + i * elem(t);
    switch (t) {
        case 0: {
            int8_t x = v;
            memcpy(p, &x, 1);
            break;
        }
        case 1: {
            uint8_t x = v;
            memcpy(p, &x, 1);
            break;
        }
        case 2: {
            int32_t x = v;
            memcpy(p, &x, 4);
            break;
        }
        case 3: {
            int64_t x = v;
            memcpy(p, &x, 8);
            break;
        }
        case 5: {
            float x = v;
            memcpy(p, &x, 4);
            break;
        }
        default: {
            uint16_t x = encode(float(v), t == 6);
            memcpy(p, &x, 2);
        }
    }
}
bool same(long double a, long double b)
{
    return a == b || (std::isnan(a) && std::isnan(b));
}
int test(ProbeCase c, int pattern, aclrtStream stream, uint32_t cores, uint64_t ub)
{
    if (c.p.tilingKey == 900001) {
        ACL(calc_topk_small_axis_params(c.outer, c.d, c.inner, c.largest, cores, ub, &c.p.smallAxis));
        c.p.usedCores = c.p.smallAxis.usedCores;
        c.p.localMemoryBytes = c.p.smallAxis.localMemoryBytes;
    }
    if (c.p.tilingKey == 900002) {
        ACL(calc_topk_fallback_params(c.outer, c.d, c.inner, c.k, c.largest, c.dtype, cores, &c.p.fallback));
        c.p.usedCores = c.p.fallback.usedCores;
    }
    size_t count = c.outer * c.d * c.inner;
    size_t outcount = c.outer * c.k * c.inner;
    size_t bytes = count * elem(c.dtype);
    size_t outbytes = outcount * elem(c.dtype);
    size_t ibytes = outcount * 8;
    std::vector<uint8_t> x(bytes);
    std::vector<uint8_t> y(outbytes + 64);
    std::vector<int64_t> idx(outcount + 8);
    for (size_t i = 0; i < count; i++) {
        long double v = pattern == 1 ? int(i % 7) - 3 : int((i * 37 + 11) % 257) - 128;
        if (c.dtype == 3)
            v += (1LL << 40);
        if (c.dtype >= 4)
            v /= 16;
        if (pattern == 2 && c.dtype == 3)
            v = i % 3 == 0 ? std::numeric_limits<int64_t>::min() :
                i % 3 == 1 ? std::numeric_limits<int64_t>::max() :
                             -(1LL << 40);
        if (pattern == 2 && c.dtype >= 4) {
            if (i % 67 == 0)
                v = std::numeric_limits<float>::quiet_NaN();
            else if (i % 71 == 0)
                v = std::numeric_limits<float>::infinity();
            else if (i % 73 == 0)
                v = -std::numeric_limits<float>::infinity();
        }
        write(x, i, c.dtype, v);
    }
    void *dx{};
    void *dy{};
    void *di{};
    void *ws{};
    ACL(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&dy, y.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&di, ibytes + 64, ACL_MEM_MALLOC_HUGE_FIRST));
    if (c.p.workspaceBytes) {
        ACL(aclrtMalloc(&ws, c.p.workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL(aclrtMemset(ws, c.p.workspaceBytes, 0, c.p.workspaceBytes));
    }
    ACL(aclrtMemcpy(dx, bytes, x.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemset(dy, y.size(), 0xa5, y.size()));
    ACL(aclrtMemset(di, ibytes + 64, 0xa5, ibytes + 64));
    std::cout << "RUN dtype=" << c.dtype << " D=" << c.d << " inner=" << c.inner << " largest=" << c.largest
              << " key=" << c.p.tilingKey << " mode=" << c.p.tiling.modeType << " pattern=" << pattern << std::endl;
    ACL(launch_topk(dx, dy, di, ws, c.dtype, c.p, stream));
    ACL(aclrtSynchronizeStreamWithTimeout(stream, 20000));
    ACL(aclrtMemcpy(y.data(), y.size(), dy, y.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    ACL(aclrtMemcpy(idx.data(), ibytes + 64, di, ibytes + 64, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t o = 0; o < c.outer; o++)
        for (int64_t n = 0; n < c.inner; n++) {
            std::vector<long double> gold(c.d);
            for (int64_t d = 0; d < c.d; d++)
                gold[d] = read(x, (o * c.d + d) * c.inner + n, c.dtype);
            std::sort(gold.begin(), gold.end(), [&](long double a, long double b) {
                if (std::isnan(a))
                    return !std::isnan(b) && c.largest;
                if (std::isnan(b))
                    return !c.largest;
                return c.largest ? a > b : a < b;
            });
            std::set<int64_t> seen;
            for (int64_t k = 0; k < c.k; k++) {
                size_t at = (o * c.k + k) * c.inner + n;
                auto value = read(y, at, c.dtype);
                auto index = idx[at];
                if (index < 0 || index >= c.d || !seen.insert(index).second || !same(value, gold[k]) ||
                    !same(value, read(x, (o * c.d + index) * c.inner + n, c.dtype))) {
                    std::cerr << "FAIL at=" << at << " value=" << value << " expected=" << gold[k] << " index=" << index
                              << "\n";
                    return 2;
                }
            }
        }
    for (size_t i = outbytes; i < y.size(); i++)
        if (y[i] != 0xa5) {
            std::cerr << "value output overrun\n";
            return 3;
        }
    auto ib = (uint8_t*)idx.data();
    for (size_t i = ibytes; i < ibytes + 64; i++)
        if (ib[i] != 0xa5) {
            std::cerr << "index output overrun\n";
            return 3;
        }
    ACL(aclrtFree(dx));
    ACL(aclrtFree(dy));
    ACL(aclrtFree(di));
    if (ws)
        ACL(aclrtFree(ws));
    return 0;
}
int main(int argc, char** argv)
{
    ACL(aclInit(nullptr));
    ACL(aclrtSetDevice(0));
    aclrtStream stream;
    ACL(aclrtCreateStream(&stream));
    auto p = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ub = 0;
    p->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    auto all = cases();
    int count = 0;
    for (size_t i = 0; i < all.size(); i++) {
        if (argc > 1 && i != size_t(std::atoi(argv[1])))
            continue;
        for (int pattern = 0; pattern < 3; pattern++) {
            if (test(all[i], pattern, stream, p->GetCoreNumAiv(), ub))
                return 2;
            count++;
        }
    }
    ACL(aclrtDestroyStream(stream));
    ACL(aclrtResetDevice(0));
    ACL(aclFinalize());
    std::cout << "PASS " << count
              << " TopK cases, exact CPU values, valid unique indices, NaN/Inf/ties and output canaries\n";
}
